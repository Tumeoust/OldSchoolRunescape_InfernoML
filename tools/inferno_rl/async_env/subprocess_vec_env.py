"""Subprocess-based vectorized environment for true CPU parallelism.

Replaces LocalVecEnv's asyncio-based sequential stepping with N independent
OS processes. Each process runs one InfernoEnv; results arrive in parallel,
so the main process receives a full batch of N results simultaneously instead
of one at a time.

Uses shared memory for bulk data (obs, reward, done, trunc, masks) to avoid
pickle serialization overhead. Pipes carry only a completion signal + the
info dict (small, infrequent on episode boundaries).

Interface mirrors LocalVecEnv so RolloutSampler works unchanged.
"""

from __future__ import annotations

import multiprocessing as mp
import time
import traceback
from multiprocessing.connection import Connection
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Iterable

import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from ..training.actions import ACTION_HEAD_SIZES, POLICY_ACTION_MASK_SIZE
from ..training.env import InfernoEnv
from ..training.rewards import RewardConfig
from ..training.observation import (
    get_observation_low,
    get_observation_size,
)

VecEnvIndices = None | int | Iterable[int]
VecEnvStepReturn = tuple[
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.bool_],
    NDArray[np.bool_],
    NDArray[np.object_],
]

_F32_BYTES = np.dtype(np.float32).itemsize  # 4

# Windows WaitForMultipleObjects supports at most 63 handles.
_WIN_WAIT_LIMIT = 61


def _batched_wait(
    connections: list[Connection], timeout: float | None,
) -> list[Connection]:
    """``mp.connection.wait`` wrapper that handles >63 connections on Windows."""
    if len(connections) <= _WIN_WAIT_LIMIT:
        return mp.connection.wait(connections, timeout)

    # Fast non-blocking sweep across all batches.
    ready: list[Connection] = []
    for start in range(0, len(connections), _WIN_WAIT_LIMIT):
        batch = connections[start : start + _WIN_WAIT_LIMIT]
        ready.extend(mp.connection.wait(batch, timeout=0))
    if ready or timeout == 0:
        return ready

    # Nothing ready — distribute remaining timeout across batches.
    n_batches = (len(connections) + _WIN_WAIT_LIMIT - 1) // _WIN_WAIT_LIMIT
    per_batch = timeout / n_batches if timeout is not None else None
    for start in range(0, len(connections), _WIN_WAIT_LIMIT):
        batch = connections[start : start + _WIN_WAIT_LIMIT]
        ready.extend(mp.connection.wait(batch, timeout=per_batch))
        if ready:
            return ready
    return ready


# Module-level function so it is picklable under Windows spawn context.
def _worker(
    conn: Connection,
    env_idx: int,
    shm_names: dict[str, str],
    start_wave: int,
    max_wave: int,
    start_wave_weights: dict[int, float] | None,
    reset_options: dict[str, Any],
    promote_after: int,
    refresh_every: int,
    min_waves_to_advance: int,
    phase: str | None = None,
    climb_sampling: str = "weighted",
    observation_version: str = "v4",
    record_reward_terms: bool = False,
    max_drill_retries: int = 10,
    episode_mode: str = "full",
    opener_tick_limit: int = 50,
    opener_min_health: int = 40,
    fixed_loadout: str | None = None,
    loadout_weights: dict[str, float] | None = None,
    sweep_death_retries: int = 0,
) -> None:
    """Worker process: runs one InfernoEnv and communicates via pipe + shared memory."""
    obs_size = get_observation_size()
    obs_slot_bytes = obs_size * np.dtype(np.float32).itemsize

    # Attach to shared memory created by parent
    shm_obs = SharedMemory(name=shm_names["obs"], create=False)
    shm_rewards = SharedMemory(name=shm_names["rewards"], create=False)
    shm_dones = SharedMemory(name=shm_names["dones"], create=False)
    shm_truncs = SharedMemory(name=shm_names["truncs"], create=False)
    shm_masks = SharedMemory(name=shm_names["masks"], create=False)

    # Numpy views for this worker's slot
    obs_view = np.ndarray(
        (1, obs_size), dtype=np.float32,
        buffer=shm_obs.buf, offset=env_idx * obs_slot_bytes,
    )
    rew_view = np.ndarray(
        (1,), dtype=np.float32,
        buffer=shm_rewards.buf, offset=env_idx * _F32_BYTES,
    )
    done_view = np.ndarray(
        (1,), dtype=np.bool_,
        buffer=shm_dones.buf, offset=env_idx,
    )
    trunc_view = np.ndarray(
        (1,), dtype=np.bool_,
        buffer=shm_truncs.buf, offset=env_idx,
    )
    mask_view = np.ndarray(
        (POLICY_ACTION_MASK_SIZE,), dtype=np.bool_,
        buffer=shm_masks.buf, offset=env_idx * POLICY_ACTION_MASK_SIZE,
    )

    try:
        env = InfernoEnv(
            start_wave=start_wave,
            max_wave=max_wave,
            start_wave_weights=start_wave_weights,
            promote_after=promote_after,
            refresh_every=refresh_every,
            min_waves_to_advance=min_waves_to_advance,
            phase=phase,
            climb_sampling=climb_sampling,
            observation_version=observation_version,
            record_reward_terms=record_reward_terms,
            max_drill_retries=max_drill_retries,
            episode_mode=episode_mode,
            opener_tick_limit=opener_tick_limit,
            opener_min_health=opener_min_health,
            fixed_loadout=fixed_loadout,
            loadout_weights=loadout_weights,
            sweep_death_retries=sweep_death_retries,
        )
        while True:
            cmd, data = conn.recv()
            if cmd == "reset":
                obs, info = env.reset(options=data)
                masks = info.get(
                    "action_mask", np.ones(POLICY_ACTION_MASK_SIZE, dtype=bool)
                )
                if "action_mask" in info:
                    info = dict(info)
                    info.pop("action_mask", None)
                obs_view[:] = obs.reshape(1, -1)
                mask_view[:] = masks
                conn.send(("reset", info))
            elif cmd == "step":
                obs, rew, done, trunc, info = env.step(data)
                obs = obs.reshape(1, -1)
                masks = info.get(
                    "action_mask", np.ones(POLICY_ACTION_MASK_SIZE, dtype=bool)
                )
                if "action_mask" in info:
                    info = dict(info)
                    info.pop("action_mask", None)
                if done or trunc:
                    info["terminal_observation"] = obs
                    obs2, reset_info = env.reset(options=reset_options)
                    obs = obs2.reshape(1, -1)
                    masks = reset_info.get(
                        "action_mask", np.ones(POLICY_ACTION_MASK_SIZE, dtype=bool)
                    )
                    if "action_mask" in reset_info:
                        reset_info = dict(reset_info)
                        reset_info.pop("action_mask", None)
                    info["reset_info"] = reset_info
                obs_view[:] = obs
                rew_view[0] = rew
                done_view[0] = done
                trunc_view[0] = trunc
                mask_view[:] = masks
                conn.send(("step", info))
            elif cmd == "prestige":
                env.apply_prestige(data)
                conn.send(("prestige", None))
            elif cmd == "sync_curriculum":
                payload = data or {}
                env.sync_curriculum(
                    frontier_wave=int(payload["frontier_wave"]),
                    min_waves_to_advance=payload.get("min_waves_to_advance"),
                )
                conn.send(("sync_curriculum", None))
            elif cmd == "reconfigure":
                payload = data or {}
                env.set_phase(payload.get("phase"))
                env.set_episode_mode(payload.get("episode_mode", "full"))
                env.set_opener_config(
                    int(payload.get("opener_tick_limit", 50)),
                    int(payload.get("opener_min_health", 40)),
                )
                wave_stats = payload.get("wave_stats")
                if wave_stats is not None:
                    env.set_wave_stats(wave_stats)
                reward_config = payload.get("reward_config")
                if reward_config is not None:
                    env.set_reward_config(RewardConfig(**reward_config))
                conn.send(("reconfigure", None))
            elif cmd == "close":
                return
    except EOFError:
        return
    except Exception as exc:
        error_payload = {
            "env_idx": env_idx,
            "error_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        try:
            conn.send(("error", error_payload))
        except Exception:
            pass
        raise
    finally:
        for shm in [shm_obs, shm_rewards, shm_dones, shm_truncs, shm_masks]:
            shm.close()
        conn.close()


class SubprocVecEnv:
    """
    Vectorized env using N subprocesses for true CPU parallelism.

    Each call to step_async dispatches N actions to N independent processes.
    poll_step uses multiprocessing.connection.wait to collect whichever
    results are ready — same non-blocking semantics as LocalVecEnv's asyncio
    version, but with actual parallel execution.

    Bulk data (obs, reward, done, trunc, masks) is transferred via shared
    memory to avoid pickle serialization overhead. Pipes carry only completion
    signals and the info dict.
    """

    def __init__(
        self,
        n_envs: int,
        start_wave: int = 1,
        max_wave: int = 66,
        start_wave_weights: dict[int, float] | None = None,
        reset_options: dict[str, Any] | None = None,
        promote_after: int = 0,
        refresh_every: int = 10,
        min_waves_to_advance: int = 1,
        phase: str | None = None,
        climb_sampling: str = "weighted",
        observation_version: str = "v4",
        record_reward_terms: bool = False,
        max_drill_retries: int = 10,
        episode_mode: str = "full",
        opener_tick_limit: int = 50,
        opener_min_health: int = 40,
        fixed_loadout: str | None = None,
        loadout_weights: dict[str, float] | None = None,
        sweep_death_retries: int = 0,
    ) -> None:
        if reset_options is None:
            reset_options = {}

        self._observation_size = get_observation_size()
        self._observation_low = get_observation_low()
        self._obs_slot_bytes = self._observation_size * np.dtype(np.float32).itemsize

        # Create shared memory blocks
        self._shm_obs = SharedMemory(create=True, size=n_envs * self._obs_slot_bytes)
        self._shm_rewards = SharedMemory(create=True, size=n_envs * _F32_BYTES)
        self._shm_dones = SharedMemory(create=True, size=n_envs)
        self._shm_truncs = SharedMemory(create=True, size=n_envs)
        self._shm_masks = SharedMemory(
            create=True, size=n_envs * POLICY_ACTION_MASK_SIZE
        )

        shm_names = {
            "obs": self._shm_obs.name,
            "rewards": self._shm_rewards.name,
            "dones": self._shm_dones.name,
            "truncs": self._shm_truncs.name,
            "masks": self._shm_masks.name,
        }

        # Numpy views on shared memory (zero-copy reads in main process)
        self._obs_buf = np.ndarray(
            (n_envs, 1, self._observation_size), dtype=np.float32,
            buffer=self._shm_obs.buf,
        )
        self._rew_buf = np.ndarray(
            (n_envs,), dtype=np.float32,
            buffer=self._shm_rewards.buf,
        )
        self._done_buf = np.ndarray(
            (n_envs,), dtype=np.bool_,
            buffer=self._shm_dones.buf,
        )
        self._trunc_buf = np.ndarray(
            (n_envs,), dtype=np.bool_,
            buffer=self._shm_truncs.buf,
        )
        self._mask_buf = np.ndarray(
            (n_envs, POLICY_ACTION_MASK_SIZE), dtype=np.bool_,
            buffer=self._shm_masks.buf,
        )

        ctx = mp.get_context("spawn")
        self._conns: list[Connection] = []
        self._processes: list[mp.Process] = []

        for i in range(n_envs):
            parent_conn, child_conn = ctx.Pipe()
            p = ctx.Process(
                target=_worker,
                args=(child_conn, i, shm_names, start_wave, max_wave,
                      start_wave_weights, reset_options,
                      promote_after, refresh_every, min_waves_to_advance, phase,
                      climb_sampling, observation_version,
                      record_reward_terms, max_drill_retries,
                      episode_mode, opener_tick_limit, opener_min_health,
                      fixed_loadout, loadout_weights, sweep_death_retries),
                daemon=True,
            )
            p.start()
            child_conn.close()  # child end not needed in parent process
            self._conns.append(parent_conn)
            self._processes.append(p)

        self.num_envs = n_envs
        self._reset_options = reset_options
        self._pending_reset: set[int] = set()
        self._pending_step: set[int] = set()
        self._pending_reset_started: dict[int, float] = {}
        self._pending_step_started: dict[int, float] = {}
        self._cached_masks = np.ones((n_envs, POLICY_ACTION_MASK_SIZE), dtype=bool)
        self._is_closed = False

        self.action_space = spaces.MultiDiscrete(ACTION_HEAD_SIZES)
        self.observation_space = spaces.Box(
            low=self._observation_low,
            high=1.0,
            shape=(1, self._observation_size),
            dtype=np.float32,
        )

    # ── reset ─────────────────────────────────────────────────────────────────

    def reset_async(self, indices: VecEnvIndices = None) -> None:
        for idx in self._resolve_indices(indices):
            assert idx not in self._pending_reset, f"Env {idx} already awaiting reset"
            # Drain any leftover step result from the previous rollout (buffer
            # can fill up while some env steps are still in-flight).
            if idx in self._pending_step:
                self._recv_or_raise(idx)  # discard stale result
                self._pending_step.discard(idx)
                self._pending_step_started.pop(idx, None)
            self._conns[idx].send(("reset", self._reset_options))
            self._pending_reset.add(idx)
            self._pending_reset_started[idx] = time.monotonic()

    def is_reset_waiting(self) -> bool:
        return bool(self._pending_reset)

    def is_step_waiting(self) -> bool:
        return bool(self._pending_step)

    def poll_reset(
        self, wait: float | None = None
    ) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
        if not self._pending_reset:
            return (
                np.empty((0,), dtype=np.int32),
                np.empty((0, 1, self._observation_size), dtype=np.float32),
            )

        pending_conns = [self._conns[i] for i in self._pending_reset]
        conn_to_idx = {id(self._conns[i]): i for i in self._pending_reset}

        ready = _batched_wait(pending_conns, timeout=wait)
        if not ready:
            self._raise_if_worker_failed()
            return (
                np.empty((0,), dtype=np.int32),
                np.empty((0, 1, self._observation_size), dtype=np.float32),
            )

        completed_indices = []
        for conn in ready:
            idx = conn_to_idx[id(conn)]
            self._recv_or_raise(idx)  # consume signal (obs/masks already in shared memory)
            completed_indices.append(idx)
            self._pending_reset.discard(idx)
            self._pending_reset_started.pop(idx, None)

        idxs = np.array(completed_indices, dtype=np.int32)
        self._cached_masks[idxs] = self._mask_buf[idxs]
        return idxs, self._obs_buf[idxs].copy()

    # ── step ──────────────────────────────────────────────────────────────────

    def step_async(
        self, actions: NDArray[np.int32], indices: VecEnvIndices = None
    ) -> None:
        resolved = self._resolve_indices(indices)
        for action_pos, idx in enumerate(resolved):
            assert idx not in self._pending_step, f"Env {idx} already awaiting step"
            assert idx not in self._pending_reset, f"Env {idx} awaiting reset"
            self._conns[idx].send(("step", np.asarray(actions[action_pos], dtype=np.int32)))
            self._pending_step.add(idx)
            self._pending_step_started[idx] = time.monotonic()

    def poll_step(
        self, wait: float | None = None
    ) -> tuple[NDArray[np.int32], VecEnvStepReturn]:
        if not self._pending_step:
            return self._empty_step()

        pending_conns = [self._conns[i] for i in self._pending_step]
        conn_to_idx = {id(self._conns[i]): i for i in self._pending_step}

        ready = _batched_wait(pending_conns, timeout=wait)
        if not ready:
            self._raise_if_worker_failed()
            return self._empty_step()

        completed_indices: list[int] = []
        info_list = []
        for conn in ready:
            idx = conn_to_idx[id(conn)]
            _cmd, info = self._recv_or_raise(idx)
            completed_indices.append(idx)
            info_list.append(info)
            self._pending_step.discard(idx)
            self._pending_step_started.pop(idx, None)

        idxs = np.array(completed_indices, dtype=np.int32)
        self._cached_masks[idxs] = self._mask_buf[idxs]
        return idxs, (
            self._obs_buf[idxs].copy(),
            self._rew_buf[idxs].copy(),
            self._done_buf[idxs].copy(),
            self._trunc_buf[idxs].copy(),
            np.array(info_list, dtype=object),
        )

    # ── misc ──────────────────────────────────────────────────────────────────

    def get_action_masks(self, indices: VecEnvIndices = None) -> NDArray[np.bool_]:
        return self._cached_masks[self._resolve_indices(indices)]

    def debug_state(self) -> dict[str, Any]:
        now = time.monotonic()
        return {
            "num_envs": self.num_envs,
            "pending_reset": sorted(self._pending_reset),
            "pending_step": sorted(self._pending_step),
            "pending_reset_age_seconds": {
                idx: round(now - started, 3)
                for idx, started in self._pending_reset_started.items()
            },
            "pending_step_age_seconds": {
                idx: round(now - started, 3)
                for idx, started in self._pending_step_started.items()
            },
            "process_alive": [process.is_alive() for process in self._processes],
            "process_exitcode": [process.exitcode for process in self._processes],
        }

    def max_pending_duration_seconds(self) -> float:
        now = time.monotonic()
        started_times = [
            *self._pending_reset_started.values(),
            *self._pending_step_started.values(),
        ]
        if not started_times:
            return 0.0
        return max(now - started for started in started_times)

    def apply_prestige(self, min_waves_to_advance: int) -> None:
        """Broadcast prestige to all workers: reset frontier, set min_waves.

        Must be called when no collection is in progress. Drains any leftover
        pending steps/resets before sending the command.
        """
        # Drain any pending operations so worker pipes are clean
        while self._pending_step:
            self.poll_step()
        while self._pending_reset:
            self.poll_reset()
        for conn in self._conns:
            conn.send(("prestige", min_waves_to_advance))
        for conn in self._conns:
            self._recv_command_or_raise(conn, "prestige")

    def sync_curriculum(self, frontier_wave: int, min_waves_to_advance: int) -> None:
        """Broadcast the highest observed climb frontier to all workers."""
        while self._pending_step:
            self.poll_step()
        while self._pending_reset:
            self.poll_reset()
        payload = {
            "frontier_wave": int(frontier_wave),
            "min_waves_to_advance": int(min_waves_to_advance),
        }
        for conn in self._conns:
            conn.send(("sync_curriculum", payload))
        for conn in self._conns:
            self._recv_command_or_raise(conn, "sync_curriculum")

    def reconfigure(
        self,
        *,
        phase: str | None,
        episode_mode: str,
        opener_tick_limit: int,
        opener_min_health: int,
        wave_stats: dict[int, dict[str, int]] | None,
        reward_config: dict[str, float] | None = None,
    ) -> None:
        """Broadcast runtime phase/episode-mode changes to all workers.

        Must be called when no collection is in progress. Drains any pending
        operations before sending the command.
        """
        while self._pending_step:
            self.poll_step()
        while self._pending_reset:
            self.poll_reset()
        payload = {
            "phase": phase,
            "episode_mode": episode_mode,
            "opener_tick_limit": opener_tick_limit,
            "opener_min_health": opener_min_health,
            "wave_stats": wave_stats,
            "reward_config": reward_config,
        }
        for conn in self._conns:
            conn.send(("reconfigure", payload))
        for conn in self._conns:
            self._recv_command_or_raise(conn, "reconfigure")

    def close(self) -> None:
        if self._is_closed:
            return
        self._is_closed = True
        for conn in self._conns:
            try:
                conn.send(("close", None))
            except Exception:
                pass
        for p in self._processes:
            p.join(timeout=5)
            if p.is_alive():
                p.kill()
        for conn in self._conns:
            conn.close()
        # Unlink shared memory (parent owns lifecycle)
        for shm in [self._shm_obs, self._shm_rewards, self._shm_dones,
                     self._shm_truncs, self._shm_masks]:
            shm.close()
            shm.unlink()

    def _resolve_indices(self, indices: VecEnvIndices) -> list[int]:
        if indices is None:
            return list(range(self.num_envs))
        if isinstance(indices, (int, np.integer)):
            return [int(indices)]
        return [int(i) for i in indices]

    def _empty_step(self) -> tuple[NDArray[np.int32], VecEnvStepReturn]:
        return (
            np.empty((0,), dtype=np.int32),
            (
                np.empty((0, 1, self._observation_size), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=bool),
                np.empty((0,), dtype=bool),
                np.empty((0,), dtype=object),
            ),
        )

    def _recv_or_raise(self, idx: int) -> tuple[str, Any]:
        try:
            cmd, payload = self._conns[idx].recv()
        except EOFError as exc:
            raise RuntimeError(
                f"SubprocVecEnv worker {idx} pipe closed. State={self.debug_state()}"
            ) from exc

        if cmd == "error":
            raise RuntimeError(
                "SubprocVecEnv worker "
                f"{payload.get('env_idx', idx)} crashed with "
                f"{payload.get('error_type', 'Exception')}: {payload.get('message', '')}\n"
                f"{payload.get('traceback', '')}"
            )
        return cmd, payload

    def _recv_command_or_raise(self, conn: Connection, expected_cmd: str) -> Any:
        idx = self._conns.index(conn)
        cmd, payload = self._recv_or_raise(idx)
        if cmd != expected_cmd:
            raise RuntimeError(
                f"Unexpected worker reply for env {idx}: expected {expected_cmd}, got {cmd}"
            )
        return payload

    def _raise_if_worker_failed(self) -> None:
        failed = [
            (idx, process.exitcode)
            for idx, process in enumerate(self._processes)
            if process.exitcode not in (None, 0)
        ]
        if failed:
            raise RuntimeError(
                f"SubprocVecEnv worker exited unexpectedly: {failed}. State={self.debug_state()}"
            )
