import itertools
import time

import gymnasium.spaces
import numpy as np
from numpy.typing import NDArray
import torch as th
from torch.utils.tensorboard import SummaryWriter

from .callback import Callback
from .async_env.local_vec_env import LocalVecEnv
from .async_env.subprocess_vec_env import SubprocVecEnv
from .ppo.buffer import Buffer
from .ppo.ppo import PPO
from .ppo.running_mean_std import TensorRunningMeanStd

POLL_INTERVAL_SECONDS = 0.1
ROLLOUT_STALL_TIMEOUT_SECONDS = 30.0


class RolloutSampler:
    def __init__(self) -> None:
        self._last_obs: np.ndarray | None = None
        self._last_episode_starts: np.ndarray | None = None
        self._lstm_states: list[th.Tensor] | None = None
        self._available_indices: NDArray[np.int32] | None = None
        self._initialized_env_id: int | None = None
        self._initialized_num_envs: int | None = None
        self._initialized_obs_shape: tuple[int, ...] | None = None
        self._initialized_lstm_hidden_size: int | None = None

    def collect(
        self,
        env: LocalVecEnv | SubprocVecEnv,
        ppo: PPO,
        steps: int,
        callback: Callback,
        eps_greedy: float = 0.0,
        gae_lambda: float = 0.95,
        gamma: float = 0.99,
        normalize_rewards: bool = False,
        novelty_reward_scale: float = 0.0,
        summary_writer: SummaryWriter | None = None,
        hindsight_death_penalty: float = 0.0,
        hindsight_death_window: int = 10,
        hindsight_death_decay: float = 0.8,
    ) -> Buffer:
        start = time.time()

        buffer = self._sample_rollout(
            env,
            ppo,
            steps,
            callback=callback,
            eps_greedy=eps_greedy,
            gae_lambda=gae_lambda,
            gamma=gamma,
        )

        callback.on_rollout_sampling_end(raw_buffer=buffer)

        reward_normalizer: TensorRunningMeanStd | None = None
        if normalize_rewards:
            reward_norm_key = "reward_norm"
            if reward_norm_key not in ppo.meta.custom_data:
                ppo.meta.custom_data[reward_norm_key] = TensorRunningMeanStd(
                    shape=(), clip_lower=-10, clip_upper=10
                )
            reward_normalizer = ppo.meta.custom_data[reward_norm_key]

        finalize_start_time = time.time()
        buffer.finalize(
            ppo, reward_normalizer, novelty_reward_scale,
            hindsight_death_penalty=hindsight_death_penalty,
            hindsight_death_window=hindsight_death_window,
            hindsight_death_decay=hindsight_death_decay,
        )
        finalize_duration = time.time() - finalize_start_time
        rollout_length = time.time() - start
        total_rollout_steps = buffer.buffer_size * buffer.n_envs
        fps = total_rollout_steps / rollout_length

        ppo.meta.custom_data["last_rollout_metrics"] = {
            "rollout_time": float(rollout_length),
            "buffer_finalize_time": float(finalize_duration),
            "rollout_fps": float(fps),
            "rollout_steps": int(total_rollout_steps),
        }

        if summary_writer is not None:
            summary_writer.add_scalar(
                "rollout/time", rollout_length, ppo.meta.trained_steps
            )
            summary_writer.add_scalar("rollout/fps", fps, ppo.meta.trained_steps)

            summary_writer.add_scalar(
                "rollout/buffer_finalize_time",
                finalize_duration,
                ppo.meta.trained_steps,
            )

            summary_writer.add_scalar(
                "rollout/num_episode_starts",
                np.sum(buffer.episode_starts),
                ppo.meta.trained_steps,
            )
            summary_writer.add_scalar(
                "rollout/num_truncates",
                np.sum(buffer.truncates),
                ppo.meta.trained_steps,
            )

            summary_writer.add_scalar(
                "rollout/value_mean", np.mean(buffer.values), ppo.meta.trained_steps
            )
            summary_writer.add_scalar(
                "rollout/advantage_mean",
                np.mean(buffer.advantages),
                ppo.meta.trained_steps,
            )
            summary_writer.add_scalar(
                "rollout/return_mean", np.mean(buffer.returns), ppo.meta.trained_steps
            )
            summary_writer.add_scalar(
                "rollout/step_reward_mean",
                np.mean(buffer.rewards),
                ppo.meta.trained_steps,
            )

            episode_lengths = list(itertools.chain(*buffer.episode_lengths))
            if episode_lengths:
                summary_writer.add_scalar(
                    "rollout/len/min_episode_length",
                    np.min(episode_lengths),
                    ppo.meta.trained_steps,
                )
                summary_writer.add_scalar(
                    "rollout/len/max_episode_length",
                    np.max(episode_lengths),
                    ppo.meta.trained_steps,
                )
                summary_writer.add_scalar(
                    "rollout/len/mean_episode_length",
                    np.mean(episode_lengths),
                    ppo.meta.trained_steps,
                )
                summary_writer.add_scalar(
                    "rollout/len/std_episode_length",
                    np.std(episode_lengths),
                    ppo.meta.trained_steps,
                )

            episode_rewards = list(itertools.chain(*buffer.episode_rewards))
            if episode_rewards:
                summary_writer.add_scalar(
                    "rollout/reward/min_episode_reward",
                    np.min(episode_rewards),
                    ppo.meta.trained_steps,
                )
                summary_writer.add_scalar(
                    "rollout/reward/max_episode_reward",
                    np.max(episode_rewards),
                    ppo.meta.trained_steps,
                )
                summary_writer.add_scalar(
                    "rollout/reward/mean_episode_reward",
                    np.mean(episode_rewards),
                    ppo.meta.trained_steps,
                )
                summary_writer.add_scalar(
                    "rollout/reward/std_episode_reward",
                    np.std(episode_rewards),
                    ppo.meta.trained_steps,
                )

            summary_writer.add_scalar(
                "rollout/num_episodes", len(episode_lengths), ppo.meta.trained_steps
            )

            for action_idx in range(buffer.actions.shape[2]):
                action_data = buffer.actions[:, :, action_idx].flatten()
                summary_writer.add_histogram(
                    f"actions/action/{action_idx}", action_data, ppo.meta.trained_steps
                )

            mask_offset = 0
            for action_idx, n in enumerate(buffer.action_space.nvec):
                mask_data = buffer.action_masks[:, :, mask_offset : mask_offset + n]
                available_actions = np.where(mask_data.flatten() == 1)[0] % n
                summary_writer.add_histogram(
                    f"actions/mask/{action_idx}",
                    available_actions,
                    ppo.meta.trained_steps,
                )
                mask_offset += n

            if ppo.meta.trained_rollouts % 10 == 0:
                for obs_idx in range(buffer.observation_space.shape[-1]):
                    data = buffer.observations[:, :, -1, obs_idx]
                    summary_writer.add_histogram(
                        f"observations/{obs_idx}", data, ppo.meta.trained_steps
                    )

            summary_writer.add_scalar(
                "rollout/eps_greedy", eps_greedy, ppo.meta.trained_steps
            )
            summary_writer.add_scalar(
                "rollout/gae_lambda", gae_lambda, ppo.meta.trained_steps
            )
            summary_writer.add_scalar("rollout/gamma", gamma, ppo.meta.trained_steps)
            summary_writer.add_scalar("rollout/steps", steps, ppo.meta.trained_steps)
            summary_writer.add_scalar(
                "rollout/num_envs", buffer.n_envs, ppo.meta.trained_steps
            )

            if reward_normalizer is not None:
                summary_writer.add_scalar(
                    "rollout/running_reward_mean",
                    reward_normalizer.mean.item(),
                    ppo.meta.trained_steps,
                )
                summary_writer.add_scalar(
                    "rollout/running_reward_var",
                    reward_normalizer.var.item(),
                    ppo.meta.trained_steps,
                )
                summary_writer.add_scalar(
                    "rollout/running_reward_count",
                    reward_normalizer.count,
                    ppo.meta.trained_steps,
                )

                clip_count = np.count_nonzero(
                    (buffer.rewards == reward_normalizer.clip_upper)
                    | (buffer.rewards == reward_normalizer.clip_lower)
                )
                summary_writer.add_scalar(
                    "rollout/reward_clip_count", clip_count, ppo.meta.trained_steps
                )

            summary_writer.add_scalar(
                "rollout/novelty_reward_scale",
                novelty_reward_scale,
                ppo.meta.trained_steps,
            )
            if ppo.meta.trained_rollouts > 0:
                # Only add after 1 rollout, otherwise running observation stats won't be set
                # so novelties won't be accurate
                summary_writer.add_scalar(
                    "rollout/novelty_mean",
                    buffer.novelty.mean(),
                    ppo.meta.trained_steps,
                )
                summary_writer.add_scalar(
                    "rollout/novelty_std", buffer.novelty.std(), ppo.meta.trained_steps
                )
                summary_writer.add_scalar(
                    "rollout/novelty_min", buffer.novelty.min(), ppo.meta.trained_steps
                )
                summary_writer.add_scalar(
                    "rollout/novelty_max", buffer.novelty.max(), ppo.meta.trained_steps
                )
                summary_writer.add_histogram(
                    "rollout/novelty", buffer.novelty, ppo.meta.trained_steps
                )

        return buffer

    def _sample_rollout(
        self,
        env: LocalVecEnv | SubprocVecEnv,
        ppo: PPO,
        steps: int,
        callback: Callback,
        eps_greedy: float = 0.0,
        gae_lambda: float = 0.95,
        gamma: float = 0.99,
    ) -> Buffer:
        env_action_space = env.action_space
        env_observation_space = env.observation_space
        # We only support these space types
        assert isinstance(env_action_space, gymnasium.spaces.MultiDiscrete)
        assert isinstance(env_observation_space, gymnasium.spaces.Box)
        buffer = Buffer(
            buffer_size=steps,
            n_envs=env.num_envs,
            action_space=env_action_space,
            observation_space=env_observation_space,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        available_indices = self._prepare_rollout_state(env, ppo, buffer)
        assert self._last_obs is not None
        assert self._last_episode_starts is not None
        last_obs = self._last_obs
        last_episode_starts = self._last_episode_starts
        lstm_states = self._lstm_states
        last_progress_time = time.monotonic()

        while not buffer.is_full():
            if env.is_reset_waiting():
                indices, obs = env.poll_reset(wait=POLL_INTERVAL_SECONDS)
                last_obs[indices] = obs
                available_indices = np.concatenate((available_indices, indices))
                if len(indices) > 0:
                    last_progress_time = time.monotonic()

            if (
                len(available_indices) == 0
                and not env.is_reset_waiting()
                and not env.is_step_waiting()
            ):
                available_indices = np.arange(env.num_envs, dtype=np.int32)
                last_progress_time = time.monotonic()

            if len(available_indices) > 0:
                # Only step envs that still need buffer space — avoids wasting
                # CPU on envs already at buffer_size and prevents stale actions
                # from carrying over into the next rollout's buffer.
                needs_step = buffer.positions[available_indices] < buffer.buffer_size
                available_indices = available_indices[needs_step]

            if len(available_indices) > 0:
                action_masks = env.get_action_masks(indices=available_indices)

                state_for_predict: tuple[th.Tensor, th.Tensor] | None = None
                if lstm_states is not None:
                    state_for_predict = (
                        lstm_states[0][:, available_indices],
                        lstm_states[1][:, available_indices],
                    )

                actions, log_probs, _, values, _, new_lstm_state = ppo.predict(
                    th.as_tensor(last_obs[available_indices], device=ppo.device),
                    th.as_tensor(action_masks, device=ppo.device),
                    deterministic=0 < eps_greedy and eps_greedy > np.random.random(),
                    return_entropy=False,
                    return_actions=True,
                    return_values=True,
                    return_log_probs=True,
                    lstm_state=state_for_predict,
                )

                if lstm_states is not None and new_lstm_state is not None:
                    lstm_states[0][:, available_indices] = new_lstm_state[0]
                    lstm_states[1][:, available_indices] = new_lstm_state[1]

                assert actions is not None
                assert log_probs is not None
                assert values is not None
                np_actions = actions.cpu().numpy()
                env.step_async(np_actions, available_indices)
                buffer.add_step_request(
                    available_indices,
                    np_actions,
                    values.cpu().numpy(),
                    log_probs.cpu().numpy(),
                    action_masks,
                )
                available_indices = np.empty((0,), dtype=np.int32)

            indices, (obs, reward, done, truncated, info) = env.poll_step(
                wait=POLL_INTERVAL_SECONDS
            )
            if len(indices) > 0:
                buffer.add_step_response(
                    indices,
                    last_obs[indices],
                    reward,
                    last_episode_starts[indices],
                    truncated,
                    obs,
                    done,
                    info,
                )
                last_obs[indices] = obs
                last_episode_starts[indices] = done

                # Reset hidden state for environments that just finished an episode
                if lstm_states is not None:
                    done_envs = indices[done.astype(bool)]
                    if len(done_envs) > 0:
                        lstm_states[0][:, done_envs] = 0.0
                        lstm_states[1][:, done_envs] = 0.0

                available_indices = indices
                callback.on_step(indices, info)
                last_progress_time = time.monotonic()
                continue

            max_pending_duration = self._max_pending_duration_seconds(env)
            if (
                env.is_step_waiting() or env.is_reset_waiting()
            ) and max_pending_duration >= ROLLOUT_STALL_TIMEOUT_SECONDS:
                raise RuntimeError(
                    "RolloutSampler stalled waiting for async env progress. "
                    f"oldest_pending={max_pending_duration:.1f}s "
                    f"State={self._debug_env_state(env)}"
                )

            if (
                env.is_step_waiting() or env.is_reset_waiting()
            ) and time.monotonic() - last_progress_time >= ROLLOUT_STALL_TIMEOUT_SECONDS:
                raise RuntimeError(
                    "RolloutSampler made no global async progress. "
                    f"State={self._debug_env_state(env)}"
                )

        # Drain any pending steps so the next rollout starts clean — prevents
        # carryover results from being recorded without matching actions/values.
        while env.is_step_waiting():
            indices, (obs, _rew, done, _trunc, _info) = env.poll_step(
                wait=POLL_INTERVAL_SECONDS
            )
            if len(indices) > 0:
                last_obs[indices] = obs
                last_episode_starts[indices] = done
                if lstm_states is not None:
                    done_envs = indices[done.astype(bool)]
                    if len(done_envs) > 0:
                        lstm_states[0][:, done_envs] = 0.0
                        lstm_states[1][:, done_envs] = 0.0

        self._last_obs = last_obs
        self._last_episode_starts = last_episode_starts
        self._lstm_states = lstm_states
        self._available_indices = np.arange(env.num_envs, dtype=np.int32)
        return buffer

    def _prepare_rollout_state(
        self,
        env: LocalVecEnv | SubprocVecEnv,
        ppo: PPO,
        buffer: Buffer,
    ) -> NDArray[np.int32]:
        lstm_hidden_size = ppo._policy_params.lstm_hidden_size
        needs_reset = (
            self._initialized_env_id != id(env)
            or self._initialized_num_envs != env.num_envs
            or self._initialized_obs_shape != tuple(buffer.observation_space.shape)
            or self._initialized_lstm_hidden_size != lstm_hidden_size
            or self._last_obs is None
            or self._last_episode_starts is None
        )

        if needs_reset:
            env.reset_async()
            self._last_obs = np.zeros(
                (buffer.n_envs, *buffer.observation_space.shape), dtype=np.float32
            )
            self._last_episode_starts = np.ones(shape=(env.num_envs,), dtype=bool)
            self._lstm_states = self._init_lstm_states(
                env.num_envs, lstm_hidden_size, ppo.device
            )
            self._available_indices = np.empty((0,), dtype=np.int32)
            self._initialized_env_id = id(env)
            self._initialized_num_envs = env.num_envs
            self._initialized_obs_shape = tuple(buffer.observation_space.shape)
            self._initialized_lstm_hidden_size = lstm_hidden_size
            return np.empty((0,), dtype=np.int32)

        if self._lstm_states is None and lstm_hidden_size is not None:
            self._lstm_states = self._init_lstm_states(
                env.num_envs, lstm_hidden_size, ppo.device
            )
        if self._available_indices is None:
            self._available_indices = np.empty((0,), dtype=np.int32)
        return self._available_indices

    @staticmethod
    def _debug_env_state(env: LocalVecEnv | SubprocVecEnv) -> object:
        debug_state = getattr(env, "debug_state", None)
        if callable(debug_state):
            return debug_state()
        return {
            "num_envs": env.num_envs,
            "reset_waiting": env.is_reset_waiting(),
            "step_waiting": env.is_step_waiting(),
        }

    @staticmethod
    def _max_pending_duration_seconds(env: LocalVecEnv | SubprocVecEnv) -> float:
        duration_fn = getattr(env, "max_pending_duration_seconds", None)
        if callable(duration_fn):
            return float(duration_fn())
        return 0.0

    @staticmethod
    def _init_lstm_states(
        n_envs: int,
        lstm_hidden_size: int | None,
        device: str,
    ) -> list[th.Tensor] | None:
        if lstm_hidden_size is None:
            return None
        h = th.zeros(1, n_envs, lstm_hidden_size, device=device)
        c = th.zeros(1, n_envs, lstm_hidden_size, device=device)
        return [h, c]
