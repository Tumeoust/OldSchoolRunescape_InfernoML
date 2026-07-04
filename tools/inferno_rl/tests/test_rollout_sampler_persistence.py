import numpy as np
import torch as th
from gymnasium import spaces

from tools.inferno_rl.callback import Callback
from tools.inferno_rl.rollout_sampler import RolloutSampler


class _FakeMeta:
    def __init__(self) -> None:
        self.trained_steps = 0
        self.trained_rollouts = 0
        self.custom_data = {}
        self.normalized_observations = False


class _FakePolicyParams:
    lstm_hidden_size = None


class _FakePPO:
    def __init__(self) -> None:
        self.device = "cpu"
        self.meta = _FakeMeta()
        self._policy_params = _FakePolicyParams()

    def predict(self, obs, action_masks, **kwargs):
        batch = obs.shape[0]
        actions = th.zeros((batch, 1), dtype=th.int32)
        log_probs = th.zeros((batch,), dtype=th.float32)
        entropy = None
        values = th.zeros((batch,), dtype=th.float32)
        probs = None
        new_lstm_state = None
        return actions, log_probs, entropy, values, probs, new_lstm_state


class _FakeVecEnv:
    def __init__(self) -> None:
        self.num_envs = 1
        self.action_space = spaces.MultiDiscrete([1])
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1, 1),
            dtype=np.float32,
        )
        self.reset_calls = 0
        self._pending_reset = False
        self._pending_step = False
        self._tick = 0
        self._pending_age = 0.0

    def reset_async(self, indices=None) -> None:
        self.reset_calls += 1
        self._pending_reset = True
        self._tick = 0

    def is_reset_waiting(self) -> bool:
        return self._pending_reset

    def is_step_waiting(self) -> bool:
        return self._pending_step

    def poll_reset(self, wait=None):
        if not self._pending_reset:
            return np.empty((0,), dtype=np.int32), np.empty((0, 1, 1), dtype=np.float32)
        self._pending_reset = False
        obs = np.zeros((1, 1, 1), dtype=np.float32)
        return np.array([0], dtype=np.int32), obs

    def get_action_masks(self, indices=None):
        return np.ones((1, 1), dtype=bool)

    def step_async(self, actions, indices=None) -> None:
        self._pending_step = True
        self._pending_age = 0.0

    def poll_step(self, wait=None):
        if not self._pending_step:
            empty = (
                np.empty((0, 1, 1), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=bool),
                np.empty((0,), dtype=bool),
                np.empty((0,), dtype=object),
            )
            return np.empty((0,), dtype=np.int32), empty
        self._pending_step = False
        self._pending_age = 0.0
        self._tick += 1
        obs = np.array([[[self._tick]]], dtype=np.float32)
        reward = np.array([0.0], dtype=np.float32)
        done = np.array([False], dtype=bool)
        truncated = np.array([False], dtype=bool)
        info = np.array([{}], dtype=object)
        return np.array([0], dtype=np.int32), (obs, reward, done, truncated, info)

    def max_pending_duration_seconds(self) -> float:
        return self._pending_age

    def debug_state(self):
        return {
            "pending_reset": [0] if self._pending_reset else [],
            "pending_step": [0] if self._pending_step else [],
        }


class _PartiallyStalledVecEnv:
    def __init__(self) -> None:
        self.num_envs = 2
        self.action_space = spaces.MultiDiscrete([1])
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1, 1),
            dtype=np.float32,
        )
        self._pending_reset = False
        self._pending_step: set[int] = set()
        self._step_ages = {0: 0.0, 1: 0.0}
        self._tick = 0

    def reset_async(self, indices=None) -> None:
        self._pending_reset = True

    def is_reset_waiting(self) -> bool:
        return self._pending_reset

    def poll_reset(self, wait=None):
        if not self._pending_reset:
            return np.empty((0,), dtype=np.int32), np.empty((0, 1, 1), dtype=np.float32)
        self._pending_reset = False
        obs = np.zeros((2, 1, 1), dtype=np.float32)
        return np.array([0, 1], dtype=np.int32), obs

    def get_action_masks(self, indices=None):
        count = self.num_envs if indices is None else len(indices)
        return np.ones((count, 1), dtype=bool)

    def step_async(self, actions, indices=None) -> None:
        assert indices is not None
        for idx in indices:
            self._pending_step.add(int(idx))
            self._step_ages[int(idx)] = 0.0

    def is_step_waiting(self) -> bool:
        return bool(self._pending_step)

    def poll_step(self, wait=None):
        if 0 in self._pending_step:
            self._step_ages[0] += 15.0
            self._pending_step.remove(0)
            self._step_ages[0] = 0.0
            self._tick += 1
            obs = np.array([[[self._tick]]], dtype=np.float32)
            reward = np.array([0.0], dtype=np.float32)
            done = np.array([False], dtype=bool)
            truncated = np.array([False], dtype=bool)
            info = np.array([{}], dtype=object)
            return np.array([0], dtype=np.int32), (obs, reward, done, truncated, info)
        if 1 in self._pending_step:
            self._step_ages[1] += 15.0
        empty = (
            np.empty((0, 1, 1), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=bool),
            np.empty((0,), dtype=bool),
            np.empty((0,), dtype=object),
        )
        return np.empty((0,), dtype=np.int32), empty

    def max_pending_duration_seconds(self) -> float:
        pending_ages = [self._step_ages[idx] for idx in self._pending_step]
        return max(pending_ages, default=0.0)

    def debug_state(self):
        return {
            "pending_reset": [0, 1] if self._pending_reset else [],
            "pending_step": sorted(self._pending_step),
            "pending_step_age_seconds": {
                idx: self._step_ages[idx] for idx in self._pending_step
            },
        }


def test_rollout_sampler_does_not_hard_reset_between_rollouts() -> None:
    sampler = RolloutSampler()
    env = _FakeVecEnv()
    ppo = _FakePPO()
    callback = Callback()

    first = sampler.collect(env, ppo, steps=2, callback=callback)
    second = sampler.collect(env, ppo, steps=2, callback=callback)

    assert env.reset_calls == 1
    assert first.episode_starts[0, 0]
    assert not second.episode_starts[0, 0]


def test_rollout_sampler_raises_when_one_env_stalls() -> None:
    sampler = RolloutSampler()
    env = _PartiallyStalledVecEnv()
    ppo = _FakePPO()
    callback = Callback()

    try:
        sampler.collect(env, ppo, steps=2, callback=callback)
    except RuntimeError as exc:
        assert "oldest_pending" in str(exc)
        assert "pending_step_age_seconds" in str(exc)
    else:
        raise AssertionError("Expected stalled env to raise RuntimeError")
