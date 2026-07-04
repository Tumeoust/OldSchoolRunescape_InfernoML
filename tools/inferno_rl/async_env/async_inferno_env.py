from asyncio import AbstractEventLoop
from typing import Any, Optional

import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from .async_io_env import AsyncIoEnv
from ..training.env import InfernoEnv
from ..training.actions import ACTION_HEAD_SIZES, POLICY_ACTION_MASK_SIZE
from ..training.observation import (
    get_observation_low,
    get_observation_size,
)


class AsyncInfernoEnv(AsyncIoEnv[NDArray[np.float32], NDArray[np.int32]]):
    """
    Async adapter wrapping InfernoEnv for use with LocalVecEnv and the custom PPO stack.

    Obs shape: (obs_size,) → (1, obs_size)  (adds sequence dim required by Policy.forward's x.dim()==3)
    Action shape: np.array([action_idx]) → int scalar passed to InfernoEnv.step()
    action_space: MultiDiscrete(action heads)
    observation_space: Box(shape=(1, 262))
    """

    def __init__(
        self,
        start_wave: int = 1,
        max_wave: int = 66,
        start_wave_weights: Optional[dict[int, float]] = None,
        observation_version: str = "v4",
        record_reward_terms: bool = False,
        loop: AbstractEventLoop | None = None,
    ):
        super().__init__(loop=loop)
        obs_size = get_observation_size()
        self._env = InfernoEnv(
            start_wave=start_wave,
            max_wave=max_wave,
            start_wave_weights=start_wave_weights,
            observation_version=observation_version,
            record_reward_terms=record_reward_terms,
        )
        self._cached_masks: NDArray[np.bool_] = np.ones(POLICY_ACTION_MASK_SIZE, dtype=bool)

        self.action_space = spaces.MultiDiscrete(ACTION_HEAD_SIZES)
        self.observation_space = spaces.Box(
            low=get_observation_low(),
            high=1.0,
            shape=(1, obs_size),
            dtype=np.float32,
        )

    async def reset_async(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        obs, info = self._env.reset(seed=seed, options=options)
        self._cached_masks = info.get(
            "action_mask", np.ones(POLICY_ACTION_MASK_SIZE, dtype=bool)
        )
        return obs.reshape(1, -1), info

    async def step_async(
        self, action: NDArray[np.int32]
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self._env.step(int(action[0]))
        self._cached_masks = info.get(
            "action_mask", np.ones(POLICY_ACTION_MASK_SIZE, dtype=bool)
        )
        return obs.reshape(1, -1), reward, terminated, truncated, info

    async def close_async(self) -> None:
        self._env.close()

    def get_action_masks(self) -> NDArray[np.bool_]:
        return self._cached_masks
