from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from torch.utils.tensorboard import SummaryWriter

if TYPE_CHECKING:
    from .ppo.buffer import Buffer
    from .ppo.ppo import PPO, Meta


class Callback:
    def __init__(self) -> None:
        self._summary_writer: SummaryWriter | None = None
        self._ppo: "PPO | None" = None

    def initialize(self, summary_writer: SummaryWriter | None, ppo: "PPO") -> None:
        self._summary_writer = summary_writer
        self._ppo = ppo

    def on_training_start(self) -> None:
        pass

    def on_training_end(self) -> None:
        pass

    def on_rollout_start(self) -> None:
        pass

    def on_step(self, indices: NDArray[np.int32], infos: NDArray[np.object_]) -> None:
        pass

    def on_rollout_sampling_end(self, raw_buffer: "Buffer") -> None:
        pass

    def on_rollout_end(self, buffer: "Buffer") -> None:
        pass

    def on_learn_end(self) -> None:
        pass
