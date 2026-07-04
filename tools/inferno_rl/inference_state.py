from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch as th

from .ppo.ppo import PPO
from .training.actions import ensure_action_mask_for_policy


@dataclass
class StepPrediction:
    action: int | np.ndarray | None
    log_prob: float | None
    entropy: np.ndarray | None
    value: float | None
    probs: np.ndarray | None


class StatefulPolicyRunner:
    def __init__(self, ppo: PPO):
        self._ppo = ppo
        self._lstm_state: tuple[th.Tensor, th.Tensor] | None = None

    def reset(self) -> None:
        self._lstm_state = None

    def predict(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        deterministic: bool = True,
        *,
        return_actions: bool = True,
        return_log_probs: bool = False,
        return_entropy: bool = False,
        return_values: bool = False,
        return_probs: bool = False,
    ) -> StepPrediction:
        action_mask = ensure_action_mask_for_policy(
            action_mask,
            self._ppo.policy_params.action_head_sizes,
        )
        obs_t = th.as_tensor(obs, dtype=th.float32).unsqueeze(0).unsqueeze(0)
        mask_t = th.as_tensor(action_mask, dtype=th.bool).unsqueeze(0)
        actions, log_probs, entropy, values, probs, self._lstm_state = self._ppo.predict(
            obs_t,
            mask_t,
            deterministic=deterministic,
            return_actions=return_actions,
            return_log_probs=return_log_probs,
            return_entropy=return_entropy,
            return_values=return_values,
            return_probs=return_probs,
            lstm_state=self._lstm_state,
        )
        return StepPrediction(
            action=(
                actions.squeeze(0).detach().cpu().numpy()
                if actions is not None
                else None
            ),
            log_prob=float(log_probs.squeeze()) if log_probs is not None else None,
            entropy=entropy.squeeze(0).detach().cpu().numpy() if entropy is not None else None,
            value=float(values.squeeze()) if values is not None else None,
            probs=probs.squeeze(0).detach().cpu().numpy() if probs is not None else None,
        )
