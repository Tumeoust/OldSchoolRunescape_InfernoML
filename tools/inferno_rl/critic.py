"""
Lightweight helper to query the critic (value function) from a loaded PPO checkpoint.
"""

import numpy as np
import torch as th

from .ppo.ppo import PPO
from .training.actions import (
    decode_policy_action,
    ensure_action_mask_for_policy,
    policy_action_mask_to_legacy_mask,
    policy_action_probabilities_to_legacy,
    uses_factored_policy_actions,
)


def get_value(ppo: PPO, obs: np.ndarray, action_mask: np.ndarray) -> float:
    """Get critic value estimate for a single observation.

    Args:
        ppo: Loaded PPO instance (non-trainable is fine).
        obs: Observation array of shape (obs_size,).
        action_mask: Action mask of shape (num_actions,).

    Returns:
        Scalar value estimate.
    """
    coerced_mask = ensure_action_mask_for_policy(
        action_mask,
        ppo.policy_params.action_head_sizes,
    )
    obs_t = th.as_tensor(obs, dtype=th.float32).unsqueeze(0).unsqueeze(0)
    mask_t = th.as_tensor(coerced_mask, dtype=th.bool).unsqueeze(0)
    _, _, _, values, *_ = ppo.predict(
        obs_t, mask_t, deterministic=True,
        return_actions=False, return_log_probs=False, return_entropy=False,
        return_values=True,
    )
    return float(values.squeeze())


def get_value_batch(ppo: PPO, obs_batch: np.ndarray,
                    mask_batch: np.ndarray) -> np.ndarray:
    """Get critic value estimates for a batch of observations.

    Args:
        ppo: Loaded PPO instance.
        obs_batch: Observations of shape (N, obs_size).
        mask_batch: Action masks of shape (N, num_actions).

    Returns:
        Value estimates of shape (N,).
    """
    coerced_masks = ensure_action_mask_for_policy(
        mask_batch,
        ppo.policy_params.action_head_sizes,
    )
    obs_t = th.as_tensor(obs_batch, dtype=th.float32).unsqueeze(1)
    mask_t = th.as_tensor(coerced_masks, dtype=th.bool)
    _, _, _, values, *_ = ppo.predict(
        obs_t, mask_t, deterministic=True,
        return_actions=False, return_log_probs=False, return_entropy=False,
        return_values=True,
    )
    return values.squeeze(-1).cpu().numpy()


def get_action_and_value(ppo: PPO, obs: np.ndarray,
                         action_mask: np.ndarray) -> tuple[int, float, list[tuple[int, str, float]]]:
    """Get action, value, and action logits for a single observation.

    Args:
        ppo: Loaded PPO instance.
        obs: Observation array of shape (obs_size,).
        action_mask: Action mask of shape (num_actions,).

    Returns:
        (action_index, value, top_actions) where top_actions is a list of
        (index, name, logit) tuples sorted by logit descending.
    """
    from .eval import _action_name

    uses_factored_actions = uses_factored_policy_actions(ppo.policy_params.action_head_sizes)
    coerced_mask = ensure_action_mask_for_policy(
        action_mask,
        ppo.policy_params.action_head_sizes,
    )
    obs_t = th.as_tensor(obs, dtype=th.float32).unsqueeze(0).unsqueeze(0)
    mask_t = th.as_tensor(coerced_mask, dtype=th.bool).unsqueeze(0)
    actions, _, _, values, probs, *_ = ppo.predict(
        obs_t, mask_t, deterministic=True,
        return_actions=True, return_log_probs=False, return_entropy=False,
        return_values=True, return_probs=True,
    )

    raw_action = actions.squeeze(0).detach().cpu().numpy()
    action = decode_policy_action(raw_action)
    value = float(values.squeeze())

    top_actions = []
    if probs is not None:
        prob_arr = probs.squeeze().cpu().numpy()
        legacy_probs = (
            policy_action_probabilities_to_legacy(prob_arr)
            if uses_factored_actions
            else prob_arr
        )
        legacy_mask = (
            policy_action_mask_to_legacy_mask(coerced_mask)
            if uses_factored_actions
            else np.asarray(coerced_mask, dtype=bool)
        )
        # Sort by probability descending, take top 5 valid actions
        sorted_indices = np.argsort(legacy_probs)[::-1]
        for idx in sorted_indices[:5]:
            if legacy_mask[idx]:
                top_actions.append((int(idx), _action_name(int(idx)), float(legacy_probs[idx])))

    return action, value, top_actions
