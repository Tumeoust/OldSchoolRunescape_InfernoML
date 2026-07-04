"""
SB3 Training modules for Inferno RL.
"""

from .actions import InfernoAction, get_action_mask


def __getattr__(name):
    if name == "InfernoEnv":
        from .env import InfernoEnv

        return InfernoEnv
    if name == "InfernoReward":
        from .rewards import InfernoReward

        return InfernoReward
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "InfernoEnv",
    "InfernoReward",
    "InfernoAction",
    "get_action_mask",
]
