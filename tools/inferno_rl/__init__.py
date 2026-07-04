"""
Inferno RL - Stable Baselines 3 training for Old School Runescape Inferno.

This package provides:
- Python port of the Inferno simulator (simulator/)
- SB3 Gymnasium environment with MaskablePPO support (training/)
- Pygame visualizer for debugging (visualizer/)

Usage:
    # Training
    python -m tools.inferno_rl.training.train --start-wave 35 --max-wave 49

    # Visualization
    python -m tools.inferno_rl.visualizer.run_visual --model models/inferno/best.zip
"""

__version__ = "1.0.0"

from .simulator import InfernoSimulator, SimulatorState, PlacedEntity, EntityTypes

_TRAINING_EXPORTS = {"InfernoEnv", "InfernoReward", "get_action_mask"}


def __getattr__(name):
    if name in _TRAINING_EXPORTS:
        from .training import InfernoEnv, InfernoReward, get_action_mask

        exports = {
            "InfernoEnv": InfernoEnv,
            "InfernoReward": InfernoReward,
            "get_action_mask": get_action_mask,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "InfernoSimulator",
    "SimulatorState",
    "PlacedEntity",
    "EntityTypes",
    "InfernoEnv",
    "InfernoReward",
    "get_action_mask",
]
