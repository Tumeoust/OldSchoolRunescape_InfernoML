"""
Legacy module name retained as an internal shim.

Observation v4 is the only supported schema. Older imports are redirected to
the v4 builder to avoid accidental import failures in stale tooling.
"""

from __future__ import annotations

from .observation_v4 import build_observation_v4


def build_observation_v32(state, tick_in_wave: int, temporal, dead_mobs):
    return build_observation_v4(state, tick_in_wave, temporal, dead_mobs)
