"""
Public observation facade.

Single supported observation version: v4.
"""

from __future__ import annotations

from .observation_common import (
    EXACT_TARGET_ENTITY_TYPES,
    EXACT_TARGET_SLOT_COUNT,
    EXACT_TARGET_SLOT_SIZE,
    EXACT_TARGET_SLOTS_TOTAL,
    GRID_HEIGHT,
    GRID_WIDTH,
    GLOBAL_V4_SIZE,
    MAX_ATTACK_COOLDOWN,
    MAX_HEALTH,
    NEIGHBORHOOD_FORECAST_SIZE,
    OBSERVATION_PRIVILEGED_SIZE,
    OBSERVATION_PUBLIC_SIZE,
    OBSERVATION_TOTAL_SIZE,
    SLOT_CORE_SIZE,
    TEMPORAL_V3_SIZE,
    THREAT_HORIZON_SIZE,
    TemporalState,
    get_observation_low,
    get_public_observation_size,
    get_observation_size,
    update_temporal_state,
)
from .observation_v4 import build_observation_v4

__all__ = [
    "EXACT_TARGET_ENTITY_TYPES",
    "EXACT_TARGET_SLOT_COUNT",
    "EXACT_TARGET_SLOT_SIZE",
    "EXACT_TARGET_SLOTS_TOTAL",
    "GRID_HEIGHT",
    "GRID_WIDTH",
    "GLOBAL_V4_SIZE",
    "MAX_ATTACK_COOLDOWN",
    "MAX_HEALTH",
    "NEIGHBORHOOD_FORECAST_SIZE",
    "OBSERVATION_PRIVILEGED_SIZE",
    "OBSERVATION_PUBLIC_SIZE",
    "OBSERVATION_TOTAL_SIZE",
    "SLOT_CORE_SIZE",
    "TEMPORAL_V3_SIZE",
    "THREAT_HORIZON_SIZE",
    "TemporalState",
    "build_observation",
    "get_observation_low",
    "get_public_observation_size",
    "get_observation_size",
    "update_temporal_state",
]


def build_observation(
    state,
    tick_in_wave: int,
    temporal: TemporalState | None = None,
    dead_mobs=None,
    neighborhood_summaries=None,
    tick_threat_cache=None,
):
    return build_observation_v4(
        state,
        tick_in_wave,
        temporal,
        dead_mobs,
        neighborhood_summaries=neighborhood_summaries,
        tick_threat_cache=tick_threat_cache,
    )
