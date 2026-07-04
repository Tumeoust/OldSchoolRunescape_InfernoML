from __future__ import annotations

from .entity import PlacedEntity
from .geometry import InfernoLineOfSight


def combat_entity_sort_key(
    entity: PlacedEntity,
    player_x: int,
    player_y: int,
    pillar_alive: list[bool],
) -> tuple[int, int, int, int, int]:
    """Stable combat ordering shared by the simulator and observations."""
    has_los = InfernoLineOfSight.can_entity_attack_player(
        entity, player_x, player_y, pillar_alive,
    )
    distance = InfernoLineOfSight.get_distance_from_npc(
        entity.x, entity.y, entity.entity_type.size_in_tiles, player_x, player_y,
    )
    return (
        0 if has_los else 1,
        0 if (has_los and entity.attack_delay <= 1 and entity.stunned <= 1) else 1,
        entity.entity_type.base_priority,
        distance,
        entity.id,
    )
