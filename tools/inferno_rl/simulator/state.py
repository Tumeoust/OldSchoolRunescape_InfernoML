"""
Simulator state management for Inferno simulation.

Central mutable state object: player, entities, pillars, wave/tick counters,
prayer state, and per-wave entity spawning.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .entity import PlacedEntity, EntityTypes, InfernoEntityType
from .equipment import AggregateStats, GearPreset, PRESET_STATS


# Constants
PLAYER_MAX_HEALTH = 99
MAX_WAVE = 66
WAVE_SPAWN_DELAY = 9  # 9 ticks between wave clear and next spawn
PILLAR_MAX_HP = 255
NUM_PILLARS = 3
PILLAR_COLLAPSE_DELAY = 2
PILLAR_COLLAPSE_DAMAGE = 45
PILLAR_COLLAPSE_RADIUS = 1

# Default starting position (Tile A)
DEFAULT_PLAYER_X = 17
DEFAULT_PLAYER_Y = 26


# Wave data: [nibblers, bats, blobs, melee, ranger, mager]
WAVE_DATA = {
    1: (3, 1, 0, 0, 0, 0),
    2: (3, 2, 0, 0, 0, 0),
    3: (6, 0, 0, 0, 0, 0),
    4: (3, 0, 1, 0, 0, 0),
    5: (3, 1, 1, 0, 0, 0),
    6: (3, 2, 1, 0, 0, 0),
    7: (3, 0, 2, 0, 0, 0),
    8: (6, 0, 0, 0, 0, 0),
    9: (3, 0, 0, 1, 0, 0),
    10: (3, 1, 0, 1, 0, 0),
    11: (3, 2, 0, 1, 0, 0),
    12: (3, 0, 1, 1, 0, 0),
    13: (3, 1, 1, 1, 0, 0),
    14: (3, 2, 1, 1, 0, 0),
    15: (3, 0, 2, 1, 0, 0),
    16: (3, 0, 0, 2, 0, 0),
    17: (6, 0, 0, 0, 0, 0),
    18: (3, 0, 0, 0, 1, 0),
    19: (3, 1, 0, 0, 1, 0),
    20: (3, 2, 0, 0, 1, 0),
    21: (3, 0, 1, 0, 1, 0),
    22: (3, 1, 1, 0, 1, 0),
    23: (3, 2, 1, 0, 1, 0),
    24: (3, 0, 2, 0, 1, 0),
    25: (3, 0, 0, 1, 1, 0),
    26: (3, 1, 0, 1, 1, 0),
    27: (3, 2, 0, 1, 1, 0),
    28: (3, 0, 1, 1, 1, 0),
    29: (3, 1, 1, 1, 1, 0),
    30: (3, 2, 1, 1, 1, 0),
    31: (3, 0, 2, 1, 1, 0),
    32: (3, 0, 0, 2, 1, 0),
    33: (3, 0, 0, 0, 2, 0),
    34: (6, 0, 0, 0, 0, 0),
    35: (3, 0, 0, 0, 0, 1),
    36: (3, 1, 0, 0, 0, 1),
    37: (3, 2, 0, 0, 0, 1),
    38: (3, 0, 1, 0, 0, 1),
    39: (3, 1, 1, 0, 0, 1),
    40: (3, 2, 1, 0, 0, 1),
    41: (3, 0, 2, 0, 0, 1),
    42: (3, 0, 0, 1, 0, 1),
    43: (3, 1, 0, 1, 0, 1),
    44: (3, 2, 0, 1, 0, 1),
    45: (3, 0, 1, 1, 0, 1),
    46: (3, 1, 1, 1, 0, 1),
    47: (3, 2, 1, 1, 0, 1),
    48: (3, 0, 2, 1, 0, 1),
    49: (3, 0, 0, 2, 0, 1),
    50: (3, 0, 0, 0, 1, 1),
    51: (3, 1, 0, 0, 1, 1),
    52: (3, 2, 0, 0, 1, 1),
    53: (3, 0, 1, 0, 1, 1),
    54: (3, 1, 1, 0, 1, 1),
    55: (3, 2, 1, 0, 1, 1),
    56: (3, 0, 2, 0, 1, 1),
    57: (3, 0, 0, 1, 1, 1),
    58: (3, 1, 0, 1, 1, 1),
    59: (3, 2, 0, 1, 1, 1),
    60: (3, 0, 1, 1, 1, 1),
    61: (3, 1, 1, 1, 1, 1),
    62: (3, 2, 1, 1, 1, 1),
    63: (3, 0, 2, 1, 1, 1),
    64: (3, 0, 0, 2, 1, 1),
    65: (3, 0, 0, 0, 2, 1),
    66: (3, 0, 0, 0, 0, 2),
}

# Regular spawn points (9 positions around the arena)
REGULAR_SPAWNS = [
    (1, 24),   # Top left
    (22, 24),  # Top right
    (3, 18),   # Mid left
    (23, 17),  # Mid right
    (16, 12),  # Center
    (5, 6),    # Bottom left
    (23, 4),   # Bottom right
    (1, 1),    # Far bottom left
    (15, 1)    # Far bottom center
]

# Nibbler spawn area (3x3 grid near center)
NIBBLER_SPAWNS = [
    (8, 16), (9, 16), (10, 16),
    (8, 17), (9, 17), (10, 17),
    (8, 18), (9, 18), (10, 18)
]

# Spawn points by entity type
SPAWN_POINTS = {
    EntityTypes.NIBBLER: NIBBLER_SPAWNS,
    EntityTypes.BAT: REGULAR_SPAWNS,
    EntityTypes.BLOB: REGULAR_SPAWNS,
    EntityTypes.MELEE: REGULAR_SPAWNS,
    EntityTypes.RANGER: REGULAR_SPAWNS,
    EntityTypes.MAGER: REGULAR_SPAWNS,
}


@dataclass
class SimulatorState:
    """Holds all mutable simulation state."""

    # Loadout fields (defaults match existing near-max crystal behavior)
    max_health: int = 99
    has_blowpipe: bool = True
    loadout_preset_stats: dict[GearPreset, AggregateStats] = field(default_factory=lambda: PRESET_STATS)

    # Entity state
    entities: List[PlacedEntity] = field(default_factory=list)

    # Player state
    player_x: int = DEFAULT_PLAYER_X
    player_y: int = DEFAULT_PLAYER_Y
    player_health: int = PLAYER_MAX_HEALTH
    player_last_attack_tick: int = -5  # Start with no cooldown
    player_attack_speed_at_last_attack: int = 4  # Speed of weapon used for last attack

    # Gear state (damage/accuracy now from combat.py lookup tables)
    player_attack_speed: int = 4  # BoFa speed
    player_attack_range: int = 10
    current_preset: GearPreset = GearPreset.BOFA
    use_blood_barrage: bool = False  # Sub-mode of MAGE preset

    # Tick state
    current_tick: int = 1

    # Wave progression state
    current_wave: int = 0
    wave_complete_timer: int = -1  # -1 = not active, >0 = countdown

    # Pillar state
    pillar_hp: List[int] = field(default_factory=lambda: [PILLAR_MAX_HP] * NUM_PILLARS)
    pillar_alive: List[bool] = field(default_factory=lambda: [True] * NUM_PILLARS)
    pending_pillar_collapses: List[Optional[int]] = field(
        default_factory=lambda: [None] * NUM_PILLARS
    )  # Ticks remaining until collapse damage, or None

    # Combat state
    attack_target: Optional[PlacedEntity] = None
    active_prayer: Optional[str] = None  # "PROTECT_FROM_MAGIC", "PROTECT_FROM_MISSILES", "PROTECT_FROM_MELEE"
    queued_prayer: Optional[str] = None
    queued_prayer_tick: int = -1

    # Per-wave kill tracking (for observation: what has died this wave)
    wave_kills: Dict[InfernoEntityType, int] = field(default_factory=dict)

    def reset_player(self):
        """Reset player state to defaults."""
        bofa_stats = self.loadout_preset_stats[GearPreset.BOFA]
        self.player_x = DEFAULT_PLAYER_X
        self.player_y = DEFAULT_PLAYER_Y
        self.player_health = self.max_health
        self.player_last_attack_tick = -5
        self.player_attack_speed_at_last_attack = bofa_stats.attack_speed
        self.player_attack_speed = bofa_stats.attack_speed
        self.player_attack_range = bofa_stats.attack_range
        self.current_preset = GearPreset.BOFA
        self.use_blood_barrage = False
        self.attack_target = None
        self.active_prayer = None
        self.queued_prayer = None
        self.queued_prayer_tick = -1

    def reset_pillars(self):
        """Reset pillar state to full health."""
        self.pillar_hp = [PILLAR_MAX_HP] * NUM_PILLARS
        self.pillar_alive = [True] * NUM_PILLARS
        self.pending_pillar_collapses = [None] * NUM_PILLARS

    def clear_entities(self):
        """Remove all entities."""
        self.entities.clear()
        self.attack_target = None
        self.wave_kills.clear()

    def add_entity(self, entity: PlacedEntity):
        """Add an entity to the simulation."""
        self.entities.append(entity)

    def remove_entity(self, entity: PlacedEntity):
        """Remove an entity from the simulation."""
        if entity in self.entities:
            self.entities.remove(entity)
        if self.attack_target == entity:
            self.attack_target = None

    def remove_dead_entities(self):
        """Remove all dead entities."""
        dead_entities = [e for e in self.entities if e.is_dead()]
        for entity in dead_entities:
            self.remove_entity(entity)

    def get_alive_entities(self) -> List[PlacedEntity]:
        """Get all alive entities."""
        return [e for e in self.entities if not e.is_dead()]

    def get_alive_pillar_indices(self) -> List[int]:
        """Get indices of alive pillars."""
        return [i for i, alive in enumerate(self.pillar_alive) if alive]

    def is_pillar_alive(self, index: int) -> bool:
        """Check if a pillar is alive."""
        return 0 <= index < NUM_PILLARS and self.pillar_alive[index]

    def get_pillar_hp(self, index: int) -> int:
        """Get pillar HP."""
        return self.pillar_hp[index] if 0 <= index < NUM_PILLARS else 0

    def damage_pillar(self, index: int, damage: int) -> bool:
        """
        Damage a pillar and return True if it died.

        Returns:
            True if pillar was destroyed
        """
        if not self.is_pillar_alive(index):
            return False

        self.pillar_hp[index] = max(0, self.pillar_hp[index] - damage)

        if self.pillar_hp[index] <= 0:
            self.pillar_alive[index] = False
            return True
        return False

    def schedule_pillar_collapse(self, pillar_index: int):
        """Schedule a pillar to collapse (damage nearby entities)."""
        self.pending_pillar_collapses[pillar_index] = PILLAR_COLLAPSE_DELAY

    def apply_gear_preset(self, preset: GearPreset):
        """Apply a gear preset (updates speed and range from pre-computed stats)."""
        self.current_preset = preset
        stats = self.loadout_preset_stats[preset]
        self.player_attack_speed = stats.attack_speed
        self.player_attack_range = stats.attack_range

    def get_player_attack_cooldown(self) -> int:
        """
        Get remaining attack cooldown in ticks.

        Uses the weapon speed from the time of the last attack, not the current weapon.
        This ensures weapon switching doesn't affect the cooldown.
        """
        cooldown = self.player_attack_speed_at_last_attack - (self.current_tick - self.player_last_attack_tick)
        return max(0, cooldown)

    def can_player_attack(self) -> bool:
        """Check if player's attack is ready."""
        return self.get_player_attack_cooldown() <= 0

    def queue_prayer(self, prayer: Optional[str], activate_tick: int):
        """Queue a prayer to activate on a specific tick."""
        self.queued_prayer = prayer
        self.queued_prayer_tick = activate_tick

    def process_action_queue(self) -> bool:
        """
        Process the action queue for the current tick.

        Returns:
            True if a prayer was activated
        """
        activated = False

        if self.queued_prayer is not None and self.queued_prayer_tick <= self.current_tick:
            self.active_prayer = self.queued_prayer
            self.queued_prayer = None
            self.queued_prayer_tick = -1
            activated = True
        elif self.queued_prayer is None and 0 < self.queued_prayer_tick <= self.current_tick:
            # Queued to turn OFF prayer
            self.active_prayer = None
            self.queued_prayer_tick = -1
            activated = True

        return activated

    def is_wave_cleared(self) -> bool:
        """Check if all enemies are dead."""
        return all(e.is_dead() for e in self.entities)

    def tick_wave_complete_timer(self) -> bool:
        """
        Decrement wave complete timer.

        Returns:
            True if timer just hit 0 (time to spawn next wave)
        """
        if self.wave_complete_timer > 0:
            self.wave_complete_timer -= 1
            return self.wave_complete_timer == 0
        return False

    def increment_tick(self):
        """Increment the current tick."""
        self.current_tick += 1


def get_wave_data(wave: int) -> Optional[Tuple[int, int, int, int, int, int]]:
    """
    Get wave composition data.

    Returns:
        (nibblers, bats, blobs, melee, ranger, mager) or None if invalid wave
    """
    return WAVE_DATA.get(wave)


def spawn_wave_entities(state: SimulatorState, wave: int) -> bool:
    """
    Spawn entities for a wave.

    Args:
        state: Simulator state to add entities to
        wave: Wave number (1-66)

    Returns:
        True if wave was spawned successfully
    """
    import random

    wave_data = get_wave_data(wave)
    if wave_data is None:
        return False

    nibblers, bats, blobs, melees, rangers, magers = wave_data
    tick = state.current_tick

    # Pick ONE random pillar for ALL nibblers (they attack as a group)
    alive_pillars = state.get_alive_pillar_indices()
    nibbler_target_pillar = random.choice(alive_pillars) if alive_pillars else -1

    # Shuffle spawn points for variety (matches real-game spawn randomization)
    shuffled_regular = list(REGULAR_SPAWNS)
    random.shuffle(shuffled_regular)
    shuffled_nibbler = list(NIBBLER_SPAWNS)
    random.shuffle(shuffled_nibbler)

    regular_spawn_idx = 0
    nibbler_spawn_idx = 0

    # Spawn each entity type
    # Order: high priority (Mager) first, low priority (Nibbler) last
    entity_counts = [
        (EntityTypes.MAGER, magers),
        (EntityTypes.RANGER, rangers),
        (EntityTypes.MELEE, melees),
        (EntityTypes.BLOB, blobs),
        (EntityTypes.BAT, bats),
        (EntityTypes.NIBBLER, nibblers),
    ]

    for entity_type, count in entity_counts:
        for i in range(count):
            if entity_type == EntityTypes.NIBBLER:
                x, y = shuffled_nibbler[nibbler_spawn_idx % len(shuffled_nibbler)]
                nibbler_spawn_idx += 1
            else:
                x, y = shuffled_regular[regular_spawn_idx % len(shuffled_regular)]
                regular_spawn_idx += 1

            entity = PlacedEntity(
                entity_type=entity_type,
                x=x,
                y=y,
                placed_tick=tick
            )

            # ALL nibblers target the SAME pillar (group behavior)
            if entity_type == EntityTypes.NIBBLER:
                entity.target_pillar_index = nibbler_target_pillar

            state.add_entity(entity)

    state.current_wave = wave
    return True

