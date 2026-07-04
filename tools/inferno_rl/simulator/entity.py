"""
Entity types and PlacedEntity for Inferno simulation.

Defines the frozen entity-type table and mutable placed-entity instances.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List


class AttackStyle(Enum):
    """Attack styles for entities."""
    MELEE = auto()
    RANGED = auto()
    MAGIC = auto()
    MAGIC_RANGED = auto()  # Blob - switches between magic and ranged
    MAGIC_RANGED_MELEE = auto()  # Jad/Zuk - uses all three styles


@dataclass(frozen=True)
class InfernoEntityType:
    """
    Types of entities that spawn in the Inferno.
    Each type has different attack styles, speeds, and threat levels.
    """
    name: str
    max_health: int
    attack_style: AttackStyle
    base_priority: int  # Lower = higher priority
    attack_range: int
    size_in_tiles: int
    attack_speed: int
    move_speed: int = 1  # All NPCs walk at 1 tile/tick

    def is_large(self) -> bool:
        """Check if entity is large (3x3 or bigger)."""
        return self.size_in_tiles >= 3

    def is_ranged_attacker(self) -> bool:
        """Check if entity can attack from range."""
        return self.attack_range > 1

    def is_melee_only(self) -> bool:
        """Check if entity is melee-only."""
        return self.attack_style == AttackStyle.MELEE and self.attack_range <= 1


# Entity type definitions
class EntityTypes:
    """All Inferno entity types."""
    NIBBLER = InfernoEntityType("Jal-Nib", 10, AttackStyle.MELEE, 2, 1, 1, 4)
    BAT = InfernoEntityType("Jal-MejRah", 25, AttackStyle.RANGED, 8, 4, 2, 3)
    BLOB = InfernoEntityType("Jal-Ak", 40, AttackStyle.MAGIC_RANGED, 5, 15, 3, 3)
    BLOB_MAGE = InfernoEntityType("Jal-AkRek-Mej", 15, AttackStyle.MAGIC, 5, 15, 1, 4)
    BLOB_RANGE = InfernoEntityType("Jal-AkRek-Xil", 15, AttackStyle.RANGED, 5, 15, 1, 4)
    BLOB_MELEE = InfernoEntityType("Jal-AkRek-Ket", 15, AttackStyle.MELEE, 5, 1, 1, 4)
    RANGER = InfernoEntityType("Jal-Xil", 125, AttackStyle.RANGED, 4, 15, 3, 4)
    MAGER = InfernoEntityType("Jal-Zek", 220, AttackStyle.MAGIC, 3, 15, 4, 4)
    MELEE = InfernoEntityType("Jal-ImKot", 75, AttackStyle.MELEE, 7, 1, 4, 4)
    JAD = InfernoEntityType("JalTok-Jad", 350, AttackStyle.MAGIC_RANGED_MELEE, 1, 10, 5, 8)
    HEALER = InfernoEntityType("Yt-HurKot", 60, AttackStyle.MELEE, 1, 1, 1, 4)
    ZUK_HEALER = InfernoEntityType("Jal-MejJak", 75, AttackStyle.MAGIC, 7, 1, 1, 4)
    ZUK = InfernoEntityType("TzKal-Zuk", 1200, AttackStyle.MAGIC_RANGED_MELEE, 0, 15, 5, -1)

    @classmethod
    def all_types(cls) -> List[InfernoEntityType]:
        """Get all entity types."""
        return [
            cls.NIBBLER, cls.BAT, cls.BLOB, cls.BLOB_MAGE, cls.BLOB_RANGE,
            cls.BLOB_MELEE, cls.RANGER, cls.MAGER, cls.MELEE, cls.JAD,
            cls.HEALER, cls.ZUK_HEALER, cls.ZUK
        ]

    @classmethod
    def from_name(cls, name: str) -> Optional[InfernoEntityType]:
        """Get entity type by name."""
        for entity_type in cls.all_types():
            if entity_type.name == name:
                return entity_type
        return None


# Pillar center positions (for nibbler targeting)
PILLAR_CENTERS = [
    (1, 21),   # NW pillar center
    (18, 23),  # NE pillar center
    (11, 7)    # S pillar center
]


@dataclass
class PlacedEntity:
    """
    Represents an entity placed in the simulator.
    Mutable to allow movement and combat simulation.

    Uses timing model where attackDelay decrements each tick
    and attacks occur when attackDelay <= 0.
    """
    entity_type: InfernoEntityType
    x: int  # SW anchor tile
    y: int
    placed_tick: int
    id: int = field(init=False)  # Unique ID for tracking across snapshots
    current_health: int = field(init=False)

    # Class-level counter for generating unique IDs
    _next_id: int = 1

    # Core timing (decrements each tick, attacks when <= 0)
    attack_delay: int = 0
    stunned: int = 1  # All NPCs spawn with 1 tick stun
    frozen: int = 0  # Cannot move when > 0 (can still attack)

    # Combat LOS tracking (used by divergence classifier)
    scanned_prayer: Optional[str] = None  # "MAGIC", "RANGED", or None
    had_los: bool = False

    # Attack tracking for rendering
    attacked_this_tick: bool = False
    last_attack_style: Optional[str] = None

    # Meleer specific tracking (dig mechanic)
    dig_sequence_time: int = 0
    dig_location: Optional[tuple] = None  # (x, y) target location after dig

    # Nibbler specific tracking
    # Index of target pillar: 0 = NW, 1 = NE, 2 = S, -1 = no target
    target_pillar_index: int = -1
    
    # Resurrection tracking - NPCs can only be resurrected once
    has_resurrected: bool = False

    def __post_init__(self):
        """Initialize health and ID from entity type."""
        self.current_health = self.entity_type.max_health
        # Assign unique ID
        self.id = PlacedEntity._next_id
        PlacedEntity._next_id += 1

    def copy(self) -> 'PlacedEntity':
        """Create a deep copy of this entity."""
        entity = PlacedEntity(
            entity_type=self.entity_type,
            x=self.x,
            y=self.y,
            placed_tick=self.placed_tick
        )
        entity.current_health = self.current_health
        entity.attack_delay = self.attack_delay
        entity.stunned = self.stunned
        entity.frozen = self.frozen
        entity.scanned_prayer = self.scanned_prayer
        entity.had_los = self.had_los
        entity.dig_sequence_time = self.dig_sequence_time
        entity.dig_location = tuple(self.dig_location) if self.dig_location else None
        entity.target_pillar_index = self.target_pillar_index
        entity.attacked_this_tick = self.attacked_this_tick
        entity.last_attack_style = self.last_attack_style
        entity.has_resurrected = self.has_resurrected
        return entity

    def is_dead(self) -> bool:
        """Check if entity is dead."""
        return self.current_health <= 0

    def take_damage(self, damage: int) -> bool:
        """Apply damage and return True if entity died."""
        self.current_health = max(0, self.current_health - damage)
        return self.is_dead()

    def can_attack(self) -> bool:
        """Check if this entity can attack (not stunned, attack ready)."""
        return self.stunned <= 0 and self.attack_delay <= 0 and not self.is_dead()

    def can_move(self) -> bool:
        """Check if this entity can move (not frozen)."""
        return self.frozen <= 0 and not self.is_dead()

    def reset_tick_flags(self):
        """Reset per-tick flags at the start of a new tick."""
        self.attacked_this_tick = False
        self.last_attack_style = None

    def decrement_timers(self):
        """Decrement all tick-based timers by 1."""
        self.attack_delay -= 1
        if self.stunned > 0:
            self.stunned -= 1
        if self.frozen > 0:
            self.frozen -= 1

    @staticmethod
    def get_pillar_center(pillar_index: int) -> Optional[tuple]:
        """
        Get the center position of a pillar for movement targeting.
        
        Args:
            pillar_index: 0 = NW, 1 = NE, 2 = S
            
        Returns:
            (x, y) center coordinates, or None if invalid index
        """
        if 0 <= pillar_index < len(PILLAR_CENTERS):
            return PILLAR_CENTERS[pillar_index]
        return None
