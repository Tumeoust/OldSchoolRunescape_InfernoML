"""
Step result data classes and result builder mixin.

PlayerDamageEvent and StepResult capture per-tick outputs for the reward
calculator.  ResultBuilderMixin assembles those outputs from simulator state.
"""

from dataclasses import dataclass
from typing import Optional, Dict, List

from .entity import PlacedEntity, EntityTypes, InfernoEntityType
from .equipment import GearPreset
from .forecast import (
    build_tick_threat_cache,
)
from .geometry import (
    SimulatorGeometry, InfernoLineOfSight,
    A_TILE_X, A_TILE_Y, is_in_ne_pillar_zone,
)
from ..training.actions import InfernoAction

# NPC types that trigger the adjacency penalty (MELEE excluded — can't avoid being near them)
_ADJACENCY_PENALTY_TYPES = {EntityTypes.MAGER, EntityTypes.RANGER, EntityTypes.BLOB}

# NPC types that count for LOS separation reward (strategic pillar-play targets).
# Excludes BAT, NIBBLER, and mini-blobs (BLOB_MAGE/BLOB_RANGE/BLOB_MELEE).
_DANGEROUS_LOS_TYPES = {
    EntityTypes.MAGER, EntityTypes.RANGER, EntityTypes.MELEE, EntityTypes.BLOB,
}

# Wave timeout
MAX_TICKS_PER_WAVE = 800


@dataclass
class PlayerDamageEvent:
    """A single NPC hit that reduced player HP on a specific tick."""
    tick: int
    attacker_id: int
    attacker_type: str
    attacker_x: int
    attacker_y: int
    attack_style: str
    damage: int


@dataclass
class StepResult:
    """Result of executing a single simulation step."""
    # Damage tracking
    damage_taken: int = 0
    pillar_damage_taken: int = 0
    ne_pillar_damage_taken: int = 0  # NE pillar (index 1) is most important for positioning
    damage_dealt: int = 0
    damage_dealt_to_mager: int = 0  # Damage dealt specifically to magers (for priority reward)
    player_damage_events: List[PlayerDamageEvent] = None
    non_mager_enemies_at_step_start: int = 0  # Non-mager, non-nibbler NPCs alive at step start
    health_gained: int = 0  # HP restored this tick (e.g. Blood Barrage), capped at 99
    health_at_step_start: int = 99  # HP at step start; only reward healing when this < 99
    # True when switched to or attacked with blood barrage while at high HP (> 95)
    used_blood_barrage_at_high_hp: bool = False
    # True when player currently has blood barrage equipped (for low HP healing incentives)
    has_blood_barrage: bool = False
    # True when player switched to blood barrage this tick (action 42)
    switched_to_blood_barrage: bool = False

    # NPC attack tracking
    npcs_attacked_player: int = 0

    # Non-nibbler NPCs with LOS to player at step end (drives multi-LOS penalty and 1-LOS bonus)
    npcs_with_los_now: int = 0
    avoidable_extra_los: int = 0
    avoidable_extra_imminent: int = 0
    # Non-nibbler NPCs with LOS on the *previous* tick (used to require sustained 1-LOS for bonus)
    npcs_with_los_prev: int = 0
    # Consecutive ticks with exactly 1 non-nibbler NPC having LOS to player.
    # Used to require sustained 1-LOS for engagement bonus (threshold: 3 ticks).
    consecutive_single_los_ticks: int = 0

    # LOS separation tracking (dangerous NPC types only: MAGER/RANGER/MELEE/BLOB*)
    dangerous_npcs_alive: int = 0
    dangerous_npcs_with_los: int = 0

    # Priority target tracking (highest threat at step start)
    priority_target_distance_start: float = -1.0
    priority_target_distance_now: float = -1.0
    priority_target_npc_los_start: bool = False
    priority_target_npc_los_now: bool = False
    priority_target_player_los_start: bool = False
    priority_target_player_los_now: bool = False
    priority_target_reachable: bool = False
    # Priority target identity (used to avoid rewarding "fake" drag/LOS vs mini-blobs)
    priority_target_entity_type: Optional[InfernoEntityType] = None

    # Kill tracking by entity type
    kills_by_type: Dict[InfernoEntityType, int] = None

    # Wave status
    wave_completed: bool = False
    wave_number: int = 0
    pillars_remaining: int = 3
    ne_pillar_alive: bool = True  # NE pillar (index 1) - most important for safespotting
    pillar_deaths: int = 0  # Number of pillars that died this tick
    ne_pillar_died: bool = False  # NE pillar (index 1) died this tick
    enemies_remaining: int = 0

    # Proximity tracking
    enemies_in_attack_range: int = 0
    nearest_enemy_distance: float = 0.0

    # Terminal conditions
    player_died: bool = False
    wave_timeout: bool = False
    inferno_complete: bool = False

    # Mager resurrection (model should prioritize killing magers to avoid this)
    mager_resurrected_count: int = 0
    melee_resurrected_count: int = 0
    bat_resurrected_count: int = 0

    # Grace period (no NPCs alive): distance to A tile for positioning reward
    # Only set when enemies_remaining == 0 (wave complete, countdown to next wave)
    distance_to_a_tile: float = -1.0  # -1 = not in grace period

    # Action validation
    action_was_valid: bool = True
    executed_action: int = 0  # Action index

    # True when player chose a valid attack action AND attack cooldown was 0 that tick.
    # Rewards tick-perfect attack timing — attacking late wastes ticks and increases
    # exposure to additional NPC attacks before the kill.
    player_attacked_on_cooldown: bool = False

    # Ticks elapsed since the current wave started. Used by reward calculator
    # to exempt early-wave ticks from idle/positioning penalties.
    ticks_in_wave: int = 0

    # True when the player is in the NE pillar zone (ring around pillar + north strip)
    player_in_ne_pillar_zone: bool = False

    # Ticks since the player last attacked (current_tick - player_last_attack_tick).
    # Used by reward calculator to condition Single-LOS bonus on recent combat.
    ticks_since_last_attack: int = 999

    # Consecutive ticks without attacking (attack cooldown going off).
    # Resets to 0 only when player fires an attack. Used for escalating stall penalty.
    ticks_since_engagement: int = 0

    # True when the player's position changed this tick (movement or attack drag).
    player_moved: bool = False

    # Player position at end of tick (for reward calculations).
    player_x: int = 0
    player_y: int = 0

    # Count of MAGER/RANGER/BLOB NPCs that attacked while player was adjacent to their footprint.
    adjacent_attacking_npc_count: int = 0

    # True when the player used ice barrage on a resolved nibbler target
    # within the first 3 ticks of a wave (ticks 0-2).
    used_early_ice_barrage_on_nibblers: bool = False

    # Player's max health from loadout (80-99). Used by reward calculator.
    max_health: int = 99

    def __post_init__(self):
        if self.kills_by_type is None:
            self.kills_by_type = {}
        if self.player_damage_events is None:
            self.player_damage_events = []

    def get_total_kills(self) -> int:
        """Get total kills this tick."""
        return sum(self.kills_by_type.values())

    def get_kills_of_type(self, entity_type: InfernoEntityType) -> int:
        """Get kills of a specific type."""
        return self.kills_by_type.get(entity_type, 0)

    def is_terminal(self) -> bool:
        """Check if this is a terminal state."""
        return self.player_died or self.inferno_complete or self.wave_timeout


class ResultBuilderMixin:
    """Mixin providing _build_step_result for InfernoSimulator."""

    def _build_step_result(self, action: int, action_valid: bool) -> StepResult:
        """Build step result for reward calculation."""
        # Calculate damage taken and health gained
        damage_taken = self.health_at_step_start - self.state.player_health
        health_gained = max(0, self.state.player_health - self.health_at_step_start)

        # Calculate pillar damage (total and NE specifically)
        pillar_damage = 0
        ne_pillar_damage = 0
        for i in range(3):
            dmg = self.pillar_hp_at_step_start[i] - self.state.get_pillar_hp(i)
            pillar_damage += dmg
            if i == 1:  # NE pillar is index 1
                ne_pillar_damage = dmg

        # Single pass: enemies remaining, in range, nearest distance, LOS counts.
        enemies_remaining = 0
        enemies_in_range = 0
        nearest_dist = 0.0
        npcs_with_los_now = 0
        current_imminent_attacks = 0
        dangerous_npcs_alive = 0
        dangerous_npcs_with_los = 0
        type_alive_counts: Dict[InfernoEntityType, int] = {}

        px, py = self.state.player_x, self.state.player_y
        p_range = self.state.player_attack_range
        pillar_alive = self.state.pillar_alive
        adjacent_attacking_npc_count = 0
        combat_entities: list[PlacedEntity] = []

        for e in self.state.entities:
            if e.is_dead():
                continue
            enemies_remaining += 1
            if e.entity_type != EntityTypes.NIBBLER:
                combat_entities.append(e)
                type_alive_counts[e.entity_type] = type_alive_counts.get(e.entity_type, 0) + 1

            # Chebyshev distance for nearest_dist metric
            dist = SimulatorGeometry.chebyshev_distance(px, py, e.x, e.y)
            if nearest_dist == 0 or dist < nearest_dist:
                nearest_dist = dist

            # Player -> NPC LOS (for enemies_in_range count)
            if InfernoLineOfSight.can_player_attack_entity(
                px, py, p_range, e, pillar_alive
            ):
                enemies_in_range += 1

            # NPC -> Player LOS (for npcs_with_los count, excludes nibblers)
            if e.entity_type != EntityTypes.NIBBLER:
                has_los = InfernoLineOfSight.can_entity_attack_player(e, px, py, pillar_alive)
                if has_los:
                    npcs_with_los_now += 1
                    if e.entity_type == EntityTypes.BLOB:
                        if e.scanned_prayer is not None and e.attack_delay <= 0:
                            current_imminent_attacks += 1
                    elif e.attack_delay <= 0 and e.stunned <= 0:
                        current_imminent_attacks += 1
                if e.entity_type in _DANGEROUS_LOS_TYPES:
                    dangerous_npcs_alive += 1
                    if has_los:
                        dangerous_npcs_with_los += 1

            # NPC proximity: event-based — only penalize when NPC attacks while player is adjacent
            if e.entity_type in _ADJACENCY_PENALTY_TYPES and e.attacked_this_tick:
                s = e.entity_type.size_in_tiles  # 4
                # On footprint: x <= px <= x+s-1, y <= py <= y+s-1
                on_footprint = (e.x <= px <= e.x + s - 1) and (e.y <= py <= e.y + s - 1)
                # Cardinal-adjacent: one tile N/S/E/W of footprint edge
                cardinal_adj = (
                    # North/South rows (within x span)
                    ((py == e.y + s or py == e.y - 1) and (e.x <= px <= e.x + s - 1))
                    # East/West columns (within y span)
                    or ((px == e.x + s or px == e.x - 1) and (e.y <= py <= e.y + s - 1))
                )
                if on_footprint or cardinal_adj:
                    adjacent_attacking_npc_count += 1

        npcs_with_los_prev = self._last_npcs_with_los
        self._last_npcs_with_los = npcs_with_los_now

        tick_threat_cache = build_tick_threat_cache(
            px,
            py,
            pillar_alive,
            combat_entities,
            self.state.active_prayer,
            npcs_with_los_now=npcs_with_los_now,
            current_imminent_attacks=current_imminent_attacks,
            type_alive_counts=type_alive_counts,
        )
        self._post_step_tick_threat_cache = tick_threat_cache
        neighborhood_summaries = tick_threat_cache.neighborhood_summaries
        best_reachable_los = npcs_with_los_now
        best_reachable_imminent = current_imminent_attacks
        for summary in neighborhood_summaries[1:]:
            if summary.blocked_move or summary.settled_step_distance <= 0:
                continue
            best_reachable_los = min(best_reachable_los, summary.los_count)
            best_reachable_imminent = min(
                best_reachable_imminent,
                summary.total_imminent,
            )
        avoidable_extra_los = max(0, npcs_with_los_now - best_reachable_los)
        avoidable_extra_imminent = max(
            0, current_imminent_attacks - best_reachable_imminent
        )

        if npcs_with_los_now == 1:
            self._consecutive_single_los_ticks += 1
        else:
            self._consecutive_single_los_ticks = 0

        # Engagement tracking: reset only when player actually attacked (cooldown
        # went off). LOS and damage alone don't count — model was exploiting
        # "stand visible but don't attack", and hitting a 0 is still attacking.
        if self.attacked_on_cooldown_this_tick:
            self._ticks_since_engagement = 0
        else:
            self._ticks_since_engagement += 1

        # Check terminal conditions
        player_died = self.state.player_health <= 0
        wave_timeout = self.get_ticks_in_wave() >= MAX_TICKS_PER_WAVE
        inferno_complete = self.is_inferno_complete()

        # Pillars remaining and death events
        pillars_remaining = sum(1 for alive in self.state.pillar_alive if alive)
        ne_pillar_alive = self.state.pillar_alive[1]  # NE pillar is index 1
        pillar_deaths = 0
        ne_pillar_died = False
        for i in range(3):
            if self.pillar_hp_at_step_start[i] > 0 and self.state.get_pillar_hp(i) <= 0:
                pillar_deaths += 1
                if i == 1:
                    ne_pillar_died = True

        # Wave completed this tick (set by _process_wave_progression)
        wave_completed = self._wave_completed_this_tick

        # During grace period (no NPCs alive), distance to A tile for reward
        distance_to_a_tile = -1.0
        if enemies_remaining == 0 and self.state.wave_complete_timer >= 0:
            distance_to_a_tile = SimulatorGeometry.chebyshev_distance(
                self.state.player_x, self.state.player_y,
                A_TILE_X, A_TILE_Y
            )

        # Used blood barrage at high HP (> 95): switch to it (42) or attack with it (33-37 and current_weapon)
        # Exclude attacking nibblers and small blobs from the penalty
        excluded_blood_barrage_targets = {
            EntityTypes.NIBBLER,
            EntityTypes.BLOB_MAGE,
            EntityTypes.BLOB_RANGE,
            EntityTypes.BLOB_MELEE,
        }
        attack_target = self._resolve_attack_target(action) if InfernoAction.is_attack(action) else None
        attack_target_excluded = (
            attack_target is not None and attack_target.entity_type in excluded_blood_barrage_targets
        )
        is_blood_barrage = self.state.current_preset == GearPreset.MAGE and self.state.use_blood_barrage
        used_blood_barrage_at_high_hp = (
            self.health_at_step_start > self.state.max_health - 4
            and (
                action == 42  # SWITCH_BLOOD_BARRAGE
                or (
                    InfernoAction.is_attack(action)
                    and is_blood_barrage
                    and not attack_target_excluded
                )
            )
        )
        # Track blood barrage equipment state for HP-based healing incentives
        has_blood_barrage = is_blood_barrage
        # Only reward switching to blood barrage if we actually changed weapons (wasn't already equipped)
        was_blood_barrage = self.preset_at_step_start == GearPreset.MAGE and self.use_blood_at_step_start
        switched_to_blood_barrage = (
            action == InfernoAction.SWITCH_BLOOD_BARRAGE
            and not was_blood_barrage
        )

        priority_target_distance_now = -1.0
        priority_target_npc_los_now = False
        priority_target_player_los_now = False
        if self.priority_target_at_step_start is not None and not self.priority_target_at_step_start.is_dead():
            priority_target_distance_now = InfernoLineOfSight.get_distance_from_npc(
                self.priority_target_at_step_start.x,
                self.priority_target_at_step_start.y,
                self.priority_target_at_step_start.entity_type.size_in_tiles,
                self.state.player_x,
                self.state.player_y,
            )
            priority_target_npc_los_now = InfernoLineOfSight.can_entity_attack_player(
                self.priority_target_at_step_start,
                self.state.player_x,
                self.state.player_y,
                self.state.pillar_alive,
            )
            priority_target_player_los_now = InfernoLineOfSight.can_player_attack_entity(
                self.state.player_x,
                self.state.player_y,
                self.state.player_attack_range,
                self.priority_target_at_step_start,
                self.state.pillar_alive,
            )
        priority_target_entity_type = (
            None
            if self.priority_target_at_step_start is None or self.priority_target_at_step_start.is_dead()
            else self.priority_target_at_step_start.entity_type
        )

        return StepResult(
            damage_taken=damage_taken,
            pillar_damage_taken=pillar_damage,
            ne_pillar_damage_taken=ne_pillar_damage,
            damage_dealt=self.damage_dealt_this_tick,
            damage_dealt_to_mager=self.damage_dealt_to_mager_this_tick,
            player_damage_events=list(self.player_damage_events_this_tick),
            non_mager_enemies_at_step_start=self.non_mager_enemies_at_step_start,
            health_gained=health_gained,
            health_at_step_start=self.health_at_step_start,
            used_blood_barrage_at_high_hp=used_blood_barrage_at_high_hp,
            has_blood_barrage=has_blood_barrage,
            switched_to_blood_barrage=switched_to_blood_barrage,
            npcs_attacked_player=self.npcs_attacked_player_this_tick,
            npcs_with_los_now=npcs_with_los_now,
            avoidable_extra_los=avoidable_extra_los,
            avoidable_extra_imminent=avoidable_extra_imminent,
            npcs_with_los_prev=npcs_with_los_prev,
            consecutive_single_los_ticks=self._consecutive_single_los_ticks,
            dangerous_npcs_alive=dangerous_npcs_alive,
            dangerous_npcs_with_los=dangerous_npcs_with_los,
            kills_by_type=dict(self.kills_this_tick),
            wave_completed=wave_completed,
            wave_number=self.state.current_wave,
            pillars_remaining=pillars_remaining,
            ne_pillar_alive=ne_pillar_alive,
            pillar_deaths=pillar_deaths,
            ne_pillar_died=ne_pillar_died,
            enemies_remaining=enemies_remaining,
            enemies_in_attack_range=enemies_in_range,
            nearest_enemy_distance=nearest_dist,
            player_died=player_died,
            wave_timeout=wave_timeout,
            inferno_complete=inferno_complete,
            mager_resurrected_count=self.mager_resurrected_this_tick,
            melee_resurrected_count=self.melee_resurrected_this_tick,
            bat_resurrected_count=self.bat_resurrected_this_tick,
            distance_to_a_tile=distance_to_a_tile,
            priority_target_distance_start=self.priority_target_distance_start,
            priority_target_distance_now=priority_target_distance_now,
            priority_target_npc_los_start=self.priority_target_npc_los_start,
            priority_target_npc_los_now=priority_target_npc_los_now,
            priority_target_player_los_start=self.priority_target_player_los_start,
            priority_target_player_los_now=priority_target_player_los_now,
            priority_target_reachable=(
                self.priority_target_player_los_start or priority_target_player_los_now
            ),
            priority_target_entity_type=priority_target_entity_type,
            action_was_valid=action_valid,
            executed_action=action,
            player_attacked_on_cooldown=self.attacked_on_cooldown_this_tick,
            ticks_in_wave=self.get_ticks_in_wave(),
            player_in_ne_pillar_zone=is_in_ne_pillar_zone(
                self.state.player_x, self.state.player_y
            ),
            ticks_since_last_attack=self.state.current_tick - self.state.player_last_attack_tick,
            ticks_since_engagement=self._ticks_since_engagement,
            player_x=self.state.player_x,
            player_y=self.state.player_y,
            adjacent_attacking_npc_count=adjacent_attacking_npc_count,
            player_moved=(
                self.state.player_x != self.player_x_at_step_start
                or self.state.player_y != self.player_y_at_step_start
            ),
            used_early_ice_barrage_on_nibblers=(
                InfernoAction.is_attack(action)
                and attack_target is not None
                and attack_target.entity_type == EntityTypes.NIBBLER
                and self.state.current_preset == GearPreset.MAGE
                and not self.state.use_blood_barrage
                and self.get_ticks_in_wave() <= 2
            ),
            max_health=self.state.max_health,
        )
