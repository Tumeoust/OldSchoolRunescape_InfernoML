"""
Main Inferno simulator for RL training.

Optimized headless simulator for RL training.
This orchestrator composes domain-specific mixins and owns the step loop,
state setup, timer management, pillar collapses, and wave progression.
"""

from typing import Optional, Dict, List

from .entity import PlacedEntity, EntityTypes, InfernoEntityType
from .equipment import GearPreset, Loadout, DEFAULT_LOADOUT
from .combat import ALL_COMBAT_TABLES, CombatTables
from .exact_targeting import get_exact_target_slot_index, select_center_nibbler
from .forecast import TickThreatCache
from .geometry import SimulatorGeometry, InfernoLineOfSight
from .priority import combat_entity_sort_key
from .state import (
    SimulatorState, spawn_wave_entities,
    PLAYER_MAX_HEALTH, PILLAR_COLLAPSE_DAMAGE, WAVE_SPAWN_DELAY,
)

# Mixins (import order: leaf mixins first, then those with cross-mixin deps)
from .npc_movement import NpcMovementMixin
from .player_actions import PlayerActionsMixin
from .prayer_prediction import PrayerPredictionMixin
from .npc_combat import NpcCombatMixin
from .step_result import (
    StepResult, PlayerDamageEvent, MAX_TICKS_PER_WAVE, ResultBuilderMixin,
)


class InfernoSimulator(
    NpcMovementMixin,
    NpcCombatMixin,
    PlayerActionsMixin,
    PrayerPredictionMixin,
    ResultBuilderMixin,
):
    """
    Optimized headless simulator for RL training.

    Implements all Inferno mechanics:
    - NPC movement with dumb pathfinding
    - NPC attacks with prayer protection
    - Player combat with attack drag
    - Wave progression
    - Pillar destruction
    - Blob scan/attack mechanics
    - Meleer dig mechanics
    """

    def __init__(self, start_wave: int = 1, max_wave: int = 66):
        """
        Create simulator with wave range for curriculum learning.

        Args:
            start_wave: Wave to start/reset to
            max_wave: Max wave - completing this ends episode successfully
        """
        self.state = SimulatorState()
        self.start_wave = start_wave
        self.max_wave = max_wave

        # Tracking for step results
        self.entities_alive_at_step_start: set = set()
        self.kills_this_tick: Dict[InfernoEntityType, int] = {}
        self.health_at_step_start: int = PLAYER_MAX_HEALTH
        self.pillar_hp_at_step_start: List[int] = [0] * 3
        self.preset_at_step_start: GearPreset = GearPreset.BOFA
        self.use_blood_at_step_start: bool = False
        self.wave_start_tick: int = 1
        self.damage_dealt_this_tick: int = 0
        self.damage_dealt_to_mager_this_tick: int = 0
        self.player_damage_events_this_tick: List[PlayerDamageEvent] = []
        self.non_mager_enemies_at_step_start: int = 0
        self.npcs_attacked_player_this_tick: int = 0
        self.attacked_on_cooldown_this_tick: bool = False

        # LOS tracking for sustained-engagement check
        self._last_npcs_with_los: int = 0
        self._consecutive_single_los_ticks: int = 0

        # Stall detection: ticks since player dealt damage or had NPC LOS
        self._ticks_since_engagement: int = 0

        # Initial barrage heuristic
        self.initial_barrage_enabled: bool = False

        # When False, _process_auto_prayer is skipped (caller manages prayer)
        self.auto_prayer_enabled: bool = True

        # Death store for mager resurrection (cleared each wave)
        self.dead_mobs: List[PlacedEntity] = []
        self.mager_resurrected_this_tick: int = 0
        self.melee_resurrected_this_tick: int = 0
        self.bat_resurrected_this_tick: int = 0
        # Priority target tracking (highest threat at step start)
        self.priority_target_at_step_start: Optional[PlacedEntity] = None
        self.priority_target_distance_start: float = -1.0
        self.priority_target_npc_los_start: bool = False
        self.priority_target_player_los_start: bool = False

        # Per-step priority list cache for reward-shaping priority features.
        self._pre_step_priority_list: Optional[List[PlacedEntity]] = None
        self._post_step_tick_threat_cache: Optional[TickThreatCache] = None
        self.suppress_attack_drag_this_tick: bool = False

        # Loadout-specific combat tables (default: crystal BP / 99s)
        self.combat_tables: CombatTables = ALL_COMBAT_TABLES[DEFAULT_LOADOUT.id]

    def set_loadout(self, loadout: Loadout) -> None:
        """Configure simulator for a specific loadout."""
        self.combat_tables = ALL_COMBAT_TABLES[loadout.id]
        self.state.max_health = loadout.levels.hitpoints
        self.state.has_blowpipe = loadout.has_blowpipe
        self.state.loadout_preset_stats = loadout.preset_stats

    def reset(self):
        """Reset to starting wave with full health and intact pillars."""
        self.state.clear_entities()
        self.state.current_tick = 1
        self.state.reset_player()
        self.state.reset_pillars()
        self.state.active_prayer = None
        self.state.current_wave = self.start_wave - 1
        self.state.wave_complete_timer = -1
        self.state.attack_target = None

        # Clear death store to prevent resurrection of mobs from previous runs
        self.dead_mobs.clear()

        self._last_npcs_with_los = 0
        self._consecutive_single_los_ticks = 0
        self._ticks_since_engagement = 0
        self._wave_completed_this_tick = False
        self._post_step_tick_threat_cache = None
        self.suppress_attack_drag_this_tick = False

        # Spawn starting wave
        spawn_wave_entities(self.state, self.start_wave)
        self.wave_start_tick = self.state.current_tick

    def reset_to_wave(self, wave: int):
        """Reset to a specific wave."""
        self.state.clear_entities()
        self.state.current_tick = 1
        self.state.reset_player()
        self.state.reset_pillars()
        self.state.active_prayer = None
        self.state.current_wave = wave - 1
        self.state.wave_complete_timer = -1
        self.state.attack_target = None

        # Clear death store to prevent resurrection of mobs from previous runs
        self.dead_mobs.clear()

        self._last_npcs_with_los = 0
        self._consecutive_single_los_ticks = 0
        self._ticks_since_engagement = 0
        self._wave_completed_this_tick = False
        self._post_step_tick_threat_cache = None
        self.suppress_attack_drag_this_tick = False

        spawn_wave_entities(self.state, wave)
        self.wave_start_tick = self.state.current_tick

    def step(self, action: int) -> StepResult:
        """
        Execute one step with the given action.

        Args:
            action: Action index (0-51)

        Returns:
            StepResult with rewards/terminal info
        """
        # Capture state before processing
        self._capture_pre_step_state()

        # Apply heuristics (mutually exclusive: wave start vs between waves)
        effective_action = self._apply_initial_barrage_heuristic(action)
        effective_action = self._apply_between_wave_heuristic(effective_action)

        # Prayer prediction phase
        self._process_auto_prayer(effective_action)

        # Execution phase
        self.state.increment_tick()

        # Apply queued prayers
        self.state.process_action_queue()

        # Execute the action
        action_valid = self._execute_action(effective_action)

        # Track attack timing: player chose a valid attack while cooldown was ready
        # AND target is in range. Without the range check, the agent can farm this
        # reward by initiating out-of-range attacks and retreating from the drag.
        action_is_attack = 33 <= effective_action <= 46
        target_in_range = (
            self.state.attack_target is not None
            and not self.state.attack_target.is_dead()
            and self._can_player_attack_entity(self.state.attack_target)
        )
        self.attacked_on_cooldown_this_tick = (
            action_is_attack and action_valid and self.state.can_player_attack()
            and target_in_range
        )

        # Decrement entity timers
        self._decrement_entity_timers()

        # Handle attack drag BEFORE NPC movement
        self._handle_attack_drag_if_needed()

        # Move NPCs
        self._process_npc_movement()

        # Process NPC attacks
        self._process_npc_attacks()

        # Process player attack
        self._process_player_attack()

        # Process pillar collapses
        self._process_pillar_collapses()

        # Remove dead entities
        self._process_dead_entities()

        # Handle wave progression
        self._process_wave_progression()

        # Build and return result
        result = self._build_step_result(effective_action, action_valid)
        self.suppress_attack_drag_this_tick = False
        return result

    def get_ticks_in_wave(self) -> int:
        """Get ticks since wave started."""
        return self.state.current_tick - self.wave_start_tick

    def get_wave_phase(self) -> str:
        """
        Get current wave phase.

        Returns:
            One of: WAVE_START, SOLVING, CLEANUP, WAVE_COMPLETE
        """
        enemies_alive = len([e for e in self.state.entities if not e.is_dead()])

        if enemies_alive == 0:
            if self.state.wave_complete_timer >= 0:
                return "CLEANUP"
            return "WAVE_COMPLETE"

        if self.get_ticks_in_wave() < 5:
            return "WAVE_START"

        return "SOLVING"

    def is_inferno_complete(self) -> bool:
        """Check if training wave range is complete."""
        return self.state.current_wave >= self.max_wave and self.state.is_wave_cleared()

    def is_player_dead(self) -> bool:
        """Check if player is dead."""
        return self.state.player_health <= 0

    # ========================================================================
    # PRIVATE METHODS
    # ========================================================================

    def _capture_pre_step_state(self):
        """Capture state before processing for reward calculation."""
        self.health_at_step_start = self.state.player_health
        self.player_x_at_step_start = self.state.player_x
        self.player_y_at_step_start = self.state.player_y
        self.pillar_hp_at_step_start = [self.state.get_pillar_hp(i) for i in range(3)]
        self.preset_at_step_start = self.state.current_preset
        self.use_blood_at_step_start = self.state.use_blood_barrage

        self.entities_alive_at_step_start.clear()
        self.kills_this_tick.clear()
        self.damage_dealt_this_tick = 0
        self.damage_dealt_to_mager_this_tick = 0
        self.player_damage_events_this_tick = []
        self.npcs_attacked_player_this_tick = 0
        self.mager_resurrected_this_tick = 0
        self.melee_resurrected_this_tick = 0
        self.bat_resurrected_this_tick = 0
        self.attacked_on_cooldown_this_tick = False

        # Single pass: build alive set, priority sorted list, non-mager count
        self._pre_step_priority_list = []
        non_mager_count = 0
        for e in self.state.entities:
            if e.is_dead():
                continue
            self.entities_alive_at_step_start.add(id(e))
            if e.entity_type != EntityTypes.NIBBLER:
                self._pre_step_priority_list.append(e)
                if e.entity_type != EntityTypes.MAGER:
                    non_mager_count += 1
        self.non_mager_enemies_at_step_start = non_mager_count

        # Sort by priority for reward-shaping features.
        self._pre_step_priority_list.sort(key=lambda e: combat_entity_sort_key(
            e, self.state.player_x, self.state.player_y, self.state.pillar_alive,
        ))

        self.priority_target_at_step_start = (
            self._pre_step_priority_list[0] if self._pre_step_priority_list else None
        )
        self.priority_target_distance_start = -1.0
        self.priority_target_npc_los_start = False
        self.priority_target_player_los_start = False
        if self.priority_target_at_step_start is not None:
            self.priority_target_distance_start = InfernoLineOfSight.get_distance_from_npc(
                self.priority_target_at_step_start.x,
                self.priority_target_at_step_start.y,
                self.priority_target_at_step_start.entity_type.size_in_tiles,
                self.state.player_x,
                self.state.player_y,
            )
            self.priority_target_npc_los_start = InfernoLineOfSight.can_entity_attack_player(
                self.priority_target_at_step_start,
                self.state.player_x,
                self.state.player_y,
                self.state.pillar_alive,
            )
            self.priority_target_player_los_start = InfernoLineOfSight.can_player_attack_entity(
                self.state.player_x,
                self.state.player_y,
                self.state.player_attack_range,
                self.priority_target_at_step_start,
                self.state.pillar_alive,
            )

    def _apply_initial_barrage_heuristic(self, model_action: int) -> int:
        """Apply initial barrage heuristic - may override model's action."""
        if not self.initial_barrage_enabled:
            return model_action

        ticks_in_wave = self.get_ticks_in_wave()

        # Tick 1: Wait
        if ticks_in_wave == 0:
            return 0  # STAY

        # Tick 2: Barrage nibblers if they exist
        if ticks_in_wave == 1:
            target = select_center_nibbler(self.state.entities)
            if target is not None:
                slot_index = get_exact_target_slot_index(self.state, target)
                if slot_index is not None:
                    self.state.apply_gear_preset(GearPreset.MAGE)
                    self.state.use_blood_barrage = False
                    return 33 + slot_index

        # Tick 3: Switch to BoFa (high DPS ranged weapon)
        if ticks_in_wave == 2:
            self.state.apply_gear_preset(GearPreset.BOFA)
            return 48  # SWITCH_BOFA

        return model_action

    def _apply_between_wave_heuristic(self, model_action: int) -> int:
        """Between-wave heuristic disabled — model learns inter-wave behavior."""
        return model_action

    def _decrement_entity_timers(self):
        """Decrement timers for all entities."""
        for entity in self.state.entities:
            if not entity.is_dead():
                entity.reset_tick_flags()
                entity.decrement_timers()

    def _process_pillar_collapses(self):
        """Process pending pillar collapse damage."""
        for i in range(3):
            if self.state.pending_pillar_collapses[i] is not None:
                self.state.pending_pillar_collapses[i] -= 1

                if self.state.pending_pillar_collapses[i] <= 0:
                    # Apply collapse damage
                    self._apply_pillar_collapse_damage(i)
                    self.state.pending_pillar_collapses[i] = None

    def _apply_pillar_collapse_damage(self, pillar_index: int):
        """Apply damage from pillar collapse to nearby entities."""
        from .geometry import PILLARS

        pillar = PILLARS[pillar_index]
        px, py = pillar[0] + 1, pillar[1] + 1  # Center of pillar

        # Damage entities within radius
        for entity in self.state.entities:
            if entity.is_dead():
                continue

            dist = SimulatorGeometry.chebyshev_distance(entity.x, entity.y, px, py)
            if dist <= 2:  # Collapse radius
                entity.take_damage(PILLAR_COLLAPSE_DAMAGE)

    def _process_wave_progression(self):
        """Handle wave progression with the normal 9-tick inter-wave grace period."""
        if not self.state.is_wave_cleared():
            self._wave_completed_this_tick = False
            return

        # First clear tick: mark the wave complete and start the grace countdown.
        if self.state.wave_complete_timer < 0:
            self._wave_completed_this_tick = True
            if self.state.current_wave < self.max_wave:
                self.state.wave_complete_timer = WAVE_SPAWN_DELAY
            return

        self._wave_completed_this_tick = False
        if not self.state.tick_wave_complete_timer():
            return

        next_wave = self.state.current_wave + 1
        if next_wave <= self.max_wave:
            # Clear death store and kill tracking for the new wave.
            self.dead_mobs.clear()
            self.state.wave_kills.clear()

            self._last_npcs_with_los = 0
            self._consecutive_single_los_ticks = 0
            self._ticks_since_engagement = 0

            spawn_wave_entities(self.state, next_wave)
            self.wave_start_tick = self.state.current_tick
            self.state.wave_complete_timer = -1
