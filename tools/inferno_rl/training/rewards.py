from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Tuple

from .actions import InfernoAction
from .schedules import ConstantSchedule, LinearSchedule, PiecewiseSchedule
from ..simulator.entity import EntityTypes
from ..simulator.simulator import StepResult


@dataclass
class RewardBreakdown:
    """Breakdown of individual reward components for a single step."""
    total: float = 0.0
    components: List[Tuple[str, float]] = field(default_factory=list)

    def add(self, name: str, value: float):
        """Add a reward component (only if non-zero)."""
        if value != 0.0:
            self.components.append((name, value))
            self.total += value

    def get_nonzero_components(self) -> List[Tuple[str, float]]:
        """Get list of (name, value) tuples for non-zero rewards."""
        return self.components


@dataclass(frozen=True)
class RewardConfig:
    # Terminal
    death_penalty: float = -20.0
    wave_timeout_penalty: float = -15.0

    # Damage
    damage_taken_per_hp: float = -0.05
    damage_dealt_per_hp: float = 0.003

    # Blood barrage
    blood_barrage_heal_per_hp: float = 0.0
    blood_barrage_high_hp_penalty: float = -0.2

    # Wave / inferno completion
    wave_complete_base: float = 3.0
    wave_progress_bonus: float = 5.0
    inferno_complete_reward: float = 15.0
    wave_end_hp_bonus: float = 3.0

    # Kill rewards — scale factor applied to _BASE_KILL_REWARDS values
    kill_reward_scale: float = 0.0

    # Stall detection
    stall_base_penalty: float = -0.08
    stall_escalation: float = 0.04

    # Invalid actions
    invalid_action_penalty: float = 0.0
    invalid_attack_penalty: float = 0.0

    # Pillar damage
    pillar_damage_per_hp: float = 0.0
    pillar_death_penalty: float = -7.5  # One-time penalty when NW/S pillar dies
    ne_pillar_death_penalty: float = -15.0  # One-time penalty when NE pillar dies

    # NE pillar zone positioning
    ne_pillar_zone_bonus: float = 0.0
    ne_pillar_zone_penalty: float = 0.0

    # Resurrection penalties
    mager_resurrection_penalty: float = 0.0
    melee_resurrection_penalty: float = 0.0

    # Mager priority / kill order
    mager_priority_per_npc: float = 0.25
    mager_early_kill_base: float = 0.6
    mager_early_kill_per_npc: float = 0.15
    mager_delay_penalty: float = 0.0

    # C tile early-wave positioning
    c_tile_on_reward: float = 0.0
    c_tile_adjacent_reward: float = 0.0

    # Tile A between-wave positioning
    tile_a_max_reward: float = 0.0

    # NPC proximity
    adjacent_npc_attack_penalty: float = 0.0

    # LOS separation
    los_separation_bonus: float = 0.01

    # Avoidable imminent
    avoidable_imminent_penalty: float = 0.0

    # Attack on cooldown
    attack_on_cooldown_bonus: float = 0.0

    # Weapon switch penalty
    weapon_switch_penalty: float = -0.005

    # Wave range (used for progress scaling)
    start_wave: int = 49
    max_wave: int = 66


# --- Legacy V44 schedule builders (kept for backward compatibility) ---

_V44_TILE_A_SCHEDULE = LinearSchedule(0.04, 0.0, 500)
_V44_AVOIDABLE_IMMINENT_SCHEDULE = ConstantSchedule(-0.01)
_V44_NE_ZONE_BONUS_SCHEDULE = LinearSchedule(0.008, 0.002, 2000)
_V44_NE_ZONE_PENALTY_SCHEDULE = LinearSchedule(-0.02, -0.005, 2000)
_V44_MAGER_PRIORITY_SCHEDULE = PiecewiseSchedule(((0, 0.25), (4000, 0.125)))
_V44_C_TILE_ON_SCHEDULE = LinearSchedule(0.5, 0.0, 500)
_V44_C_TILE_ADJ_SCHEDULE = LinearSchedule(0.25, 0.0, 500)


def build_legacy_reward_config() -> RewardConfig:
    return RewardConfig()


def build_v44_reward_config(
        trained_rollouts: int,
        start_wave: int = 49,
        max_wave: int = 66,
) -> RewardConfig:
    """Build a full-rewards config with V44 schedule-decayed values.

    Returns a RewardConfig with all original (non-minimal) reward values,
    plus schedule-driven decay for positioning/priority terms. Use this to
    restore pre-V51 reward behavior or as a reference for "full" values.
    """
    return RewardConfig(
        # Terminal
        death_penalty=-20.0,
        wave_timeout_penalty=0.0,
        # Damage
        damage_taken_per_hp=-0.05,
        damage_dealt_per_hp=0.006,
        # Blood barrage
        blood_barrage_heal_per_hp=0.06,
        blood_barrage_high_hp_penalty=-0.2,
        # Wave completion
        wave_complete_base=3.0,
        wave_progress_bonus=5.0,
        inferno_complete_reward=15.0,
        wave_end_hp_bonus=3.0,
        # Kill rewards
        kill_reward_scale=1.0,
        # Stall
        stall_base_penalty=-0.08,
        stall_escalation=0.04,
        # Invalid actions
        invalid_action_penalty=-0.1,
        invalid_attack_penalty=-0.05,
        # Pillar damage
        pillar_damage_per_hp=-0.01,
        # NE pillar zone (scheduled)
        ne_pillar_zone_bonus=_V44_NE_ZONE_BONUS_SCHEDULE.value(trained_rollouts),
        ne_pillar_zone_penalty=_V44_NE_ZONE_PENALTY_SCHEDULE.value(trained_rollouts),
        # Resurrection
        mager_resurrection_penalty=-0.6,
        melee_resurrection_penalty=-0.3,
        # Mager priority (scheduled)
        mager_priority_per_npc=_V44_MAGER_PRIORITY_SCHEDULE.value(trained_rollouts),
        mager_early_kill_base=0.6,
        mager_early_kill_per_npc=0.15,
        mager_delay_penalty=-0.02,
        # C tile (scheduled)
        c_tile_on_reward=_V44_C_TILE_ON_SCHEDULE.value(trained_rollouts),
        c_tile_adjacent_reward=_V44_C_TILE_ADJ_SCHEDULE.value(trained_rollouts),
        # Tile A (scheduled)
        tile_a_max_reward=_V44_TILE_A_SCHEDULE.value(trained_rollouts),
        # NPC proximity
        adjacent_npc_attack_penalty=-0.40,
        # LOS separation
        los_separation_bonus=0.025,
        # Avoidable imminent (scheduled)
        avoidable_imminent_penalty=_V44_AVOIDABLE_IMMINENT_SCHEDULE.value(trained_rollouts),
        # Attack on cooldown
        attack_on_cooldown_bonus=0.0,
        # Wave range
        start_wave=start_wave,
        max_wave=max_wave,
    )


class InfernoReward:
    """Reward calculator for the Inferno RL agent.

    All reward magnitudes are read from `self.config` (RewardConfig).
    Structural/timing constants remain as class-level attributes.
    """

    def __init__(self, config: RewardConfig | None = None):
        self.config = config or RewardConfig()

    def set_config(self, config: RewardConfig) -> None:
        self.config = config

    # Base kill rewards per entity type. Multiplied by config.kill_reward_scale.
    _BASE_KILL_REWARDS = {
        EntityTypes.MAGER: 0.6,
        EntityTypes.RANGER: 0.15,
        EntityTypes.MELEE: 0.35,
        EntityTypes.BLOB: 0.25,
        EntityTypes.BLOB_MAGE: 0.3,
        EntityTypes.BLOB_RANGE: 0.3,
        EntityTypes.BLOB_MELEE: 0.3,
        EntityTypes.BAT: 0.3,
        EntityTypes.NIBBLER: 0.25,
    }

    # Structural / timing constants (not configurable via CLI)
    TILE_A_REWARD_RADIUS = 5
    STALL_WINDOW = 15
    WAVE_START_GRACE_TICKS = 17
    ENGAGEMENT_SHAPING_WINDOW_TICKS = 4
    C_TILE_X = 19
    C_TILE_Y = 25
    C_TILE_ACTIVE_TICKS = 5
    NE_PILLAR_DAMAGE_MULTIPLIER = 3.0

    @staticmethod
    def _killed_magers(result: StepResult) -> int:
        if not result.kills_by_type:
            return 0
        return result.kills_by_type.get(EntityTypes.MAGER, 0)

    @staticmethod
    def _has_safely_focusable_priority_mager(result: StepResult) -> bool:
        if result.priority_target_entity_type != EntityTypes.MAGER:
            return False
        if result.npcs_with_los_now > 1 and result.avoidable_extra_los > 0:
            return False
        return result.priority_target_reachable

    def _has_recent_engagement(self, result: StepResult) -> bool:
        return result.ticks_since_engagement <= self.ENGAGEMENT_SHAPING_WINDOW_TICKS

    def calculate(self, result: StepResult) -> float:
        return self._calculate_internal(result, breakdown=None)

    def calculate_with_breakdown(self, result: StepResult) -> RewardBreakdown:
        breakdown = RewardBreakdown()
        self._calculate_internal(result, breakdown=breakdown)
        return breakdown

    def _calculate_internal(
            self,
            result: StepResult,
            breakdown: RewardBreakdown | None,
    ) -> float:
        def add(name: str, value: float) -> None:
            if value == 0.0:
                return
            if breakdown is None:
                nonlocal total
                total += value
                return
            breakdown.add(name, value)

        cfg = self.config
        total = 0.0
        killed_magers = self._killed_magers(result)
        safe_focus_mager = self._has_safely_focusable_priority_mager(result)
        made_mager_progress = result.damage_dealt_to_mager > 0 or killed_magers > 0

        # === TERMINALS ===
        if result.wave_timeout:
            add("Wave Timeout", cfg.wave_timeout_penalty)
            return breakdown.total if breakdown is not None else total
        if result.player_died:
            add("Death", cfg.death_penalty)
            return breakdown.total if breakdown is not None else total

        # === DAMAGE TAKEN ===
        if result.damage_taken > 0:
            add(
                f"Damage Taken ({result.damage_taken}HP)",
                result.damage_taken * cfg.damage_taken_per_hp,
            )

        # === LOS EXPOSURE (only during active combat, after grace period) ===
        past_grace = result.ticks_in_wave >= self.WAVE_START_GRACE_TICKS
        recent_engagement = self._has_recent_engagement(result)
        if result.enemies_remaining > 0 and past_grace:
            if result.avoidable_extra_imminent > 0:
                add(
                    "Avoidable Imminent",
                    result.avoidable_extra_imminent * cfg.avoidable_imminent_penalty,
                )
            if result.player_attacked_on_cooldown:
                add("Attack on Cooldown", cfg.attack_on_cooldown_bonus)

        # === COMBAT PROGRESS ===
        if result.damage_dealt > 0:
            add(
                f"Damage Dealt ({result.damage_dealt}HP)",
                result.damage_dealt * cfg.damage_dealt_per_hp,
            )

        # === MAGER PRIORITY DAMAGE BONUS ===
        if result.damage_dealt_to_mager > 0 and result.non_mager_enemies_at_step_start > 0:
            multiplier = (
                1.0
                + cfg.mager_priority_per_npc
                * result.non_mager_enemies_at_step_start
            )
            bonus_multiplier = multiplier - 1.0
            mager_priority_bonus = result.damage_dealt_to_mager * cfg.damage_dealt_per_hp * bonus_multiplier
            add("Mager Priority", mager_priority_bonus)

        self._calculate_kill_rewards_breakdown(result, add)
        if killed_magers > 0 and result.non_mager_enemies_at_step_start > 0:
            per_mager_bonus = (
                    cfg.mager_early_kill_base
                    + cfg.mager_early_kill_per_npc * result.non_mager_enemies_at_step_start
            )
            add("Early Mager Kill", killed_magers * per_mager_bonus)
        if safe_focus_mager and not made_mager_progress:
            add("Mager Delay", cfg.mager_delay_penalty)

        # === BLOOD BARRAGE HEALING ===
        if result.health_gained > 0:
            add(
                f"Blood Barrage Heal ({result.health_gained}HP)",
                result.health_gained * cfg.blood_barrage_heal_per_hp,
            )
        if result.used_blood_barrage_at_high_hp:
            add("Blood Barrage at High HP", cfg.blood_barrage_high_hp_penalty)

        # === WAVE / INFERNO COMPLETION ===
        if result.wave_completed:
            wave_range = max(1, cfg.max_wave - cfg.start_wave)
            progress = max(0.0, min(1.0, (result.wave_number - cfg.start_wave) / wave_range))
            scaled_base = cfg.wave_complete_base + cfg.wave_progress_bonus * progress
            add(f"Wave {result.wave_number} Complete", scaled_base)
            hp_bonus = (result.health_at_step_start / result.max_health) * cfg.wave_end_hp_bonus
            add("Wave End HP Bonus", hp_bonus)
        if result.inferno_complete:
            add("Inferno Complete!", cfg.inferno_complete_reward)

        # === PILLAR DAMAGE (all pillars, NE weighted 3x) ===
        if result.pillar_damage_taken > result.ne_pillar_damage_taken:
            non_ne = result.pillar_damage_taken - result.ne_pillar_damage_taken
            add("Pillar Damage", non_ne * cfg.pillar_damage_per_hp)
        if result.ne_pillar_damage_taken > 0:
            add(
                "NE Pillar Damage",
                result.ne_pillar_damage_taken
                * cfg.pillar_damage_per_hp
                * self.NE_PILLAR_DAMAGE_MULTIPLIER,
            )

        # === PILLAR DEATH EVENTS ===
        if result.pillar_deaths > 0:
            non_ne_deaths = result.pillar_deaths - (1 if result.ne_pillar_died else 0)
            if non_ne_deaths > 0:
                add("Pillar Death", non_ne_deaths * cfg.pillar_death_penalty)
            if result.ne_pillar_died:
                add("NE Pillar Death", cfg.ne_pillar_death_penalty)

        # === MAGER RESURRECTION ===
        other_resurrected = (
                result.mager_resurrected_count
                - result.melee_resurrected_count
                - result.bat_resurrected_count
        )
        if other_resurrected > 0:
            add("Mager Resurrection", other_resurrected * cfg.mager_resurrection_penalty)
        if result.melee_resurrected_count > 0:
            add("Melee Resurrection", result.melee_resurrected_count * cfg.melee_resurrection_penalty)

        # === STALL PENALTY (after grace period) ===
        if result.enemies_remaining > 0 and past_grace:
            stall_ticks = result.ticks_since_engagement - self.STALL_WINDOW
            if stall_ticks > 0:
                penalty = cfg.stall_base_penalty - (stall_ticks - 1) * cfg.stall_escalation
                add("Stall Penalty", penalty)

        # === NE PILLAR ZONE POSITIONING (during combat) ===
        if result.enemies_remaining > 0:
            if result.player_in_ne_pillar_zone:
                if (not past_grace or (result.npcs_with_los_now >= 1 and recent_engagement)) and not (
                        safe_focus_mager and not made_mager_progress
                ):
                    add("NE Pillar Zone", cfg.ne_pillar_zone_bonus)
            elif past_grace:
                add("NE Pillar Zone Penalty", cfg.ne_pillar_zone_penalty)

        # === C TILE EARLY-WAVE POSITIONING ===
        if result.ticks_in_wave < self.C_TILE_ACTIVE_TICKS:
            dx = abs(result.player_x - self.C_TILE_X)
            dy = abs(result.player_y - self.C_TILE_Y)
            chebyshev = max(dx, dy)
            if chebyshev == 0:
                add("C Tile Position", cfg.c_tile_on_reward)
            elif chebyshev == 1:
                add("C Tile Position", cfg.c_tile_adjacent_reward)

        # === NPC PROXIMITY PENALTY (event-based) ===
        if result.adjacent_attacking_npc_count > 0:
            add("NPC Proximity", cfg.adjacent_npc_attack_penalty * result.adjacent_attacking_npc_count)

        # === LOS SEPARATION BONUS (gradient for pillar use) ===
        if (past_grace
                and result.dangerous_npcs_alive > 1
                and result.dangerous_npcs_with_los >= 1):
            blocked = result.dangerous_npcs_alive - result.dangerous_npcs_with_los
            max_blockable = result.dangerous_npcs_alive - 1
            fraction = blocked / max_blockable if max_blockable > 0 else 1.0
            add("LOS Separation", cfg.los_separation_bonus * fraction)

        if not result.action_was_valid and not InfernoAction.is_attack(result.executed_action):
            add("Invalid Action", cfg.invalid_action_penalty)
        if InfernoAction.is_attack(result.executed_action) and not result.action_was_valid:
            add("Invalid Attack", cfg.invalid_attack_penalty)

        # === WEAPON SWITCH PENALTY ===
        if InfernoAction.is_weapon_switch(result.executed_action):
            add("Weapon Switch", cfg.weapon_switch_penalty)

        # === TILE A PROXIMITY (between waves only) ===
        if result.distance_to_a_tile >= 0:
            tile_a_reward = max(
                0.0,
                cfg.tile_a_max_reward
                * (1.0 - result.distance_to_a_tile / self.TILE_A_REWARD_RADIUS),
            )
            add("Tile A Proximity", tile_a_reward)

        return breakdown.total if breakdown is not None else total

    def _calculate_kill_rewards_breakdown(self, result: StepResult, add) -> None:
        scale = self.config.kill_reward_scale
        if scale == 0.0:
            return
        for entity_type, count in result.kills_by_type.items():
            base_reward = self._BASE_KILL_REWARDS.get(entity_type, 0.0)
            kill_reward = base_reward * scale
            if kill_reward > 0 and count > 0:
                name = entity_type.name if hasattr(entity_type, 'name') else str(entity_type)
                if name.startswith("Jal-"):
                    name = name[4:]
                add(f"Kill {name}" + (f" x{count}" if count > 1 else ""), count * kill_reward)


_REWARD_TERM_NORMALIZERS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^Damage Taken \(\d+HP\)$"), "Damage Taken"),
    (re.compile(r"^Damage Dealt \(\d+HP\)$"), "Damage Dealt"),
    (re.compile(r"^Blood Barrage Heal \(\d+HP\)$"), "Blood Barrage Heal"),
    (re.compile(r"^Kill (.+?) x\d+$"), r"Kill \1"),
    (re.compile(r"^Wave \d+ Complete$"), "Wave Complete"),
]


def normalize_reward_term_name(name: str) -> str:
    """Normalize dynamic breakdown labels to stable names for logging/aggregation."""
    for pattern, replacement in _REWARD_TERM_NORMALIZERS:
        if pattern.match(name):
            return pattern.sub(replacement, name)
    return name
