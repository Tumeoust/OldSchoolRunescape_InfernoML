import pytest
from dataclasses import asdict

from tools.inferno_rl.simulator.entity import EntityTypes
from tools.inferno_rl.simulator.step_result import StepResult
from tools.inferno_rl.training.rewards import InfernoReward, RewardConfig, build_v44_reward_config


def _minimal_config(**overrides) -> RewardConfig:
    """Build a minimal (default) RewardConfig with optional overrides."""
    return RewardConfig(**overrides)


def _full_config(**overrides) -> RewardConfig:
    """Build a full-rewards config matching pre-V51 values."""
    defaults = dict(
        death_penalty=-20.0,
        wave_timeout_penalty=0.0,
        damage_taken_per_hp=-0.05,
        damage_dealt_per_hp=0.006,
        blood_barrage_heal_per_hp=0.06,
        blood_barrage_high_hp_penalty=-0.2,
        kill_reward_scale=1.0,
        invalid_action_penalty=-0.1,
        invalid_attack_penalty=-0.05,
        pillar_damage_per_hp=-0.01,
        ne_pillar_zone_bonus=0.008,
        ne_pillar_zone_penalty=-0.02,
        mager_resurrection_penalty=-0.6,
        melee_resurrection_penalty=-0.3,
        adjacent_npc_attack_penalty=-0.40,
        los_separation_bonus=0.025,
        avoidable_imminent_penalty=-0.01,
    )
    defaults.update(overrides)
    return RewardConfig(**defaults)


def test_single_los_no_longer_gives_positive_reward() -> None:
    reward = InfernoReward(_minimal_config())
    result = StepResult(
        enemies_remaining=1,
        ticks_in_wave=20,
        npcs_with_los_now=1,
        player_in_ne_pillar_zone=True,
        ticks_since_engagement=InfernoReward.ENGAGEMENT_SHAPING_WINDOW_TICKS + 1,
        priority_target_reachable=True,
    )

    assert reward.calculate(result) == 0.0


def test_los_separation_bonus_scales_with_blocked_fraction() -> None:
    config = _minimal_config(los_separation_bonus=0.025)
    reward = InfernoReward(config)
    # 3 dangerous NPCs alive, 1 with LOS, 2 blocked -> full bonus
    result = StepResult(
        enemies_remaining=3,
        ticks_in_wave=20,
        dangerous_npcs_alive=3,
        dangerous_npcs_with_los=1,
        player_in_ne_pillar_zone=True,
    )

    assert reward.calculate(result) == config.los_separation_bonus


def test_los_separation_not_given_with_single_dangerous_npc() -> None:
    config = _minimal_config(los_separation_bonus=0.025)
    reward = InfernoReward(config)
    result = StepResult(
        enemies_remaining=1,
        ticks_in_wave=20,
        dangerous_npcs_alive=1,
        dangerous_npcs_with_los=1,
        player_in_ne_pillar_zone=True,
    )
    assert reward.calculate(result) == 0.0


def test_multi_los_alone_has_no_penalty() -> None:
    reward = InfernoReward(_minimal_config())
    result = StepResult(
        enemies_remaining=2,
        ticks_in_wave=20,
        npcs_with_los_now=2,
        player_in_ne_pillar_zone=True,
        ticks_since_engagement=InfernoReward.ENGAGEMENT_SHAPING_WINDOW_TICKS + 1,
    )

    assert reward.calculate(result) == 0.0


def test_mager_resurrection_is_a_penalty() -> None:
    config = _full_config()
    reward = InfernoReward(config)
    result = StepResult(
        mager_resurrected_count=1,
        melee_resurrected_count=0,
        bat_resurrected_count=0,
    )

    assert reward.calculate(result) == config.mager_resurrection_penalty


def test_mager_delay_penalty_zeroed_by_default() -> None:
    config = _minimal_config()
    reward = InfernoReward(config)
    result = StepResult(
        priority_target_entity_type=EntityTypes.MAGER,
        priority_target_reachable=True,
        npcs_with_los_now=1,
        enemies_remaining=2,
    )

    assert config.mager_delay_penalty == 0.0
    assert reward.calculate(result) == 0.0


def test_post_wave_66_kill_rewards_are_absent() -> None:
    config = _minimal_config(kill_reward_scale=1.0)
    reward = InfernoReward(config)
    result = StepResult(
        kills_by_type={
            EntityTypes.JAD: 1,
            EntityTypes.HEALER: 1,
            EntityTypes.ZUK: 1,
            EntityTypes.ZUK_HEALER: 1,
        }
    )

    assert reward.calculate(result) == 0.0


def test_death_penalty() -> None:
    config = _minimal_config()
    reward = InfernoReward(config)
    result = StepResult(player_died=True)

    assert reward.calculate(result) == config.death_penalty
    assert config.death_penalty == -20.0


def test_wave_complete_at_start_wave() -> None:
    config = _minimal_config(start_wave=49, max_wave=66)
    reward = InfernoReward(config)
    result = StepResult(
        wave_completed=True,
        wave_number=49,
        health_at_step_start=99,
        max_health=99,
    )

    breakdown = reward.calculate_with_breakdown(result)
    components = dict(breakdown.get_nonzero_components())
    assert components["Wave 49 Complete"] == pytest.approx(config.wave_complete_base)
    assert components["Wave End HP Bonus"] == pytest.approx(config.wave_end_hp_bonus)


def test_wave_complete_at_max_wave() -> None:
    config = _minimal_config(start_wave=49, max_wave=66)
    reward = InfernoReward(config)
    result = StepResult(
        wave_completed=True,
        wave_number=66,
        health_at_step_start=99,
        max_health=99,
    )

    breakdown = reward.calculate_with_breakdown(result)
    components = dict(breakdown.get_nonzero_components())
    expected = config.wave_complete_base + config.wave_progress_bonus
    assert components["Wave 66 Complete"] == pytest.approx(expected)


def test_wave_complete_mid_range() -> None:
    config = _minimal_config(start_wave=49, max_wave=66)
    reward = InfernoReward(config)
    result = StepResult(
        wave_completed=True,
        wave_number=57,
        health_at_step_start=99,
        max_health=99,
    )

    breakdown = reward.calculate_with_breakdown(result)
    components = dict(breakdown.get_nonzero_components())
    progress = (57 - 49) / (66 - 49)
    expected = config.wave_complete_base + config.wave_progress_bonus * progress
    assert components["Wave 57 Complete"] == pytest.approx(expected)


def test_wave_progress_clamps_below_start() -> None:
    config = _minimal_config(start_wave=49, max_wave=66)
    reward = InfernoReward(config)
    result = StepResult(
        wave_completed=True,
        wave_number=45,
        health_at_step_start=99,
        max_health=99,
    )

    breakdown = reward.calculate_with_breakdown(result)
    components = dict(breakdown.get_nonzero_components())
    assert components["Wave 45 Complete"] == pytest.approx(config.wave_complete_base)


def test_inferno_complete_reward() -> None:
    config = _minimal_config(start_wave=49, max_wave=66)
    reward = InfernoReward(config)
    result = StepResult(inferno_complete=True)

    breakdown = reward.calculate_with_breakdown(result)
    components = dict(breakdown.get_nonzero_components())
    assert components["Inferno Complete!"] == pytest.approx(config.inferno_complete_reward)
    assert config.inferno_complete_reward == 15.0


# --- New tests for minimal config and kill scaling ---


def test_minimal_defaults_zero_removed_terms() -> None:
    """Minimal config produces zero reward for kills, healing, resurrection, etc."""
    config = _minimal_config()
    reward = InfernoReward(config)

    # Kill reward should be 0
    result = StepResult(kills_by_type={EntityTypes.MAGER: 1})
    assert reward.calculate(result) == 0.0

    # Blood barrage healing should be 0
    result = StepResult(health_gained=20)
    assert reward.calculate(result) == 0.0

    # Resurrection should be 0
    result = StepResult(mager_resurrected_count=1, melee_resurrected_count=0, bat_resurrected_count=0)
    assert reward.calculate(result) == 0.0

    # Damage taken is still active in minimal config
    result = StepResult(damage_taken=50)
    assert reward.calculate(result) == pytest.approx(50 * config.damage_taken_per_hp)


def test_kill_reward_scale() -> None:
    """kill_reward_scale multiplies base kill values."""
    half_config = _minimal_config(kill_reward_scale=0.5)
    full_config = _minimal_config(kill_reward_scale=1.0)

    result = StepResult(kills_by_type={EntityTypes.MAGER: 1})

    half_reward = InfernoReward(half_config).calculate(result)
    full_reward = InfernoReward(full_config).calculate(result)

    base_mager = InfernoReward._BASE_KILL_REWARDS[EntityTypes.MAGER]
    assert full_reward == pytest.approx(base_mager * 1.0)
    assert half_reward == pytest.approx(base_mager * 0.5)


def test_config_roundtrip_through_asdict() -> None:
    """RewardConfig survives asdict() -> RewardConfig(**dict) round-trip."""
    config = RewardConfig(
        death_penalty=-15.0,
        damage_dealt_per_hp=0.005,
        los_separation_bonus=0.02,
        kill_reward_scale=0.7,
        start_wave=50,
        max_wave=65,
    )
    roundtripped = RewardConfig(**asdict(config))
    assert roundtripped == config


# --- Timeout and stall penalty tests ---


def test_wave_timeout_penalty() -> None:
    """Default config applies -15.0 on wave timeout."""
    reward = InfernoReward(_minimal_config())
    result = StepResult(wave_timeout=True)
    assert reward.calculate(result) == pytest.approx(-15.0)


def test_wave_timeout_penalty_v44_is_zero() -> None:
    """V44 backward-compat config keeps timeout penalty at 0."""
    config = build_v44_reward_config(trained_rollouts=0)
    reward = InfernoReward(config)
    result = StepResult(wave_timeout=True)
    assert reward.calculate(result) == 0.0


def test_stall_penalty_no_doubling() -> None:
    """Repeated stall periods get the same penalty (no multiplier)."""
    config = _minimal_config(stall_base_penalty=-0.08, stall_escalation=0.04)
    reward = InfernoReward(config)

    def _stall_result(ticks_in_wave: int) -> StepResult:
        return StepResult(
            enemies_remaining=1,
            ticks_in_wave=ticks_in_wave,
            ticks_since_engagement=16,  # STALL_WINDOW(15) + 1 = first stall tick
        )

    r1 = reward.calculate(_stall_result(20))
    assert r1 == pytest.approx(-0.08)

    # 2nd stall period: same penalty, no doubling
    r2 = reward.calculate(_stall_result(50))
    assert r2 == pytest.approx(-0.08)

    # 3rd stall period: still the same
    r3 = reward.calculate(_stall_result(80))
    assert r3 == pytest.approx(-0.08)


def test_stall_escalation_per_tick() -> None:
    """Per-tick escalation increases penalty linearly within a stall period."""
    config = _minimal_config(stall_base_penalty=-0.08, stall_escalation=0.04)
    reward = InfernoReward(config)

    # First stall tick (stall_ticks=1): base penalty
    r1 = reward.calculate(StepResult(
        enemies_remaining=1, ticks_in_wave=20, ticks_since_engagement=16,
    ))
    assert r1 == pytest.approx(-0.08)

    # Second stall tick (stall_ticks=2): base - 1*escalation
    r2 = reward.calculate(StepResult(
        enemies_remaining=1, ticks_in_wave=21, ticks_since_engagement=17,
    ))
    assert r2 == pytest.approx(-0.08 - 0.04)
