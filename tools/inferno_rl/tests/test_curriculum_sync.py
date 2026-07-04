from tools.inferno_rl.training.env import InfernoEnv


def test_sync_curriculum_advances_frontier_and_resets_streak() -> None:
    env = InfernoEnv(start_wave=1, max_wave=66, promote_after=5)
    env._frontier_wave = 32
    env._consecutive_completions = 4

    env.sync_curriculum(frontier_wave=33, min_waves_to_advance=1)

    assert env._frontier_wave == 33
    assert env._min_waves_to_advance == 1
    assert env._consecutive_completions == 0


def test_sync_curriculum_never_rolls_back_frontier_or_phase_requirement() -> None:
    env = InfernoEnv(start_wave=1, max_wave=66, promote_after=5)
    env._frontier_wave = 40
    env._min_waves_to_advance = 2
    env._consecutive_completions = 3

    env.sync_curriculum(frontier_wave=39, min_waves_to_advance=1)

    assert env._frontier_wave == 40
    assert env._min_waves_to_advance == 2
    assert env._consecutive_completions == 3


def test_prestige_then_sync_reclimbs_from_reset_state() -> None:
    env = InfernoEnv(start_wave=1, max_wave=66, promote_after=5)
    env._frontier_wave = 66
    env._consecutive_completions = 2

    env.apply_prestige(min_waves_to_advance=2)
    env.sync_curriculum(frontier_wave=5, min_waves_to_advance=2)

    assert env._frontier_wave == 5
    assert env._min_waves_to_advance == 2
    assert env._consecutive_completions == 0
