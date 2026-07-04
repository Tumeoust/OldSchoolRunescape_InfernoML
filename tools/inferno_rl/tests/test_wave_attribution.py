from tools.inferno_rl.simulator.step_result import StepResult
from tools.inferno_rl.training.env import InfernoEnv


def test_full_mode_terminal_fail_is_recorded_on_death_wave_only() -> None:
    env = InfernoEnv(start_wave=49, max_wave=66, phase="sweep", episode_mode="full")
    env.current_episode_start_wave = 49
    env.simulator.state.current_wave = 63
    env._episode_cleared_waves = []

    updates = env._build_wave_stat_updates(StepResult(player_died=True), episode_success=False)

    assert updates == [{"wave": 63, "success": False}]


def test_full_mode_records_successes_for_cleared_waves_and_fail_for_terminal_wave() -> None:
    env = InfernoEnv(start_wave=49, max_wave=66, phase="sweep", episode_mode="full")
    env.current_episode_start_wave = 49
    env.simulator.state.current_wave = 63
    env._episode_cleared_waves = list(range(49, 63))

    updates = env._build_wave_stat_updates(StepResult(player_died=True), episode_success=False)

    assert updates[:-1] == [{"wave": wave, "success": True} for wave in range(49, 63)]
    assert updates[-1] == {"wave": 63, "success": False}


def test_opener_mode_keeps_single_wave_attribution() -> None:
    env = InfernoEnv(start_wave=49, max_wave=66, phase="backfill", episode_mode="opener")
    env.current_episode_start_wave = 49
    env.simulator.state.current_wave = 63
    env._episode_cleared_waves = list(range(49, 63))

    updates = env._build_wave_stat_updates(StepResult(player_died=True), episode_success=False)

    assert updates == [{"wave": 49, "success": False}]


def test_backfill_sampling_uses_corrected_per_wave_stats() -> None:
    env = InfernoEnv(start_wave=49, max_wave=63, phase="backfill", episode_mode="full")
    env._total_episodes = InfernoEnv.BACKFILL_WARMUP_EPISODES + 1
    env._wave_stats = {
        49: {"fails": 0, "successes": 100},
        50: {"fails": 0, "successes": 100},
        51: {"fails": 0, "successes": 100},
        52: {"fails": 0, "successes": 100},
        53: {"fails": 0, "successes": 100},
        54: {"fails": 0, "successes": 100},
        55: {"fails": 0, "successes": 100},
        56: {"fails": 0, "successes": 100},
        57: {"fails": 0, "successes": 100},
        58: {"fails": 0, "successes": 100},
        59: {"fails": 0, "successes": 100},
        60: {"fails": 0, "successes": 100},
        61: {"fails": 0, "successes": 100},
        62: {"fails": 0, "successes": 100},
        63: {"fails": 100, "successes": 0},
    }

    samples = [env._sample_backfill_wave() for _ in range(512)]

    assert samples.count(63) > samples.count(49)
