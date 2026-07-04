from tools.inferno_rl.training.env import InfernoEnv


def test_backfill_warmup_uses_per_env_episode_count_not_global_stats() -> None:
    env = InfernoEnv(start_wave=1, max_wave=66, phase="sweep")
    env._wave_stats = {wave: {"fails": 10, "successes": 0} for wave in range(1, 67)}
    env._total_episodes = InfernoEnv.BACKFILL_WARMUP_EPISODES

    samples = {env._sample_backfill_wave() for _ in range(128)}

    assert len(samples) > 1


def test_backfill_sampling_uses_failure_weighting_after_warmup() -> None:
    env = InfernoEnv(start_wave=1, max_wave=2, phase="sweep")
    env._total_episodes = InfernoEnv.BACKFILL_WARMUP_EPISODES + 1
    env._wave_stats = {
        1: {"fails": 100, "successes": 0},
        2: {"fails": 0, "successes": 100},
    }

    samples = [env._sample_backfill_wave() for _ in range(512)]

    assert samples.count(1) > samples.count(2)
