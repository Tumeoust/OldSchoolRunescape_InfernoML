import pytest
import torch as th

from tools.inferno_rl.ppo.mlp_helper import MlpConfig, default_mlp_config
from tools.inferno_rl.ppo.policy import Policy
from tools.inferno_rl.train_gpu import _validate_training_configuration
from tools.inferno_rl.training.actions import (
    ACTION_HEAD_SIZES,
    POLICY_ACTION_DEPENDENCIES,
    POLICY_ACTION_MASK_SIZE,
)
from tools.inferno_rl.training.observation import (
    get_observation_size,
    get_public_observation_size,
)


def test_flat_lstm_residual_forward_shapes() -> None:
    actor_obs_size = get_public_observation_size()
    critic_obs_size = get_observation_size()
    policy = Policy(
        max_sequence_length=1,
        actor_input_size=actor_obs_size,
        critic_input_size=critic_obs_size,
        action_head_sizes=ACTION_HEAD_SIZES,
        feature_extractor_config=MlpConfig(),
        actor_config=default_mlp_config([32]),
        critic_config=default_mlp_config([32]),
        action_dependencies=POLICY_ACTION_DEPENDENCIES,
        autoregressive_actions=True,
        lstm_hidden_size=128,
        observation_version="v4",
        policy_arch="flat_lstm_residual",
    )

    obs = th.randn(2, 4, critic_obs_size)
    masks = th.ones((8, POLICY_ACTION_MASK_SIZE), dtype=th.bool)
    episode_starts = th.zeros((2, 4), dtype=th.bool)
    episode_starts[:, 0] = True

    actions, log_probs, entropy, values, probs, lstm_state = policy(
        obs,
        masks,
        return_actions=True,
        return_values=True,
        return_entropy=True,
        return_log_probs=True,
        return_probs=True,
        episode_starts=episode_starts,
    )

    assert policy._actor_input_size == actor_obs_size + 128
    assert policy._critic_input_size == critic_obs_size + 128
    assert actions is not None and actions.shape == (8, len(ACTION_HEAD_SIZES))
    assert log_probs is not None and log_probs.shape == (8,)
    assert entropy is not None and entropy.shape == (8, len(ACTION_HEAD_SIZES))
    assert values is not None and values.shape == (8,)
    assert probs is not None and probs.shape == (8, POLICY_ACTION_MASK_SIZE)
    assert lstm_state is not None
    assert lstm_state[0].shape == (1, 2, 128)
    assert lstm_state[1].shape == (1, 2, 128)


def test_validation_accepts_v4_flat_lstm_residual() -> None:
    _validate_training_configuration(
        observation_version="v4",
        policy_arch="flat_lstm_residual",
        lstm_hidden_size=128,
        curriculum_mode="static",
        episode_mode="full",
        opener_min_health=40,
        opener_tick_limit=50,
        load_path=None,
    )


@pytest.mark.parametrize(
    ("observation_version", "policy_arch", "lstm_hidden_size", "expected_message"),
    [
        ("v4", "flat_lstm_residual", None, "flat_lstm_residual requires --lstm-hidden-size"),
        ("v4", "flat", 128, "policy_arch='flat' does not support LSTM"),
        ("v3.2", "flat_lstm_residual", 128, "Unsupported observation_version='v3.2'; expected 'v4'"),
    ],
)
def test_validation_rejects_invalid_combinations(
    observation_version: str,
    policy_arch: str,
    lstm_hidden_size: int | None,
    expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        _validate_training_configuration(
            observation_version=observation_version,
            policy_arch=policy_arch,
            lstm_hidden_size=lstm_hidden_size,
            curriculum_mode="static",
            episode_mode="full",
            opener_min_health=40,
            opener_tick_limit=50,
            load_path=None,
        )
