import numpy as np
import torch as th

from tools.inferno_rl.inference_state import StatefulPolicyRunner
from tools.inferno_rl.ppo.mlp_helper import default_mlp_config
from tools.inferno_rl.ppo.ppo import PPO, PolicyParams
from tools.inferno_rl.training.actions import (
    ACTION_HEAD_SIZES,
    POLICY_ACTION_DEPENDENCIES,
    POLICY_ACTION_MASK_SIZE,
)
from tools.inferno_rl.training.observation import (
    get_observation_size,
    get_public_observation_size,
)


def _build_ppo(*, recurrent: bool) -> PPO:
    actor_obs_size = get_public_observation_size()
    critic_obs_size = get_observation_size()
    params = PolicyParams(
        max_sequence_length=1,
        actor_input_size=actor_obs_size,
        critic_input_size=critic_obs_size,
        action_head_sizes=ACTION_HEAD_SIZES,
        actor_config=default_mlp_config([32]),
        critic_config=default_mlp_config([32]),
        action_dependencies=POLICY_ACTION_DEPENDENCIES,
        autoregressive_actions=True,
        lstm_hidden_size=128 if recurrent else None,
        observation_version="v4",
        policy_arch="flat_lstm_residual" if recurrent else "flat",
    )
    return PPO.new_instance(params, normalize_observations=False)


def test_stateful_runner_matches_manual_lstm_stepping() -> None:
    ppo = _build_ppo(recurrent=True)
    runner = StatefulPolicyRunner(ppo)
    obs_size = get_observation_size()
    observations = np.random.randn(4, obs_size).astype(np.float32)
    masks = np.ones((4, POLICY_ACTION_MASK_SIZE), dtype=bool)

    manual_state = None
    manual_actions = []
    manual_values = []
    for obs, mask in zip(observations, masks):
        obs_t = th.as_tensor(obs, dtype=th.float32).unsqueeze(0).unsqueeze(0)
        mask_t = th.as_tensor(mask, dtype=th.bool).unsqueeze(0)
        actions, _, _, values, _, manual_state = ppo.predict(
            obs_t,
            mask_t,
            deterministic=True,
            return_values=True,
            lstm_state=manual_state,
        )
        manual_actions.append(tuple(int(v) for v in actions.squeeze().tolist()))
        manual_values.append(float(values.squeeze()))

    runner_actions = []
    runner_values = []
    for obs, mask in zip(observations, masks):
        prediction = runner.predict(
            obs,
            mask,
            deterministic=True,
            return_values=True,
        )
        runner_actions.append(tuple(int(v) for v in np.asarray(prediction.action).tolist()))
        runner_values.append(prediction.value)

    assert runner._lstm_state is not None
    assert runner_actions == manual_actions
    np.testing.assert_allclose(runner_values, manual_values)

    runner.reset()
    assert runner._lstm_state is None
    first_prediction = runner.predict(
        observations[0],
        masks[0],
        deterministic=True,
        return_values=True,
    )
    assert tuple(int(v) for v in np.asarray(first_prediction.action).tolist()) == manual_actions[0]
    assert first_prediction.value == manual_values[0]


def test_stateful_runner_reset_is_noop_for_non_recurrent_policy() -> None:
    ppo = _build_ppo(recurrent=False)
    runner = StatefulPolicyRunner(ppo)
    obs_size = get_observation_size()
    obs = np.random.randn(obs_size).astype(np.float32)
    mask = np.ones(POLICY_ACTION_MASK_SIZE, dtype=bool)

    prediction = runner.predict(obs, mask, deterministic=True)
    assert prediction.action is not None
    assert runner._lstm_state is None

    runner.reset()
    assert runner._lstm_state is None
