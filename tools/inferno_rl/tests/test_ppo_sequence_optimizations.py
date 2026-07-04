import numpy as np
import torch as th
from gymnasium import spaces

from tools.inferno_rl.ppo.buffer import Buffer
from tools.inferno_rl.ppo.mlp_helper import MlpConfig, default_mlp_config
from tools.inferno_rl.ppo.policy import Policy
from tools.inferno_rl.training.actions import (
    ACTION_HEAD_SIZES,
    POLICY_ACTION_DEPENDENCIES,
    POLICY_ACTION_MASK_SIZE,
)
from tools.inferno_rl.training.observation import (
    get_observation_size,
    get_public_observation_size,
)


def _make_recurrent_policy() -> Policy:
    th.manual_seed(0)
    return Policy(
        max_sequence_length=1,
        actor_input_size=get_public_observation_size(),
        critic_input_size=get_observation_size(),
        action_head_sizes=ACTION_HEAD_SIZES,
        feature_extractor_config=MlpConfig(),
        actor_config=default_mlp_config([32]),
        critic_config=default_mlp_config([32]),
        action_dependencies=POLICY_ACTION_DEPENDENCIES,
        autoregressive_actions=True,
        lstm_hidden_size=64,
        observation_version="v4",
        policy_arch="flat_lstm_residual",
    )


def _legacy_flat_lstm_residual_forward(
    policy: Policy,
    obs: th.Tensor,
    action_masks: th.Tensor,
    input_actions: th.Tensor,
    episode_starts: th.Tensor,
    lstm_state: tuple[th.Tensor, th.Tensor],
) -> tuple[
    th.Tensor,
    th.Tensor,
    th.Tensor,
    th.Tensor,
    th.Tensor,
    tuple[th.Tensor, th.Tensor],
]:
    assert policy.lstm is not None
    assert policy.lstm_input_norm is not None
    assert policy.lstm_input_encoder is not None

    batch_size, seq_len, _ = obs.shape
    actor_x = obs[..., : policy._actor_obs_size]
    critic_x = obs[..., : policy._critic_obs_size]
    normed_x = policy.lstm_input_norm(actor_x)
    encoded_x = policy.lstm_input_encoder(normed_x)

    h = lstm_state[0].clone()
    c = lstm_state[1].clone()
    outputs = []
    for t in range(seq_len):
        reset_mask = (1.0 - episode_starts[:, t].float()).view(1, batch_size, 1)
        h = h * reset_mask
        c = c * reset_mask
        step_out, (h, c) = policy.lstm(encoded_x[:, t:t + 1, :], (h, c))
        outputs.append(step_out)

    lstm_out = th.cat(outputs, dim=1)
    actor_features = th.cat([actor_x, lstm_out], dim=-1).reshape(batch_size * seq_len, -1)
    critic_features = th.cat([critic_x, lstm_out], dim=-1).reshape(batch_size * seq_len, -1)
    actions, log_probs, entropy, probs = policy.actor(
        actor_features,
        action_masks,
        input_actions=input_actions,
        return_entropy=True,
        return_log_probs=True,
        return_probs=True,
    )
    values = policy.critic(critic_features)
    assert log_probs is not None
    assert entropy is not None
    assert probs is not None
    return actions, log_probs, entropy, values, probs, (h, c)


def test_segmented_lstm_matches_legacy_step_loop() -> None:
    policy = _make_recurrent_policy()
    batch_size = 3
    seq_len = 7
    obs_size = get_observation_size()
    head_count = len(ACTION_HEAD_SIZES)

    th.manual_seed(1)
    obs = th.randn(batch_size, seq_len, obs_size)
    action_masks = th.ones((batch_size * seq_len, POLICY_ACTION_MASK_SIZE), dtype=th.bool)
    input_actions = th.randint(
        low=0,
        high=1,
        size=(batch_size * seq_len, head_count),
        dtype=th.int32,
    )
    for index, head_size in enumerate(ACTION_HEAD_SIZES):
        input_actions[:, index] = th.randint(
            low=0,
            high=head_size,
            size=(batch_size * seq_len,),
            dtype=th.int32,
        )

    episode_starts = th.tensor(
        [
            [True, False, False, True, False, False, False],
            [True, False, True, False, False, True, False],
            [True, False, False, False, True, False, False],
        ],
        dtype=th.bool,
    )
    lstm_state = (
        th.randn(1, batch_size, 64),
        th.randn(1, batch_size, 64),
    )

    expected = _legacy_flat_lstm_residual_forward(
        policy,
        obs,
        action_masks,
        input_actions,
        episode_starts,
        lstm_state,
    )
    actual = policy(
        obs,
        action_masks,
        input_actions=input_actions,
        return_actions=True,
        return_values=True,
        return_entropy=True,
        return_log_probs=True,
        return_probs=True,
        lstm_state=(lstm_state[0].clone(), lstm_state[1].clone()),
        episode_starts=episode_starts,
    )

    assert actual[0] is not None
    assert th.equal(actual[0], expected[0])
    for expected_tensor, actual_tensor in zip(expected[1:5], actual[1:5]):
        assert actual_tensor is not None
        assert th.allclose(actual_tensor, expected_tensor, atol=1e-6, rtol=1e-6)
    assert actual[5] is not None
    assert th.allclose(actual[5][0], expected[5][0], atol=1e-6, rtol=1e-6)
    assert th.allclose(actual[5][1], expected[5][1], atol=1e-6, rtol=1e-6)


def test_value_only_forward_does_not_require_action_masks() -> None:
    policy = _make_recurrent_policy()
    obs = th.randn(2, 4, get_observation_size())
    episode_starts = th.tensor(
        [[True, False, False, False], [True, False, True, False]],
        dtype=th.bool,
    )
    dummy_masks = th.ones((8, POLICY_ACTION_MASK_SIZE), dtype=th.bool)

    _, _, _, values_with_masks, _, state_with_masks = policy(
        obs,
        dummy_masks,
        return_actions=False,
        return_values=True,
        return_entropy=False,
        return_log_probs=False,
        return_probs=False,
        episode_starts=episode_starts,
    )
    _, _, _, values_without_masks, _, state_without_masks = policy(
        obs,
        None,
        return_actions=False,
        return_values=True,
        return_entropy=False,
        return_log_probs=False,
        return_probs=False,
        episode_starts=episode_starts,
    )

    assert values_with_masks is not None
    assert values_without_masks is not None
    assert state_with_masks is not None
    assert state_without_masks is not None
    assert th.allclose(values_without_masks, values_with_masks, atol=1e-6, rtol=1e-6)
    assert th.allclose(state_without_masks[0], state_with_masks[0], atol=1e-6, rtol=1e-6)
    assert th.allclose(state_without_masks[1], state_with_masks[1], atol=1e-6, rtol=1e-6)


def _manual_sequence_batches(
    buffer: Buffer,
    seq_len: int,
    batch_size: int,
    burn_in_len: int,
    perm: np.ndarray,
) -> list[dict[str, np.ndarray | None]]:
    n_windows = buffer.buffer_size // seq_len
    total_windows = n_windows * buffer.n_envs
    env_indices = np.repeat(np.arange(buffer.n_envs), n_windows)
    start_ts = np.tile(np.arange(n_windows) * seq_len, buffer.n_envs)
    env_indices = env_indices[perm]
    start_ts = start_ts[perm]
    batches: list[dict[str, np.ndarray | None]] = []

    for start in range(0, total_windows, batch_size):
        envs = env_indices[start:start + batch_size]
        ts = start_ts[start:start + batch_size]
        bsz = len(envs)

        def extract(arr: np.ndarray) -> np.ndarray:
            return np.stack([arr[t:t + seq_len, e] for t, e in zip(ts, envs)])

        burn_obs = None
        burn_ep = None
        if burn_in_len > 0:
            obs_shape = buffer.observations.shape[2:]
            burn_obs = np.zeros((bsz, burn_in_len, *obs_shape), dtype=np.float32)
            burn_ep = np.ones((bsz, burn_in_len), dtype=bool)
            for index, (t, env) in enumerate(zip(ts, envs)):
                available = min(burn_in_len, int(t))
                if available <= 0:
                    continue
                src_start = t - available
                dest = burn_in_len - available
                burn_obs[index, dest:] = buffer.observations[src_start:t, env]
                burn_ep[index, dest:] = buffer.episode_starts[src_start:t, env]
                if dest > 0:
                    burn_ep[index, dest] = True

        batches.append(
            {
                "observations": extract(buffer.observations).reshape(bsz * seq_len, -1),
                "actions": extract(buffer.actions).reshape(bsz * seq_len, -1),
                "action_masks": extract(buffer.action_masks).reshape(bsz * seq_len, -1),
                "old_log_prob": extract(buffer.log_probs).reshape(-1),
                "old_values": extract(buffer.values).reshape(-1),
                "advantages": extract(buffer.advantages).reshape(-1),
                "returns": extract(buffer.returns).reshape(-1),
                "episode_starts": extract(buffer.episode_starts),
                "burn_in_observations": (
                    None if burn_obs is None else burn_obs.reshape(bsz * burn_in_len, -1)
                ),
                "burn_in_episode_starts": burn_ep,
            }
        )

    return batches


def test_generate_sequence_batches_matches_legacy_extraction() -> None:
    observation_space = spaces.Box(low=-1.0, high=1.0, shape=(1, 4), dtype=np.float32)
    action_space = spaces.MultiDiscrete([3, 2])
    buffer = Buffer(
        buffer_size=8,
        n_envs=3,
        observation_space=observation_space,
        action_space=action_space,
    )

    buffer.observations[:] = np.arange(buffer.observations.size, dtype=np.float32).reshape(buffer.observations.shape)
    buffer.actions[..., 0] = (
        np.arange(buffer.buffer_size * buffer.n_envs, dtype=np.int32).reshape(buffer.buffer_size, buffer.n_envs) % 3
    )
    buffer.actions[..., 1] = (
        np.arange(buffer.buffer_size * buffer.n_envs, dtype=np.int32).reshape(buffer.buffer_size, buffer.n_envs) % 2
    )
    buffer.action_masks[:] = (
        np.arange(buffer.action_masks.size, dtype=np.int32).reshape(buffer.action_masks.shape) % 2
    ) == 0
    buffer.log_probs[:] = np.arange(buffer.log_probs.size, dtype=np.float32).reshape(buffer.log_probs.shape) / 10.0
    buffer.values[:] = np.arange(buffer.values.size, dtype=np.float32).reshape(buffer.values.shape) / 7.0
    buffer.advantages[:] = np.arange(buffer.advantages.size, dtype=np.float32).reshape(buffer.advantages.shape) / 5.0
    buffer.returns[:] = np.arange(buffer.returns.size, dtype=np.float32).reshape(buffer.returns.shape) / 3.0
    buffer.episode_starts[:] = False
    buffer.episode_starts[0, :] = True
    buffer.episode_starts[3, 1] = True
    buffer.episode_starts[4, 2] = True

    seq_len = 2
    batch_size = 4
    burn_in_len = 2
    total_windows = (buffer.buffer_size // seq_len) * buffer.n_envs

    np.random.seed(7)
    perm = np.random.permutation(total_windows)
    expected_batches = _manual_sequence_batches(
        buffer,
        seq_len=seq_len,
        batch_size=batch_size,
        burn_in_len=burn_in_len,
        perm=perm,
    )

    np.random.seed(7)
    actual_batches = list(
        buffer.generate_sequence_batches(
            seq_len=seq_len,
            batch_size=batch_size,
            device="cpu",
            burn_in_len=burn_in_len,
        )
    )

    assert len(actual_batches) == len(expected_batches)
    for expected, actual in zip(expected_batches, actual_batches):
        assert np.array_equal(actual.observations.cpu().numpy(), expected["observations"])
        assert np.array_equal(actual.actions.cpu().numpy(), expected["actions"])
        assert np.array_equal(actual.action_masks.cpu().numpy(), expected["action_masks"])
        assert np.allclose(actual.old_log_prob.cpu().numpy(), expected["old_log_prob"])
        assert np.allclose(actual.old_values.cpu().numpy(), expected["old_values"])
        assert np.allclose(actual.advantages.cpu().numpy(), expected["advantages"])
        assert np.allclose(actual.returns.cpu().numpy(), expected["returns"])
        assert np.array_equal(actual.episode_starts.cpu().numpy(), expected["episode_starts"])
        assert actual.burn_in_observations is not None
        assert actual.burn_in_episode_starts is not None
        assert np.array_equal(
            actual.burn_in_observations.cpu().numpy(),
            expected["burn_in_observations"],
        )
        assert np.array_equal(
            actual.burn_in_episode_starts.cpu().numpy(),
            expected["burn_in_episode_starts"],
        )
