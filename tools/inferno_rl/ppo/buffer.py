from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import torch as th
from gymnasium import spaces
from numpy.typing import NDArray

from .running_mean_std import TensorRunningMeanStd


@dataclass(frozen=True)
class BufferSamples:
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    action_masks: th.Tensor
    episode_starts: th.Tensor | None = None  # (batch_windows, seq_len) — only for sequence batches
    burn_in_observations: th.Tensor | None = None  # (batch_windows * burn_in, obs_dim)
    burn_in_episode_starts: th.Tensor | None = None  # (batch_windows, burn_in)


class Buffer:
    def __init__(
        self,
        buffer_size: int,
        n_envs: int,
        observation_space: spaces.Box,
        action_space: spaces.MultiDiscrete,
        gae_lambda: float = 0.95,
        gamma: float = 0.99,
    ):
        self.buffer_size = buffer_size
        self.n_envs = n_envs
        self.observation_space = observation_space
        self.action_space = action_space
        self.observations = np.zeros(
            (self.buffer_size, self.n_envs, *self.observation_space.shape),
            dtype=np.float32,
        )
        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, len(self.action_space.nvec)), dtype=np.int32
        )
        self.action_masks = np.zeros(
            (self.buffer_size, self.n_envs, sum(self.action_space.nvec)), dtype=bool
        )
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.novelty = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=bool)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.infos = np.array(
            [[{} for _ in range(self.n_envs)] for _ in range(self.buffer_size)]
        )
        self.truncates = np.zeros((self.buffer_size, self.n_envs), dtype=bool)
        self.episode_rewards: list[list[float]] = [[] for _ in range(self.n_envs)]
        self.episode_lengths: list[list[int]] = [[] for _ in range(self.n_envs)]
        self.last_step_obs = np.zeros(
            (n_envs, *self.observation_space.shape), dtype=np.float32
        )
        self.last_step_dones = np.ones(shape=(n_envs,), dtype=bool)
        self.positions = np.zeros((self.n_envs,), dtype=np.int32)
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.finalized = False

    def is_full(self) -> bool:
        return np.all(self.positions >= self.buffer_size).item()

    def add_step_request(
        self,
        environment_indices: NDArray[np.int32],
        action: NDArray[np.int32],
        value: NDArray[np.float32],
        log_prob: NDArray[np.float32],
        action_masks: NDArray[np.bool_],
    ) -> None:
        remaining_input_indices = self.positions[environment_indices] < self.buffer_size
        remaining_env_indices = environment_indices[remaining_input_indices]

        positions = self.positions[remaining_env_indices]

        self.actions[positions, remaining_env_indices] = action[remaining_input_indices]
        self.log_probs[positions, remaining_env_indices] = log_prob[
            remaining_input_indices
        ]
        self.action_masks[positions, remaining_env_indices] = action_masks[
            remaining_input_indices
        ]
        self.values[positions, remaining_env_indices] = value[
            remaining_input_indices
        ].flatten()

    def add_step_response(
        self,
        environment_indices: NDArray[np.int32],
        obs: NDArray[np.float32],
        reward: NDArray[np.float32],
        episode_start: NDArray[np.bool_],
        truncate: NDArray[np.bool_],
        next_obs: NDArray[np.float32],
        done: NDArray[np.bool_],
        infos: NDArray[np.object_],
    ) -> None:
        remaining_input_indices = self.positions[environment_indices] < self.buffer_size
        remaining_env_indices = environment_indices[remaining_input_indices]

        positions = self.positions[remaining_env_indices]

        self.observations[positions, remaining_env_indices] = obs[
            remaining_input_indices
        ]
        self.rewards[positions, remaining_env_indices] = reward[remaining_input_indices]
        self.episode_starts[positions, remaining_env_indices] = episode_start[
            remaining_input_indices
        ]
        self.truncates[positions, remaining_env_indices] = truncate[
            remaining_input_indices
        ]
        self.infos[positions, remaining_env_indices] = infos[remaining_input_indices]

        self.positions[remaining_env_indices] += 1

        final_step_inputs = self.positions[environment_indices] == self.buffer_size
        final_step_indexes = environment_indices[final_step_inputs]
        self.last_step_obs[final_step_indexes] = next_obs[final_step_inputs]
        self.last_step_dones[final_step_indexes] = done[final_step_inputs]

    def generate_batches(self, batch_size: int, device: str) -> Iterator[BufferSamples]:
        n_samples = self.buffer_size * self.n_envs
        idxs = np.random.permutation(n_samples)

        reshaped_observations = self.observations.reshape(
            n_samples, *self.observation_space.shape
        )
        reshaped_actions = self.actions.reshape(n_samples, -1)
        reshaped_action_masks = self.action_masks.reshape(n_samples, -1)

        reshaped_values = self.values.flatten()
        reshaped_log_probs = self.log_probs.flatten()
        reshaped_advantages = self.advantages.flatten()
        reshaped_returns = self.returns.flatten()

        for start_idx in range(0, n_samples, batch_size):
            end_idx = start_idx + batch_size
            batch_indices = idxs[start_idx:end_idx]

            samples = BufferSamples(
                observations=th.as_tensor(
                    reshaped_observations[batch_indices],
                    dtype=th.float32,
                    device=device,
                ),
                actions=th.as_tensor(
                    reshaped_actions[batch_indices], dtype=th.int32, device=device
                ),
                old_values=th.as_tensor(
                    reshaped_values[batch_indices], dtype=th.float32, device=device
                ),
                old_log_prob=th.as_tensor(
                    reshaped_log_probs[batch_indices], dtype=th.float32, device=device
                ),
                advantages=th.as_tensor(
                    reshaped_advantages[batch_indices], dtype=th.float32, device=device
                ),
                returns=th.as_tensor(
                    reshaped_returns[batch_indices], dtype=th.float32, device=device
                ),
                action_masks=th.as_tensor(
                    reshaped_action_masks[batch_indices], dtype=th.bool, device=device
                ),
            )

            yield samples

    def generate_sequence_batches(
        self, seq_len: int, batch_size: int, device: str, burn_in_len: int = 0
    ) -> Iterator[BufferSamples]:
        """Yield non-overlapping windows of seq_len ticks for LSTM training.

        Each yielded batch has flat shape (batch_windows * seq_len, ...) for
        observations/actions/masks and scalar arrays, plus episode_starts shaped
        (batch_windows, seq_len) for hidden-state masking in the LSTM forward pass.
        When burn_in_len > 0, additional warmup observations are returned so the
        LSTM can reconstruct a better hidden state before the training window.
        """
        n_windows = self.buffer_size // seq_len
        total_windows = n_windows * self.n_envs

        env_indices = np.repeat(np.arange(self.n_envs), n_windows)
        window_indices = np.tile(np.arange(n_windows), self.n_envs)
        perm = np.random.permutation(total_windows)
        env_indices, window_indices = env_indices[perm], window_indices[perm]

        used_steps = n_windows * seq_len

        def reshape_windows(arr: np.ndarray) -> np.ndarray:
            base = arr[:used_steps]
            reshaped = base.reshape(n_windows, seq_len, self.n_envs, *arr.shape[2:])
            axes = (2, 0, 1, *range(3, reshaped.ndim))
            return reshaped.transpose(axes)

        obs_windows = reshape_windows(self.observations)
        action_windows = reshape_windows(self.actions)
        mask_windows = reshape_windows(self.action_masks)
        episode_start_windows = reshape_windows(self.episode_starts)
        log_prob_windows = reshape_windows(self.log_probs)
        value_windows = reshape_windows(self.values)
        advantage_windows = reshape_windows(self.advantages)
        return_windows = reshape_windows(self.returns)

        for b in range(0, total_windows, batch_size):
            envs = env_indices[b:b + batch_size]
            windows = window_indices[b:b + batch_size]
            bsz = len(envs)

            obs_w = obs_windows[envs, windows]
            ep_w = episode_start_windows[envs, windows]
            act_w = action_windows[envs, windows]
            mask_w = mask_windows[envs, windows]
            lp_w = log_prob_windows[envs, windows]
            val_w = value_windows[envs, windows]
            adv_w = advantage_windows[envs, windows]
            ret_w = return_windows[envs, windows]

            burn_obs_t: th.Tensor | None = None
            burn_ep_t: th.Tensor | None = None
            if burn_in_len > 0:
                obs_shape = self.observations.shape[2:]
                burn_obs = np.zeros((bsz, burn_in_len, *obs_shape), dtype=np.float32)
                burn_ep = np.ones((bsz, burn_in_len), dtype=bool)

                for i, (window_index, e) in enumerate(zip(windows, envs)):
                    t = int(window_index) * seq_len
                    available = min(burn_in_len, int(t))
                    if available <= 0:
                        continue

                    start = t - available
                    dest = burn_in_len - available
                    burn_obs[i, dest:] = self.observations[start:t, e]
                    burn_ep[i, dest:] = self.episode_starts[start:t, e]
                    if dest > 0:
                        # The first real warmup step should start from zero state,
                        # not from the synthetic zero-padding that precedes it.
                        burn_ep[i, dest] = True

            def flat(a: np.ndarray) -> np.ndarray:
                return a.reshape(bsz * seq_len, -1) if a.ndim > 2 else a.reshape(-1)

            if burn_in_len > 0:
                burn_obs_t = th.as_tensor(
                    burn_obs.reshape(bsz * burn_in_len, -1),
                    dtype=th.float32,
                    device=device,
                )
                burn_ep_t = th.as_tensor(burn_ep, dtype=th.bool, device=device)

            yield BufferSamples(
                observations=th.as_tensor(
                    flat(obs_w.reshape(bsz, seq_len, -1)), dtype=th.float32, device=device
                ),
                actions=th.as_tensor(flat(act_w), dtype=th.int32, device=device),
                action_masks=th.as_tensor(flat(mask_w), dtype=th.bool, device=device),
                old_log_prob=th.as_tensor(flat(lp_w), dtype=th.float32, device=device),
                old_values=th.as_tensor(flat(val_w), dtype=th.float32, device=device),
                advantages=th.as_tensor(flat(adv_w), dtype=th.float32, device=device),
                returns=th.as_tensor(flat(ret_w), dtype=th.float32, device=device),
                episode_starts=th.as_tensor(ep_w, dtype=th.bool, device=device),
                burn_in_observations=burn_obs_t,
                burn_in_episode_starts=burn_ep_t,
            )

    def finalize(
        self,
        ppo: "PPO",  # noqa: F821
        reward_normalizer: TensorRunningMeanStd | None = None,
        novelty_reward_scale: float = 0.0,
        hindsight_death_penalty: float = 0.0,
        hindsight_death_window: int = 10,
        hindsight_death_decay: float = 0.8,
    ) -> None:
        assert not self.finalized, "Buffer is already finalized"
        self.finalized = True
        self._compute_novelty_reward(ppo, novelty_reward_scale)
        self._apply_hindsight_death_penalty(
            hindsight_death_penalty, hindsight_death_window, hindsight_death_decay
        )
        self._bootstrap_truncates(ppo)
        self._compute_returns_and_advantage(ppo, reward_normalizer)
        self._calculate_episode_reward_and_length()

    def _apply_hindsight_death_penalty(
        self, peak: float, window: int, decay: float
    ) -> None:
        if peak <= 0.0:
            return

        total_penalties = 0
        for env in range(self.n_envs):
            # Find all terminal steps in this env's buffer
            terminal_steps: list[int] = []
            for t in range(self.buffer_size - 1):
                if self.episode_starts[t + 1, env]:
                    terminal_steps.append(t)
            if self.last_step_dones[env]:
                terminal_steps.append(self.buffer_size - 1)

            for term_t in terminal_steps:
                # Only penalize death/timeout, not successful completions
                term_info = self.infos[term_t, env]
                if not (term_info.get("player_died") or term_info.get("wave_timeout")):
                    continue

                # Find the start of this episode (first step where episode_starts is True
                # scanning backward from term_t)
                ep_start = 0
                for s in range(term_t, -1, -1):
                    if self.episode_starts[s, env]:
                        ep_start = s
                        break

                # Walk backward from term_t-1, applying decaying penalty
                for k in range(window):
                    target = term_t - 1 - k
                    if target < ep_start:
                        break
                    self.rewards[target, env] -= peak * (decay ** k)
                    total_penalties += 1

        if total_penalties > 0:
            print(f"  Hindsight death penalty: {total_penalties} steps penalized "
                  f"(peak={peak}, window={window}, decay={decay})")

    def _bootstrap_truncates(self, ppo: "PPO") -> None:  # noqa: F821
        for i in range(0, self.buffer_size):
            if np.any(self.truncates[i]):
                terminal_obs = np.stack(
                    [
                        info["terminal_observation"]
                        for info in self.infos[i, self.truncates[i]]
                    ]
                )
                _, _, _, terminal_values, *_ = ppo.predict(
                    th.as_tensor(terminal_obs, device=ppo.device),
                    None,
                    deterministic=False,
                    return_log_probs=False,
                    return_values=True,
                    return_actions=False,
                    return_entropy=False,
                )
                assert terminal_values is not None
                self.rewards[i][self.truncates[i]] += (
                    self.gamma * terminal_values.cpu().numpy().flatten()
                )

    def _compute_returns_and_advantage(
        self, ppo: "PPO", reward_normalizer: TensorRunningMeanStd | None  # noqa: F821
    ) -> None:
        _, _, _, last_values_tensor, *_ = ppo.predict(
            th.as_tensor(self.last_step_obs, device=ppo.device),
            None,
            deterministic=False,
            return_log_probs=False,
            return_values=True,
            return_actions=False,
            return_entropy=False,
        )

        assert last_values_tensor is not None
        last_values = last_values_tensor.cpu().numpy().flatten()

        if reward_normalizer is not None:
            self._normalize_rewards(last_values, reward_normalizer)

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - self.last_step_dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = (
                self.rewards[step]
                + self.gamma * next_values * next_non_terminal
                - self.values[step]
            )
            last_gae_lam = (
                delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            )
            self.advantages[step] = last_gae_lam

        self.returns = self.advantages + self.values

    def _calculate_episode_reward_and_length(self) -> None:
        current_rewards, current_lengths = np.zeros(self.n_envs), np.zeros(self.n_envs)

        for t in range(self.buffer_size):
            for i, episode_start in enumerate(self.episode_starts[t]):
                if episode_start:
                    if current_lengths[i] > 0:
                        self.episode_rewards[i].append(current_rewards[i].item())
                        self.episode_lengths[i].append(current_lengths[i].item())
                    current_rewards[i] = 0
                    current_lengths[i] = 0

            current_rewards += self.rewards[t]
            current_lengths += 1

        # Check last step (if episode ended last step of the rollout)
        for i, done in enumerate(self.last_step_dones):
            if done and current_lengths[i] > 0:
                self.episode_rewards[i].append(current_rewards[i].item())
                self.episode_lengths[i].append(current_lengths[i].item())

    def _normalize_rewards(
        self, last_values: NDArray[np.float32], reward_normalizer: TensorRunningMeanStd
    ) -> None:
        # Normalize rewards based on standard deviation of historical cumulative episodic rewards
        cumulative_rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - self.last_step_dones
                next_rewards = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_rewards = cumulative_rewards[step + 1]
            cumulative_rewards[step] = (
                self.rewards[step] + self.gamma * next_non_terminal * next_rewards
            )

        # Adapt the rewards to a tensor, so we can use existing logic for it
        reward_normalizer.to(
            "cpu"
        )  # There's no reason to do this computation on the GPU, so always move to CPU

        flattened_reward_tensor = th.as_tensor(cumulative_rewards.reshape(-1, 1))
        reward_normalizer.update(flattened_reward_tensor)

        reward_tensor = th.as_tensor(self.rewards)
        normalized_reward_tensor = reward_normalizer.normalize(
            reward_tensor, apply_mean=False, clip=True
        )
        normalized_rewards = normalized_reward_tensor.numpy()

        assert normalized_rewards.shape == self.rewards.shape
        assert normalized_rewards.dtype == self.rewards.dtype
        self.rewards[:] = normalized_rewards  # Update in-place

    def _compute_novelty_reward(self, ppo: "PPO", novelty_reward_scale: float) -> None:  # noqa: F821
        if novelty_reward_scale == 0.0:
            return
        scaled_observations = ppo.meta.running_observation_stats.normalize(
            th.as_tensor(self.observations, device=ppo.device),
            clip=True,
        )

        # Subtract by 1, so we don't reward observations within 1 standard deviation since they aren't novel
        env_step_novelty_rewards = (
            (scaled_observations.abs() - 1).clamp(min=0).sum(dim=-1)
        )
        # Take first value of last dimension to get latest frame (if frame stacking)
        env_step_novelty_rewards = env_step_novelty_rewards[..., 0].cpu().numpy()

        self.rewards += env_step_novelty_rewards * novelty_reward_scale
        self.novelty += env_step_novelty_rewards


def merge_buffers(buffers: list[Buffer]) -> Buffer:
    assert buffers, "No buffers to merge"

    buffer_size = buffers[0].buffer_size
    observation_space = buffers[0].observation_space
    action_space = buffers[0].action_space
    gae_lambda = buffers[0].gae_lambda
    gamma = buffers[0].gamma
    finalized = buffers[0].finalized

    for buffer in buffers[1:]:
        assert buffer.buffer_size == buffer_size
        assert buffer.observation_space == observation_space
        assert buffer.action_space == action_space
        assert buffer.gae_lambda == gae_lambda
        assert buffer.gamma == gamma
        assert buffer.finalized == finalized

    merged_buffer = Buffer(
        buffer_size=buffer_size,
        n_envs=sum(buffer.n_envs for buffer in buffers),
        observation_space=observation_space,
        action_space=action_space,
        gae_lambda=gae_lambda,
        gamma=gamma,
    )

    merged_buffer.finalized = finalized

    merged_buffer.observations = np.concatenate(
        [buffer.observations for buffer in buffers], axis=1
    )
    merged_buffer.actions = np.concatenate(
        [buffer.actions for buffer in buffers], axis=1
    )
    merged_buffer.action_masks = np.concatenate(
        [buffer.action_masks for buffer in buffers], axis=1
    )
    merged_buffer.log_probs = np.concatenate(
        [buffer.log_probs for buffer in buffers], axis=1
    )
    merged_buffer.values = np.concatenate([buffer.values for buffer in buffers], axis=1)
    merged_buffer.rewards = np.concatenate(
        [buffer.rewards for buffer in buffers], axis=1
    )
    merged_buffer.novelty = np.concatenate(
        [buffer.novelty for buffer in buffers], axis=1
    )
    merged_buffer.episode_starts = np.concatenate(
        [buffer.episode_starts for buffer in buffers], axis=1
    )
    merged_buffer.advantages = np.concatenate(
        [buffer.advantages for buffer in buffers], axis=1
    )
    merged_buffer.returns = np.concatenate(
        [buffer.returns for buffer in buffers], axis=1
    )
    merged_buffer.infos = np.concatenate([buffer.infos for buffer in buffers], axis=1)
    merged_buffer.truncates = np.concatenate(
        [buffer.truncates for buffer in buffers], axis=1
    )
    merged_buffer.episode_rewards = sum(
        (buffer.episode_rewards for buffer in buffers), []
    )
    merged_buffer.episode_lengths = sum(
        (buffer.episode_lengths for buffer in buffers), []
    )
    merged_buffer.last_step_obs = np.concatenate(
        [buffer.last_step_obs for buffer in buffers], axis=0
    )
    merged_buffer.last_step_dones = np.concatenate(
        [buffer.last_step_dones for buffer in buffers], axis=0
    )
    merged_buffer.positions = np.concatenate(
        [buffer.positions for buffer in buffers], axis=0
    )

    return merged_buffer
