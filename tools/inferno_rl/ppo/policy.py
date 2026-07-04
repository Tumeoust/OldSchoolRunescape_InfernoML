from functools import partial
from typing import Dict, List, Optional, Tuple, cast

import numpy as np
import torch as th
import torch.nn as nn

from .mlp_helper import (
    MlpConfig,
    create_mlp,
    default_mlp_config,
    init_weights,
)

# Type alias replacing pvp_ml.util.contract_loader.ActionDependencies
ActionDependencies = dict[int, dict[int, dict[str, list[tuple[int, int]]]]]


class Actor(nn.Module):
    action_dependencies: Dict[int, Dict[int, Dict[str, List[Tuple[int, int]]]]]

    def __init__(
        self,
        input_size: int,
        action_head_sizes: list[int],
        config: MlpConfig = MlpConfig(),
        action_dependencies: ActionDependencies = {},
        head_configs: MlpConfig | list[MlpConfig] | None = None,
        autoregressive_actions: bool = True,
        append_future_action_masks: bool = False,
        normalize_autoregressive_actions: bool = True,
    ):
        super(Actor, self).__init__()
        self.autoregressive_actions = autoregressive_actions
        self.normalize_autoregressive_actions = normalize_autoregressive_actions
        self.append_future_action_masks = append_future_action_masks

        hidden_mlp, hidden_size = create_mlp(config, input_size)

        self.hidden = hidden_mlp
        self.hidden.apply(partial(init_weights, gain=np.sqrt(2)))

        action_stats_shape = (sum(action_head_sizes),)
        self.register_buffer("action_mean", th.zeros(action_stats_shape))
        self.register_buffer("action_var", th.ones(action_stats_shape))
        self.register_buffer("action_count", th.tensor([1e-4]))

        # Explicit types for type-checking
        self.action_mean: th.Tensor
        self.action_var: th.Tensor
        self.action_count: th.Tensor

        def _create_head(index: int) -> nn.Module:
            head_config = (
                (
                    head_configs[index]
                    if isinstance(head_configs, list)
                    else head_configs
                )
                if head_configs is not None
                else MlpConfig()
            )

            autoregressive_size = (
                sum(action_head_sizes[:index]) if autoregressive_actions else 0
            )
            future_masks_size = (
                sum(action_head_sizes[index + 1 :]) if append_future_action_masks else 0
            )
            head_input_size = hidden_size + autoregressive_size + future_masks_size

            mlp, mlp_output_size = create_mlp(head_config, head_input_size)
            head = nn.Linear(mlp_output_size, action_head_sizes[index])

            mlp.apply(partial(init_weights, gain=np.sqrt(2)))
            head.apply(partial(init_weights, gain=0.01))

            return nn.Sequential(mlp, head)

        self.heads = nn.ModuleList(
            [_create_head(i) for i in range(len(action_head_sizes))]
        )

        self.action_head_sizes = th.tensor(action_head_sizes, dtype=th.long)

        self.action_dependencies = action_dependencies
        self._float32_eps = th.finfo(th.float32).eps

    def forward(
        self,
        x: th.Tensor,
        flattened_action_masks: th.Tensor,
        sample_deterministic: Optional[th.Tensor] = None,
        input_actions: Optional[th.Tensor] = None,
        return_log_probs: bool = True,
        return_entropy: bool = True,
        return_probs: bool = False,
    ) -> Tuple[
        th.Tensor, Optional[th.Tensor], Optional[th.Tensor], Optional[th.Tensor]
    ]:
        # action masks should be a flattened array of the action masks, so convert to action space dims
        action_head_sizes: List[int] = self.action_head_sizes.tolist()
        action_masks = th.split(flattened_action_masks, action_head_sizes, dim=1)
        actor_hidden = th.relu(self.hidden(x))

        actions: List[th.Tensor] = []
        log_probs: List[th.Tensor] = []
        one_hot_actions: List[th.Tensor] = []
        entropy: List[th.Tensor] = []
        probabilities: List[th.Tensor] = []

        for i, head in enumerate(self.heads):
            current_actor_hidden = actor_hidden

            if self.autoregressive_actions and i > 0:
                current_actions = th.cat(one_hot_actions, dim=-1)
                if self.normalize_autoregressive_actions:
                    current_actions = self._normalize(current_actions)
                current_actor_hidden = th.cat(
                    [current_actor_hidden, current_actions], dim=-1
                )

            if self.append_future_action_masks and i < len(self.heads) - 1:
                offset = sum(action_head_sizes[:i]) + 1
                current_actor_hidden = th.cat(
                    [current_actor_hidden, flattened_action_masks[..., offset:]], dim=-1
                )

            action_mask = action_masks[i]
            dependency_mask = self._get_action_dependency_mask(
                actions, i, x.shape[0], device=x.device
            )
            mask = action_mask & dependency_mask

            # If no actions are available, default to action 0 (the no-op action)
            no_action_mask = ~mask.any(dim=-1)
            mask[no_action_mask, 0] = True

            logits = head(current_actor_hidden)
            masked_logits = logits - ((~mask) * 1e8)
            probs = th.softmax(masked_logits, dim=-1)

            if input_actions is None:
                action = (
                    probs.argmax(dim=-1)
                    if sample_deterministic is not None and sample_deterministic[i]
                    else th.multinomial(probs, 1).squeeze(-1)
                )
            else:
                action = input_actions[:, i].long()

            actions.append(action)
            one_hot_actions.append(
                th.nn.functional.one_hot(action.detach(), action_head_sizes[i])
            )

            if return_log_probs:
                log_probs.append(self._log_prob(probs, action))

            if return_entropy:
                entropy.append(self._entropy(probs))

            if return_probs:
                probabilities.append(probs)

        combined_log_probs: Optional[th.Tensor] = None
        if return_log_probs:
            combined_log_probs = th.stack(log_probs, dim=1).sum(dim=1)

        combined_entropy: Optional[th.Tensor] = None
        if return_entropy:
            combined_entropy = th.stack(entropy, dim=1)

        combined_probs: Optional[th.Tensor] = None
        if return_probs:
            combined_probs = th.cat(probabilities, dim=1)

        return (
            th.stack(actions, dim=1),
            combined_log_probs,
            combined_entropy,
            combined_probs,
        )

    def _get_action_dependency_mask(
        self,
        previous_actions: List[th.Tensor],
        action_index: int,
        batch_size: int,
        device: th.device,
    ) -> th.Tensor:
        action_head_size = int(self.action_head_sizes[action_index].item())
        mask = th.ones(
            size=(batch_size, action_head_size), dtype=th.bool, device=device
        )

        if action_index not in self.action_dependencies:
            return mask

        action_dependencies = self.action_dependencies[action_index]

        for single_action_index, action_config in action_dependencies.items():
            single_mask = th.ones((batch_size,), dtype=th.bool, device=device)

            if "require_all" in action_config:
                for action_head_idx, action in action_config["require_all"]:
                    single_mask = single_mask & (
                        previous_actions[action_head_idx] == action
                    )

            if "require_any" in action_config:
                require_any_mask = th.zeros((batch_size,), dtype=th.bool)
                for action_head_idx, action in action_config["require_any"]:
                    require_any_mask = require_any_mask | (
                        previous_actions[action_head_idx] == action
                    )
                single_mask = single_mask & require_any_mask

            if "require_none" in action_config:
                for action_head_idx, action in action_config["require_none"]:
                    single_mask = single_mask & (
                        previous_actions[action_head_idx] != action
                    )

            mask[:, single_action_index] = single_mask

        return mask

    def _log_prob(self, probs: th.Tensor, value: th.Tensor) -> th.Tensor:
        clamped_probs = probs.clamp(min=self._float32_eps, max=1 - self._float32_eps)
        logits = th.log(clamped_probs)
        value = value.long().unsqueeze(-1)
        value, log_pmf = th.broadcast_tensors(value, logits)
        assert isinstance(value, th.Tensor)
        assert isinstance(log_pmf, th.Tensor)
        value = value[..., :1]
        return log_pmf.gather(-1, value).squeeze(-1)

    def _entropy(self, probs: th.Tensor) -> th.Tensor:
        clamped_probs = probs.clamp(min=self._float32_eps, max=1 - self._float32_eps)
        return -th.sum(probs * th.log(clamped_probs), dim=-1)

    def _normalize(self, actions: th.Tensor) -> th.Tensor:
        mean = self.action_mean[..., : actions.shape[-1]]
        var = self.action_var[..., : actions.shape[-1]]
        actions = actions - mean
        actions = actions / th.sqrt(var + 1e-8)
        actions = th.clamp(actions, -5, 5)
        return actions

    def update_action_normalization(self, actions: th.Tensor) -> None:
        assert len(actions.shape) == 2
        # Convert list of selected actions ints into one-hot-encoded
        tensors = []
        for i in range(0, len(self.action_head_sizes)):
            tensors.append(
                th.nn.functional.one_hot(
                    actions[..., i].to(th.int64), int(self.action_head_sizes[i].item())
                )
            )
        actions = th.cat(tensors, dim=-1).to(th.float32)
        batch_mean = th.mean(actions, dim=0)
        batch_var = th.var(actions, dim=0, unbiased=False)
        batch_count = actions.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(
        self, batch_mean: th.Tensor, batch_var: th.Tensor, batch_count: int
    ) -> None:
        delta = batch_mean - self.action_mean
        tot_count = self.action_count + batch_count

        new_mean = self.action_mean + delta * batch_count / tot_count
        m_a = self.action_var * self.action_count
        m_b = batch_var * batch_count
        m_2 = (
            m_a
            + m_b
            + th.square(delta)
            * self.action_count
            * batch_count
            / (self.action_count + batch_count)
        )
        new_var = m_2 / (self.action_count + batch_count)

        new_count = batch_count + self.action_count

        self.action_mean = new_mean
        self.action_var = new_var
        self.action_count = new_count


class Critic(nn.Module):
    def __init__(self, input_size: int, config: MlpConfig = MlpConfig()):
        super(Critic, self).__init__()
        hidden_mlp, hidden_size = create_mlp(config, input_size)
        self.hidden = hidden_mlp
        self.head = nn.Linear(hidden_size, 1)
        self.hidden.apply(partial(init_weights, gain=np.sqrt(2)))
        self.head.apply(partial(init_weights, gain=1))

    def forward(self, x: th.Tensor) -> th.Tensor:
        critic_hidden = th.relu(self.hidden(x))
        value: th.Tensor = self.head(critic_hidden)
        return value.squeeze(-1)


class EntityPoolEncoder(nn.Module):
    def __init__(
        self,
        global_feature_size: int,
        entity_slot_size: int,
        max_entity_slots: int,
        entity_encoder_size: int = 128,
        fused_output_size: int = 256,
    ):
        super().__init__()
        self.global_feature_size = global_feature_size
        self.entity_slot_size = entity_slot_size
        self.max_entity_slots = max_entity_slots
        self.entity_encoder_size = entity_encoder_size

        self.entity_input_norm = nn.LayerNorm(entity_slot_size)
        self.entity_encoder = nn.Sequential(
            nn.Linear(entity_slot_size, entity_encoder_size),
            nn.ReLU(),
            nn.Linear(entity_encoder_size, entity_encoder_size),
            nn.ReLU(),
        )
        self.global_input_norm = nn.LayerNorm(global_feature_size)
        self.global_encoder = nn.Sequential(
            nn.Linear(global_feature_size, entity_encoder_size),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(entity_encoder_size * 3, fused_output_size),
            nn.ReLU(),
        )

        self.entity_encoder.apply(partial(init_weights, gain=np.sqrt(2)))
        self.global_encoder.apply(partial(init_weights, gain=np.sqrt(2)))
        self.fusion.apply(partial(init_weights, gain=np.sqrt(2)))

    def forward(self, x: th.Tensor) -> th.Tensor:
        batch_size, seq_len, obs_size = x.shape
        expected_obs_size = self.global_feature_size + (
            self.max_entity_slots * self.entity_slot_size
        )
        assert (
            obs_size == expected_obs_size
        ), f"Expected obs_size={expected_obs_size}, got {obs_size}"

        global_features = x[:, :, : self.global_feature_size]
        entity_features = x[:, :, self.global_feature_size :]
        entity_features = entity_features.reshape(
            batch_size, seq_len, self.max_entity_slots, self.entity_slot_size
        )
        entity_mask = entity_features[..., 0] > 0.5

        flat_entities = entity_features.reshape(-1, self.entity_slot_size)
        normed_entities = self.entity_input_norm(flat_entities)
        encoded_entities = self.entity_encoder(normed_entities).reshape(
            batch_size,
            seq_len,
            self.max_entity_slots,
            self.entity_encoder_size,
        )

        mask = entity_mask.unsqueeze(-1).to(encoded_entities.dtype)
        masked_sum = (encoded_entities * mask).sum(dim=2)
        entity_counts = mask.sum(dim=2).clamp(min=1.0)
        mean_pool = masked_sum / entity_counts

        neg_inf = th.full_like(encoded_entities, float("-inf"))
        masked_for_max = th.where(mask.bool(), encoded_entities, neg_inf)
        max_pool = masked_for_max.max(dim=2).values
        has_any_entities = entity_mask.any(dim=2, keepdim=True)
        max_pool = th.where(has_any_entities, max_pool, th.zeros_like(max_pool))

        flat_global = global_features.reshape(-1, self.global_feature_size)
        normed_global = self.global_input_norm(flat_global)
        global_embedding = self.global_encoder(normed_global).reshape(
            batch_size,
            seq_len,
            self.entity_encoder_size,
        )

        fused = th.cat([global_embedding, mean_pool, max_pool], dim=-1)
        fused = self.fusion(fused.reshape(-1, fused.shape[-1])).reshape(
            batch_size, seq_len, -1
        )
        return fused


class Policy(nn.Module):
    def __init__(
        self,
        max_sequence_length: int,
        actor_input_size: int,
        critic_input_size: int,
        action_head_sizes: list[int],
        feature_extractor_config: MlpConfig = MlpConfig(),
        share_feature_extractor: bool = False,
        critic_config: MlpConfig = default_mlp_config([64, 64]),
        actor_config: MlpConfig = default_mlp_config([128, 128, 128]),
        action_head_configs: MlpConfig | list[MlpConfig] | None = None,
        action_dependencies: ActionDependencies = {},
        autoregressive_actions: bool = True,
        append_future_action_masks: bool = False,
        normalize_autoregressive_actions: bool = True,
        lstm_hidden_size: int | None = None,
        observation_version: str = "v1",
        policy_arch: str = "flat",
        global_feature_size: int = 0,
        entity_slot_size: int = 0,
        max_entity_slots: int = 0,
        entity_encoder_size: int = 128,
    ):
        super(Policy, self).__init__()
        self._max_sequence_length = max_sequence_length
        self._share_feature_extractor = share_feature_extractor
        self._actor_obs_size = actor_input_size
        self._critic_obs_size = critic_input_size
        self._observation_version = observation_version
        self._policy_arch = policy_arch
        self.feature_extractor: nn.Module | None
        self.actor_feature_extractor: nn.Module | None
        self.critic_feature_extractor: nn.Module | None
        self.lstm: nn.LSTM | None
        self.lstm_input_norm: nn.LayerNorm | None
        self.lstm_input_encoder: nn.Module | None
        self.entity_pool_encoder: EntityPoolEncoder | None

        if policy_arch not in ("flat", "entity_pool_lstm", "flat_lstm_residual"):
            raise ValueError(f"Unsupported policy_arch={policy_arch!r}")

        if policy_arch == "entity_pool_lstm":
            if observation_version != "v2":
                raise ValueError("entity_pool_lstm requires observation_version='v2'")
            if lstm_hidden_size is None:
                raise ValueError("entity_pool_lstm requires lstm_hidden_size")
            if actor_input_size != critic_input_size:
                raise ValueError("entity_pool_lstm requires identical actor/critic input sizes")
            expected_input_size = global_feature_size + (
                entity_slot_size * max_entity_slots
            )
            if expected_input_size <= 0:
                raise ValueError("entity_pool_lstm requires positive V2 layout sizes")
            if actor_input_size != expected_input_size:
                raise ValueError(
                    f"entity_pool_lstm expected input size {expected_input_size}, got {actor_input_size}"
                )

            self._lstm_hidden_size = lstm_hidden_size
            self.lstm_input_norm = None
            self.lstm_input_encoder = None
            self.entity_pool_encoder = EntityPoolEncoder(
                global_feature_size=global_feature_size,
                entity_slot_size=entity_slot_size,
                max_entity_slots=max_entity_slots,
                entity_encoder_size=entity_encoder_size,
            )
            self.lstm = nn.LSTM(256, lstm_hidden_size, batch_first=True)
            nn.init.orthogonal_(self.lstm.weight_ih_l0)
            nn.init.orthogonal_(self.lstm.weight_hh_l0)
            nn.init.zeros_(self.lstm.bias_ih_l0)
            nn.init.zeros_(self.lstm.bias_hh_l0)
            actor_input_size = lstm_hidden_size
            critic_input_size = lstm_hidden_size
            self.feature_extractor = None
            self.actor_feature_extractor = None
            self.critic_feature_extractor = None
        elif policy_arch == "flat_lstm_residual":
            if lstm_hidden_size is None:
                raise ValueError("flat_lstm_residual requires lstm_hidden_size")
            self._lstm_hidden_size = lstm_hidden_size
            self.lstm_input_norm = nn.LayerNorm(actor_input_size)
            self.lstm_input_encoder = nn.Sequential(
                nn.Linear(actor_input_size, lstm_hidden_size),
                nn.ReLU(),
            )
            encoder_linear = cast(nn.Linear, self.lstm_input_encoder[0])
            nn.init.orthogonal_(encoder_linear.weight, gain=np.sqrt(2))
            nn.init.zeros_(encoder_linear.bias)
            self.lstm = nn.LSTM(lstm_hidden_size, lstm_hidden_size, batch_first=True)
            nn.init.orthogonal_(self.lstm.weight_ih_l0)
            nn.init.orthogonal_(self.lstm.weight_hh_l0)
            nn.init.zeros_(self.lstm.bias_ih_l0)
            nn.init.zeros_(self.lstm.bias_hh_l0)
            actor_input_size = actor_input_size + lstm_hidden_size
            critic_input_size = critic_input_size + lstm_hidden_size
            self.feature_extractor = None
            self.actor_feature_extractor = None
            self.critic_feature_extractor = None
            self.entity_pool_encoder = None
        elif lstm_hidden_size is not None:
            # LSTM path: bypass feature extractors entirely
            self._lstm_hidden_size = lstm_hidden_size
            self.lstm_input_norm = nn.LayerNorm(actor_input_size)
            self.lstm_input_encoder = nn.Sequential(
                nn.Linear(actor_input_size, actor_input_size),
                nn.ReLU(),
            )
            encoder_linear = cast(nn.Linear, self.lstm_input_encoder[0])
            nn.init.orthogonal_(encoder_linear.weight, gain=np.sqrt(2))
            nn.init.zeros_(encoder_linear.bias)
            self.lstm = nn.LSTM(actor_input_size, lstm_hidden_size, batch_first=True)
            nn.init.orthogonal_(self.lstm.weight_ih_l0)
            nn.init.orthogonal_(self.lstm.weight_hh_l0)
            nn.init.zeros_(self.lstm.bias_ih_l0)
            nn.init.zeros_(self.lstm.bias_hh_l0)
            actor_input_size = lstm_hidden_size
            critic_input_size = lstm_hidden_size
            self.feature_extractor = None
            self.actor_feature_extractor = None
            self.critic_feature_extractor = None
            self.entity_pool_encoder = None
        else:
            self._lstm_hidden_size = 0
            self.lstm_input_norm = None
            self.lstm_input_encoder = None
            self.lstm = None
            self.entity_pool_encoder = None
            if share_feature_extractor:
                assert (
                    actor_input_size == critic_input_size
                ), "Actor/critic input sizes must equal for shared layers"
                feature_extractor, hidden_size = create_mlp(
                    feature_extractor_config, max_sequence_length * actor_input_size
                )
                actor_input_size = hidden_size
                critic_input_size = hidden_size
                self.feature_extractor = feature_extractor
                self.actor_feature_extractor = None
                self.critic_feature_extractor = None
                self.feature_extractor.apply(partial(init_weights, gain=np.sqrt(2)))
            else:
                self.feature_extractor = None
                actor_feature_extractor, actor_input_size = create_mlp(
                    feature_extractor_config, max_sequence_length * actor_input_size
                )
                critic_feature_extractor, critic_input_size = create_mlp(
                    feature_extractor_config, max_sequence_length * critic_input_size
                )
                self.actor_feature_extractor = actor_feature_extractor
                self.critic_feature_extractor = critic_feature_extractor
                self.actor_feature_extractor.apply(partial(init_weights, gain=np.sqrt(2)))
                self.critic_feature_extractor.apply(partial(init_weights, gain=np.sqrt(2)))

        self._actor_input_size = actor_input_size
        self._critic_input_size = critic_input_size

        self.actor = Actor(
            actor_input_size,
            action_head_sizes,
            actor_config,
            action_dependencies,
            action_head_configs,
            autoregressive_actions=autoregressive_actions,
            append_future_action_masks=append_future_action_masks,
            normalize_autoregressive_actions=normalize_autoregressive_actions,
        )
        self.critic = Critic(critic_input_size, critic_config)

    def forward(
        self,
        x: th.Tensor,
        action_masks: Optional[th.Tensor],
        sample_deterministic: Optional[th.Tensor] = None,
        input_actions: Optional[th.Tensor] = None,
        return_actions: bool = True,
        return_values: bool = True,
        return_entropy: bool = True,
        return_log_probs: bool = True,
        return_probs: bool = False,
        lstm_state: Optional[Tuple[th.Tensor, th.Tensor]] = None,
        episode_starts: Optional[th.Tensor] = None,
    ) -> Tuple[
        Optional[th.Tensor],
        Optional[th.Tensor],
        Optional[th.Tensor],
        Optional[th.Tensor],
        Optional[th.Tensor],
        Optional[Tuple[th.Tensor, th.Tensor]],
    ]:
        if self.lstm is not None:
            return self._forward_lstm(
                x, action_masks, sample_deterministic, input_actions,
                return_actions, return_values, return_entropy, return_log_probs,
                return_probs, lstm_state, episode_starts,
            )
        return self._forward_mlp(
            x, action_masks, sample_deterministic, input_actions,
            return_actions, return_values, return_entropy, return_log_probs, return_probs,
        )

    def _forward_lstm(
        self,
        x: th.Tensor,
        action_masks: Optional[th.Tensor],
        sample_deterministic: Optional[th.Tensor],
        input_actions: Optional[th.Tensor],
        return_actions: bool,
        return_values: bool,
        return_entropy: bool,
        return_log_probs: bool,
        return_probs: bool,
        lstm_state: Optional[Tuple[th.Tensor, th.Tensor]],
        episode_starts: Optional[th.Tensor],
    ) -> Tuple[
        Optional[th.Tensor],
        Optional[th.Tensor],
        Optional[th.Tensor],
        Optional[th.Tensor],
        Optional[th.Tensor],
        Optional[Tuple[th.Tensor, th.Tensor]],
    ]:
        assert self.lstm is not None
        batch_size, seq_len, _ = x.shape
        needs_actor = return_actions or return_log_probs or return_entropy or return_probs

        if self._policy_arch == "entity_pool_lstm":
            assert self.entity_pool_encoder is not None
            x = self.entity_pool_encoder(x)
            base_x = None
        else:
            assert self.lstm_input_norm is not None
            assert self.lstm_input_encoder is not None
            base_x = x if self._policy_arch == "flat_lstm_residual" else None
            actor_x = x[..., : self._actor_obs_size]
            normed_x = self.lstm_input_norm(actor_x)
            if self._policy_arch == "flat_lstm_residual":
                x = self.lstm_input_encoder(normed_x)
            else:
                x = actor_x + self.lstm_input_encoder(normed_x)

        if lstm_state is not None:
            h, c = lstm_state[0], lstm_state[1]
        else:
            h = th.zeros(1, batch_size, self._lstm_hidden_size, device=x.device, dtype=x.dtype)
            c = th.zeros(1, batch_size, self._lstm_hidden_size, device=x.device, dtype=x.dtype)

        lstm_out, (h, c) = self._run_segmented_lstm(
            x,
            h,
            c,
            episode_starts,
        )
        actor_features: Optional[th.Tensor] = None
        critic_features: Optional[th.Tensor] = None
        if self._policy_arch == "flat_lstm_residual":
            assert base_x is not None
            actor_base_x = base_x[..., : self._actor_obs_size]
            critic_base_x = base_x[..., : self._critic_obs_size]
            actor_features = th.cat([actor_base_x, lstm_out], dim=-1).reshape(
                batch_size * seq_len, -1
            )
            critic_features = th.cat([critic_base_x, lstm_out], dim=-1).reshape(
                batch_size * seq_len, -1
            )
        else:
            features = lstm_out.reshape(batch_size * seq_len, -1)
            actor_features = features
            critic_features = features

        actions: Optional[th.Tensor] = None
        log_probs: Optional[th.Tensor] = None
        entropy: Optional[th.Tensor] = None
        probs: Optional[th.Tensor] = None
        if needs_actor:
            if action_masks is None:
                raise ValueError("action_masks is required when actor outputs are requested")
            assert actor_features is not None
            sampled_actions, log_probs, entropy, probs = self.actor(
                actor_features,
                action_masks,
                sample_deterministic=sample_deterministic,
                input_actions=input_actions,
                return_entropy=return_entropy,
                return_log_probs=return_log_probs,
                return_probs=return_probs,
            )
            if return_actions:
                actions = sampled_actions

        values: Optional[th.Tensor] = None
        if return_values:
            assert critic_features is not None
            values = self.critic(critic_features)

        new_lstm_state: Optional[Tuple[th.Tensor, th.Tensor]] = (h, c)
        return actions, log_probs, entropy, values, probs, new_lstm_state

    def _forward_mlp(
        self,
        x: th.Tensor,
        action_masks: Optional[th.Tensor],
        sample_deterministic: Optional[th.Tensor],
        input_actions: Optional[th.Tensor],
        return_actions: bool,
        return_values: bool,
        return_entropy: bool,
        return_log_probs: bool,
        return_probs: bool,
    ) -> Tuple[
        Optional[th.Tensor],
        Optional[th.Tensor],
        Optional[th.Tensor],
        Optional[th.Tensor],
        Optional[th.Tensor],
        Optional[Tuple[th.Tensor, th.Tensor]],
    ]:
        assert x.dim() == 3  # (batch_size, max_sequence_length, num_features)
        needs_actor = return_actions or return_log_probs or return_entropy or return_probs
        assert (
            x.shape[1] <= self._max_sequence_length
        ), f"Got {x.shape[1]} when expecting at most {self._max_sequence_length} for shape[1]"

        if needs_actor:
            assert (
                x.shape[2] >= self._actor_obs_size
            ), f"Got {x.shape[2]} when expecting >= {self._actor_obs_size} for shape[2]"
        if return_values:
            assert (
                x.shape[2] >= self._critic_obs_size
            ), f"Got {x.shape[2]} when expecting >= {self._critic_obs_size} for shape[2]"

        x = x.reshape(x.size(0), -1)  # Flatten frame stacked input

        actor_features: Optional[th.Tensor] = None
        critic_features: Optional[th.Tensor] = None
        if needs_actor or return_values:
            if self._share_feature_extractor:
                assert self.feature_extractor is not None
                actor_features = critic_features = self.feature_extractor(
                    x[..., : self._actor_input_size]
                )
            else:
                if needs_actor:
                    assert self.actor_feature_extractor is not None
                    actor_features = self.actor_feature_extractor(
                        x[..., : self._actor_input_size]
                    )
                if return_values:
                    assert self.critic_feature_extractor is not None
                    critic_features = self.critic_feature_extractor(
                        x[..., : self._critic_input_size]
                    )

        actions: Optional[th.Tensor] = None
        log_probs: Optional[th.Tensor] = None
        entropy: Optional[th.Tensor] = None
        probs: Optional[th.Tensor] = None
        if needs_actor:
            if action_masks is None:
                raise ValueError("action_masks is required when actor outputs are requested")
            assert actor_features is not None
            sampled_actions, log_probs, entropy, probs = self.actor(
                actor_features,
                action_masks,
                sample_deterministic=sample_deterministic,
                input_actions=input_actions,
                return_entropy=return_entropy,
                return_log_probs=return_log_probs,
                return_probs=return_probs,
            )
            if return_actions:
                actions = sampled_actions

        values: Optional[th.Tensor] = None
        if return_values:
            assert critic_features is not None
            values = self.critic(critic_features)

        return actions, log_probs, entropy, values, probs, None

    def _run_segmented_lstm(
        self,
        x: th.Tensor,
        h: th.Tensor,
        c: th.Tensor,
        episode_starts: Optional[th.Tensor],
    ) -> Tuple[th.Tensor, Tuple[th.Tensor, th.Tensor]]:
        assert self.lstm is not None
        batch_size, seq_len, _ = x.shape
        if seq_len == 0:
            empty = x.new_zeros((batch_size, 0, self._lstm_hidden_size))
            return empty, (h, c)

        boundaries = [0, seq_len]
        if episode_starts is not None:
            reset_columns = th.nonzero(
                episode_starts.any(dim=0),
                as_tuple=False,
            ).flatten().tolist()
            boundaries = sorted(set([0, seq_len, *reset_columns]))

        outputs: list[th.Tensor] = []
        for index in range(len(boundaries) - 1):
            start = boundaries[index]
            end = boundaries[index + 1]
            if episode_starts is not None:
                reset_rows = episode_starts[:, start]
                if bool(reset_rows.any()):
                    reset_mask = (~reset_rows).to(dtype=x.dtype).view(1, batch_size, 1)
                    h = h * reset_mask
                    c = c * reset_mask
            if end <= start:
                continue
            segment_out, (h, c) = self.lstm(x[:, start:end, :], (h, c))
            outputs.append(segment_out)

        if not outputs:
            empty = x.new_zeros((batch_size, 0, self._lstm_hidden_size))
            return empty, (h, c)
        return th.cat(outputs, dim=1), (h, c)
