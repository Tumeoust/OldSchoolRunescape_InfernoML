import dataclasses
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import torch as th
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from .buffer import Buffer
from .policy import Policy, ActionDependencies
from .mlp_helper import MlpConfig, default_mlp_config
from .running_mean_std import TensorRunningMeanStd


@dataclass(frozen=True)
class PolicyParams:
    max_sequence_length: int
    actor_input_size: int
    critic_input_size: int
    action_head_sizes: list[int]
    feature_extractor_config: MlpConfig = field(default_factory=lambda: MlpConfig())
    share_feature_extractor: bool = False
    critic_config: MlpConfig = field(
        default_factory=lambda: default_mlp_config([64, 64])
    )
    actor_config: MlpConfig = field(
        default_factory=lambda: default_mlp_config([128, 128, 128])
    )
    action_head_configs: MlpConfig | list[MlpConfig] | None = None
    action_dependencies: ActionDependencies = field(default_factory=dict)
    autoregressive_actions: bool = True
    append_future_action_masks: bool = False
    normalize_autoregressive_actions: bool = True
    lstm_hidden_size: int | None = None
    lstm_seq_len: int = 16
    observation_version: str = "v1"
    policy_arch: str = "flat"
    global_feature_size: int = 0
    entity_slot_size: int = 0
    max_entity_slots: int = 0
    entity_encoder_size: int = 128


@dataclass
class Meta:
    running_observation_stats: TensorRunningMeanStd
    normalized_observations: bool
    trained_steps: int = 0
    trained_rollouts: int = 0
    num_updates: int = 0
    custom_data: dict[str, Any] = field(default_factory=dict)


logger = logging.getLogger(__name__)
# th.jit.compile seems to not be threadsafe
# ex. 'RuntimeError: Can't redefine method: forward on class:' ...
_jit_lock = threading.Lock()
_JIT_EVAL_POLICY = os.getenv("TORCH_SCRIPT_INFERENCE", "true").lower() == "true"


def _coerce_policy_params(raw: Any) -> PolicyParams:
    """Coerce legacy checkpoint policy params into the current dataclass shape."""
    values: dict[str, Any] = {}
    is_dict = isinstance(raw, dict)
    for field_info in dataclasses.fields(PolicyParams):
        if is_dict:
            present = field_info.name in raw
            value = raw.get(field_info.name) if present else None
        else:
            present = hasattr(raw, field_info.name)
            value = getattr(raw, field_info.name, None)

        if present:
            values[field_info.name] = value
            continue

        if field_info.default is not dataclasses.MISSING:
            values[field_info.name] = field_info.default
            continue

        if field_info.default_factory is not dataclasses.MISSING:
            values[field_info.name] = field_info.default_factory()
            continue

        raise ValueError(
            f"Checkpoint policy_params missing required field {field_info.name!r}"
        )
    return PolicyParams(**values)


def _same_policy_family(left: PolicyParams, right: PolicyParams) -> bool:
    return (
        left.observation_version == right.observation_version
        and left.policy_arch == right.policy_arch
    )


class PPO:
    def __init__(
        self,
        policy_params: PolicyParams,
        meta: Meta,
        device: str = "cpu",
        trainable: bool = True,
        policy_state: dict[str, Any] | None = None,
        optimizer_state: dict[str, Any] | None = None,
    ):
        # Note: don't call constructor directly, use one of the static factory methods to load or create a new instance
        self._policy_params = _coerce_policy_params(policy_params)
        self.device = device
        self.meta = meta
        policy_kwargs = {k: v for k, v in dataclasses.asdict(self._policy_params).items()
                         if k != "lstm_seq_len"}
        self._policy: Policy | None = Policy(**policy_kwargs)
        self._policy.to(device=th.device(device))
        self._policy.eval()
        if policy_state is not None:
            self._policy.load_state_dict(policy_state, strict=False)
        self._use_amp = (device == "cuda")
        use_jit = _JIT_EVAL_POLICY and self._policy_params.lstm_hidden_size is None
        if use_jit:
            with _jit_lock:
                self._eval_policy = th.jit.freeze(th.jit.script(self._policy))
        else:
            self._eval_policy = self._policy
        self._grad_scaler: th.amp.GradScaler | None = None
        self._optimizer: optim.Adam | None
        if trainable:
            self._optimizer = optim.Adam(self._policy.parameters(), eps=1e-5)
            if self._use_amp:
                self._grad_scaler = th.amp.GradScaler()
            if optimizer_state is not None:
                try:
                    self._optimizer.load_state_dict(optimizer_state)
                except ValueError:
                    logger.info("Optimizer state incompatible (likely new parameters added), starting fresh optimizer")
        else:
            self._policy = None
            self._optimizer = None

    @property
    def policy_params(self) -> PolicyParams:
        return self._policy_params

    def create_inference_copy(self) -> "PPO":
        """Create a non-trainable copy for pipelined rollout collection.

        Deep-copies the policy weights and observation stats. Shares the
        custom_data dict (reward normalizer) by shallow copy — safe because
        learn() never touches custom_data.
        """
        import copy
        return PPO(
            policy_params=self._policy_params,
            meta=Meta(
                running_observation_stats=copy.deepcopy(
                    self.meta.running_observation_stats
                ),
                normalized_observations=self.meta.normalized_observations,
                trained_steps=self.meta.trained_steps,
                trained_rollouts=self.meta.trained_rollouts,
                custom_data=dict(self.meta.custom_data),
            ),
            device=self.device,
            trainable=False,
            policy_state=self._policy.state_dict() if self._policy is not None else self._eval_policy.state_dict(),
        )

    def predict(
        self,
        obs: th.Tensor,
        action_masks: th.Tensor | None,
        deterministic: bool | th.Tensor = False,
        return_device: str | None = None,
        return_actions: bool = True,
        return_log_probs: bool = True,
        return_entropy: bool = True,
        return_values: bool = True,
        return_probs: bool = False,
        lstm_state: tuple[th.Tensor, th.Tensor] | None = None,
    ) -> tuple[
        th.Tensor | None,
        th.Tensor | None,
        th.Tensor | None,
        th.Tensor | None,
        th.Tensor | None,
        tuple[th.Tensor, th.Tensor] | None,
    ]:
        with th.inference_mode():
            obs = obs.to(self.device)
            if action_masks is not None:
                action_masks = action_masks.to(self.device)
            if lstm_state is not None:
                lstm_state = (lstm_state[0].to(self.device), lstm_state[1].to(self.device))

            if deterministic is True:
                deterministic = th.ones(
                    len(self._policy_params.action_head_sizes),
                    dtype=th.bool,
                    device=self.device,
                )
            elif deterministic is False:
                deterministic = th.zeros(
                    len(self._policy_params.action_head_sizes),
                    dtype=th.bool,
                    device=self.device,
                )

            if self.meta.normalized_observations:
                obs = self.meta.running_observation_stats.normalize(obs, clip=True)

            with th.amp.autocast('cuda', enabled=self._use_amp):
                actions, log_probs, entropy, values, probs, new_lstm_state = self._eval_policy(
                    obs,
                    action_masks,
                    sample_deterministic=deterministic,
                    return_actions=return_actions,
                    return_entropy=return_entropy,
                    return_log_probs=return_log_probs,
                    return_values=return_values,
                    return_probs=return_probs,
                    lstm_state=lstm_state,
                )

            if self._use_amp:
                if log_probs is not None:
                    log_probs = log_probs.float()
                if entropy is not None:
                    entropy = entropy.float()
                if values is not None:
                    values = values.float()
                if probs is not None:
                    probs = probs.float()
                if new_lstm_state is not None:
                    new_lstm_state = (new_lstm_state[0].float(), new_lstm_state[1].float())

            if return_device is not None:
                if actions is not None:
                    actions = actions.to(return_device)
                if log_probs is not None:
                    log_probs = log_probs.to(return_device)
                if entropy is not None:
                    entropy = entropy.to(return_device)
                if values is not None:
                    values = values.to(return_device)
                if probs is not None:
                    probs = probs.to(return_device)

            return actions, log_probs, entropy, values, probs, new_lstm_state

    def learn(
        self,
        buffer: Buffer,
        summary_writer: SummaryWriter | None = None,
        num_updates: int = 5,
        batch_size: int = 64,
        clip_coef: float = 0.2,
        vf_coef: float = 0.5,
        entropy_coef: float = 0.0,
        max_grad_norm: float = 0.5,
        grad_accum: int = 1,
        learning_rate: float = 0.0003,
        normalize_advantages: bool = True,
        target_kl: float | None = None,
        lstm_burn_in: int = 0,
    ) -> None:
        assert self.is_trainable(), "PPO instance not trainable"
        assert self._optimizer is not None
        assert buffer.is_full(), "Buffer is not full"
        assert self._policy is not None
        for param_group in self._optimizer.param_groups:
            param_group["lr"] = learning_rate

        if self.meta.trained_rollouts == 0:
            logger.info(
                "Skipping training on first rollout to accumulate observation statistics"
            )
            # Skip training on first rollout to collect observation statistics, since normalizations may change
            num_updates = 0

        has_lstm = self._policy_params.lstm_hidden_size is not None
        seq_len = self._policy_params.lstm_seq_len
        burn_in = max(0, lstm_burn_in) if has_lstm else 0

        self._policy.train()

        start_time = time.time()
        start_updates = self.meta.num_updates
        epochs_completed = 0
        entropy_losses = []
        pg_losses = []
        value_losses = []
        clip_fractions = []
        approx_kls = []
        losses = []
        grad_norms = []
        action_entropy_losses = []
        stopped_early = False
        early_stop_kl = 0.0

        accumulated_gradients = 0
        for _ in range(num_updates):
            epochs_completed += 1
            if has_lstm:
                batch_gen = buffer.generate_sequence_batches(
                    seq_len=seq_len,
                    batch_size=max(1, batch_size // seq_len),
                    device=self.device,
                    burn_in_len=burn_in,
                )
            else:
                batch_gen = buffer.generate_batches(batch_size, device=self.device)

            for batch in batch_gen:
                observations = batch.observations
                if self.meta.normalized_observations:
                    observations = self.meta.running_observation_stats.normalize(
                        observations, clip=True
                    )

                if has_lstm:
                    warm_lstm_state: tuple[th.Tensor, th.Tensor] | None = None
                    if (
                        burn_in > 0
                        and batch.burn_in_observations is not None
                        and batch.burn_in_episode_starts is not None
                    ):
                        warm_observations = batch.burn_in_observations
                        if self.meta.normalized_observations:
                            warm_observations = self.meta.running_observation_stats.normalize(
                                warm_observations, clip=True
                            )
                        warm_obs_3d = warm_observations.reshape(
                            -1, burn_in, warm_observations.shape[-1]
                        )
                        with th.no_grad(), th.amp.autocast('cuda', enabled=self._use_amp):
                            *_, warm_lstm_state = self._policy(
                                warm_obs_3d,
                                None,
                                return_actions=False,
                                return_values=False,
                                return_entropy=False,
                                return_log_probs=False,
                                episode_starts=batch.burn_in_episode_starts,
                            )
                    obs_3d = observations.reshape(-1, seq_len, observations.shape[-1])
                    with th.amp.autocast('cuda', enabled=self._use_amp):
                        outputs = self._policy(
                            obs_3d,
                            batch.action_masks,
                            input_actions=batch.actions,
                            lstm_state=warm_lstm_state,
                            episode_starts=batch.episode_starts,
                            return_entropy=True,
                            return_values=True,
                            return_log_probs=True,
                        )
                else:
                    with th.amp.autocast('cuda', enabled=self._use_amp):
                        outputs = self._policy(
                            observations,
                            batch.action_masks,
                            input_actions=batch.actions,
                            return_entropy=True,
                            return_values=True,
                            return_log_probs=True,
                        )

                _, new_log_probs, individual_entropies, new_values, _ = outputs[:5]
                new_log_probs = new_log_probs.float()
                individual_entropies = individual_entropies.float()
                new_values = new_values.float()

                old_log_probs = batch.old_log_prob
                log_prob_ratios = new_log_probs - old_log_probs
                prob_ratios = th.exp(log_prob_ratios)

                advantages = batch.advantages
                if normalize_advantages and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )

                surrogate1 = prob_ratios * advantages
                surrogate2 = (
                    th.clamp(prob_ratios, 1 - clip_coef, 1 + clip_coef) * advantages
                )
                policy_loss = -th.mean(th.min(surrogate1, surrogate2))

                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(prob_ratios - 1) > clip_coef).float())
                clip_fractions.append(clip_fraction.item())

                approx_kl = (prob_ratios - 1) - log_prob_ratios
                approx_kl_mean = approx_kl.mean().item()
                approx_kls.append(approx_kl_mean)
                if target_kl is not None and target_kl > 0.0 and approx_kl_mean > target_kl:
                    stopped_early = True
                    early_stop_kl = approx_kl_mean
                    accumulated_gradients = 0
                    self._optimizer.zero_grad()
                    break

                entropy_loss = -th.mean(individual_entropies.sum(dim=1))
                entropy_losses.append(entropy_loss.item())
                individual_entropy_losses = (
                    -individual_entropies.mean(dim=0).detach().cpu().numpy()
                )
                action_entropy_losses.append(individual_entropy_losses)

                value_loss = th.nn.functional.mse_loss(
                    new_values.squeeze(), batch.returns
                )
                value_losses.append(value_loss.item())

                loss = policy_loss + entropy_coef * entropy_loss + value_loss * vf_coef

                losses.append(loss.item())

                loss = loss / grad_accum
                if self._grad_scaler is not None:
                    self._grad_scaler.scale(loss).backward()
                else:
                    loss.backward()
                accumulated_gradients += 1

                if accumulated_gradients == grad_accum:
                    if self._grad_scaler is not None:
                        self._grad_scaler.unscale_(self._optimizer)
                    grad_norm = th.nn.utils.clip_grad_norm_(
                        self._policy.parameters(), max_grad_norm
                    )
                    grad_norms.append(th.mean(grad_norm).item())
                    if self._grad_scaler is not None:
                        self._grad_scaler.step(self._optimizer)
                        self._grad_scaler.update()
                    else:
                        self._optimizer.step()
                    self._optimizer.zero_grad()
                    accumulated_gradients = 0
                    self.meta.num_updates += 1

            if stopped_early:
                break

        self._optimizer.zero_grad()

        flattened_obs = buffer.observations.reshape(-1, buffer.observations.shape[-1])
        self.meta.running_observation_stats.update(
            th.as_tensor(flattened_obs, device=self.device)
        )
        flattened_actions = buffer.actions.reshape(-1, buffer.actions.shape[-1])
        self._policy.actor.update_action_normalization(
            th.as_tensor(flattened_actions, dtype=th.float32, device=self.device)
        )

        explained_var = float("nan")
        mean_entropy_loss = float(np.mean(entropy_losses)) if entropy_losses else float("nan")
        mean_pg_loss = float(np.mean(pg_losses)) if pg_losses else float("nan")
        mean_value_loss = float(np.mean(value_losses)) if value_losses else float("nan")
        mean_clip_fraction = float(np.mean(clip_fractions)) if clip_fractions else float("nan")
        mean_grad_norm = float(np.mean(grad_norms)) if grad_norms else float("nan")
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        mean_kl = float(np.mean(approx_kls)) if approx_kls else float("nan")
        if losses:
            y_pred = buffer.values.flatten()
            y_true = buffer.returns.flatten()
            var_y = np.var(y_true)
            explained_var = float(
                np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            )
        self.meta.custom_data["last_train_metrics"] = {
            "explained_variance": explained_var,
            "entropy_loss": mean_entropy_loss,
            "policy_gradient_loss": mean_pg_loss,
            "value_loss": mean_value_loss,
            "clip_fraction": mean_clip_fraction,
            "grad_norm": mean_grad_norm,
            "loss": mean_loss,
            "kl": mean_kl,
            "early_stop_kl": float(early_stop_kl),
            "stopped_early": float(stopped_early),
        }

        train_time = time.time() - start_time
        rollout_metrics = self.meta.custom_data.get("last_rollout_metrics", {})
        rollout_time = float(rollout_metrics.get("rollout_time", 0.0))
        total_wall_time = rollout_time + train_time
        trained_steps_this_rollout = buffer.buffer_size * buffer.n_envs
        effective_steps_per_sec = (
            trained_steps_this_rollout / total_wall_time if total_wall_time > 0.0 else float("nan")
        )
        rollout_wall_time_share = (
            rollout_time / total_wall_time if total_wall_time > 0.0 else float("nan")
        )
        learn_wall_time_share = (
            train_time / total_wall_time if total_wall_time > 0.0 else float("nan")
        )
        self.meta.custom_data["last_train_metrics"].update(
            {
                "train_time": float(train_time),
                "effective_steps_per_sec": float(effective_steps_per_sec),
                "rollout_wall_time_share": float(rollout_wall_time_share),
                "learn_wall_time_share": float(learn_wall_time_share),
            }
        )

        if summary_writer is not None:
            summary_writer.add_scalar(
                "train/total_steps", self.meta.trained_steps, self.meta.trained_rollouts
            )
            summary_writer.add_scalar(
                "train/epochs", epochs_completed, self.meta.trained_steps
            )
            summary_writer.add_scalar("train/time", train_time, self.meta.trained_steps)
            if rollout_time > 0.0:
                summary_writer.add_scalar(
                    "throughput/effective_steps_per_sec",
                    effective_steps_per_sec,
                    self.meta.trained_steps,
                )
                summary_writer.add_scalar(
                    "throughput/rollout_wall_time_share",
                    rollout_wall_time_share,
                    self.meta.trained_steps,
                )
                summary_writer.add_scalar(
                    "throughput/learn_wall_time_share",
                    learn_wall_time_share,
                    self.meta.trained_steps,
                )
            summary_writer.add_scalar(
                "train/entropy_coef", entropy_coef, self.meta.trained_steps
            )
            summary_writer.add_scalar(
                "train/clip_coef", clip_coef, self.meta.trained_steps
            )
            summary_writer.add_scalar("train/vf_coef", vf_coef, self.meta.trained_steps)
            summary_writer.add_scalar(
                "train/num_updates",
                self.meta.num_updates - start_updates,
                self.meta.trained_steps,
            )
            summary_writer.add_scalar(
                "train/max_grad_norm", max_grad_norm, self.meta.trained_steps
            )
            summary_writer.add_scalar(
                "train/batch_size", batch_size, self.meta.trained_steps
            )
            summary_writer.add_scalar(
                "train/grad_accum", grad_accum, self.meta.trained_steps
            )
            summary_writer.add_scalar(
                "train/learning_rate", learning_rate, self.meta.trained_steps
            )
            if target_kl is not None:
                summary_writer.add_scalar("train/target_kl", target_kl, self.meta.trained_steps)
            if has_lstm:
                summary_writer.add_scalar("train/lstm_burn_in", burn_in, self.meta.trained_steps)
            summary_writer.add_scalar(
                "train/early_stop", 1.0 if stopped_early else 0.0, self.meta.trained_steps
            )
            if stopped_early:
                summary_writer.add_scalar(
                    "train/early_stop_kl", early_stop_kl, self.meta.trained_steps
                )
            if losses:
                summary_writer.add_scalar(
                    "train/explained_variance", explained_var, self.meta.trained_steps
                )
                summary_writer.add_scalar(
                    "train/entropy_loss", mean_entropy_loss, self.meta.trained_steps
                )
                summary_writer.add_scalar(
                    "train/policy_gradient_loss", mean_pg_loss, self.meta.trained_steps
                )
                summary_writer.add_scalar(
                    "train/value_loss", mean_value_loss, self.meta.trained_steps
                )
                summary_writer.add_scalar(
                    "train/clip_fraction", mean_clip_fraction, self.meta.trained_steps
                )
                summary_writer.add_scalar(
                    "train/grad_norm", mean_grad_norm, self.meta.trained_steps
                )
                summary_writer.add_scalar(
                    "train/loss", mean_loss, self.meta.trained_steps
                )
                summary_writer.add_scalar(
                    "train/kl", mean_kl, self.meta.trained_steps
                )
                for i, entropy_loss in enumerate(
                    np.mean(action_entropy_losses, axis=0)
                ):
                    summary_writer.add_scalar(
                        f"train/entropy_loss/action/{i}",
                        entropy_loss,
                        self.meta.trained_steps,
                    )
            summary_writer.add_scalar(
                "observations/stats_count",
                self.meta.running_observation_stats.count,
                self.meta.trained_steps,
            )

            if self.meta.trained_rollouts % 10 == 0:
                for i in range(buffer.observations.shape[-1]):
                    obs_key = f"{i}"
                    key_obs = buffer.observations[..., i]
                    summary_writer.add_scalar(
                        f"observations/{obs_key}_rollout_mean",
                        np.mean(key_obs),
                        self.meta.trained_steps,
                    )
                    summary_writer.add_scalar(
                        f"observations/{obs_key}_rollout_std",
                        np.std(key_obs),
                        self.meta.trained_steps,
                    )
                    summary_writer.add_scalar(
                        f"observations/{obs_key}_rollout_min",
                        np.min(key_obs),
                        self.meta.trained_steps,
                    )
                    summary_writer.add_scalar(
                        f"observations/{obs_key}_rollout_max",
                        np.max(key_obs),
                        self.meta.trained_steps,
                    )
                    summary_writer.add_scalar(
                        f"observations/{obs_key}_running_mean",
                        self.meta.running_observation_stats.mean[i],
                        self.meta.trained_steps,
                    )
                    summary_writer.add_scalar(
                        f"observations/{obs_key}_running_var",
                        self.meta.running_observation_stats.var[i],
                        self.meta.trained_steps,
                    )

        self.meta.trained_steps += buffer.buffer_size * buffer.n_envs
        self.meta.trained_rollouts += 1
        self._policy.eval()

        use_jit = _JIT_EVAL_POLICY and self._policy_params.lstm_hidden_size is None
        if use_jit:
            self._eval_policy = th.jit.freeze(th.jit.script(self._policy))

    def is_trainable(self) -> bool:
        return self._optimizer is not None

    def save(self, save_path: str) -> None:
        assert self.is_trainable(), "Can't save non-trainable model"
        assert self._policy is not None
        assert self._optimizer is not None
        # Create directory if needed
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        # Save model weights
        th.save(
            {
                "policy": self._policy.state_dict(),
                "optimizer": self._optimizer.state_dict(),
                "policy_params": self._policy_params,
                "meta": self.meta,
                "extensions": [],
            },
            save_path,
        )

    @staticmethod
    def load(
        load_path: str, device: str = "cpu", trainable: bool | None = None
    ) -> "PPO":
        if not os.path.exists(load_path):
            raise ValueError(f"{load_path} not found")
        checkpoint = th.load(load_path, map_location=device, weights_only=False)
        policy_params = _coerce_policy_params(checkpoint["policy_params"])
        # Ensure the loaded model is actually trainable, if requested
        if trainable is None:
            trainable = "optimizer" in checkpoint
        assert (
            not trainable or "optimizer" in checkpoint
        ), f"Cannot load non-trainable model as trainable: {load_path}"
        return PPO(
            policy_params=policy_params,
            meta=checkpoint["meta"],
            device=device,
            trainable=trainable,
            policy_state=checkpoint["policy"],
            optimizer_state=checkpoint.get("optimizer"),
        )

    @staticmethod
    def load_with_resize(
        load_path: str,
        target_policy_params: PolicyParams,
        device: str = "cpu",
    ) -> "PPO":
        """Load checkpoint and resize layers to match target_policy_params.

        Handles mismatches in LSTM hidden size, actor/critic input size (obs dim),
        and MLP layer sizes. Old weights are zero-padded into new dimensions.
        Observation normalization stats are resized to match the new obs dim
        (new dims start with mean=0, var=1). Optimizer state is discarded.
        """
        if not os.path.exists(load_path):
            raise ValueError(f"{load_path} not found")
        checkpoint = th.load(load_path, map_location=device, weights_only=False)
        source_policy_params = _coerce_policy_params(checkpoint["policy_params"])
        target_policy_params = _coerce_policy_params(target_policy_params)

        if not _same_policy_family(source_policy_params, target_policy_params):
            raise ValueError(
                "load_with_resize does not support cross-architecture loads "
                f"({source_policy_params.observation_version}/{source_policy_params.policy_arch} "
                f"-> {target_policy_params.observation_version}/{target_policy_params.policy_arch})"
            )

        old_state = checkpoint["policy"]
        policy_kwargs = {k: v for k, v in dataclasses.asdict(target_policy_params).items()
                         if k != "lstm_seq_len"}
        new_policy = Policy(**policy_kwargs)
        new_state = new_policy.state_dict()

        resized_keys = []
        for key in new_state:
            if key not in old_state:
                continue  # new layer, keep random init
            old_tensor = old_state[key]
            new_tensor = new_state[key]
            if old_tensor.shape == new_tensor.shape:
                new_state[key] = old_tensor
                continue
            insert_offset = source_policy_params.actor_input_size
            insert_count = target_policy_params.actor_input_size - insert_offset

            padded = th.zeros_like(new_tensor)
            last_dim = old_tensor.ndim - 1
            old_last = old_tensor.shape[last_dim]
            new_last = new_tensor.shape[last_dim]

            if insert_count > 0 and new_last - old_last == insert_count:
                # Insertion-aware: split at insert_offset, shift tail by insert_count
                before = [slice(None)] * last_dim + [slice(0, insert_offset)]
                padded[before] = old_tensor[before]
                tail_len = old_last - insert_offset
                if tail_len > 0:
                    src_tail = [slice(None)] * last_dim + [slice(insert_offset, old_last)]
                    dst_tail = [slice(None)] * last_dim + [slice(insert_offset + insert_count, new_last)]
                    padded[dst_tail] = old_tensor[src_tail]
            else:
                # Fallback: zero-pad top-left corner (unchanged behavior)
                slices = tuple(slice(0, min(o, n)) for o, n in zip(old_tensor.shape, new_tensor.shape))
                padded[slices] = old_tensor[slices]

            new_state[key] = padded
            resized_keys.append(f"  {key}: {list(old_tensor.shape)} -> {list(new_tensor.shape)}")

        if resized_keys:
            logger.info(f"Resized {len(resized_keys)} tensors:\n" + "\n".join(resized_keys))

        new_policy.load_state_dict(new_state)
        meta: Meta = checkpoint["meta"]

        # Resize observation normalization stats if obs dim changed
        new_obs_size = max(
            target_policy_params.actor_input_size,
            target_policy_params.critic_input_size,
        )
        old_stats = meta.running_observation_stats
        old_obs_size = old_stats.mean.shape[0]
        if old_obs_size != new_obs_size:
            logger.info(f"Resizing observation stats: {old_obs_size} -> {new_obs_size}")
            insert_offset = source_policy_params.actor_input_size
            insert_count = target_policy_params.actor_input_size - insert_offset
            new_stats = TensorRunningMeanStd(shape=(new_obs_size,), device=device)

            if insert_count > 0 and new_obs_size - old_obs_size == insert_count:
                new_stats.mean[:insert_offset] = old_stats.mean[:insert_offset]
                new_stats.var[:insert_offset] = old_stats.var[:insert_offset]
                tail = old_obs_size - insert_offset
                if tail > 0:
                    new_stats.mean[insert_offset + insert_count:] = old_stats.mean[insert_offset:]
                    new_stats.var[insert_offset + insert_count:] = old_stats.var[insert_offset:]
            else:
                n = min(old_obs_size, new_obs_size)
                new_stats.mean[:n] = old_stats.mean[:n]
                new_stats.var[:n] = old_stats.var[:n]

            new_stats.count = old_stats.count
            meta.running_observation_stats = new_stats

        # Don't load optimizer state — dimensions changed, Adam moments are invalid
        return PPO(
            policy_params=target_policy_params,
            meta=meta,
            device=device,
            trainable=True,
            policy_state=new_policy.state_dict(),
            optimizer_state=None,
        )

    @staticmethod
    def load_meta(load_path: str) -> Meta:
        # Optimized version of load, to just load the model meta
        if not os.path.exists(load_path):
            raise ValueError(f"{load_path} not found")
        checkpoint = th.load(load_path, map_location="cpu", weights_only=False)
        return cast(Meta, checkpoint["meta"])

    @staticmethod
    def load_policy_params(load_path: str) -> PolicyParams:
        if not os.path.exists(load_path):
            raise ValueError(f"{load_path} not found")
        checkpoint = th.load(load_path, map_location="cpu", weights_only=False)
        return _coerce_policy_params(checkpoint["policy_params"])

    @staticmethod
    def save_meta(save_path: str, meta: Meta) -> None:
        if not os.path.exists(save_path):
            raise ValueError(f"{save_path} not found")
        checkpoint = th.load(save_path, map_location="cpu", weights_only=False)
        checkpoint["meta"] = meta
        th.save(checkpoint, save_path)

    @staticmethod
    def optimize_for_inference(model_path: str) -> None:
        # Optimize the model for deployment by removing unnecessary information (ex. optimizer state)
        checkpoint = th.load(model_path, map_location="cpu", weights_only=False)
        checkpoint.pop("optimizer", None)
        th.save(checkpoint, model_path)

    @staticmethod
    def new_instance(
        policy_params: PolicyParams,
        device: str = "cpu",
        normalize_observations: bool = False,
    ) -> "PPO":
        return PPO(
            policy_params=policy_params,
            meta=Meta(
                normalized_observations=normalize_observations,
                running_observation_stats=TensorRunningMeanStd(
                    shape=(
                        max(
                            policy_params.actor_input_size,
                            policy_params.critic_input_size,
                        ),
                    ),
                    device=device,
                ),
            ),
            device=device,
        )

    def pretrain_bc(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        action_masks: np.ndarray,
        n_epochs: int = 10,
        batch_size: int = 512,
        learning_rate: float = 1e-3,
        summary_writer=None,
        teacher_logits: np.ndarray | None = None,
    ) -> None:
        """
        Supervised behaviour-cloning pre-training.

        When teacher_logits is None, uses cross-entropy on hard action labels.
        When teacher_logits is provided, uses head-wise KL divergence to match
        the teacher's action distributions.

        Args:
            observations:    (N, obs_size) float32 array
            actions:         (N,) or (N, num_heads) int32 array of expert actions
            action_masks:    (N, num_actions) bool array
            n_epochs:        number of passes over the dataset
            batch_size:      mini-batch size
            learning_rate:   Adam learning rate for BC phase
            summary_writer:  optional TensorBoard writer for bc/loss, bc/accuracy
            teacher_logits:  (N, num_actions) float32 array of teacher log-probs
        """
        assert self.is_trainable(), "PPO instance is not trainable"
        assert self._policy is not None
        assert self._optimizer is not None

        import torch as th
        import torch.nn.functional as F

        distill = teacher_logits is not None
        mode = "distillation (KL)" if distill else "cross-entropy"
        print(f"  BC mode: {mode}")

        for param_group in self._optimizer.param_groups:
            param_group["lr"] = learning_rate

        obs_tensor = th.as_tensor(observations, dtype=th.float32, device=self.device)
        expected_obs_size = max(
            self._policy_params.actor_input_size,
            self._policy_params.critic_input_size,
        )
        if obs_tensor.shape[-1] > expected_obs_size:
            raise ValueError(
                f"Expected observations with at most {expected_obs_size} features, got {obs_tensor.shape[-1]}"
            )
        if obs_tensor.shape[-1] < expected_obs_size:
            obs_tensor = F.pad(obs_tensor, (0, expected_obs_size - obs_tensor.shape[-1]))
        act_tensor = th.as_tensor(actions, dtype=th.long, device=self.device)
        if act_tensor.ndim == 1:
            act_tensor = act_tensor.unsqueeze(-1)
        mask_tensor = th.as_tensor(action_masks, dtype=th.bool, device=self.device)
        action_head_sizes = self._policy_params.action_head_sizes
        expected_num_heads = len(action_head_sizes)
        if act_tensor.shape[-1] != expected_num_heads:
            raise ValueError(
                f"Expected {expected_num_heads} action heads, got tensor shape {tuple(act_tensor.shape)}"
            )

        if distill:
            teacher_logits_tensor = th.as_tensor(
                teacher_logits, dtype=th.float32, device=self.device
            )
            expected_prob_size = sum(action_head_sizes)
            if teacher_logits_tensor.shape[-1] != expected_prob_size:
                raise ValueError(
                    f"Expected teacher logits width {expected_prob_size}, got {teacher_logits_tensor.shape[-1]}"
                )

        N = len(observations)
        self._policy.train()

        for epoch in range(n_epochs):
            indices = th.randperm(N, device=self.device)
            epoch_losses = []
            epoch_corrects = []

            for start in range(0, N, batch_size):
                idx = indices[start:start + batch_size]
                obs_batch = obs_tensor[idx]
                act_batch = act_tensor[idx]
                mask_batch = mask_tensor[idx]

                if self.meta.normalized_observations:
                    obs_batch = self.meta.running_observation_stats.normalize(
                        obs_batch, clip=True
                    )

                # Policy.forward expects (B, seq_len, features); add seq dim
                obs_3d = obs_batch.unsqueeze(1)

                # Forward: get log_probs of expert actions and probs for accuracy
                _, log_probs, _, _, probs, _ = self._policy(
                    obs_3d,
                    mask_batch,
                    input_actions=act_batch,
                    return_log_probs=True,
                    return_probs=True,
                )

                if distill:
                    student_heads = th.split(probs, action_head_sizes, dim=-1)
                    teacher_heads = th.split(teacher_logits_tensor[idx], action_head_sizes, dim=-1)
                    loss = th.zeros((), dtype=th.float32, device=self.device)
                    for student_head, teacher_head_logits in zip(student_heads, teacher_heads):
                        teacher_head_p = F.softmax(teacher_head_logits, dim=-1)
                        student_head_log_p = th.log(student_head.clamp(min=1e-8))
                        loss = loss + F.kl_div(
                            student_head_log_p,
                            teacher_head_p,
                            reduction="batchmean",
                        )
                    loss = loss / len(action_head_sizes)
                else:
                    loss = -log_probs.mean()

                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

                epoch_losses.append(loss.item())

                if probs is not None:
                    predicted = th.stack(
                        [head.argmax(dim=-1) for head in th.split(probs, action_head_sizes, dim=-1)],
                        dim=-1,
                    )
                    epoch_corrects.append(
                        (predicted == act_batch).all(dim=-1).float().mean().item()
                    )

            mean_loss = sum(epoch_losses) / len(epoch_losses)
            mean_acc = (
                sum(epoch_corrects) / len(epoch_corrects) if epoch_corrects else 0.0
            )
            print(
                f"  BC epoch {epoch + 1}/{n_epochs} | "
                f"loss={mean_loss:.4f} | accuracy={100*mean_acc:.1f}%"
            )

            if summary_writer is not None:
                summary_writer.add_scalar("bc/loss", mean_loss, epoch)
                summary_writer.add_scalar("bc/accuracy", mean_acc, epoch)

        # Populate obs normalisation stats so RL fine-tuning inherits them
        self.meta.running_observation_stats.update(obs_tensor)

        self._policy.eval()
        use_jit = _JIT_EVAL_POLICY and self._policy_params.lstm_hidden_size is None
        if use_jit:
            self._eval_policy = th.jit.freeze(th.jit.script(self._policy))

    def __str__(self) -> str:
        return str(self._policy)
