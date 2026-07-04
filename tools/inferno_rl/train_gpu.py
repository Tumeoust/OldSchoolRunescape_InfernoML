"""
GPU-accelerated training script for Inferno RL agent using custom PyTorch PPO.

Usage:
    python -m tools.inferno_rl.train_gpu --device cpu --n-envs 2 --n-steps 64 --timesteps 1000

This training stack provides:
- Custom PPO with full GPU support
- TorchScript inference compilation
- Running observation normalization (orthogonal weight init)
- Separate actor/critic network sizes
- Async vectorized environment (pipelines inference with env stepping)
"""

import argparse
import ctypes
import os
import platform
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from typing import Optional, Dict, List

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from .adaptive_curriculum import AdaptiveConfig, AdaptiveController, GlobalWaveStats
from .async_env.subprocess_vec_env import SubprocVecEnv
from .callback import Callback
from .eval import evaluate_policy
from .ppo.buffer import Buffer
from .ppo.mlp_helper import MlpConfig, default_mlp_config
from .ppo.ppo import PPO, PolicyParams
from .rollout_sampler import RolloutSampler
from .training.observation import get_observation_size, get_public_observation_size
from .training.rewards import RewardConfig
from .training.schedules import LinearSchedule
from .training.actions import ACTION_HEAD_SIZES, POLICY_ACTION_DEPENDENCIES

_V44_NOVELTY_SCHEDULE = LinearSchedule(0.0003, 0.0, 300)


# ──────────────────────────────────────────────────────────────────────────────
# Callbacks
# ──────────────────────────────────────────────────────────────────────────────

class WaveProgressCallback(Callback):
    """Tracks mean wave reached and clear rate per start-wave category."""

    def __init__(self) -> None:
        super().__init__()
        self.final_waves: Dict[int, List[int]] = {}
        self.clears: Dict[int, int] = {}

    def on_step(self, indices: np.ndarray, infos: np.ndarray) -> None:
        for info in infos:
            if not info.get("episode_done"):
                continue
            start_wave = info.get("start_wave")
            if start_wave is None:
                continue
            self.final_waves.setdefault(start_wave, []).append(info.get("wave", 0))
            if info.get("inferno_complete"):
                self.clears[start_wave] = self.clears.get(start_wave, 0) + 1

    def log(self, summary_writer: SummaryWriter, step: int) -> None:
        for start_wave, waves in sorted(self.final_waves.items()):
            mean_wave = sum(waves) / len(waves)
            clear_rate = self.clears.get(start_wave, 0) / len(waves)
            summary_writer.add_scalar(f"rollout/mean_wave_from_{start_wave}", mean_wave, step)
            summary_writer.add_scalar(f"rollout/clear_rate_from_{start_wave}", clear_rate, step)

    def reset(self) -> None:
        self.final_waves.clear()
        self.clears.clear()


class OutcomeStatsCallback(Callback):
    """Logs outcome stats per rollout."""

    def __init__(self) -> None:
        super().__init__()
        self._deaths = 0
        self._timeouts = 0
        self._waves_completed = 0
        self._death_waves: list[int] = []
        self._death_loadouts: list[str] = []

    def on_step(self, indices: np.ndarray, infos: np.ndarray) -> None:
        for info in infos:
            if info.get("player_died"):
                self._deaths += 1
                self._death_waves.append(info.get("wave", -1))
                self._death_loadouts.append(info.get("loadout", "unknown"))
            if info.get("wave_timeout"):
                self._timeouts += 1
            if info.get("wave_completed"):
                self._waves_completed += 1

    def log(self, summary_writer: SummaryWriter, step: int) -> None:
        summary_writer.add_scalar("rollout/deaths", self._deaths, step)
        summary_writer.add_scalar("rollout/wave_timeouts", self._timeouts, step)
        summary_writer.add_scalar("rollout/waves_completed", self._waves_completed, step)
        if self._death_waves:
            summary_writer.add_scalar(
                "rollout/death_wave_mean", sum(self._death_waves) / len(self._death_waves), step
            )
            summary_writer.add_scalar("rollout/death_wave_min", min(self._death_waves), step)
            for loadout, count in Counter(self._death_loadouts).items():
                summary_writer.add_scalar(f"deaths/by_loadout/{loadout}", count, step)

    def reset(self) -> None:
        self._deaths = 0
        self._timeouts = 0
        self._waves_completed = 0
        self._death_waves.clear()
        self._death_loadouts.clear()


class CurriculumCallback(Callback):
    """Tracks curriculum frontier wave and phase across envs.

    Only logs when curriculum advancement is active (climb phase or legacy
    promote_after). Suppressed for harden/backfill where frontier is static.
    Also tracks drill-phase metrics when phase == "drill".
    """

    def __init__(self, phase: str | None = None) -> None:
        super().__init__()
        self._phase = phase
        self._mastery_count: int = 0
        self._step_count: int = 0
        # Drill phase tracking
        self._drill_waves: list[int] = []
        self._drill_cycles_total: int = 0
        # Global prestige sync
        self.pending_prestige: int | None = None
        self.pending_frontier_sync: tuple[int, int] | None = None
        # Promotion diagnostics
        self._promotion_clears: int = 0
        self._promotion_fails: int = 0
        self._max_streak: int = 0
        self._max_wave_cleared_values: list[int] = []

    def on_step(self, indices: np.ndarray, infos: np.ndarray) -> None:
        if self._phase in ("harden", "backfill"):
            return
        for info in infos:
            self._step_count += 1
            if info.get("mastery_mode"):
                self._mastery_count += 1
            if info.get("prestige_event"):
                self.pending_prestige = info["new_min_waves"]
            if info.get("episode_done") and "frontier_wave" in info:
                frontier_wave = int(info["frontier_wave"])
                min_waves_required = int(info.get("min_waves_required", 1))
                if self.pending_frontier_sync is None:
                    self.pending_frontier_sync = (frontier_wave, min_waves_required)
                else:
                    self.pending_frontier_sync = (
                        max(self.pending_frontier_sync[0], frontier_wave),
                        max(self.pending_frontier_sync[1], min_waves_required),
                    )
            # Drill metrics
            if "drill_wave" in info:
                self._drill_waves.append(info["drill_wave"])
            if "drill_cycles" in info:
                self._drill_cycles_total = max(self._drill_cycles_total, info["drill_cycles"])
            # Promotion diagnostics
            if info.get("promotion_cleared"):
                self._promotion_clears += 1
                streak = info.get("promotion_streak", 0)
                self._max_streak = max(self._max_streak, streak)
            if info.get("promotion_failed"):
                self._promotion_fails += 1
            if info.get("episode_done") and "max_wave_cleared" in info:
                self._max_wave_cleared_values.append(info["max_wave_cleared"])

    def log(self, summary_writer: SummaryWriter, step: int) -> None:
        if self._step_count > 0:
            summary_writer.add_scalar(
                "rollout/mastery_mode_pct",
                self._mastery_count / self._step_count,
                step,
            )
        # Promotion diagnostics
        total_promo = self._promotion_clears + self._promotion_fails
        if total_promo > 0:
            summary_writer.add_scalar(
                "rollout/promotion_clear_rate",
                self._promotion_clears / total_promo,
                step,
            )
            summary_writer.add_scalar(
                "rollout/promotion_clears", self._promotion_clears, step
            )
            summary_writer.add_scalar(
                "rollout/promotion_fails", self._promotion_fails, step
            )
            summary_writer.add_scalar(
                "rollout/promotion_max_streak", self._max_streak, step
            )
        if self._max_wave_cleared_values:
            summary_writer.add_scalar(
                "rollout/max_wave_cleared_mean",
                sum(self._max_wave_cleared_values) / len(self._max_wave_cleared_values),
                step,
            )
        # Drill phase scalars
        if self._drill_waves:
            summary_writer.add_scalar(
                "rollout/drill_wave_mean",
                sum(self._drill_waves) / len(self._drill_waves),
                step,
            )
            summary_writer.add_scalar(
                "rollout/drill_wave_min", min(self._drill_waves), step
            )
            summary_writer.add_scalar(
                "rollout/drill_wave_max", max(self._drill_waves), step
            )
            summary_writer.add_scalar(
                "rollout/drill_cycles", self._drill_cycles_total, step
            )

    def reset(self) -> None:
        self._mastery_count = 0
        self._step_count = 0
        self._drill_waves.clear()
        self._drill_cycles_total = 0
        self.pending_frontier_sync = None
        self._promotion_clears = 0
        self._promotion_fails = 0
        self._max_streak = 0
        self._max_wave_cleared_values.clear()


class PhaseStatsCallback(Callback):
    """Logs per-phase TensorBoard metrics for harden/backfill phases."""

    def __init__(self, phase: str | None = None) -> None:
        super().__init__()
        self._phase = phase
        self._failures = 0
        self._episodes = 0
        self._wave_updates: list[tuple[int, bool]] = []
        self._global_wave_stats: dict[int, dict[str, int]] | None = None
        self._opener_episodes = 0
        self._opener_failures = 0
        self._opener_successes = 0
        self._opener_resolved = 0
        self._opener_survive_window = 0
        self._opener_low_hp_fails = 0
        self._opener_end_hp_sum = 0.0
        self._opener_end_tick_sum = 0.0
        self._opener_magers_remaining_sum = 0.0
        self._opener_melees_remaining_sum = 0.0

    def set_phase(self, phase: str | None) -> None:
        self._phase = phase

    def set_global_wave_stats(self, stats: dict[int, dict[str, int]] | None) -> None:
        self._global_wave_stats = stats

    def on_step(self, indices: np.ndarray, infos: np.ndarray) -> None:
        if self._phase not in ("harden", "backfill", "sweep"):
            return
        for info in infos:
            if info.get("episode_done"):
                self._episodes += 1
                success = bool(info.get("episode_success"))
                if not success:
                    self._failures += 1
                updates = info.get("wave_stat_updates")
                if isinstance(updates, list):
                    for update in updates:
                        if isinstance(update, dict) and "wave" in update and "success" in update:
                            self._wave_updates.append(
                                (int(update["wave"]), bool(update["success"]))
                            )
                if info.get("episode_mode") == "opener":
                    self._opener_episodes += 1
                    if success:
                        self._opener_successes += 1
                    else:
                        self._opener_failures += 1
                    reason = info.get("opener_end_reason")
                    if reason == "resolved":
                        self._opener_resolved += 1
                    elif reason == "tick_limit":
                        self._opener_survive_window += 1
                    elif reason == "low_hp":
                        self._opener_low_hp_fails += 1
                    self._opener_end_hp_sum += float(info.get("player_health", 0))
                    self._opener_end_tick_sum += float(info.get("ticks_in_wave", 0))
                    self._opener_magers_remaining_sum += float(info.get("opener_magers_remaining", 0))
                    self._opener_melees_remaining_sum += float(info.get("opener_melees_remaining", 0))

    def log(self, summary_writer: SummaryWriter, step: int) -> None:
        if self._phase not in ("harden", "backfill", "sweep"):
            return

        if self._episodes > 0:
            summary_writer.add_scalar(
                "rollout/phase_failure_rate",
                self._failures / self._episodes,
                step,
            )

        if self._global_wave_stats:
            worst_wave = None
            worst_rate = -1.0
            mastered = 0
            for wave, stats in self._global_wave_stats.items():
                total = stats["fails"] + stats["successes"]
                if total > 0:
                    rate = stats["fails"] / total
                    if rate > worst_rate:
                        worst_rate = rate
                        worst_wave = wave
                    if total >= 25 and rate < 0.02:
                        mastered += 1

            if worst_wave is not None:
                summary_writer.add_scalar("rollout/phase_worst_wave_global", worst_wave, step)
                summary_writer.add_scalar("rollout/backfill_worst_wave_fail_rate_global", worst_rate, step)

            if self._phase == "backfill":
                summary_writer.add_scalar("rollout/backfill_waves_mastered_global", mastered, step)

        if self._opener_episodes > 0:
            denom = float(self._opener_episodes)
            summary_writer.add_scalar("rollout/opener_success_rate", self._opener_successes / denom, step)
            summary_writer.add_scalar("rollout/opener_failure_rate", self._opener_failures / denom, step)
            summary_writer.add_scalar("rollout/opener_resolved_rate", self._opener_resolved / denom, step)
            summary_writer.add_scalar("rollout/opener_survive_window_rate", self._opener_survive_window / denom, step)
            summary_writer.add_scalar("rollout/opener_low_hp_fail_rate", self._opener_low_hp_fails / denom, step)
            summary_writer.add_scalar("rollout/opener_mean_end_hp", self._opener_end_hp_sum / denom, step)
            summary_writer.add_scalar("rollout/opener_mean_end_tick", self._opener_end_tick_sum / denom, step)
            summary_writer.add_scalar(
                "rollout/opener_mean_magers_remaining",
                self._opener_magers_remaining_sum / denom,
                step,
            )
            summary_writer.add_scalar(
                "rollout/opener_mean_melees_remaining",
                self._opener_melees_remaining_sum / denom,
                step,
            )

    @property
    def wave_updates(self) -> list[tuple[int, bool]]:
        return list(self._wave_updates)

    def reset(self) -> None:
        self._failures = 0
        self._episodes = 0
        self._wave_updates.clear()
        self._opener_episodes = 0
        self._opener_failures = 0
        self._opener_successes = 0
        self._opener_resolved = 0
        self._opener_survive_window = 0
        self._opener_low_hp_fails = 0
        self._opener_end_hp_sum = 0.0
        self._opener_end_tick_sum = 0.0
        self._opener_magers_remaining_sum = 0.0
        self._opener_melees_remaining_sum = 0.0


class RewardTermsCallback(Callback):
    """Logs per-episode raw reward term contributions (averaged over episodes in a rollout)."""

    def __init__(self) -> None:
        super().__init__()
        self._episodes = 0
        self._sum_by_term: Dict[str, float] = {}
        self._mean_per_tick_by_term: Dict[str, float] = {}
        # Per-outcome accumulators
        self._death_episodes = 0
        self._death_sum_by_term: Dict[str, float] = {}
        self._clear_episodes = 0
        self._clear_sum_by_term: Dict[str, float] = {}

    @staticmethod
    def _sanitize_term(term: str) -> str:
        term = term.replace(" ", "_")
        term = term.replace("(", "").replace(")", "")
        term = term.replace("!", "")
        term = term.replace(":", "")
        term = term.replace("+", "plus")
        return term

    def _accumulate(
        self, terms: dict, ep_len: int, sum_dict: Dict[str, float], tick_dict: Dict[str, float] | None
    ) -> None:
        for term, value in terms.items():
            if not isinstance(value, (int, float)):
                continue
            v = float(value)
            sum_dict[term] = sum_dict.get(term, 0.0) + v
            if tick_dict is not None:
                tick_dict[term] = tick_dict.get(term, 0.0) + v / ep_len

    def on_step(self, indices: np.ndarray, infos: np.ndarray) -> None:
        for info in infos:
            terms = info.get("episode_reward_terms")
            if not isinstance(terms, dict) or not terms:
                continue

            ep_len = info.get("episode_reward_term_steps")
            ep_len = max(1, int(ep_len or 1))

            self._episodes += 1
            self._accumulate(terms, ep_len, self._sum_by_term, self._mean_per_tick_by_term)

            if info.get("player_died"):
                self._death_episodes += 1
                self._accumulate(terms, ep_len, self._death_sum_by_term, None)
            elif info.get("episode_success"):
                self._clear_episodes += 1
                self._accumulate(terms, ep_len, self._clear_sum_by_term, None)

    def _log_bucket(
        self, summary_writer: SummaryWriter, step: int, prefix: str, count: int, sum_dict: Dict[str, float]
    ) -> None:
        if count == 0:
            return
        summary_writer.add_scalar(f"{prefix}/episodes", count, step)
        for term, total in sorted(sum_dict.items()):
            key = self._sanitize_term(term)
            summary_writer.add_scalar(f"{prefix}/ep_sum_mean/{key}", total / count, step)

    def log(self, summary_writer: SummaryWriter, step: int) -> None:
        if self._episodes == 0:
            return

        summary_writer.add_scalar("raw_reward_terms/episodes", self._episodes, step)
        for term, total in sorted(self._sum_by_term.items()):
            key = self._sanitize_term(term)
            summary_writer.add_scalar(f"raw_reward_terms/ep_sum_mean/{key}", total / self._episodes, step)
        for term, total_mean_per_tick in sorted(self._mean_per_tick_by_term.items()):
            key = self._sanitize_term(term)
            summary_writer.add_scalar(
                f"raw_reward_terms/ep_mean_per_tick_mean/{key}",
                total_mean_per_tick / self._episodes,
                step,
            )

        self._log_bucket(summary_writer, step, "raw_reward_terms/death", self._death_episodes, self._death_sum_by_term)
        self._log_bucket(summary_writer, step, "raw_reward_terms/clear", self._clear_episodes, self._clear_sum_by_term)

    def summary(self) -> dict[str, float]:
        if self._episodes == 0:
            return {}
        return {
            term: total / self._episodes
            for term, total in self._sum_by_term.items()
        }

    def reset(self) -> None:
        self._episodes = 0
        self._sum_by_term.clear()
        self._mean_per_tick_by_term.clear()
        self._death_episodes = 0
        self._death_sum_by_term.clear()
        self._clear_episodes = 0
        self._clear_sum_by_term.clear()


class TrainingChampionTracker(Callback):
    """Training-only champion selection using reweighted sweep metrics."""

    def __init__(self, alpha: float = 0.05) -> None:
        super().__init__()
        self._alpha = alpha
        self._by_wave: dict[int, dict[str, float]] = {}
        self.best_score = float("-inf")
        self.best_checkpoint_path: str | None = None
        self.best_steps = 0
        self.best_shape_share = float("inf")
        self._consecutive_guardrail_failures = 0

    def on_step(self, indices: np.ndarray, infos: np.ndarray) -> None:
        for info in infos:
            if not info.get("episode_done"):
                continue
            start_wave = int(info.get("start_wave", 1))
            max_wave_cleared = int(info.get("max_wave_cleared", start_wave - 1))
            span = max(1, 66 - start_wave + 1)
            normalized_progress = max(
                0.0, min(1.0, (max_wave_cleared - start_wave + 1) / span)
            )
            success = 1.0 if info.get("episode_success") else 0.0
            death = 1.0 if info.get("player_died") else 0.0
            timeout = 1.0 if info.get("wave_timeout") else 0.0
            stats = self._by_wave.setdefault(
                start_wave,
                {
                    "success_rate": 0.0,
                    "death_rate": 0.0,
                    "timeout_rate": 0.0,
                    "normalized_progress": 0.0,
                },
            )
            for key, value in (
                ("success_rate", success),
                ("death_rate", death),
                ("timeout_rate", timeout),
                ("normalized_progress", normalized_progress),
            ):
                stats[key] = (1.0 - self._alpha) * stats[key] + self._alpha * value

    def score(self) -> float:
        weighted_sum = 0.0
        total_weight = 0.0
        for wave in range(1, 67):
            stats = self._by_wave.get(wave)
            if stats is None:
                continue
            weight = 2.0 if wave >= 55 else 1.0
            wave_score = 0.7 * stats["normalized_progress"] + 0.3 * stats["success_rate"]
            weighted_sum += weight * wave_score
            total_weight += weight
        if total_weight == 0.0:
            return 0.0
        return 100.0 * weighted_sum / total_weight

    @staticmethod
    def _shape_share(reward_summary: dict[str, float]) -> float:
        shaping = sum(
            reward_summary.get(key, 0.0)
            for key in (
                "Single-LOS Engagement",
                "NE Pillar Zone",
                "C Tile Position",
                "Tile A Proximity",
                "Mager Priority",
            )
        )
        progress = sum(
            reward_summary.get(key, 0.0)
            for key in (
                "Wave Complete",
                "Wave End HP Bonus",
                "Damage Dealt",
                "Blood Barrage Heal",
                "Kill Mager",
                "Kill Ranger",
                "Kill Melee",
                "Kill Blob",
                "Kill Bat",
                "Kill Nibbler",
            )
        )
        return shaping / max(progress, 1e-6)

    def guardrails_pass(
        self,
        train_metrics: dict[str, float],
        reward_summary: dict[str, float],
        reward_var: float,
    ) -> tuple[bool, float]:
        shape_share = self._shape_share(reward_summary)
        ev = float(train_metrics.get("explained_variance", float("nan")))
        value_loss = float(train_metrics.get("value_loss", float("nan")))
        kl = float(train_metrics.get("kl", float("nan")))
        grad_norm = float(train_metrics.get("grad_norm", float("nan")))
        reward_var_ok = np.isfinite(reward_var)
        passed = (
            np.isfinite(ev) and ev >= 0.90
            and np.isfinite(value_loss) and value_loss <= 0.08
            and np.isfinite(kl) and kl >= 0.0015
            and np.isfinite(grad_norm) and grad_norm <= 2.5
            and reward_var_ok
            and shape_share <= 1.0
        )
        return passed, shape_share

    def maybe_update(
        self,
        *,
        checkpoint_path: str,
        trained_steps: int,
        train_metrics: dict[str, float],
        reward_summary: dict[str, float],
        reward_var: float,
    ) -> tuple[bool, bool, float]:
        score = self.score()
        guardrails_pass, shape_share = self.guardrails_pass(
            train_metrics, reward_summary, reward_var,
        )
        improved = guardrails_pass and score >= self.best_score + 0.5
        if improved:
            self.best_score = score
            self.best_checkpoint_path = checkpoint_path
            self.best_steps = trained_steps
            self.best_shape_share = shape_share
            self._consecutive_guardrail_failures = 0
            return True, False, score

        if guardrails_pass:
            self._consecutive_guardrail_failures = 0
        else:
            self._consecutive_guardrail_failures += 1

        should_reload = (
            self._consecutive_guardrail_failures >= 3
            and self.best_checkpoint_path is not None
            and score < self.best_score + 0.5
        )
        return False, should_reload, score


class CompositeCallback(Callback):
    """Combines multiple callbacks into one."""

    def __init__(self, callbacks: list[Callback]) -> None:
        super().__init__()
        self._callbacks = callbacks

    def on_step(self, indices: np.ndarray, infos: np.ndarray) -> None:
        for cb in self._callbacks:
            cb.on_step(indices, infos)

    def on_rollout_sampling_end(self, raw_buffer: Buffer) -> None:
        for cb in self._callbacks:
            cb.on_rollout_sampling_end(raw_buffer)

    def on_rollout_end(self, buffer: Buffer) -> None:
        for cb in self._callbacks:
            cb.on_rollout_end(buffer)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * max(0.0, min(1.0, t))


def _parse_wave_weights(weights_str: Optional[str]) -> Optional[Dict[int, float]]:
    if not weights_str:
        return None
    try:
        weights = {}
        for part in weights_str.split(","):
            wave, weight = part.split(":")
            weights[int(wave)] = float(weight)
        return weights
    except ValueError:
        raise ValueError(
            f"Invalid wave weights format: {weights_str}. Expected: 'wave:weight,...'"
        )


def _save_checkpoint(ppo: PPO, save_dir: str, run_name: str, suffix: str) -> str:
    checkpoint_path = os.path.join(save_dir, f"{run_name}_{suffix}.pt")
    ppo.save(checkpoint_path)
    return checkpoint_path


def _validate_training_configuration(
    *,
    observation_version: str,
    policy_arch: str,
    lstm_hidden_size: Optional[int],
    curriculum_mode: str,
    episode_mode: str,
    opener_min_health: int,
    opener_tick_limit: int,
    load_path: Optional[str],
) -> None:
    if policy_arch not in ("flat", "flat_lstm_residual"):
        raise ValueError(
            f"Unsupported policy_arch={policy_arch!r}; expected 'flat' or 'flat_lstm_residual'"
        )
    if policy_arch == "flat" and lstm_hidden_size is not None:
        raise ValueError("policy_arch='flat' does not support LSTM")
    if policy_arch == "flat_lstm_residual" and lstm_hidden_size is None:
        raise ValueError("flat_lstm_residual requires --lstm-hidden-size")
    if observation_version != "v4":
        raise ValueError(
            f"Unsupported observation_version={observation_version!r}; expected 'v4'"
        )
    if curriculum_mode not in ("static", "adaptive_v36"):
        raise ValueError(
            f"Unsupported curriculum_mode={curriculum_mode!r}; expected 'static' or 'adaptive_v36'"
        )
    if episode_mode not in ("full", "opener"):
        raise ValueError(
            f"Unsupported episode_mode={episode_mode!r}; expected 'full' or 'opener'"
        )
    if not (1 <= opener_min_health <= 98):
        raise ValueError("opener_min_health must be within 1..98")
    if opener_tick_limit < 1:
        raise ValueError("opener_tick_limit must be >= 1")
    if curriculum_mode == "adaptive_v36" and load_path is None:
        raise ValueError("adaptive_v36 requires --load")


def _log_rollout_callbacks(
    summary_writer: SummaryWriter,
    ppo: PPO,
    cur_mw: WaveProgressCallback,
    cur_oc: OutcomeStatsCallback,
    cur_cu: CurriculumCallback,
    cur_ph: PhaseStatsCallback,
    cur_rt: RewardTermsCallback | None,
    global_wave_stats: GlobalWaveStats | None = None,
) -> tuple[int | None, tuple[int, int] | None, dict[str, float]]:
    if global_wave_stats is not None:
        global_wave_stats.merge_updates(cur_ph.wave_updates)
        cur_ph.set_global_wave_stats(global_wave_stats.snapshot())

    cur_mw.log(summary_writer, ppo.meta.trained_steps)
    cur_mw.reset()
    cur_oc.log(summary_writer, ppo.meta.trained_steps)
    cur_oc.reset()
    cur_cu.log(summary_writer, ppo.meta.trained_steps)
    prestige_from_cur = cur_cu.pending_prestige
    frontier_sync_from_cur = cur_cu.pending_frontier_sync
    cur_cu.reset()
    cur_ph.log(summary_writer, ppo.meta.trained_steps)
    cur_ph.reset()
    reward_summary: dict[str, float] = {}
    if cur_rt is not None:
        reward_summary = cur_rt.summary()
        cur_rt.log(summary_writer, ppo.meta.trained_steps)
        cur_rt.reset()
    return prestige_from_cur, frontier_sync_from_cur, reward_summary


def _log_adaptive_metrics(
    summary_writer: SummaryWriter,
    step: int,
    summary,
    controller: AdaptiveController,
    switch_reason: str,
) -> None:
    summary_writer.add_scalar("adaptive/current_score", summary.full_clear_rate, step)
    summary_writer.add_scalar("adaptive/current_death_rate", summary.death_rate, step)
    summary_writer.add_scalar("adaptive/current_timeout_rate", summary.timeout_rate, step)
    summary_writer.add_scalar("adaptive/current_mean_max_wave", summary.mean_max_wave, step)
    summary_writer.add_scalar("adaptive/current_regime", controller.current_regime_code, step)
    summary_writer.add_scalar("adaptive/regime_window", controller.current_regime_window, step)
    summary_writer.add_scalar("adaptive/plateau_count", controller.plateau_windows, step)
    summary_writer.add_scalar("adaptive/rollback_count", controller.rollback_count, step)
    summary_writer.add_scalar(
        "adaptive/champion_score", controller.champion.score, step
    )
    summary_writer.add_scalar(
        "adaptive/champion_death_rate", controller.champion.death_rate, step
    )
    summary_writer.add_scalar(
        "adaptive/champion_timeout_rate", controller.champion.timeout_rate, step
    )
    summary_writer.add_scalar(
        "adaptive/champion_mean_max_wave", controller.champion.mean_max_wave, step
    )
    summary_writer.add_scalar(
        "adaptive/switch_reason",
        controller.switch_reason_code(switch_reason),
        step,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train(
    total_timesteps: int = 1_000_000,
    start_wave: int = 1,
    max_wave: int = 66,
    start_wave_weights: Optional[Dict[int, float]] = None,
    promote_after: int = 0,
    refresh_every: int = 10,
    min_waves_to_advance: int = 1,
    climb_sampling: str = "weighted",
    n_envs: int = 48,
    device: str = "cpu",
    save_dir: str = "models/inferno_gpu",
    log_dir: str = "logs/inferno_gpu",
    learning_rate: float = 3e-4,
    n_steps: int = 256,
    batch_size: int = 4096,
    n_epochs: int = 1,
    gamma: float = 0.995,
    gae_lambda: float = 0.95,
    clip_coef: float = 0.2,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    target_kl: Optional[float] = 0.02,
    entropy_start: float = 0.05,
    entropy_end: float = 0.002,
    normalize_obs: bool = True,
    normalize_reward: bool = True,
    novelty_scale: float = 0.0,  # ignored; V44 uses _V44_NOVELTY_SCHEDULE
    checkpoint_every: int = 100,
    load_path: Optional[str] = None,
    phase: Optional[str] = "sweep",
    log_reward_terms: bool = False,
    actor_sizes: Optional[list[int]] = None,
    critic_sizes: Optional[list[int]] = None,
    resize_lstm: Optional[int] = None,
    lstm_hidden_size: Optional[int] = 128,
    lstm_seq_len: int = 16,
    lstm_burn_in: Optional[int] = None,
    max_drill_retries: int = 10,
    hindsight_death_penalty: float = 0.0,
    hindsight_death_window: int = 10,
    hindsight_death_decay: float = 0.8,
    observation_version: str = "v4",
    policy_arch: str = "flat_lstm_residual",
    curriculum_mode: str = "static",
    adaptive_eval_every: int = 50,
    adaptive_eval_episodes: int = 100,
    adaptive_eval_start_wave: int = 49,
    adaptive_eval_max_wave: int = 66,
    adaptive_harden_max_windows: int = 3,
    adaptive_backfill_max_windows: int = 6,
    adaptive_opener_max_windows: int = 1,
    adaptive_improve_threshold: float = 0.5,
    adaptive_regress_threshold: float = 2.0,
    adaptive_plateau_windows: int = 2,
    episode_mode: str = "full",
    opener_tick_limit: int = 50,
    opener_min_health: int = 40,
    fixed_loadout: Optional[str] = None,
    loadout_weights: Optional[Dict[str, float]] = None,
    sweep_death_retries: int = 0,
    reward_config: RewardConfig | None = None,
) -> PPO:
    import torch as th

    if platform.system() == "Windows":
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        )
        print("Windows sleep prevention enabled")

    if device == "cuda" and not th.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    _validate_training_configuration(
        observation_version=observation_version,
        policy_arch=policy_arch,
        lstm_hidden_size=lstm_hidden_size,
        curriculum_mode=curriculum_mode,
        episode_mode=episode_mode,
        opener_min_health=opener_min_health,
        opener_tick_limit=opener_tick_limit,
        load_path=load_path,
    )

    if reward_config is None:
        reward_config = RewardConfig(start_wave=start_wave, max_wave=max_wave)
    reward_config_dict = asdict(reward_config)
    print(f"  Reward config: {reward_config_dict}")

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    adaptive_config = None
    adaptive_controller = None
    if curriculum_mode == "adaptive_v36":
        adaptive_config = AdaptiveConfig(
            eval_every_rollouts=adaptive_eval_every,
            eval_episodes=adaptive_eval_episodes,
            eval_start_wave=adaptive_eval_start_wave,
            eval_max_wave=adaptive_eval_max_wave,
            harden_max_windows=adaptive_harden_max_windows,
            backfill_max_windows=adaptive_backfill_max_windows,
            opener_max_windows=adaptive_opener_max_windows,
            improve_threshold_pp=adaptive_improve_threshold,
            regress_threshold_pp=adaptive_regress_threshold,
            plateau_windows=adaptive_plateau_windows,
            opener_tick_limit=opener_tick_limit,
            opener_min_health=opener_min_health,
        )
        adaptive_controller = AdaptiveController(adaptive_config)
        phase = adaptive_controller.current_phase
        episode_mode = adaptive_controller.current_episode_mode

    # Phase constraint enforcement
    if phase == "climb":
        if promote_after <= 0:
            promote_after = 5  # default for climb
        # refresh_every handled in env.py (hardcoded to 5 for climb)
    elif phase == "drill":
        promote_after = 0  # drill handles its own advancement
    elif phase in ("harden", "backfill"):
        promote_after = 0  # no curriculum advancement
    elif phase == "sweep":
        promote_after = 0  # sweep: failure-weighted sampling, no frontier

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"inferno_gpu_w{start_wave}-{max_wave}_{timestamp}"

    print(f"Run: {run_name}")
    print(f"  Device: {device}")
    print(f"  Envs: {n_envs}  Steps/rollout: {n_steps}  Batch: {batch_size}")
    print(f"  Waves: {start_wave}-{max_wave}  Weights: {start_wave_weights}")
    print(f"  Phase: {phase or 'none'}")
    print(f"  Curriculum mode: {curriculum_mode}  Episode mode: {episode_mode}")
    print(
        f"  Curriculum: promote_after={promote_after}  refresh_every={refresh_every}  "
        f"min_waves_to_advance={min_waves_to_advance}  climb_sampling={climb_sampling}"
    )
    print(
        f"  Observation: {observation_version}  Policy arch: {policy_arch}"
    )
    print(f"  Normalize obs: {normalize_obs}  Normalize reward: {normalize_reward}")
    if target_kl is not None:
        print(f"  Target KL: {target_kl}")
    effective_lstm_burn_in = 0
    if lstm_hidden_size is not None:
        effective_lstm_burn_in = max(
            0,
            lstm_seq_len // 2 if lstm_burn_in is None else lstm_burn_in,
        )
        print(f"  Recurrent: hidden={lstm_hidden_size}  seq_len={lstm_seq_len}  burn_in={effective_lstm_burn_in}")
    if hindsight_death_penalty > 0:
        print(f"  Hindsight death penalty: peak={hindsight_death_penalty}  window={hindsight_death_window}  decay={hindsight_death_decay}")

    env = SubprocVecEnv(
        n_envs=n_envs,
        start_wave=start_wave,
        max_wave=max_wave,
        start_wave_weights=start_wave_weights,
        promote_after=promote_after,
        refresh_every=refresh_every,
        min_waves_to_advance=min_waves_to_advance,
        phase=phase,
        climb_sampling=climb_sampling,
        observation_version=observation_version,
        record_reward_terms=log_reward_terms,
        max_drill_retries=max_drill_retries,
        episode_mode=episode_mode,
        opener_tick_limit=opener_tick_limit,
        opener_min_health=opener_min_health,
        fixed_loadout=fixed_loadout,
        loadout_weights=loadout_weights,
        sweep_death_retries=sweep_death_retries,
    )
    env.reconfigure(
        phase=phase,
        episode_mode=episode_mode,
        opener_tick_limit=opener_tick_limit,
        opener_min_health=opener_min_health,
        wave_stats=None,
        reward_config=reward_config_dict,
    )

    actor_observation_size = get_public_observation_size()
    critic_observation_size = get_observation_size()

    policy_params = PolicyParams(
        max_sequence_length=1,
        actor_input_size=actor_observation_size,
        critic_input_size=critic_observation_size,
        action_head_sizes=ACTION_HEAD_SIZES,
        actor_config=default_mlp_config(actor_sizes or [512, 512]),
        critic_config=default_mlp_config(critic_sizes or [512, 512]),
        feature_extractor_config=MlpConfig(),
        action_dependencies=POLICY_ACTION_DEPENDENCIES,
        autoregressive_actions=True,
        lstm_hidden_size=lstm_hidden_size,
        lstm_seq_len=lstm_seq_len,
        observation_version=observation_version,
        policy_arch=policy_arch,
        global_feature_size=0,
        entity_slot_size=0,
        max_entity_slots=0,
    )

    if load_path is not None:
        old_params = PPO.load_policy_params(load_path)
        if (
            old_params.observation_version != policy_params.observation_version
            or old_params.policy_arch != policy_params.policy_arch
        ):
            raise ValueError(
                "Cannot load checkpoint across observation/policy families "
                f"({old_params.observation_version}/{old_params.policy_arch} "
                f"-> {policy_params.observation_version}/{policy_params.policy_arch})"
            )
        needs_resize = (
            old_params.actor_config != policy_params.actor_config
            or old_params.critic_config != policy_params.critic_config
            or old_params.actor_input_size != policy_params.actor_input_size
            or old_params.lstm_hidden_size != policy_params.lstm_hidden_size
            or old_params.global_feature_size != policy_params.global_feature_size
            or old_params.entity_slot_size != policy_params.entity_slot_size
            or old_params.max_entity_slots != policy_params.max_entity_slots
            or old_params.entity_encoder_size != policy_params.entity_encoder_size
        )
        if needs_resize:
            print(f"Loading model from {load_path} with resize to match target architecture")
            ppo = PPO.load_with_resize(load_path, policy_params, device=device)
        else:
            print(f"Loading model from {load_path}")
            ppo = PPO.load(load_path, device=device, trainable=True)
        if normalize_obs and not ppo.meta.normalized_observations:
            ppo.meta.normalized_observations = True
            print("  Observation normalization enabled on loaded model (stats will warm up)")
    else:
        ppo = PPO.new_instance(
            policy_params, device=device, normalize_observations=normalize_obs
        )

    env.reconfigure(
        phase=phase,
        episode_mode=episode_mode,
        opener_tick_limit=opener_tick_limit,
        opener_min_health=opener_min_health,
        wave_stats=None,
        reward_config=reward_config_dict,
    )

    print(f"Policy:\n{ppo}")

    sampler = RolloutSampler()
    summary_writer = SummaryWriter(log_dir=os.path.join(log_dir, run_name))
    champion_tracker = TrainingChampionTracker()

    def _create_callbacks() -> tuple[
        CompositeCallback,
        WaveProgressCallback,
        OutcomeStatsCallback,
        CurriculumCallback,
        PhaseStatsCallback,
        RewardTermsCallback | None,
    ]:
        mw = WaveProgressCallback()
        oc = OutcomeStatsCallback()
        cu = CurriculumCallback(phase=phase)
        ph = PhaseStatsCallback(phase=phase)
        rt = RewardTermsCallback() if log_reward_terms else None
        cbs: list[Callback] = [mw, oc, cu, ph, champion_tracker]
        if rt is not None:
            cbs.append(rt)
        return CompositeCallback(cbs), mw, oc, cu, ph, rt

    def _maybe_update_training_champion(
        checkpoint_path: str,
        reward_summary: dict[str, float],
    ) -> PPO:
        reward_norm = ppo.meta.custom_data.get("reward_norm")
        reward_var = (
            float(reward_norm.var.item())
            if reward_norm is not None and hasattr(reward_norm, "var")
            else float("nan")
        )
        train_metrics = ppo.meta.custom_data.get("last_train_metrics", {})
        champion_updated, should_reload, champion_score = champion_tracker.maybe_update(
            checkpoint_path=checkpoint_path,
            trained_steps=ppo.meta.trained_steps,
            train_metrics=train_metrics,
            reward_summary=reward_summary,
            reward_var=reward_var,
        )
        summary_writer.add_scalar(
            "train_champion/score", champion_score, ppo.meta.trained_steps
        )
        summary_writer.add_scalar(
            "train_champion/best_score",
            champion_tracker.best_score if np.isfinite(champion_tracker.best_score) else 0.0,
            ppo.meta.trained_steps,
        )
        summary_writer.add_scalar(
            "train_champion/shape_share",
            champion_tracker._shape_share(reward_summary) if reward_summary else 0.0,
            ppo.meta.trained_steps,
        )
        if champion_updated:
            print(
                f"  [train-champion] score={champion_score:.2f} "
                f"checkpoint={checkpoint_path}"
            )
        if should_reload and champion_tracker.best_checkpoint_path is not None:
            wall_steps = ppo.meta.trained_steps
            wall_rollouts = ppo.meta.trained_rollouts
            reloaded = PPO.load(
                champion_tracker.best_checkpoint_path, device=device, trainable=True
            )
            reloaded.meta.trained_steps = wall_steps
            reloaded.meta.trained_rollouts = wall_rollouts
            print(
                f"  [train-champion] guardrail reload -> "
                f"{champion_tracker.best_checkpoint_path}"
            )
            return reloaded
        return ppo

    total_rollouts = max(1, total_timesteps // (n_steps * n_envs))
    print(f"  Total rollouts: {total_rollouts}  (~{total_rollouts * n_steps * n_envs:,} steps)")

    global_wave_stats = GlobalWaveStats() if curriculum_mode == "adaptive_v36" else None

    if adaptive_controller is not None and adaptive_config is not None:
        print(
            "  Adaptive control: "
            f"eval every {adaptive_config.eval_every_rollouts} rollouts, "
            f"{adaptive_config.eval_episodes} episodes on "
            f"W{adaptive_config.eval_start_wave}-{adaptive_config.eval_max_wave}"
        )
        print("  Seeding adaptive champion from loaded checkpoint...")
        seed_summary = evaluate_policy(
            ppo.create_inference_copy(),
            adaptive_config.eval_start_wave,
            adaptive_config.eval_max_wave,
            adaptive_config.eval_episodes,
        )
        adaptive_controller.seed_champion(
            load_path,
            seed_summary,
            ppo.meta.trained_steps,
        )
        env.reconfigure(
            phase=adaptive_controller.current_phase,
            episode_mode=adaptive_controller.current_episode_mode,
            opener_tick_limit=adaptive_config.opener_tick_limit,
            opener_min_health=adaptive_config.opener_min_health,
            wave_stats=global_wave_stats.snapshot(),
            reward_config=reward_config_dict,
        )
        _log_adaptive_metrics(
            summary_writer,
            ppo.meta.trained_steps,
            seed_summary,
            adaptive_controller,
            "seed",
        )
        print(
            f"  Seed champion: {seed_summary.full_clear_rate:.1f}% clear, "
            f"{seed_summary.death_rate:.1f}% death, {seed_summary.timeout_rate:.1f}% timeout"
        )

    # Initial collection (not pipelined — no previous training to overlap with)
    cur_cb, cur_mw, cur_oc, cur_cu, cur_ph, cur_rt = _create_callbacks()
    if global_wave_stats is not None:
        cur_ph.set_global_wave_stats(global_wave_stats.snapshot())
    buffer = sampler.collect(
        env,
        ppo,
        steps=n_steps,
        callback=cur_cb,
        gae_lambda=gae_lambda,
        gamma=gamma,
        normalize_rewards=normalize_reward,
        novelty_reward_scale=_V44_NOVELTY_SCHEDULE.value(ppo.meta.trained_rollouts),
        summary_writer=summary_writer,
        hindsight_death_penalty=hindsight_death_penalty,
        hindsight_death_window=hindsight_death_window,
        hindsight_death_decay=hindsight_death_decay,
    )

    if adaptive_controller is not None and adaptive_config is not None:
        for rollout in range(total_rollouts):
            entropy_coef = _lerp(
                entropy_start, entropy_end, rollout / max(1, total_rollouts - 1)
            )

            ppo.learn(
                buffer,
                summary_writer=summary_writer,
                num_updates=n_epochs,
                batch_size=batch_size,
                clip_coef=clip_coef,
                vf_coef=vf_coef,
                entropy_coef=entropy_coef,
                max_grad_norm=max_grad_norm,
                learning_rate=learning_rate,
                target_kl=target_kl,
                lstm_burn_in=effective_lstm_burn_in,
            )

            prestige_from_cur, frontier_sync_from_cur, reward_summary = _log_rollout_callbacks(
                summary_writer,
                ppo,
                cur_mw,
                cur_oc,
                cur_cu,
                cur_ph,
                cur_rt,
                global_wave_stats=global_wave_stats,
            )

            if prestige_from_cur is not None:
                env.apply_prestige(prestige_from_cur)
                print(
                    f"  [prestige] Global sync: all envs reset to start_wave, "
                    f"min_waves_to_advance={prestige_from_cur}"
                )
            elif frontier_sync_from_cur is not None:
                frontier_wave, frontier_min_waves = frontier_sync_from_cur
                env.sync_curriculum(frontier_wave, frontier_min_waves)

            should_eval = (
                (rollout + 1) % adaptive_config.eval_every_rollouts == 0
                or rollout == total_rollouts - 1
            )
            should_checkpoint = (
                (rollout + 1) % checkpoint_every == 0
                or rollout == total_rollouts - 1
                or should_eval
            )
            checkpoint_path: str | None = None
            if should_checkpoint:
                checkpoint_path = _save_checkpoint(
                    ppo, save_dir, run_name, str(rollout + 1)
                )
                print(
                    f"[rollout {rollout + 1}/{total_rollouts}] "
                    f"steps={ppo.meta.trained_steps:,}  "
                    f"checkpoint -> {checkpoint_path}"
                )
                ppo = _maybe_update_training_champion(checkpoint_path, reward_summary)

            if should_eval:
                assert checkpoint_path is not None
                eval_summary = evaluate_policy(
                    ppo.create_inference_copy(),
                    adaptive_config.eval_start_wave,
                    adaptive_config.eval_max_wave,
                    adaptive_config.eval_episodes,
                )
                decision = adaptive_controller.evaluate_window(
                    checkpoint_path,
                    eval_summary,
                    ppo.meta.trained_steps,
                )
                _log_adaptive_metrics(
                    summary_writer,
                    ppo.meta.trained_steps,
                    eval_summary,
                    adaptive_controller,
                    decision.switch_reason,
                )
                print(
                    f"  [adaptive] eval={eval_summary.full_clear_rate:.1f}% clear, "
                    f"death={eval_summary.death_rate:.1f}% timeout={eval_summary.timeout_rate:.1f}% "
                    f"regime={adaptive_controller.current_regime} reason={decision.switch_reason}"
                )

                if decision.should_reload_champion and decision.champion_path is not None:
                    wall_steps = ppo.meta.trained_steps
                    wall_rollouts = ppo.meta.trained_rollouts
                    ppo = PPO.load(decision.champion_path, device=device, trainable=True)
                    ppo.meta.trained_steps = wall_steps
                    ppo.meta.trained_rollouts = wall_rollouts
                    print(f"  [adaptive] reloaded champion -> {decision.champion_path}")

                env.reconfigure(
                    phase=adaptive_controller.current_phase,
                    episode_mode=adaptive_controller.current_episode_mode,
                    opener_tick_limit=adaptive_config.opener_tick_limit,
                    opener_min_health=adaptive_config.opener_min_health,
                    wave_stats=global_wave_stats.snapshot(),
                    reward_config=reward_config_dict,
                )
                phase = adaptive_controller.current_phase

            if rollout < total_rollouts - 1:
                env.reconfigure(
                    phase=adaptive_controller.current_phase,
                    episode_mode=adaptive_controller.current_episode_mode,
                    opener_tick_limit=adaptive_config.opener_tick_limit,
                    opener_min_health=adaptive_config.opener_min_health,
                    wave_stats=global_wave_stats.snapshot(),
                    reward_config=reward_config_dict,
                )
                cur_cb, cur_mw, cur_oc, cur_cu, cur_ph, cur_rt = _create_callbacks()
                cur_ph.set_global_wave_stats(global_wave_stats.snapshot())
                buffer = sampler.collect(
                    env,
                    ppo,
                    steps=n_steps,
                    callback=cur_cb,
                    gae_lambda=gae_lambda,
                    gamma=gamma,
                    normalize_rewards=normalize_reward,
                    novelty_reward_scale=_V44_NOVELTY_SCHEDULE.value(ppo.meta.trained_rollouts),
                    summary_writer=summary_writer,
                    hindsight_death_penalty=hindsight_death_penalty,
                    hindsight_death_window=hindsight_death_window,
                    hindsight_death_decay=hindsight_death_decay,
                )
    else:
        for rollout in range(total_rollouts):
            entropy_coef = _lerp(entropy_start, entropy_end, rollout / max(1, total_rollouts - 1))

            ppo.learn(
                buffer,
                summary_writer=summary_writer,
                num_updates=n_epochs,
                batch_size=batch_size,
                clip_coef=clip_coef,
                vf_coef=vf_coef,
                entropy_coef=entropy_coef,
                max_grad_norm=max_grad_norm,
                learning_rate=learning_rate,
                target_kl=target_kl,
                lstm_burn_in=effective_lstm_burn_in,
            )

            prestige_from_cur, frontier_sync_from_cur, reward_summary = _log_rollout_callbacks(
                summary_writer,
                ppo,
                cur_mw,
                cur_oc,
                cur_cu,
                cur_ph,
                cur_rt,
                global_wave_stats=None,
            )

            if (rollout + 1) % checkpoint_every == 0 or rollout == total_rollouts - 1:
                checkpoint_path = _save_checkpoint(ppo, save_dir, run_name, str(rollout + 1))
                print(
                    f"[rollout {rollout + 1}/{total_rollouts}] "
                    f"steps={ppo.meta.trained_steps:,}  "
                    f"checkpoint -> {checkpoint_path}"
                )
                ppo = _maybe_update_training_champion(checkpoint_path, reward_summary)

            if prestige_from_cur is not None:
                env.apply_prestige(prestige_from_cur)
                print(
                    f"  [prestige] Global sync: all envs reset to start_wave, "
                    f"min_waves_to_advance={prestige_from_cur}"
                )
            elif frontier_sync_from_cur is not None:
                frontier_wave, frontier_min_waves = frontier_sync_from_cur
                env.sync_curriculum(frontier_wave, frontier_min_waves)

            if rollout < total_rollouts - 1:
                env.reconfigure(
                    phase=phase,
                    episode_mode=episode_mode,
                    opener_tick_limit=opener_tick_limit,
                    opener_min_health=opener_min_health,
                    wave_stats=None,
                    reward_config=reward_config_dict,
                )
                cur_cb, cur_mw, cur_oc, cur_cu, cur_ph, cur_rt = _create_callbacks()
                buffer = sampler.collect(
                    env,
                    ppo,
                    steps=n_steps,
                    callback=cur_cb,
                    gae_lambda=gae_lambda,
                    gamma=gamma,
                    normalize_rewards=normalize_reward,
                    novelty_reward_scale=_V44_NOVELTY_SCHEDULE.value(ppo.meta.trained_rollouts),
                    summary_writer=summary_writer,
                    hindsight_death_penalty=hindsight_death_penalty,
                    hindsight_death_window=hindsight_death_window,
                    hindsight_death_decay=hindsight_death_decay,
                )

    env.close()
    summary_writer.close()
    print("Training complete.")
    return ppo


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _reward_config_from_args(args: argparse.Namespace, start_wave: int, max_wave: int) -> RewardConfig:
    """Build a RewardConfig from --rw-* CLI args."""
    return RewardConfig(
        death_penalty=args.rw_death_penalty,
        wave_timeout_penalty=args.rw_wave_timeout_penalty,
        damage_taken_per_hp=args.rw_damage_taken_per_hp,
        damage_dealt_per_hp=args.rw_damage_dealt_per_hp,
        blood_barrage_heal_per_hp=args.rw_blood_barrage_heal_per_hp,
        blood_barrage_high_hp_penalty=args.rw_blood_barrage_high_hp_penalty,
        wave_complete_base=args.rw_wave_complete_base,
        wave_progress_bonus=args.rw_wave_progress_bonus,
        inferno_complete_reward=args.rw_inferno_complete_reward,
        wave_end_hp_bonus=args.rw_wave_end_hp_bonus,
        kill_reward_scale=args.rw_kill_reward_scale,
        stall_base_penalty=args.rw_stall_base_penalty,
        stall_escalation=args.rw_stall_escalation,
        invalid_action_penalty=args.rw_invalid_action_penalty,
        invalid_attack_penalty=args.rw_invalid_attack_penalty,
        pillar_damage_per_hp=args.rw_pillar_damage_per_hp,
        pillar_death_penalty=args.rw_pillar_death_penalty,
        ne_pillar_death_penalty=args.rw_ne_pillar_death_penalty,
        ne_pillar_zone_bonus=args.rw_ne_pillar_zone_bonus,
        ne_pillar_zone_penalty=args.rw_ne_pillar_zone_penalty,
        mager_resurrection_penalty=args.rw_mager_resurrection_penalty,
        melee_resurrection_penalty=args.rw_melee_resurrection_penalty,
        mager_priority_per_npc=args.rw_mager_priority_per_npc,
        mager_early_kill_base=args.rw_mager_early_kill_base,
        mager_early_kill_per_npc=args.rw_mager_early_kill_per_npc,
        mager_delay_penalty=args.rw_mager_delay_penalty,
        c_tile_on_reward=args.rw_c_tile_on_reward,
        c_tile_adjacent_reward=args.rw_c_tile_adjacent_reward,
        tile_a_max_reward=args.rw_tile_a_max_reward,
        adjacent_npc_attack_penalty=args.rw_adjacent_npc_attack_penalty,
        los_separation_bonus=args.rw_los_separation_bonus,
        avoidable_imminent_penalty=args.rw_avoidable_imminent_penalty,
        attack_on_cooldown_bonus=args.rw_attack_on_cooldown_bonus,
        weapon_switch_penalty=args.rw_weapon_switch_penalty,
        start_wave=start_wave,
        max_wave=max_wave,
    )


def main() -> None:
    import torch as th

    parser = argparse.ArgumentParser(description="Train Inferno RL Agent (GPU PPO)")

    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--start-wave", type=int, default=1)
    parser.add_argument("--max-wave", type=int, default=66)
    parser.add_argument("--mixed-waves", type=str, default=None,
                        help="Mixed start waves (e.g. '50:0.6,35:0.4')")
    parser.add_argument("--promote-after", type=int, default=0,
                        help="Curriculum: consecutive completions before frontier advances. 0=off.")
    parser.add_argument("--refresh-every", type=int, default=10,
                        help="Curriculum: revisit a mastered wave every N episodes.")
    parser.add_argument("--min-waves-to-advance", type=int, default=1,
                        help="Curriculum: waves that must be cleared from frontier in one episode to count as a completion.")
    parser.add_argument("--climb-sampling", type=str, default="weighted",
                        choices=["weighted", "legacy"],
                        help="Climb phase non-refresh starts: weighted frontier band (V32) or fixed frontier-3 (V31).")
    parser.add_argument("--n-envs", type=int, default=48)
    parser.add_argument("--device", type=str,
                        default="cuda" if th.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", type=str, default="models/inferno_gpu")
    parser.add_argument("--log-dir", type=str, default="logs/inferno_gpu")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--n-epochs", type=int, default=1)
    parser.add_argument("--entropy-start", type=float, default=0.05)
    parser.add_argument("--entropy-end", type=float, default=0.002)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--normalize-obs", dest="normalize_obs", action="store_true")
    parser.add_argument("--no-normalize-obs", dest="normalize_obs", action="store_false")
    parser.add_argument("--normalize-reward", dest="normalize_reward", action="store_true")
    parser.add_argument("--no-normalize-reward", dest="normalize_reward", action="store_false")
    parser.add_argument("--novelty-scale", type=float, default=0.0)
    parser.add_argument("--log-reward-terms", action="store_true",
                        help="Log per-episode raw reward terms (TensorBoard: raw_reward_terms/*)")
    parser.add_argument("--checkpoint-every", type=int, default=100,
                        help="Save checkpoint every N rollouts")
    parser.add_argument("--gamma", type=float, default=0.995,
                        help="Discount factor for GAE (default: 0.995)")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="GAE lambda for advantage estimation (default: 0.95)")
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.02,
                        help="Early-stop PPO updates when approximate KL exceeds this threshold (0=disabled)")
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--phase", type=str, default="sweep",
                        choices=["climb", "harden", "backfill", "drill", "sweep", "none"],
                        help="Training phase: climb (forward curriculum), "
                             "harden (uniform random), backfill (failure-weighted), "
                             "drill (retry-on-failure), sweep (all waves, failure-weighted), "
                             "none (always start at --start-wave)")
    parser.add_argument("--curriculum-mode", type=str, default="static",
                        choices=["static", "adaptive_v36"],
                        help="Static phase training or adaptive V36 controller")
    parser.add_argument("--adaptive-eval-every", type=int, default=50,
                        help="Adaptive mode: evaluate every N rollouts")
    parser.add_argument("--adaptive-eval-episodes", type=int, default=100,
                        help="Adaptive mode: evaluation episodes per control window")
    parser.add_argument("--adaptive-eval-start-wave", type=int, default=49,
                        help="Adaptive mode: eval start wave")
    parser.add_argument("--adaptive-eval-max-wave", type=int, default=66,
                        help="Adaptive mode: eval max wave")
    parser.add_argument("--adaptive-harden-max-windows", type=int, default=3,
                        help="Adaptive mode: max eval windows to stay in harden_full")
    parser.add_argument("--adaptive-backfill-max-windows", type=int, default=6,
                        help="Adaptive mode: max eval windows to stay in backfill_full")
    parser.add_argument("--adaptive-opener-max-windows", type=int, default=1,
                        help="Adaptive mode: max eval windows to stay in backfill_opener")
    parser.add_argument("--adaptive-improve-threshold", type=float, default=0.5,
                        help="Adaptive mode: percentage-point improvement needed to update champion")
    parser.add_argument("--adaptive-regress-threshold", type=float, default=2.0,
                        help="Adaptive mode: percentage-point drop vs champion before rollback")
    parser.add_argument("--adaptive-plateau-windows", type=int, default=2,
                        help="Adaptive mode: consecutive plateau windows before switching regime")
    parser.add_argument("--max-drill-retries", type=int, default=10,
                        help="Drill phase: max consecutive deaths on same wave before auto-advancing.")
    parser.add_argument("--episode-mode", type=str, default="full",
                        choices=["full", "opener"],
                        help="Episode termination mode for static training")
    parser.add_argument("--opener-tick-limit", type=int, default=50,
                        help="Opener mode: succeed/fail boundary in ticks")
    parser.add_argument("--opener-min-health", type=int, default=40,
                        help="Opener mode: minimum HP threshold for success")
    parser.add_argument("--actor-sizes", type=str, default="512,512",
                        help="Actor hidden layer sizes, comma-separated (default: 512,512)")
    parser.add_argument("--critic-sizes", type=str, default="512,512",
                        help="Critic hidden layer sizes, comma-separated (default: 512,512)")
    parser.add_argument("--resize-lstm", type=int, default=None,
                        help="Resize LSTM hidden from checkpoint size to this target size")
    parser.add_argument("--lstm-hidden-size", type=int, default=128,
                        help="LSTM hidden size (default: 128)")
    parser.add_argument("--lstm-seq-len", type=int, default=16,
                        help="LSTM sequence length for training (default: 16)")
    parser.add_argument("--lstm-burn-in", type=int, default=None,
                        help="Warmup ticks before each LSTM training window (default: seq_len // 2)")
    parser.add_argument("--observation-version", type=str, default="v4",
                        help="Observation schema version (v4)")
    parser.add_argument("--policy-arch", type=str, default="flat_lstm_residual",
                        choices=["flat", "flat_lstm_residual"],
                        help="Policy front-end architecture")
    parser.set_defaults(normalize_obs=True, normalize_reward=True)
    parser.add_argument("--loadout", type=str, default=None,
                        help="Force a single loadout by name (e.g. BUDGET_RCB, CRYSTAL_BP)")
    parser.add_argument("--loadout-weights", type=str, default=None,
                        help='Loadout sampling weights as JSON: \'{"BUDGET_RCB":0.25,"CRYSTAL_BP":0.25,...}\'')
    parser.add_argument("--sweep-death-retries", type=int, default=0,
                        help="After death in sweep mode, retry the death wave this many times before resuming normal sampling")
    parser.add_argument("--hindsight-death-penalty", type=float, default=0.0,
                        help="Peak penalty injected on ticks before death/timeout (0=disabled)")
    parser.add_argument("--hindsight-death-window", type=int, default=10,
                        help="How many ticks before terminal to penalize")
    parser.add_argument("--hindsight-death-decay", type=float, default=0.8,
                        help="Exponential decay factor per tick (e.g. 0.8 = 20%% drop per tick)")

    # --- Reward configuration (--rw-* flags) ---
    rw = parser.add_argument_group(
        "Reward Configuration",
        "Override individual reward parameters. Defaults are the minimal config (V51). "
        "To restore full V50 rewards, pass the 'full' values shown in each help string.",
    )
    rw.add_argument("--rw-death-penalty", type=float, default=-20.0,
                    help="Terminal death penalty (full: -20.0, default: %(default)s)")
    rw.add_argument("--rw-wave-timeout-penalty", type=float, default=-15.0,
                    help="Terminal wave timeout penalty (V44: 0.0, default: %(default)s)")
    rw.add_argument("--rw-damage-taken-per-hp", type=float, default=-0.05,
                    help="Per-HP damage taken penalty (full: -0.05, default: %(default)s)")
    rw.add_argument("--rw-damage-dealt-per-hp", type=float, default=0.003,
                    help="Per-HP damage dealt reward (full: 0.006, default: %(default)s)")
    rw.add_argument("--rw-blood-barrage-heal-per-hp", type=float, default=0.0,
                    help="Per-HP blood barrage heal reward (full: 0.06, default: %(default)s)")
    rw.add_argument("--rw-blood-barrage-high-hp-penalty", type=float, default=-0.2,
                    help="Penalty for blood barrage at high HP (full: -0.2, default: %(default)s)")
    rw.add_argument("--rw-wave-complete-base", type=float, default=3.0,
                    help="Base wave completion reward (full: 3.0, default: %(default)s)")
    rw.add_argument("--rw-wave-progress-bonus", type=float, default=5.0,
                    help="Wave progress scaling bonus (full: 5.0, default: %(default)s)")
    rw.add_argument("--rw-inferno-complete-reward", type=float, default=15.0,
                    help="Inferno completion reward (full: 15.0, default: %(default)s)")
    rw.add_argument("--rw-wave-end-hp-bonus", type=float, default=3.0,
                    help="HP bonus at wave end (full: 3.0, default: %(default)s)")
    rw.add_argument("--rw-kill-reward-scale", type=float, default=0.0,
                    help="Scale factor for all kill rewards (full: 1.0, default: %(default)s)")
    rw.add_argument("--rw-stall-base-penalty", type=float, default=-0.08,
                    help="Base stall penalty (full: -0.08, default: %(default)s)")
    rw.add_argument("--rw-stall-escalation", type=float, default=0.04,
                    help="Stall escalation per tick (full: 0.04, default: %(default)s)")
    rw.add_argument("--rw-invalid-action-penalty", type=float, default=0.0,
                    help="Invalid non-attack action penalty (full: -0.1, default: %(default)s)")
    rw.add_argument("--rw-invalid-attack-penalty", type=float, default=0.0,
                    help="Invalid attack action penalty (full: -0.05, default: %(default)s)")
    rw.add_argument("--rw-pillar-damage-per-hp", type=float, default=0.0,
                    help="Per-HP pillar damage penalty (full: -0.01, default: %(default)s)")
    rw.add_argument("--rw-pillar-death-penalty", type=float, default=-7.5,
                    help="One-time penalty when NW/S pillar dies (default: %(default)s)")
    rw.add_argument("--rw-ne-pillar-death-penalty", type=float, default=-15.0,
                    help="One-time penalty when NE pillar dies (default: %(default)s)")
    rw.add_argument("--rw-ne-pillar-zone-bonus", type=float, default=0.0,
                    help="Per-tick NE pillar zone bonus (full: 0.008, default: %(default)s)")
    rw.add_argument("--rw-ne-pillar-zone-penalty", type=float, default=0.0,
                    help="Per-tick NE pillar zone penalty (full: -0.02, default: %(default)s)")
    rw.add_argument("--rw-mager-resurrection-penalty", type=float, default=0.0,
                    help="Mager resurrection penalty (full: -0.6, default: %(default)s)")
    rw.add_argument("--rw-melee-resurrection-penalty", type=float, default=0.0,
                    help="Melee resurrection penalty (full: -0.3, default: %(default)s)")
    rw.add_argument("--rw-mager-priority-per-npc", type=float, default=0.25,
                    help="Mager priority bonus per non-mager NPC (full: 0.25, default: %(default)s)")
    rw.add_argument("--rw-mager-early-kill-base", type=float, default=0.6,
                    help="Base bonus for early mager kill (full: 0.6, default: %(default)s)")
    rw.add_argument("--rw-mager-early-kill-per-npc", type=float, default=0.15,
                    help="Per-NPC bonus for early mager kill (full: 0.15, default: %(default)s)")
    rw.add_argument("--rw-mager-delay-penalty", type=float, default=-0.02,
                    help="Penalty for not progressing mager when safe (full: -0.02, default: %(default)s)")
    rw.add_argument("--rw-c-tile-on-reward", type=float, default=0.0,
                    help="Per-tick C tile position reward (full: 0.5, default: %(default)s)")
    rw.add_argument("--rw-c-tile-adjacent-reward", type=float, default=0.0,
                    help="Per-tick C tile adjacent reward (full: 0.25, default: %(default)s)")
    rw.add_argument("--rw-tile-a-max-reward", type=float, default=0.0,
                    help="Tile A proximity reward (full: 0.04, default: %(default)s)")
    rw.add_argument("--rw-adjacent-npc-attack-penalty", type=float, default=0.0,
                    help="Per-NPC adjacent attack penalty (full: -0.40, default: %(default)s)")
    rw.add_argument("--rw-los-separation-bonus", type=float, default=0.01,
                    help="LOS separation bonus (full: 0.025, default: %(default)s)")
    rw.add_argument("--rw-avoidable-imminent-penalty", type=float, default=0.0,
                    help="Per-NPC avoidable imminent penalty (full: -0.01, default: %(default)s)")
    rw.add_argument("--rw-attack-on-cooldown-bonus", type=float, default=0.0,
                    help="Bonus for attacking on cooldown (full: 0.0, default: %(default)s)")
    rw.add_argument("--rw-weapon-switch-penalty", type=float, default=-0.005,
                    help="Per-switch penalty for weapon swapping (default: %(default)s)")

    args = parser.parse_args()

    import json as _json
    start_wave_weights = _parse_wave_weights(args.mixed_waves)
    actor_sizes = [int(x) for x in args.actor_sizes.split(",")] if args.actor_sizes else None
    critic_sizes = [int(x) for x in args.critic_sizes.split(",")] if args.critic_sizes else None
    loadout_weights = _json.loads(args.loadout_weights) if args.loadout_weights else None
    rw_config = _reward_config_from_args(args, args.start_wave, args.max_wave)

    train(
        total_timesteps=args.timesteps,
        start_wave=args.start_wave,
        max_wave=args.max_wave,
        start_wave_weights=start_wave_weights,
        promote_after=args.promote_after,
        refresh_every=args.refresh_every,
        min_waves_to_advance=args.min_waves_to_advance,
        climb_sampling=args.climb_sampling,
        n_envs=args.n_envs,
        device=args.device,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        entropy_start=args.entropy_start,
        entropy_end=args.entropy_end,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        vf_coef=args.vf_coef,
        target_kl=args.target_kl,
        normalize_obs=args.normalize_obs,
        normalize_reward=args.normalize_reward,
        novelty_scale=args.novelty_scale,
        log_reward_terms=args.log_reward_terms,
        checkpoint_every=args.checkpoint_every,
        max_grad_norm=args.max_grad_norm,
        load_path=args.load,
        phase=None if args.phase == "none" else args.phase,
        actor_sizes=actor_sizes,
        critic_sizes=critic_sizes,
        resize_lstm=args.resize_lstm,
        lstm_hidden_size=args.lstm_hidden_size,
        lstm_seq_len=args.lstm_seq_len,
        lstm_burn_in=args.lstm_burn_in,
        max_drill_retries=args.max_drill_retries,
        hindsight_death_penalty=args.hindsight_death_penalty,
        hindsight_death_window=args.hindsight_death_window,
        hindsight_death_decay=args.hindsight_death_decay,
        observation_version=args.observation_version,
        policy_arch=args.policy_arch,
        curriculum_mode=args.curriculum_mode,
        adaptive_eval_every=args.adaptive_eval_every,
        adaptive_eval_episodes=args.adaptive_eval_episodes,
        adaptive_eval_start_wave=args.adaptive_eval_start_wave,
        adaptive_eval_max_wave=args.adaptive_eval_max_wave,
        adaptive_harden_max_windows=args.adaptive_harden_max_windows,
        adaptive_backfill_max_windows=args.adaptive_backfill_max_windows,
        adaptive_opener_max_windows=args.adaptive_opener_max_windows,
        adaptive_improve_threshold=args.adaptive_improve_threshold,
        adaptive_regress_threshold=args.adaptive_regress_threshold,
        adaptive_plateau_windows=args.adaptive_plateau_windows,
        episode_mode=args.episode_mode,
        opener_tick_limit=args.opener_tick_limit,
        opener_min_health=args.opener_min_health,
        fixed_loadout=args.loadout,
        loadout_weights=loadout_weights,
        sweep_death_retries=args.sweep_death_retries,
        reward_config=rw_config,
    )


if __name__ == "__main__":
    main()
