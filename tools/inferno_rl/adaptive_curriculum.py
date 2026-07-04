from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal


AdaptiveRegime = Literal["harden_full", "backfill_full", "backfill_opener"]
EpisodeMode = Literal["full", "opener"]
SwitchReason = Literal["seed", "stay", "improved", "plateau", "regress", "max_windows"]

_REGIME_SEQUENCE: tuple[AdaptiveRegime, ...] = (
    "harden_full",
    "backfill_full",
    "backfill_opener",
)

_REGIME_PHASE: dict[AdaptiveRegime, str] = {
    "harden_full": "harden",
    "backfill_full": "backfill",
    "backfill_opener": "backfill",
}

_REGIME_EPISODE_MODE: dict[AdaptiveRegime, EpisodeMode] = {
    "harden_full": "full",
    "backfill_full": "full",
    "backfill_opener": "opener",
}

_REGIME_CODE: dict[AdaptiveRegime, int] = {
    "harden_full": 0,
    "backfill_full": 1,
    "backfill_opener": 2,
}

_SWITCH_REASON_CODE: dict[SwitchReason, int] = {
    "seed": 0,
    "stay": 1,
    "improved": 2,
    "plateau": 3,
    "regress": 4,
    "max_windows": 5,
}


@dataclass(frozen=True)
class AdaptiveConfig:
    eval_every_rollouts: int = 50
    eval_episodes: int = 100
    eval_start_wave: int = 49
    eval_max_wave: int = 66
    harden_max_windows: int = 3
    backfill_max_windows: int = 6
    opener_max_windows: int = 1
    improve_threshold_pp: float = 0.5
    regress_threshold_pp: float = 2.0
    plateau_windows: int = 2
    opener_tick_limit: int = 50
    opener_min_health: int = 40

    def max_windows_for(self, regime: AdaptiveRegime) -> int:
        if regime == "harden_full":
            return self.harden_max_windows
        if regime == "backfill_full":
            return self.backfill_max_windows
        return self.opener_max_windows


@dataclass(frozen=True)
class EvalSummary:
    full_clear_rate: float
    death_rate: float
    timeout_rate: float
    mean_max_wave: float
    death_counts_by_wave: Dict[int, int]
    episodes: int


@dataclass(frozen=True)
class ChampionState:
    checkpoint_path: str
    score: float
    death_rate: float
    timeout_rate: float
    mean_max_wave: float
    regime: AdaptiveRegime
    trained_steps: int


@dataclass(frozen=True)
class AdaptiveDecision:
    champion_updated: bool
    should_reload_champion: bool
    champion_path: str | None
    next_regime: AdaptiveRegime
    switch_reason: SwitchReason


class GlobalWaveStats:
    """Global backfill stats aggregated across rollouts and workers."""

    def __init__(self) -> None:
        self._stats: dict[int, dict[str, int]] = {}

    def merge_episode(self, wave: int, success: bool) -> None:
        bucket = self._stats.setdefault(wave, {"fails": 0, "successes": 0})
        if success:
            bucket["successes"] += 1
        else:
            bucket["fails"] += 1

    def merge_updates(self, updates: list[tuple[int, bool]]) -> None:
        for wave, success in updates:
            self.merge_episode(wave, success)

    def snapshot(self) -> dict[int, dict[str, int]]:
        return {
            wave: {"fails": stats["fails"], "successes": stats["successes"]}
            for wave, stats in self._stats.items()
        }

    def summarize(self, mastery_fail_threshold: float = 0.02, mastery_min_samples: int = 25) -> tuple[int | None, float, int]:
        worst_wave: int | None = None
        worst_rate = -1.0
        mastered = 0
        for wave, stats in self._stats.items():
            total = stats["fails"] + stats["successes"]
            if total <= 0:
                continue
            fail_rate = stats["fails"] / total
            if fail_rate > worst_rate:
                worst_rate = fail_rate
                worst_wave = wave
            if total >= mastery_min_samples and fail_rate < mastery_fail_threshold:
                mastered += 1
        if worst_rate < 0.0:
            worst_rate = 0.0
        return worst_wave, worst_rate, mastered


class AdaptiveController:
    """Owns regime sequencing, champion selection, and rollback decisions."""

    def __init__(
        self,
        config: AdaptiveConfig,
        initial_regime: AdaptiveRegime = "harden_full",
    ) -> None:
        self.config = config
        self._current_regime: AdaptiveRegime = initial_regime
        self._current_regime_window = 0
        self._plateau_windows = 0
        self._rollback_count = 0
        self._champion: ChampionState | None = None

    @property
    def current_regime(self) -> AdaptiveRegime:
        return self._current_regime

    @property
    def current_phase(self) -> str:
        return _REGIME_PHASE[self._current_regime]

    @property
    def current_episode_mode(self) -> EpisodeMode:
        return _REGIME_EPISODE_MODE[self._current_regime]

    @property
    def current_regime_code(self) -> int:
        return _REGIME_CODE[self._current_regime]

    @property
    def current_regime_window(self) -> int:
        return self._current_regime_window

    @property
    def plateau_windows(self) -> int:
        return self._plateau_windows

    @property
    def rollback_count(self) -> int:
        return self._rollback_count

    @property
    def champion(self) -> ChampionState | None:
        return self._champion

    def switch_reason_code(self, reason: SwitchReason) -> int:
        return _SWITCH_REASON_CODE[reason]

    def seed_champion(
        self,
        checkpoint_path: str,
        summary: EvalSummary,
        trained_steps: int,
    ) -> None:
        self._champion = ChampionState(
            checkpoint_path=checkpoint_path,
            score=summary.full_clear_rate,
            death_rate=summary.death_rate,
            timeout_rate=summary.timeout_rate,
            mean_max_wave=summary.mean_max_wave,
            regime=self._current_regime,
            trained_steps=trained_steps,
        )
        self._current_regime_window = 0
        self._plateau_windows = 0

    def evaluate_window(
        self,
        candidate_path: str,
        summary: EvalSummary,
        trained_steps: int,
    ) -> AdaptiveDecision:
        if self._champion is None:
            self.seed_champion(candidate_path, summary, trained_steps)
            return AdaptiveDecision(
                champion_updated=True,
                should_reload_champion=False,
                champion_path=None,
                next_regime=self._current_regime,
                switch_reason="seed",
            )

        self._current_regime_window += 1
        champion_updated = False
        reason: SwitchReason = "stay"

        if self._is_better(summary, self._champion):
            self._champion = ChampionState(
                checkpoint_path=candidate_path,
                score=summary.full_clear_rate,
                death_rate=summary.death_rate,
                timeout_rate=summary.timeout_rate,
                mean_max_wave=summary.mean_max_wave,
                regime=self._current_regime,
                trained_steps=trained_steps,
            )
            self._plateau_windows = 0
            champion_updated = True
            reason = "improved"
        else:
            delta = summary.full_clear_rate - self._champion.score
            if delta < -self.config.regress_threshold_pp:
                return self._switch("regress")
            if self._is_plateau(summary, self._champion):
                self._plateau_windows += 1
            else:
                self._plateau_windows = 0
            if self._plateau_windows >= self.config.plateau_windows:
                return self._switch("plateau")

        if self._current_regime_window >= self.config.max_windows_for(self._current_regime):
            decision = self._switch("max_windows")
            if champion_updated:
                return AdaptiveDecision(
                    champion_updated=True,
                    should_reload_champion=decision.should_reload_champion,
                    champion_path=decision.champion_path,
                    next_regime=decision.next_regime,
                    switch_reason=decision.switch_reason,
                )
            return decision

        return AdaptiveDecision(
            champion_updated=champion_updated,
            should_reload_champion=False,
            champion_path=None,
            next_regime=self._current_regime,
            switch_reason=reason,
        )

    def _switch(self, reason: SwitchReason) -> AdaptiveDecision:
        self._current_regime = self._next_regime(self._current_regime)
        self._current_regime_window = 0
        self._plateau_windows = 0
        self._rollback_count += 1
        return AdaptiveDecision(
            champion_updated=False,
            should_reload_champion=self._champion is not None,
            champion_path=None if self._champion is None else self._champion.checkpoint_path,
            next_regime=self._current_regime,
            switch_reason=reason,
        )

    @staticmethod
    def _next_regime(regime: AdaptiveRegime) -> AdaptiveRegime:
        idx = _REGIME_SEQUENCE.index(regime)
        return _REGIME_SEQUENCE[(idx + 1) % len(_REGIME_SEQUENCE)]

    def _is_plateau(self, summary: EvalSummary, champion: ChampionState) -> bool:
        if self._is_better(summary, champion):
            return False
        return abs(summary.full_clear_rate - champion.score) <= self.config.improve_threshold_pp

    def _is_better(self, summary: EvalSummary, champion: ChampionState) -> bool:
        if summary.full_clear_rate > champion.score + self.config.improve_threshold_pp:
            return True
        if summary.full_clear_rate < champion.score - self.config.improve_threshold_pp:
            return False
        # Tie-breakers for near-equal scores.
        if summary.death_rate < champion.death_rate - 1e-9:
            return True
        if summary.death_rate > champion.death_rate + 1e-9:
            return False
        if summary.timeout_rate < champion.timeout_rate - 1e-9:
            return True
        if summary.timeout_rate > champion.timeout_rate + 1e-9:
            return False
        return summary.mean_max_wave > champion.mean_max_wave + 1e-9
