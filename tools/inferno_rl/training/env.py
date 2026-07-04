"""
Gymnasium environment for Inferno RL training with SB3.

This environment wraps the Inferno simulator and provides:
- Action masking for MaskablePPO
- Observation space construction
- Reward shaping
"""

import random
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..adaptive_curriculum import EpisodeMode
from ..simulator.entity import EntityTypes
from ..simulator.equipment import Loadout, LoadoutId, LOADOUTS, DEFAULT_LOADOUT
from ..simulator.forecast import (
    TickThreatCache,
    build_tick_threat_cache,
)
from ..simulator.simulator import InfernoSimulator
from .observation import (
    TemporalState,
    build_observation,
    get_observation_low,
    get_observation_size,
    update_temporal_state,
)
from .rewards import InfernoReward, RewardConfig, normalize_reward_term_name
from .actions import (
    ACTION_HEAD_SIZES,
    decode_policy_action,
    get_policy_action_mask,
)


class InfernoEnv(gym.Env):
    """
    Gymnasium environment for Inferno RL training.

    Supports action masking for sb3_contrib.MaskablePPO.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    OPENER_RESOLVED_SUCCESS_BONUS = 2.0
    OPENER_SURVIVE_WINDOW_BONUS = 0.5
    OPENER_FAILURE_PENALTY = -1.0
    BACKFILL_WARMUP_EPISODES = 100

    def __init__(
        self,
        start_wave: int = 1,
        max_wave: int = 66,
        render_mode: Optional[str] = None,
        start_wave_weights: Optional[Dict[int, float]] = None,
        promote_after: int = 0,
        refresh_every: int = 10,
        min_waves_to_advance: int = 1,
        phase: Optional[str] = None,
        climb_sampling: str = "weighted",
        observation_version: str = "v4",
        record_reward_terms: bool = False,
        max_drill_retries: int = 10,
        episode_mode: EpisodeMode = "full",
        opener_tick_limit: int = 50,
        opener_min_health: int = 40,
        fixed_loadout: Optional[str] = None,
        loadout_weights: Optional[Dict[str, float]] = None,
        sweep_death_retries: int = 0,
    ):
        """
        Create Inferno environment.

        Args:
            start_wave: Wave to start training from (default if no weights)
            max_wave: Maximum wave (episode ends when cleared)
            render_mode: Render mode (None, "human", "rgb_array")
            start_wave_weights: Dict of {wave: probability} for mixed starts
            promote_after: Consecutive completions before frontier advances.
                0 = disabled (use start_wave / start_wave_weights as before).
            refresh_every: Every N episodes, revisit a random already-mastered
                wave instead of the frontier. Only active when promote_after > 0.
            min_waves_to_advance: Number of waves that must be cleared from the
                frontier in a single episode to count as a completion. Default 1
                (just clear the start wave). Set to e.g. 3 to require the agent
                to survive frontier, frontier+1, frontier+2 before it counts.
            phase: Training phase ("climb", "harden", "backfill", or None).
                climb = forward curriculum with no refresh.
                harden = uniform random wave from [start_wave, max_wave].
                backfill = failure-weighted wave sampling.
                None = legacy behavior (uses promote_after / start_wave_weights).
            climb_sampling: How climb selects non-refresh start waves.
                "weighted" = bias across [frontier-3, frontier] toward frontier.
                "legacy" = always start at max(start_wave, frontier-3).
            observation_version: Observation schema version ("v4").
            record_reward_terms: When True, accumulates per-episode raw reward term
                totals and emits them in info on terminal steps (episode end).
        """
        super().__init__()

        self.start_wave = start_wave
        self.max_wave = max_wave
        self.render_mode = render_mode
        self.current_episode_start_wave = start_wave
        self.start_wave_weights = start_wave_weights
        self._promote_after = promote_after
        self._refresh_every = refresh_every
        self._min_waves_to_advance = min_waves_to_advance
        self._phase = phase
        if climb_sampling not in ("weighted", "legacy"):
            raise ValueError(
                f"Unsupported climb_sampling={climb_sampling!r}; expected 'weighted' or 'legacy'"
            )
        self._climb_sampling = climb_sampling
        self._observation_version = observation_version
        self._temporal = TemporalState()
        self._wave_stats: Dict[int, Dict[str, int]] = {}  # {wave: {"fails": N, "successes": N}}
        self._record_reward_terms = record_reward_terms
        self._episode_reward_terms: Dict[str, float] = {}
        self._episode_reward_term_steps: int = 0
        self._episode_cleared_waves: list[int] = []
        self._episode_mode: EpisodeMode = "full"
        self._opener_tick_limit = 50
        self._opener_min_health = 40
        self.set_episode_mode(episode_mode)
        self.set_opener_config(opener_tick_limit, opener_min_health)

        if self.start_wave_weights:
            # Normalize weights to sum to 1.0 just in case
            total = sum(self.start_wave_weights.values())
            self.start_wave_weights = {k: v / total for k, v in self.start_wave_weights.items()}
            self.start_waves = list(self.start_wave_weights.keys())
            self.start_probs = list(self.start_wave_weights.values())

        # Curriculum state — only used when promote_after > 0
        self._frontier_wave: int = start_wave
        self._consecutive_completions: int = 0
        self._total_episodes: int = 0
        self._current_is_refresh: bool = False
        self._current_max_wave_cleared: int = start_wave - 1
        self._mastery_mode: bool = False
        self._prestige_event: bool = False

        # Drill phase state
        self._drill_wave: int = start_wave
        self._drill_retries: int = 0
        self._drill_cycles: int = 0
        self._max_drill_retries: int = max_drill_retries

        # Sweep death retry state
        self._sweep_death_retries = sweep_death_retries
        self._death_retry_wave: Optional[int] = None
        self._death_retry_remaining: int = 0

        # Loadout configuration
        self._current_loadout_name: str = ""
        self._fixed_loadout: Optional[Loadout] = None
        self._loadout_ids: list[LoadoutId] = list(LOADOUTS.keys())
        self._loadout_weights: Optional[list[float]] = None
        if fixed_loadout is not None:
            self._fixed_loadout = LOADOUTS[LoadoutId[fixed_loadout]]
        elif loadout_weights is not None:
            ids = []
            weights = []
            for name, weight in loadout_weights.items():
                ids.append(LoadoutId[name])
                weights.append(weight)
            self._loadout_ids = ids
            self._loadout_weights = weights

        # Create simulator
        self.simulator = InfernoSimulator(start_wave, max_wave)
        self.simulator.initial_barrage_enabled = True

        # Create reward calculator
        self.reward_calculator = InfernoReward()

        # Define action and observation spaces
        self.action_space = spaces.MultiDiscrete(ACTION_HEAD_SIZES)
        self.observation_space = spaces.Box(
            low=get_observation_low(),
            high=1.0,
            shape=(get_observation_size(),),
            dtype=np.float32
        )
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment.
        
        Args:
            seed: Random seed
            options: Optional reset options
            
        Returns:
            (observation, info) tuple
        """
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)  # Spawn positions use Python random; seed so each env gets varied spawns

        # Reset simulator
        if options and "wave" in options:
            # Explicit override (testing / external control) — skip curriculum.
            start_wave = options["wave"]
            self._current_is_refresh = False
            self._current_max_wave_cleared = start_wave - 1
            self._total_episodes += 1
        elif self._phase == "drill":
            self._total_episodes += 1
            start_wave = self._drill_wave
            self._current_max_wave_cleared = start_wave - 1
        elif self._phase == "harden":
            # Uniform random wave from [start_wave, max_wave]
            self._total_episodes += 1
            start_wave = random.randint(self.start_wave, self.max_wave)
        elif self._phase == "backfill":
            # Failure-weighted wave sampling
            self._total_episodes += 1
            start_wave = self._sample_backfill_wave()
        elif self._phase == "sweep":
            # Sweep: failure-weighted sampling across all waves, no frontier
            self._total_episodes += 1
            if self._death_retry_remaining > 0 and self._death_retry_wave is not None:
                start_wave = self._death_retry_wave
                self._death_retry_remaining -= 1
            else:
                start_wave = self._sample_backfill_wave()
        elif self._phase == "climb" or self._promote_after > 0:
            # Forward curriculum: advance frontier after consecutive completions.
            # climb phase uses refresh_every=5 (revisit mastered wave every 5 episodes).
            self._total_episodes += 1
            refresh_every = 5 if self._phase == "climb" else self._refresh_every
            is_refresh = (
                refresh_every > 0
                and self._total_episodes % refresh_every == 0
                and self._frontier_wave > self.start_wave
            )
            start_wave = (
                random.randint(self.start_wave, self._frontier_wave - 1)
                if is_refresh
                else self._sample_climb_wave()
            )
            self._current_is_refresh = is_refresh
            self._current_max_wave_cleared = start_wave - 1
        elif self.start_wave_weights:
            # Sample start wave from distribution
            start_wave = random.choices(self.start_waves, weights=self.start_probs, k=1)[0]
        else:
            start_wave = self.start_wave

        selected_loadout = self._sample_loadout()
        self._current_loadout_name = selected_loadout.id.name
        self.simulator.set_loadout(selected_loadout)
        self.simulator.reset_to_wave(start_wave)
        self.current_episode_start_wave = start_wave
        if self._record_reward_terms:
            self._episode_reward_terms = {}
            self._episode_reward_term_steps = 0
        self._episode_cleared_waves = []
        if self._temporal is not None:
            self._temporal.reset()

        combat_entities = [
            entity for entity in self.simulator.state.entities
            if not entity.is_dead() and entity.entity_type != EntityTypes.NIBBLER
        ]
        tick_threat_cache = build_tick_threat_cache(
            self.simulator.state.player_x,
            self.simulator.state.player_y,
            self.simulator.state.pillar_alive,
            combat_entities,
            self.simulator.state.active_prayer,
        )

        # Build observation
        obs = build_observation(
            self.simulator.state,
            self.simulator.get_ticks_in_wave(),
            temporal=self._temporal,
            dead_mobs=self.simulator.dead_mobs,
            tick_threat_cache=tick_threat_cache,
        )

        info = {
            "wave": self.simulator.state.current_wave,
            "tick": self.simulator.state.current_tick,
            "ticks_in_wave": self.simulator.get_ticks_in_wave(),
            "start_wave": self.current_episode_start_wave,
            "frontier_wave": self._frontier_wave,
            "min_waves_required": self._min_waves_to_advance,
            "player_health": self.simulator.state.player_health,
            "episode_mode": self._episode_mode,
            "action_mask": self.action_masks(tick_threat_cache=tick_threat_cache),
            "loadout": selected_loadout.id.name,
        }

        return obs, info
    
    def step(self, action: int | np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step.
        
        Args:
            action: Action index (0-42)
            
        Returns:
            (observation, reward, terminated, truncated, info) tuple
        """
        legacy_action = decode_policy_action(action)

        # Execute action
        result = self.simulator.step(legacy_action)
        if result.wave_completed:
            if not self._episode_cleared_waves or self._episode_cleared_waves[-1] != result.wave_number:
                self._episode_cleared_waves.append(result.wave_number)

        # Track the highest wave cleared this episode.
        if (self._promote_after > 0 or self._phase == "drill") and result.wave_completed:
            self._current_max_wave_cleared = max(
                self._current_max_wave_cleared, result.wave_number
            )

        # Update temporal state for v3
        update_temporal_state(self._temporal, result.executed_action, result)

        tick_threat_cache = self.simulator._post_step_tick_threat_cache
        if tick_threat_cache is None:
            combat_entities = [
                entity for entity in self.simulator.state.entities
                if not entity.is_dead() and entity.entity_type != EntityTypes.NIBBLER
            ]
            tick_threat_cache = build_tick_threat_cache(
                self.simulator.state.player_x,
                self.simulator.state.player_y,
                self.simulator.state.pillar_alive,
                combat_entities,
                self.simulator.state.active_prayer,
            )

        # Build observation
        obs = build_observation(
            self.simulator.state,
            self.simulator.get_ticks_in_wave(),
            temporal=self._temporal,
            dead_mobs=self.simulator.dead_mobs,
            tick_threat_cache=tick_threat_cache,
        )

        # Calculate reward
        if self._record_reward_terms:
            breakdown = self.reward_calculator.calculate_with_breakdown(result)
            reward = breakdown.total
            self._episode_reward_term_steps += 1
            for name, value in breakdown.get_nonzero_components():
                key = normalize_reward_term_name(name)
                self._episode_reward_terms[key] = self._episode_reward_terms.get(key, 0.0) + value
        else:
            reward = self.reward_calculator.calculate(result)

        # Check termination
        terminated = result.is_terminal()
        truncated = False  # We use terminated for wave timeout
        opener_triggered = False
        opener_success = False
        opener_reason: str | None = None
        opener_bonus = 0.0
        magers_remaining, melees_remaining = self._count_alive_priority_entities()
        if self._episode_mode == "opener":
            (
                opener_triggered,
                opener_success,
                opener_reason,
                opener_bonus,
            ) = self._evaluate_opener_terminal(result, magers_remaining, melees_remaining)
            if opener_triggered:
                terminated = True
                reward += opener_bonus
                if self._record_reward_terms and opener_bonus != 0.0:
                    opener_term = {
                        "resolved": "Opener Resolved Success",
                        "tick_limit": "Opener Survive Window",
                    }.get(opener_reason, "Opener Failure")
                    self._episode_reward_terms[opener_term] = (
                        self._episode_reward_terms.get(opener_term, 0.0) + opener_bonus
                    )

        episode_done = terminated or truncated
        episode_success = False
        if episode_done:
            if self._episode_mode == "opener" and opener_triggered:
                episode_success = opener_success
            else:
                episode_success = not result.player_died and not result.wave_timeout

        wave_stat_updates: list[dict[str, int | bool]] = []
        if episode_done and self._phase in ("harden", "backfill", "sweep"):
            wave_stat_updates = self._build_wave_stat_updates(result, episode_success)
            for update in wave_stat_updates:
                wave = int(update["wave"])
                success = bool(update["success"])
                bucket = self._wave_stats.setdefault(wave, {"fails": 0, "successes": 0})
                if success:
                    bucket["successes"] += 1
                else:
                    bucket["fails"] += 1

        # Update curriculum on episode end (not refresher episodes).
        _promotion_cleared = False
        _promotion_failed = False
        if terminated and self._promote_after > 0 and not self._current_is_refresh:
            required = min(
                self._frontier_wave + self._min_waves_to_advance - 1,
                self.max_wave,
            )
            if self._current_max_wave_cleared >= required:
                _promotion_cleared = True
                self._consecutive_completions += 1
                if self._consecutive_completions >= self._promote_after:
                    self._consecutive_completions = 0
                    if self._frontier_wave < self.max_wave:
                        self._frontier_wave += 1
                    else:
                        # Prestige: reset frontier, require longer consecutive clears
                        self._frontier_wave = self.start_wave
                        self._min_waves_to_advance += 1
                        self._prestige_event = True
            else:
                _promotion_failed = True
                if self._consecutive_completions > 0:
                    pass
                self._consecutive_completions = 0

        # Drill phase: retry-on-failure advancement
        if terminated and self._phase == "drill":
            death_wave = self.simulator.state.current_wave
            if result.player_died or result.wave_timeout:
                if death_wave == self._drill_wave:
                    self._drill_retries += 1
                    if self._drill_retries >= self._max_drill_retries:
                        self._drill_wave += 1
                        self._drill_retries = 0
                        if self._drill_wave > self.max_wave:
                            self._drill_wave = self.start_wave
                            self._drill_cycles += 1
                else:
                    # Died on a later wave — drill that wave next
                    self._drill_wave = death_wave
                    self._drill_retries = 0
            elif result.inferno_complete or self._current_max_wave_cleared >= self.max_wave:
                # Full clear — loop back
                self._drill_wave = self.start_wave
                self._drill_retries = 0
                self._drill_cycles += 1
            else:
                # Cleared some waves but didn't reach max — advance to next uncleared
                self._drill_wave = self._current_max_wave_cleared + 1
                self._drill_retries = 0

        # Build info
        info = {
            "wave": self.simulator.state.current_wave,
            "tick": self.simulator.state.current_tick,
            "ticks_in_wave": self.simulator.get_ticks_in_wave(),
            "start_wave": self.current_episode_start_wave,
            "frontier_wave": self._frontier_wave,
            "min_waves_required": self._min_waves_to_advance,
            "kills": result.get_total_kills(),
            "damage_dealt": result.damage_dealt,
            "damage_taken": result.damage_taken,
            "player_health": self.simulator.state.player_health,
            "wave_completed": result.wave_completed,
            "player_died": result.player_died,
            "wave_timeout": result.wave_timeout,
            "inferno_complete": result.inferno_complete,
            "phase": self._phase,
            "episode_mode": self._episode_mode,
            "episode_done": episode_done,
            "episode_success": episode_success,
            "opener_success": opener_success,
            "opener_end_reason": opener_reason,
            "opener_magers_remaining": magers_remaining,
            "opener_melees_remaining": melees_remaining,
            "mastery_mode": self._mastery_mode,
            "wave_stats": self._wave_stats if self._phase in ("harden", "backfill", "sweep") else None,
            "promotion_cleared": _promotion_cleared,
            "promotion_failed": _promotion_failed,
            "promotion_streak": self._consecutive_completions,
            "max_wave_cleared": self._current_max_wave_cleared,
            "loadout": self._current_loadout_name,
            "action_mask": self.action_masks(tick_threat_cache=tick_threat_cache),
        }
        # Queue death retry for sweep mode
        if (episode_done and result.player_died
                and self._phase == "sweep" and self._sweep_death_retries > 0):
            self._death_retry_wave = self.simulator.state.current_wave
            self._death_retry_remaining = self._sweep_death_retries

        if episode_done and self._phase in ("harden", "backfill", "sweep"):
            info["wave_stat_updates"] = list(wave_stat_updates)
        if self._prestige_event:
            info["prestige_event"] = True
            info["new_min_waves"] = self._min_waves_to_advance
            self._prestige_event = False
        if self._phase == "drill":
            info["drill_wave"] = self._drill_wave
            info["drill_cycles"] = self._drill_cycles
        if terminated and self._record_reward_terms:
            info["episode_reward_terms"] = dict(self._episode_reward_terms)
            info["episode_reward_term_steps"] = self._episode_reward_term_steps

        return obs, reward, terminated, truncated, info

    def apply_prestige(self, min_waves_to_advance: int) -> None:
        """Global prestige sync: reset frontier and set min_waves_to_advance."""
        self._frontier_wave = self.start_wave
        self._min_waves_to_advance = min_waves_to_advance
        self._consecutive_completions = 0

    def sync_curriculum(
        self,
        frontier_wave: int,
        min_waves_to_advance: int | None = None,
    ) -> None:
        """Sync climb curriculum state from another worker.

        Frontier only ever moves forward here; prestige/reset remains a separate
        operation handled by ``apply_prestige``.
        """
        target_frontier = max(self.start_wave, min(int(frontier_wave), self.max_wave))
        target_min_waves = self._min_waves_to_advance
        if min_waves_to_advance is not None:
            target_min_waves = max(self._min_waves_to_advance, int(min_waves_to_advance))

        frontier_advanced = target_frontier > self._frontier_wave
        min_waves_changed = target_min_waves != self._min_waves_to_advance
        self._frontier_wave = max(self._frontier_wave, target_frontier)
        self._min_waves_to_advance = target_min_waves
        if frontier_advanced or min_waves_changed:
            self._consecutive_completions = 0

    def set_phase(self, phase: Optional[str]) -> None:
        if phase not in (None, "climb", "harden", "backfill", "drill", "sweep"):
            raise ValueError(f"Unsupported phase={phase!r}")
        self._phase = phase
        if phase == "drill":
            self._drill_wave = self.start_wave
            self._drill_retries = 0

    def set_episode_mode(self, mode: EpisodeMode) -> None:
        if mode not in ("full", "opener"):
            raise ValueError(f"Unsupported episode_mode={mode!r}")
        self._episode_mode = mode

    def set_opener_config(self, tick_limit: int, min_health: int) -> None:
        if tick_limit < 1:
            raise ValueError(f"opener_tick_limit must be >= 1, got {tick_limit}")
        if not 1 <= min_health <= 98:
            raise ValueError(f"opener_min_health must be in [1, 98], got {min_health}")
        self._opener_tick_limit = tick_limit
        self._opener_min_health = min_health

    def set_wave_stats(self, stats: Dict[int, Dict[str, int]]) -> None:
        self._wave_stats = {
            int(wave): {
                "fails": int(values.get("fails", 0)),
                "successes": int(values.get("successes", 0)),
            }
            for wave, values in stats.items()
        }

    def reset_wave_stats(self) -> None:
        self._wave_stats = {}

    def set_reward_config(self, config: RewardConfig) -> None:
        self.reward_calculator.set_config(config)

    def action_masks(
        self,
        movement_table=None,
        tick_threat_cache: TickThreatCache | None = None,
    ) -> np.ndarray:
        """
        Get action mask for current state.
        
        Returns the flattened MultiDiscrete mask expected by the custom PPO
        actor. Head order matches ACTION_HEAD_SIZES.
        """
        return get_policy_action_mask(
            self.simulator.state,
            movement_table=movement_table,
            tick_threat_cache=tick_threat_cache,
        )
    
    def render(self):
        """Render the environment (optional)."""
        if self.render_mode == "human":
            # Print current state
            state = self.simulator.state
            print(f"Wave {state.current_wave} | Tick {state.current_tick} | "
                  f"HP {state.player_health} | "
                  f"Enemies {len([e for e in state.entities if not e.is_dead()])}")
    
    def close(self):
        """Clean up resources."""
        pass

    @property
    def wave_stats(self) -> Dict[int, Dict[str, int]]:
        """Per-wave failure/success stats (populated in harden/backfill phases)."""
        return self._wave_stats

    # Mastery mode: difficulty-weighted wave distribution.
    # W63 is the hardest composition (3 magers + blob + melee + ranger + nibblers).
    # Weights focus training time on the bottleneck waves.
    _MASTERY_WEIGHTS = {
        49: 1, 50: 1, 51: 1, 52: 1, 53: 1, 54: 1,
        55: 2, 56: 2, 57: 2, 58: 2, 59: 3, 60: 3,
        61: 4, 62: 5, 63: 5, 64: 4, 65: 3, 66: 2,
    }

    def _sample_loadout(self) -> Loadout:
        """Sample a loadout for this episode."""
        if self._fixed_loadout is not None:
            return self._fixed_loadout
        if self._loadout_weights is not None:
            lid = random.choices(self._loadout_ids, weights=self._loadout_weights, k=1)[0]
            return LOADOUTS[lid]
        lid = random.choice(self._loadout_ids)
        return LOADOUTS[lid]

    def _sample_climb_wave(self) -> int:
        """Sample from the recent frontier band, biased toward the bottleneck wave."""
        if self._climb_sampling == "legacy":
            return max(self.start_wave, self._frontier_wave - 3)

        low = max(self.start_wave, self._frontier_wave - 3)
        waves = list(range(low, self._frontier_wave + 1))
        weights_by_offset = {
            0: 6,  # frontier
            1: 3,
            2: 2,
            3: 1,
        }
        weights = [weights_by_offset.get(self._frontier_wave - wave, 1) for wave in waves]
        return random.choices(waves, weights=weights, k=1)[0]

    def _sample_mastery_wave(self) -> int:
        """Sample from difficulty-weighted distribution for post-climb training."""
        waves = [w for w in self._MASTERY_WEIGHTS if w <= self.max_wave]
        weights = [self._MASTERY_WEIGHTS[w] for w in waves]
        return random.choices(waves, weights=weights, k=1)[0]

    def _sample_backfill_wave(self) -> int:
        """Sample a wave weighted by failure rate, with 0.02 floor."""
        if self._total_episodes <= self.BACKFILL_WARMUP_EPISODES:
            # Warmup is per env. Using global stat count here would flip into
            # weighted sampling almost immediately once a handful of episodes finish.
            return random.randint(self.start_wave, self.max_wave)

        waves = list(range(self.start_wave, self.max_wave + 1))
        weights = []
        for w in waves:
            stats = self._wave_stats.get(w, {"fails": 0, "successes": 0})
            total = stats["fails"] + stats["successes"]
            fail_rate = stats["fails"] / total if total > 0 else 0.5
            weights.append(max(fail_rate, 0.02))

        return random.choices(waves, weights=weights, k=1)[0]

    def _count_alive_priority_entities(self) -> tuple[int, int]:
        magers = 0
        melees = 0
        for entity in self.simulator.state.get_alive_entities():
            if entity.entity_type == EntityTypes.MAGER:
                magers += 1
            elif entity.entity_type == EntityTypes.MELEE:
                melees += 1
        return magers, melees

    def _evaluate_opener_terminal(
        self,
        result,
        magers_remaining: int,
        melees_remaining: int,
    ) -> tuple[bool, bool, str | None, float]:
        if result.player_died:
            return True, False, "death", self.OPENER_FAILURE_PENALTY
        if result.wave_timeout:
            return True, False, "timeout", self.OPENER_FAILURE_PENALTY

        resolved = magers_remaining == 0 and melees_remaining == 0
        survived_window = self.simulator.get_ticks_in_wave() >= self._opener_tick_limit
        if not resolved and not survived_window:
            return False, False, None, 0.0

        hp_ok = self.simulator.state.player_health > self._opener_min_health
        if resolved and hp_ok:
            return True, True, "resolved", self.OPENER_RESOLVED_SUCCESS_BONUS
        if survived_window and hp_ok:
            return True, True, "tick_limit", self.OPENER_SURVIVE_WINDOW_BONUS
        return True, False, "low_hp", self.OPENER_FAILURE_PENALTY

    def _build_wave_stat_updates(
        self,
        result,
        episode_success: bool,
    ) -> list[dict[str, int | bool]]:
        if self._episode_mode == "opener":
            return [{"wave": self.current_episode_start_wave, "success": episode_success}]

        updates = [{"wave": wave, "success": True} for wave in self._episode_cleared_waves]
        if not episode_success:
            terminal_wave = self.simulator.state.current_wave
            updates.append({"wave": terminal_wave, "success": False})
        return updates


def make_inferno_env(
    start_wave: int = 35,
    max_wave: int = 49,
    seed: Optional[int] = None,
    start_wave_weights: Optional[Dict[int, float]] = None,
) -> InfernoEnv:
    """
    Create an Inferno environment with common defaults.
    
    Args:
        start_wave: Wave to start from (default 35 for curriculum)
        max_wave: Maximum wave (default 49 for early curriculum)
        seed: Random seed
        start_wave_weights: Optional dict of {wave: weight} for mixed starting waves
        
    Returns:
        Configured InfernoEnv
    """
    env = InfernoEnv(
        start_wave=start_wave, 
        max_wave=max_wave, 
        start_wave_weights=start_wave_weights
    )
    if seed is not None:
        env.reset(seed=seed)
    return env
