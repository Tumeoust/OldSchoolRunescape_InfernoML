# Inferno RL Training System — Architecture

This document is the structural/technical reference for the Python RL training system for OSRS
Inferno: the module map, key classes, and execution order. For the *narrative* (why the problem is
hard, how the observation/reward/curriculum designs were arrived at, and the V9→V53 version history),
see the top-level [`README.md`](../../README.md).

> **Educational / research project.** Everything runs against an offline, self-contained simulator.
> There is no connection to the live game — no screen reading, no input automation, no client
> integration. The interesting part is the reinforcement-learning problem, not playing the real game.

## Why Python?

The whole stack — simulator, environment, and a custom PyTorch PPO — is pure Python:

1. **PyTorch ecosystem**: mature autograd/AMP/TorchScript, easy custom recurrent policies and
   factored action heads.
2. **Tooling**: TensorBoard, quick eval/replay CLIs, and Pygame visualizers for inspecting behavior.
3. **Fast iteration**: reward functions, observation layouts, and curricula are plain Python and
   change without a build step; the hot simulator paths drop to Cython (`.pyx`) for speed.

## Directory Structure

```
tools/inferno_rl/
├── train_gpu.py             # ★ Main training entry point (custom PyTorch PPO)
├── eval.py                  # Headless evaluation / clear-rate benchmarking
├── analyze_trace.py         # Replay recorded traces through the sim, diff LOS/attack
├── death_analysis.py        # Bucket failures by wave / loadout / cause
├── adaptive_curriculum.py   # Adaptive regime controller (champion tracking, rollback)
├── rollout_sampler.py       # Async experience collector → PPO Buffer
├── callback.py              # Callback base class (concrete callbacks live in train_gpu.py)
├── critic.py                # Value-function query helpers
├── inference_state.py       # StatefulPolicyRunner (persists LSTM state across single steps)
├── setup_cython.py          # Build the .pyx backends in-place
├── verify_cython_backend.py # Report compiled (.pyd/.so) vs pure-Python fallback
├── requirements.txt
│
├── simulator/               # Tick-accurate headless fight engine
│   ├── simulator.py         # InfernoSimulator: composes mixins, owns the step loop
│   ├── npc_movement.py      # NpcMovementMixin: dumb pathfinding, meleer dig
│   ├── npc_combat.py        # NpcCombatMixin: NPC attacks, blob scan/split, mager resurrection
│   ├── player_actions.py    # PlayerActionsMixin: move/attack/switch, attack-drag, barrage AoE
│   ├── prayer_prediction.py # PrayerPredictionMixin: auto-pray one tick ahead of the forecast
│   ├── step_result.py       # ResultBuilderMixin, StepResult, PlayerDamageEvent
│   ├── state.py             # SimulatorState (central mutable state), wave spawning
│   ├── entity.py            # AttackStyle, InfernoEntityType (frozen), PlacedEntity (mutable)
│   ├── combat.py            # OSRS accuracy/max-hit formulas + precomputed combat tables
│   ├── npc_stats.py         # NpcCombatStats tables (OSRS-wiki values)
│   ├── equipment.py         # Loadouts, gear presets, player levels, prayer multipliers
│   ├── geometry.py / .pyx   # Grid/world math, Chebyshev distance, fixed-point Bresenham LOS
│   ├── pathfinding.py / .pyx# OSRSPathfinding + NpcCollisionResolver
│   ├── forecast.py          # Threat/neighborhood-safety forecasting engine
│   ├── forecast_fast.pyx    #   └─ Cython backend for the hot forecast functions
│   ├── priority.py          # combat_entity_sort_key — shared target ordering
│   ├── exact_targeting.py   # 14 exact-target slot resolution (get_exact_target_slots)
│   └── movement_actions.py  # Movement direction/distance tables (8 dirs × 4 distances)
│
├── training/                # Gymnasium env, observation, actions, rewards
│   ├── env.py               # InfernoEnv: builds obs v4 + factored action masks
│   ├── observation_v4.py    # ★ Current 602-dim observation
│   ├── observation.py       # Thin facade → build_observation_v4
│   ├── observation_common.py# Shared block-size constants + temporal state
│   ├── observation_v3.py    # Superseded earlier observation
│   ├── actions.py           # Factored MultiDiscrete action space + per-head masking
│   ├── rewards.py           # InfernoReward, RewardConfig, reward-term breakdown
│   ├── schedules.py         # Linear/Piecewise schedules for reward-shaping decay
│   └── train.py             # ⚠ Legacy SB3 MaskablePPO trainer — superseded by train_gpu.py
│
├── ppo/                     # Custom PyTorch PPO (no external RL framework)
│   ├── ppo.py               # PPO: recurrent rollout update loop, checkpointing
│   ├── policy.py            # Policy / Actor / Critic: flat_lstm_residual, factored heads
│   ├── buffer.py            # Rollout buffer, GAE, recurrent sequence batching
│   ├── running_mean_std.py  # TensorRunningMeanStd (observation + reward normalization)
│   └── mlp_helper.py        # MLP builder, orthogonal init
│
├── async_env/               # Vectorized environment layer
│   ├── subprocess_vec_env.py# SubprocVecEnv: N worker processes, shared-memory transport
│   ├── local_vec_env.py     # LocalVecEnv: single-process asyncio alternative
│   ├── async_inferno_env.py # AsyncInfernoEnv adapter over InfernoEnv
│   └── async_io_env.py      # Sync↔async bridge base class
│
├── cli/                     # Command-line analysis tools
│   ├── eval_model.py        #   evaluate a checkpoint
│   ├── replay_episode.py    #   replay a recorded episode tick-by-tick
│   ├── analyze_play.py      #   analyze a play session
│   ├── query_value.py       #   query the critic's value estimates
│   ├── snapshot_wave.py     #   snapshot a wave state
│   ├── run_hybrid.py        #   heuristic + policy hybrid runner
│   └── state_decoder.py     #   decode an observation vector back to state
│
├── visualizer/              # Pygame tools
│   ├── visualizer.py        # Base grid/entity/LOS renderer
│   ├── play_human.py        # Play the simulator manually
│   ├── review_deaths.py     # Step through recorded deaths
│   ├── debug_sandbox.py     # Interactive sandbox (place entities, step ticks)
│   └── run_visual.py        # ⚠ Watch a legacy SB3 .zip model
│
├── pretrain/                # Behavior-cloning warmstart
│   ├── collect_from_model.py# collect BC demonstrations from a policy
│   ├── pretrain_bc.py       # supervised pretraining
│   └── transform_bc_obs.py  # migrate BC data across observation versions
│
├── tuning/                  # eval_report.py, tb_summary.py, export_reward_terms_csv.py
├── tests/                   # ~20 unit / parity / optimization test modules
└── docs/                    # Design docs + per-version training record (V*_TB_TRACKING.md)
```

## Execution Flow

### Training (`train_gpu.py`)

Entry point: `python -m tools.inferno_rl.train_gpu ...`. `main()` parses CLI flags into `train()`,
which drives the loop below.

```
python -m tools.inferno_rl.train_gpu   →   train_gpu.py: train()
        │
        ▼
  SubprocVecEnv            N worker processes, each running an InfernoEnv.
  (async_env/)             Obs / reward / done / masks move over shared memory.
        │
        ▼
  RolloutSampler.collect() Async step/predict loop for n_steps × n_envs.
  (rollout_sampler.py)     Per env-step: get_policy_action_mask → PPO.predict
        │                  → step_async/poll_step → Buffer. LSTM hidden state is
        │                  threaded per env and zeroed on episode end.
        ▼
  Buffer.finalize()        Reward normalization → GAE(γ, λ) → returns.
  (ppo/buffer.py)
        │
        ▼
  PPO.learn(buffer)        Recurrent sequence batches (lstm_seq_len with burn-in warmup),
  (ppo/ppo.py)             clipped surrogate + entropy + value loss, per-batch advantage
        │                  normalization, approx-KL early stop, grad clipping.
        │                  After each rollout: update observation RunningMeanStd.
        ▼
  Policy                   flat_lstm_residual: LayerNorm → Linear+ReLU encoder → LSTM,
  (ppo/policy.py)          then concat the raw observation (residual skip) → separate
                           512×512 actor (4 factored heads) and 512×512 critic MLPs.
        │
        └── checkpoint + TensorBoard + TrainingChampionTracker → collect next rollout
```

Collection and learning are sequential per rollout (not pipelined): each iteration calls
`ppo.learn(buffer)` then samples the next `buffer`. The entropy coefficient is linearly annealed
across training; a first-rollout warmup skips updates so the observation normalizer can prime.

Two curriculum modes select which waves each env plays: `static` (fixed `--phase`, e.g. `sweep`) and
`adaptive_v36` (an `AdaptiveController` that sequences regimes and rolls back on regression).

### Simulator tick order (`simulator.py: step()`)

`InfernoSimulator.step(action)` executes one 600 ms game tick in this order:

```
 1. _capture_pre_step_state()          snapshot state for reward attribution / forecasting
 2. _apply_*_heuristic(action)         opener barrage heuristic (early ticks only)
 3. _process_auto_prayer(action)       predict next-tick position, queue the right prayer
 4. state.increment_tick()
 5. state.process_action_queue()       queued prayer becomes active (1-tick delay)
 6. _execute_action(action)            move / attack / weapon switch → action_valid
 7. (record attack-on-cooldown info)
 8. _decrement_entity_timers()
 9. _handle_attack_drag_if_needed()    player auto-walks toward LOS — BEFORE NPC movement
10. _process_npc_movement()            dumb pathfinding, meleer dig
11. _process_npc_attacks()             prayer protection, blob scan, mager resurrection
12. _process_player_attack()           damage, kill tracking, barrage AoE
13. _process_pillar_collapses()        nibbler-destroyed pillars + splash damage
14. _process_dead_entities()           remove dead, spawn blob splits
15. _process_wave_progression()
16. _build_step_result(...)            → StepResult (damage, kills, terminal flags)
```

Wave timeout is `MAX_TICKS_PER_WAVE = 800` (`step_result.py`).

## Core Components

### 1. Simulator (`simulator/`)

`InfernoSimulator` is an orchestrator that composes five mixins by inheritance and owns the step
loop, reset/wave setup, timers, pillar collapses, and wave progression:

| Mixin | Module | Responsibility |
|-------|--------|----------------|
| `NpcMovementMixin`      | `npc_movement.py`      | NPC dumb pathfinding, nibbler pillar-seeking, meleer dig |
| `NpcCombatMixin`        | `npc_combat.py`        | NPC attacks, blob scan/split, mager attack + resurrection, dead-entity handling |
| `PlayerActionsMixin`    | `player_actions.py`    | Player move/attack/switch, attack-drag, barrage AoE, LOS helpers |
| `PrayerPredictionMixin` | `prayer_prediction.py` | Predict next-tick player position and queue the correct protection prayer |
| `ResultBuilderMixin`    | `step_result.py`       | Assemble the per-tick `StepResult` |

Supporting modules:

- **`state.py`** — `SimulatorState`: player position/health/weapon/cooldown/target, entity list,
  pillar HP/alive, wave/tick counters, prayer queue. `spawn_wave_entities()` places a wave.
- **`entity.py`** — `AttackStyle`, frozen `InfernoEntityType` (per-type stats), the `EntityTypes`
  table, and mutable `PlacedEntity` instances.
- **`combat.py`** — OSRS accuracy/max-hit formulas as pure functions plus precomputed `CombatTables`
  built once at import (tbow modifiers, hit-chance, max-hit rolls). `roll_player_damage`,
  `roll_npc_damage`. Stat values come from `npc_stats.py` and `equipment.py`.
- **`equipment.py`** — `Loadout`, `GearPreset`, `LoadoutId`, `PlayerLevels`, `PrayerMultipliers`;
  `LOADOUTS` / `DEFAULT_LOADOUT` (budget crossbow → max tbow).
- **`geometry.py`** — `SimulatorGeometry` (grid/world conversion, Chebyshev distance, pillar
  collision) and `InfernoLineOfSight` (fixed-point Bresenham). `PILLARS`, NE-pillar-zone helpers.
- **`pathfinding.py`** — `OSRSPathfinding` (direct "dumb" movement, cardinal fallbacks, diagonal
  clipping) and `NpcCollisionResolver`.
- **`forecast.py`** — the neighborhood/threat forecasting engine that powers the observation's
  one-tick-ahead safety features (`build_tick_threat_cache`, `forecast_neighborhood_safety`,
  `forecast_threat_styles`).
- **`priority.py`** — `combat_entity_sort_key`, the single stable target ordering shared by the
  simulator and the observation (LOS → imminent → base priority → distance → id).
- **`exact_targeting.py`** — resolves the `MAX_TARGET_SLOTS = 14` exact-target slots used by both
  the attack action head and the observation.

#### Cython backends

Three modules have compiled backends that shadow the `.py` on import for a 3–5× speedup:
`geometry.pyx`, `pathfinding.pyx`, and `forecast_fast.pyx`. Build them in-place with
`setup_cython.py`; `verify_cython_backend.py` reports whether each is running compiled or falling
back to pure Python (`geometry`/`pathfinding` are expected compiled; `forecast_fast` may be absent).

### 2. Observation (`training/observation_v4.py`)

The current observation is **v4**, a flat **602-dim** float32 vector (`observation.py` is a thin
facade that always delegates to `build_observation_v4`). Layout:

| Block                 | Size | Contents |
|-----------------------|------|----------|
| Global                | 51   | Player state, weapon/prayer one-hots, pillar HP, wave, kill/dead-pool counts, blob-scan counts, nibbler-per-pillar counts, resurrection hazard |
| Neighborhood forecast | 108  | 9 tiles × 12 features: one-tick-ahead prediction per adjacent tile (and stay) — LOS count/delta, imminent attacks by style, auto-prayer resolution, blob-scan triggers, multi-step LOS lookahead |
| Threat horizon        | 9    | Incoming attack counts by style, forecast 3 ticks ahead |
| Temporal              | 7    | Rolling damage sums, ticks since last attack/engagement, previous-action flags |
| Exact-target slots    | 420  | 14 slots × 30 features (21 core + 9-wide type one-hot), sorted by threat priority |
| Loadout               | 7    | Weapon speed/range/bonuses, max HP for the current gear set |

The 14 exact-target slots share their ordering with the attack action head, so "attack slot *k*"
always refers to the entity described in slot *k*. Full field-by-field reference:
[`docs/OBSERVATION_SPACE.md`](docs/OBSERVATION_SPACE.md). Block-size constants live in
`observation_common.py`; the actor consumes the public observation and the critic the full one
(currently the same 602 dims — the privileged-critic block is size 0).

### 3. Action space (`training/actions.py`)

The policy outputs a **factored MultiDiscrete** action with four autoregressive heads
(`ACTION_HEAD_SIZES = [5, 32, 14, 4]`):

| Head   | Size | Meaning |
|--------|------|---------|
| mode   | 5    | stay / move / attack / switch / no-op |
| move   | 32   | 8 directions × 4 distances |
| attack | 14   | one per exact-target slot |
| switch | 4    | BoFa / blowpipe / ice barrage / blood barrage |

Sampled actions are decoded (`decode_policy_action`) into the simulator's legacy 52-action
interface. **Per-head masking** (`get_policy_action_mask` → a 55-wide bool mask) removes invalid
choices before sampling: movement bits that don't change tile, empty target slots, weapons not in the
current loadout, and — via `POLICY_ACTION_DEPENDENCIES` — sub-heads whose mode isn't selected. Prayer
is not an action: the environment auto-prays against the forecast, and the observation reports how
that auto-prayer will resolve.

### 4. Rewards (`training/rewards.py`, `training/schedules.py`)

`InfernoReward` reads magnitudes from a frozen `RewardConfig`; the environment uses the default
config. Active default terms (≈11):

- Damage taken (−0.05/HP) and dealt (+0.003/HP)
- Mager priority + early-mager-kill shaping
- Wave complete (+3 plus a wave-scaled progress bonus) and end-of-wave HP bonus
- Inferno complete (+15), death (−20), wave timeout (−15)
- One-time pillar-death penalties (−7.5 non-NE, −15 NE)
- Escalating stall penalty (after a 15-tick idle window)
- LOS-separation bonus (+0.01 × blocked fraction), high-HP blood-barrage and weapon-switch penalties

Many earlier positioning/priority terms are zeroed in the default config and only re-enabled through
`build_v44_reward_config`, which evaluates named `schedules.py` schedules at the current
`trained_rollouts` to **decay** shaping over training (so bonuses can't be farmed). `RewardBreakdown`
records per-term contributions for TensorBoard logging. Rationale and history:
[`docs/TRAINING_FINDINGS.md`](docs/TRAINING_FINDINGS.md).

### 5. Environment (`training/env.py`)

`InfernoEnv(gym.Env)` (Gymnasium) wraps `InfernoSimulator`:

- `observation_space = Box(-1, 1, shape=(602,))`, `action_space = MultiDiscrete([5, 32, 14, 4])`.
- `reset()` / `step()` build the v4 observation and place the 55-wide factored mask in
  `info["action_mask"]`.
- Decodes the factored action, steps the simulator, and computes reward via `InfernoReward`.
- `terminated = StepResult.is_terminal()`; `truncated` is always False (wave timeout is terminal).
- Owns per-env **wave selection** for the curriculum (see §7).

### 6. PPO stack (`ppo/`)

A self-contained PyTorch PPO — no external RL framework.

- **`PPO` (`ppo/ppo.py`)** — holds the `Policy`, optimizer (Adam), AMP scaler, and a `Meta` record
  (observation `RunningMeanStd`, trained steps/rollouts). `predict()` runs masked inference threading
  LSTM state; `learn(buffer)` runs the clipped-surrogate update over recurrent sequence batches with
  burn-in, entropy + value losses, and approx-KL early stopping. Supports checkpoint save/load,
  including `load_with_resize` to migrate weights across observation/action dim changes.
- **`Policy` / `Actor` / `Critic` (`ppo/policy.py`)** — the `flat_lstm_residual` architecture
  (default) normalizes the observation, encodes it to the LSTM width, runs a segmented LSTM (hidden
  state zeroed at episode boundaries), then **concatenates the raw observation** with the LSTM output
  before the actor and critic MLPs. `Actor` emits one linear head per `ACTION_HEAD_SIZES` entry,
  applies per-head + dependency masks, and conditions later heads on earlier ones (autoregressive).
  Other architectures (`flat`, `entity_pool_lstm`) exist but training permits only `flat` and
  `flat_lstm_residual`.
- **`Buffer` (`ppo/buffer.py`)** — stores rollout tensors; `finalize()` runs reward normalization
  then GAE. `generate_sequence_batches()` yields non-overlapping `seq_len` windows with optional
  burn-in warmup observations for the recurrent update.
- **`TensorRunningMeanStd` (`ppo/running_mean_std.py`)** — Welford-style normalizer used for
  observations (updated per rollout, applied in predict + learn) and, separately, for rewards
  (applied in the buffer before GAE).

### 7. Vectorized envs & curriculum

- **`async_env/`** — `SubprocVecEnv` spawns N worker processes (default 48), each running an
  `InfernoEnv`, moving bulk data over shared memory and only completion signals over pipes.
  `LocalVecEnv` is a single-process asyncio alternative. `RolloutSampler` (`rollout_sampler.py`)
  drives either to fill the PPO `Buffer` — it is the experience collector, *not* the wave curriculum.
- **Wave curriculum** — per-env wave selection lives in `env.py: reset()`, keyed by `_phase`:
  `climb` (frontier advance + prestige), `harden` (uniform over the range), `backfill`
  (failure-weighted sampling), `sweep` (failure-weighted + persistent death-retry), and `drill`
  (retry-on-failure). Per-wave `{fails, successes}` feed the failure weights.
- **`AdaptiveController` (`adaptive_curriculum.py`)** — the `adaptive_v36` mode: sequences regimes
  (`harden_full → backfill_full → backfill_opener`), and on each eval window compares the new
  `EvalSummary` to the tracked champion (`full_clear_rate`, tie-broken by death/timeout/mean-wave) to
  promote, roll back, or advance the regime.

### 8. Tooling

- **`eval.py`** — headless evaluation (drives the simulator directly via a `StatefulPolicyRunner`),
  per-loadout sweeps, multiprocess eval, and model comparison; also the source of clear-rate
  benchmarks. `inference_state.py` persists LSTM state across single-env steps; `critic.py` queries
  value estimates.
- **`death_analysis.py`** — buckets failures by wave/loadout/cause; drove most observation/reward
  changes.
- **`analyze_trace.py`** — replays recorded RuneLite traces through the simulator and diffs LOS /
  can-attack decisions tick-by-tick (see [`docs/TRACE_VALIDATION.md`](docs/TRACE_VALIDATION.md)).
- **`cli/`** — replay episodes, evaluate checkpoints, query the value function, decode observations.
- **`visualizer/`** — Pygame renderer, manual play, death review, and an interactive sandbox.
- **`pretrain/`** — behavior-cloning warmstart used to bootstrap early versions.
- **`tuning/`** — eval reports and TensorBoard metric summaries.

## OSRS Tick Model

The game runs on 600 ms ticks. Actions queued on tick N take effect on tick N+1:

```
TICK N:
├── Player observes enemy state from TICK N-1
├── Player queues action (attack / move) and the env queues a prayer
└── Results resolve on TICK N+1

TICK N+1:
├── Queued prayer becomes active
├── Player movement, then NPC movement
├── Attacks resolve (prayer protection applied)
└── Deaths processed
```

**Prayer timing is critical**: a mager winding up on tick N must be prayed against by tick N to block
the hit on tick N+1. The environment auto-prays one tick ahead based on the forecast; the policy's job
is to *position* so a single prayer is enough (see the neighborhood-forecast observation block).

## Key Mechanics

- **Attack drag** — clicking an out-of-reach NPC auto-walks the player toward the nearest tile with
  LOS; resolved *before* NPC movement each tick, and stopped early once LOS is achieved.
- **Blob scan** — blobs have no fixed style; on gaining LOS they scan (MAGIC or RANGED), wind up over
  several ticks, and can then fire even through temporary LOS loss.
- **Meleer dig** — a meleer blocked from reaching the player digs (a fixed multi-tick sequence),
  teleports adjacent, and attacks after a short post-dig delay.
- **Nibbler group behavior** — all nibblers in a wave target the same random pillar; when that pillar
  dies, those nibblers die with it.
- **Mager resurrection** — magers can resurrect a dead meleer/ranger, a mid-wave hazard the
  observation surfaces.

## Running the System

```bash
cd tools/inferno_rl
pip install -r requirements.txt

# (Optional) build the Cython backends for a 3-5x simulator speedup
python setup_cython.py build_ext --inplace
python verify_cython_backend.py            # report compiled vs pure-python

# Train (custom PyTorch PPO). Run as a module from the repo root:
python -m tools.inferno_rl.train_gpu \
    --device cuda --n-envs 48 \
    --observation-version v4 --policy-arch flat_lstm_residual --lstm-hidden-size 256 \
    --curriculum-mode static --phase sweep --start-wave 1 --max-wave 66 \
    --normalize-obs --normalize-reward --actor-sizes 512,512 --critic-sizes 512,512 \
    --save-dir models/run --log-dir logs/run

# Evaluate a checkpoint (clear-rate benchmark)
python -m tools.inferno_rl.eval --load models/run/<checkpoint>.pt

# Watch / debug
python -m tools.inferno_rl.visualizer.debug_sandbox --wave 35
```

> **Legacy note.** `training/train.py` (with `training/env.py`'s SB3 masking path and
> `visualizer/run_visual.py`) is the earlier Stable-Baselines3 `MaskablePPO` baseline. It is
> superseded by the custom PPO in `train_gpu.py` / `ppo/` and kept only for reference; the
> `stable-baselines3` / `sb3-contrib` entries in `requirements.txt` exist for that legacy path.

## Documentation

| Topic                        | Location                                     |
|------------------------------|----------------------------------------------|
| Project narrative & history  | [`README.md`](../../README.md)               |
| Observation space (full)     | [`docs/OBSERVATION_SPACE.md`](docs/OBSERVATION_SPACE.md) |
| Training findings            | [`docs/TRAINING_FINDINGS.md`](docs/TRAINING_FINDINGS.md) |
| Simulator-vs-reality checks  | [`docs/TRACE_VALIDATION.md`](docs/TRACE_VALIDATION.md)   |
| Per-version training record  | `docs/V*_TB_TRACKING.md`                     |
| Inferno tactics knowledge    | `docs/inferno_knowledge/`                    |
