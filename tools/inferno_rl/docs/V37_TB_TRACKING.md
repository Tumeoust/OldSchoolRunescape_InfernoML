# V37 TB Tracking

## Goal

Switch to **observation V3.1** (295 dims, +26 from V3's 269) and **remove the between-wave heuristic override** so the model controls its
own inter-wave behavior (repositioning, weapon switching, prayer setup during the 9-tick grace period).

V36 established that:

- **adaptive curriculum** works as a training controller
- the `S2-300` checkpoint (V35) remained the champion through V36
- the LSTM entity-pool architecture (V2 obs) could not reliably improve on the champion
- the V36 regime cycle (harden/backfill/opener) did not produce a new champion after a full cycle

V37 changes the observation and action space (inter-wave freedom) rather than the training regime. The hypothesis is that hidden state gaps
and forced inter-wave behavior are limiting the model's ability to learn the full wave lifecycle.

## Base Checkpoint

- Base checkpoint: **V36 champion** (originally V35 `S2-300`)
- Path: **`models/V36_adaptive/` champion checkpoint**
- Note: V37 changes observation size (269→295) and policy arch (flat MLP), so this is a **fresh start**, not a warmstart from V36

## What Changed (V36 -> V37)

### 1. Observation V3.1 (+26 Features)

V3.1 extends the V3 global block (indices 0-51 unchanged) with 26 new features at indices 52-77, shifting safety/temporal/entity blocks by

26.

**New global features (indices 52-77):**

| Index | Feature                                | Source                                            | Normalization       |
|-------|----------------------------------------|---------------------------------------------------|---------------------|
| 52    | blob_scanned_magic_count               | blobs with `scanned_prayer == "MAGIC"`            | / 2.0, clip [0,1]   |
| 53    | blob_scanned_ranged_count              | blobs with `scanned_prayer == "RANGED"`           | / 2.0, clip [0,1]   |
| 54    | blob_scanned_imminent_count            | scanned blobs with `attack_delay <= 0`            | / 2.0, clip [0,1]   |
| 55    | wave_spawn_timer_norm                  | `wave_complete_timer / 9` when >= 0               | [0,1]               |
| 56-59 | queued_prayer_one_hot                  | `state.queued_prayer` (none/magic/missiles/melee) | binary              |
| 60    | nibblers_targeting_nw                  | nibblers with `target_pillar_index == 0`          | / 10.0, clip [0,1]  |
| 61    | nibblers_targeting_ne                  | nibblers with `target_pillar_index == 1`          | / 10.0, clip [0,1]  |
| 62    | nibblers_targeting_s                   | nibblers with `target_pillar_index == 2`          | / 10.0, clip [0,1]  |
| 63    | dead_pool_bat                          | dead BAT count                                    | / 5.0, clip [0,1]   |
| 64    | dead_pool_blob                         | dead BLOB count                                   | / 5.0, clip [0,1]   |
| 65    | dead_pool_melee                        | dead MELEE count                                  | / 5.0, clip [0,1]   |
| 66    | dead_pool_ranger                       | dead RANGER count                                 | / 5.0, clip [0,1]   |
| 67    | dead_pool_mager                        | dead MAGER count                                  | / 5.0, clip [0,1]   |
| 68    | resurrection_hazard                    | `alive_magers * len(dead_mobs)`                   | / 10.0, clip [0,1]  |
| 69-71 | p1_player_los, p1_npc_los, p1_distance | priority target 1 affordances                     | binary / normalized |
| 72-74 | p2_player_los, p2_npc_los, p2_distance | priority target 2 affordances                     | binary / normalized |
| 75-77 | p3_player_los, p3_npc_los, p3_distance | priority target 3 affordances                     | binary / normalized |

**Why these features matter:**

- **Blob scan**: the model currently cannot see whether a blob has already scanned, which determines whether the next hit is magic or
  ranged — critical for prayer switching
- **Wave spawn timer**: tells the model exactly how many ticks remain before the next wave spawns
- **Queued prayer**: the model cannot see its own pending prayer, creating a hidden-state gap where it may redundantly queue or fail to
  react
- **Nibbler pillar targets**: which pillar each nibbler is heading to — needed for barrage triage decisions
- **Dead pool + resurrection hazard**: magers can resurrect dead mobs; the model needs to see what's in the dead pool to assess this risk
- **Priority affordances**: LOS and distance for top-3 priority targets give the model fast tactical read without parsing all entity slots

### 2. Safety Map V3.1 Variant

`_compute_tile_safety_v31` adds blob imminent count (blobs with scan + `attack_delay <= 0`) to the `imminent_attacks` feature for each safe
tile. Blob attacks bypass LOS after scanning, so this count is tile-independent.

### 3. Between-Wave Heuristic Removed

`_apply_between_wave_heuristic` now returns `model_action` unchanged. Previously it forced:

- movement toward Tile A during the 9-tick grace period
- weapon switch to ice barrage

The model must now learn these behaviors itself. The 9-tick grace period (`WAVE_SPAWN_DELAY = 9`) and `distance_to_a_tile` reward signal
remain intact — the model has incentive but not forced override.

### 4. Bug Fix: min_attack_delay Clamp

Fixed for both V3 and V3.1: negative `attack_delay` values (from entities with delays < 0) were not clamped to 0 before normalization. Now
uses `np.clip(min_attack_delay / 8.0, 0.0, 1.0)`.

### 5. Policy Architecture: Flat MLP

V3.1 requires `policy_arch="flat"` (same constraint as V3). This is a switch from V36's `entity_pool_lstm` architecture. The flat MLP with
295 inputs should be simpler and faster to train.

## V37 Hypothesis

If the model's main remaining gaps are:

- hidden state (blob scan, dead pool, queued prayer, nibbler targets) preventing optimal decisions
- forced inter-wave behavior preventing the model from learning wave transitions

then V37 should:

- enable better prayer switching around blob attacks (scan visibility)
- enable better resurrection risk management (dead pool visibility)
- enable more strategic inter-wave repositioning (model-controlled)
- produce a stronger generalist on the broad benchmark

## Training Settings

| Setting             | Value               | Notes                                                                |
|---------------------|---------------------|----------------------------------------------------------------------|
| warmstart           | none (fresh)        | New obs size, new arch — cannot warmstart from V36                   |
| curriculum-mode     | `static`            | **Changed from V36's `adaptive_v36`** — simple climb for fresh start |
| phase               | `climb`             | Forward curriculum with weighted frontier sampling                   |
| climb-sampling      | `weighted`          | Bias toward frontier wave                                            |
| promote-after       | `5`                 | 5 consecutive clears before frontier advances                        |
| start-wave          | `1`                 | **Start from W1** — fresh model needs full curriculum                |
| max-wave            | `66`                | Full Inferno                                                         |
| observation-version | `v3.1`              | **Changed from V36's `v2`**                                          |
| policy-arch         | `flat`              | **Changed from V36's `entity_pool_lstm`**                            |
| actor/critic sizes  | `512,512 / 512,512` | Same as V36                                                          |
| n-envs              | `48`                | Same as V36                                                          |
| n-steps             | `1024`              | Same as V36                                                          |
| batch-size          | `2048`              | Same as V36                                                          |
| n-epochs            | `3`                 | **Increased from V36's 1** — fresh start benefits from more epochs   |
| lr                  | `3e-4`              | **Increased from V36's 3e-5** — fresh start needs higher LR          |
| target-kl           | `0.02`              | Slightly relaxed for fresh start                                     |
| entropy-start/end   | `0.05 / 0.005`      | **Higher than V36** — fresh start needs more exploration             |
| gamma               | `0.995`             | Same as V36                                                          |
| gae-lambda          | `0.95`              | Same as V36                                                          |
| vf-coef             | `0.5`               | Same as V36                                                          |
| max-grad-norm       | `0.5`               | Same as V36                                                          |
| normalize-reward    | yes                 | Same as V36                                                          |
| normalize-obs       | yes                 | Same as V36                                                          |
| checkpoint-every    | `50`                | Same as V36                                                          |
| total budget        | `50M`               | Larger than V36 — fresh start needs more training                    |

## Run Command

### V37 Fresh Start

```powershell
python -m tools.inferno_rl.train_gpu --curriculum-mode static --phase climb --climb-sampling weighted --promote-after 5 --start-wave 1 --max-wave 66 --observation-version v3.1 --policy-arch flat --episode-mode full --n-envs 16 --n-steps 1024 --batch-size 2048 --n-epochs 5 --lr 3e-4 --target-kl 0.02 --entropy-start 0.05 --entropy-end 0.002 --gamma 0.995 --gae-lambda 0.95 --vf-coef 0.5 --max-grad-norm 0.5 --normalize-obs --normalize-reward --actor-sizes 512,512 --critic-sizes 512,512 --save-dir models/V37_obs31 --log-dir logs/V37_obs31 --checkpoint-every 100 --timesteps 200000000 --device cuda --log-reward-terms
```

## Files Changed

| File                                       | Changes                                                                                                                |
|--------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| `tools/inferno_rl/training/observation.py` | V3.1 constants, `_build_observation_v31`, `_compute_tile_safety_v31`, type/routing updates, min_attack_delay clamp fix |
| `tools/inferno_rl/simulator/simulator.py`  | `_apply_between_wave_heuristic` → passthrough                                                                          |
| `tools/inferno_rl/training/env.py`         | Accept `"v3.1"`, pass `dead_mobs`, create temporal for v3.1                                                            |
| `tools/inferno_rl/train_gpu.py`            | Accept `"v3.1"` in validation + argparse                                                                               |

## Metrics Log

| Ckpt | Steps | Frontier | EV | Entropy | KL | Grad | Ep Len | FPS | Notes |
|------|-------|----------|----|---------|----|------|--------|-----|-------|
| ~232 | 3.8M | 10 | 0.91 | 0.049 | 0.013 | 0.32 | — | 3055 | Early learning, W15 max wave |
| ~329 | 5.4M | 14 | 0.90 | 0.049 | 0.015 | 0.31 | 531 | 2817 | Reward +49%, pillar penalty improving |
| ~424 | 6.9M | 15 | 0.94 | 0.048 | 0.016 | 0.32 | 207 | 2661 | Reward dip on harder waves, EV improving |
| ~627 | 10.3M | 20 | 0.93 | 0.048 | 0.014 | 0.29 | 178 | 2492 | W19 max, pillar penalty -2.1, stall near zero |
| ~859 | 14.1M | 31 | 0.95 | 0.047 | 0.016 | 0.28 | 264 | 2271 | First Zek kills, mager waves entered |
| ~2553 | 41.8M | 32 | 0.97 | 0.040 | 0.015 | 0.34 | — | 1324 | Early-stop triggered, return 1.41, max W40 from W32 |

## Eval Results

### Broad Eval (`W49-66`, 100 seeds)

Run manually at key checkpoints (e.g. when frontier reaches W49+, or at regular intervals).

| Ckpt | Steps | Frontier | Clear | Death | Timeout | Mean Max Wave | Notes |
|------|-------|----------|-------|-------|---------|---------------|-------|
|      |       |          |       |       |         |               |       |

### Narrow Eval (`W55-66`, 100 seeds)

| Ckpt | Steps | Clear | Death | Timeout | Top Death Waves | Notes |
|------|-------|-------|-------|---------|-----------------|-------|
|      |       |       |       |         |                 |       |

## Success Criteria

V37 is a success if it does at least one of:

1. Frontier reaches W49+ within 30M steps, demonstrating the V3.1 obs + flat MLP can learn the full wave range.
2. Once frontier is past W49, broad `W49-66` eval beats the V36 champion (V35 `S2-300`).
3. Shows clear evidence that inter-wave learning is happening (model repositions to Tile A and equips barrage without forced override).

## Failure / Stop Criteria

Stop the run early if any of these happen:

1. Frontier stalls below W20 after 15M steps — the fresh-start policy is not converging on basic waves.
2. Frontier stalls below W49 after 30M steps — climb is too slow to reach the interesting waves.
3. The model fails to learn inter-wave repositioning after 10M steps (consistently starts waves away from Tile A without barrage equipped).
4. Training becomes unstable: EV collapses, KL stays near cap, frontier oscillates without advancing.

## Key Risks

1. **Fresh start cost**: V37 cannot warmstart from V36 due to obs size and arch change. It starts from W1 and must climb through all waves
   before reaching the W49-66 range where V36 was evaluated.
2. **Inter-wave learning difficulty**: removing the heuristic override means the model must learn to reposition to Tile A and equip barrage
   on its own. The `distance_to_a_tile` reward signal should guide this, but it may take significant training time.
3. **Climb speed**: starting from W1 means the first 10-20M steps are spent on easy waves. This is necessary for a fresh policy but delays
   the point where we can compare to V36.

## Notes

- V37 is a **fresh start from W1** — frontier progress is the primary metric until it reaches W49+. Broad eval is only meaningful after
  that.
- The between-wave heuristic removal is the highest-risk change. If the model fails to learn inter-wave behavior, consider re-enabling the
  heuristic as a fallback.
- Once frontier passes W49, switch to harden/backfill phase (or adaptive) for a continuation run. The climb phase is just the bootstrap.
- If V37 flat MLP works well, this validates the V3.x observation approach over entity-pool LSTM for future iterations.
