# V39 TB Tracking

## Goal

Test whether the new forecast-first **observation V3.2** improves tactical anticipation, path selection, and inter-wave setup over
V38's **observation V3.1** baseline while keeping the same lightweight `flat_lstm_residual` policy family.

## Base Checkpoint

- Base checkpoint: **none (fresh start)**
- Start point: **W35**
- Note: V39 changes the observation shape from `295 -> 363`, so a direct warmstart from V38 is not a clean comparison.

## What Changed (V38 -> V39)

### 1. Observation Version: `v3.1 -> v3.2`

V39 switches the recurrent policy from the 295-dim V3.1 snapshot to the 363-dim V3.2 schema.

V3.2 keeps the V3.1 global block and safety map, then adds:

- a 9-dim threat-horizon block: `T+1..T+3` magic / ranged / melee forecast counts for a stationary player
- a 24-dim directional movement-forecast block: 8 directions x settled distance / LOS count / imminent attacks
- a 5-dim nibbler summary block: centroid position, pillar distance, and cluster spread — nibblers no longer occupy entity slots
- richer 21-float entity slots with closest-footprint geometry, per-entity `player->npc LOS`, and melee `dig_pressure`

### 2. Forecast Features Are the Main New Signal

The core V39 bet is that explicit short-horizon tactical prediction is more valuable than adding more hidden memory alone.

Compared with V3.1, V3.2 now tells the policy:

- what styles are likely to land over the next 3 ticks if it holds position
- how dangerous each canonical 2-tile move looks after settlement
- where large NPC footprints are attackable from, using closest-footprint geometry instead of SW-anchor coords

This should make blob/mager/ranger timing and pillar-side movement less guessy.

### 3. Observation/Simulator Parity Tightened

The V39 observation rollout includes parity fixes so the new forecast features match simulator behavior:

- unscanned blobs are **not** treated as imminent until they finish the scan delay
- scanned blobs still count through LOS loss where appropriate
- temporal previous-action features now track the **executed** simulator action, not just the requested one
- v3.2 temporal / dead-pool plumbing is shared across env, eval, replay, and visualizer tooling

This matters because V3.2 adds more future-facing features; any simulator/observation mismatch would be more damaging than in V3.1.

### 4. Architecture Held Constant

V39 keeps the V38 policy family:

- `policy_arch = flat_lstm_residual`
- `lstm_hidden_size = 128`
- `lstm_seq_len = 16`
- `lstm_burn_in = 8`

So the main experiment is the observation upgrade, not a new recurrent architecture.

### 5. Rewards and Curriculum Held Constant

V39 keeps the current reward setup and the same climb-style curriculum used in V38.

That makes the comparison cleaner:

- V38 = recurrent architecture gain on V3.1
- V39 = observation gain on top of that recurrent baseline

## V39 Hypothesis

V38's residual LSTM should already help with multi-tick continuity.

V39 should help with the parts that still require tactical prediction rather than generic memory:

- anticipating near-future threat style mix
- choosing which 2-tile move actually reduces exposure next tick
- interpreting large-NPC geometry around the NE pillar
- handling blob scan/attack windows and melee dig threat with less implicit inference

If the hypothesis is correct, V39 should learn later-wave movement and inter-wave setup faster than V38 at comparable step counts.

## Training Settings

| Setting             | Value                | Notes                                                 |
|---------------------|----------------------|-------------------------------------------------------|
| warmstart           | `none (fresh)`       | Observation shape changed                             |
| curriculum-mode     | `static`             | Same as V38                                           |
| phase               | `climb`              | Same as V38                                           |
| climb-sampling      | `weighted`           | Same as V38                                           |
| promote-after       | `5`                  | Same as V38                                           |
| start-wave          | `1`                  | Fresh from W1                                         |
| max-wave            | `66`                 | Full Inferno                                          |
| observation-version | `v3.2`               | New for V39                                           |
| policy-arch         | `flat_lstm_residual` | Same as V38                                           |
| lstm-hidden-size    | `128`                | Same as V38                                           |
| lstm-seq-len        | `16`                 | Same as V38                                           |
| lstm-burn-in        | `8`                  | Same as V38                                           |
| actor/critic sizes  | `512,512 / 512,512`  | Same head sizes as V38                                |
| n-envs              | `16`                 | Same as V38                                           |
| n-steps             | `1024`               | Same as V38                                           |
| batch-size          | `2048`               | Same as V38                                           |
| n-epochs            | `3`                  | Same as V38                                           |
| lr                  | `2e-4`               | Same as V38 table; keep optimizer pressure comparable |
| target-kl           | `0.015`              | Same as V38                                           |
| entropy-start/end   | `0.05 / 0.002`       | Same as V38                                           |
| gamma               | `0.995`              | Same                                                  |
| gae-lambda          | `0.95`               | Same                                                  |
| vf-coef             | `0.5`                | Same                                                  |
| max-grad-norm       | `0.5`                | Same                                                  |
| normalize-reward    | yes                  | Same                                                  |
| normalize-obs       | yes                  | Same                                                  |
| checkpoint-every    | `100`                | Same                                                  |
| total budget        | `200M`               | Same first-pass budget as V38                         |

## Run Command

```powershell
python -m tools.inferno_rl.train_gpu --curriculum-mode static --phase climb --climb-sampling weighted --promote-after 5 --start-wave 35 --max-wave 66 --observation-version v3.2 --policy-arch flat_lstm_residual --lstm-hidden-size 128 --lstm-seq-len 16 --lstm-burn-in 8 --episode-mode full --n-envs 16 --n-steps 1024 --batch-size 2048 --n-epochs 3 --lr 2e-4 --target-kl 0.015 --entropy-start 0.05 --entropy-end 0.002 --gamma 0.995 --gae-lambda 0.95 --vf-coef 0.5 --max-grad-norm 0.5 --normalize-obs --normalize-reward --actor-sizes 512,512 --critic-sizes 512,512 --save-dir models/V39_obs32_lstm128 --log-dir logs/V39_obs32_lstm128 --checkpoint-every 100 --timesteps 200000000 --device cuda
```

## Files Changed

| File                                              | Changes                                                                          |
|---------------------------------------------------|----------------------------------------------------------------------------------|
| `tools/inferno_rl/training/observation.py`        | Stable facade now dispatches `v3.2`                                              |
| `tools/inferno_rl/training/observation_common.py` | Shared constants plus centralized temporal update helper                         |
| `tools/inferno_rl/training/observation_v3.py`     | Added `v3.2` builder and tightened blob-imminence parity                         |
| `tools/inferno_rl/simulator/forecast.py`          | Added threat-horizon / directional forecast helpers and aligned blob scan timing |
| `tools/inferno_rl/simulator/priority.py`          | Shared combat sort key for simulator and observation ordering                    |
| `tools/inferno_rl/train_gpu.py`                   | Allows `v3.2` with `flat` and `flat_lstm_residual`                               |
| `tools/inferno_rl/ppo/policy.py`                  | Residual LSTM architecture reused for `v3.2`                                     |
| `tools/inferno_rl/training/env.py`                | V3.2 temporal + dead-pool wiring; temporal updates follow executed action        |
| `tools/inferno_rl/eval.py`                        | Stateful recurrent inference and correct temporal updates for `v3.2`             |
| `tools/inferno_rl/model_log_run.py`               | Stateful recurrent inference and correct temporal updates for `v3.2`             |
| `tools/inferno_rl/cli/replay_episode.py`          | Recurrent replay and `v3.2` temporal / dead-pool support                         |
| `tools/inferno_rl/visualizer/run_visual.py`       | Stateful recurrent inference and `v3.2` observation plumbing                     |
| `tools/inferno_rl/visualizer/play_human.py`       | Human-vs-model comparison path updated for `v3.2` temporal handling              |
| `tools/inferno_rl/visualizer/review_deaths.py`    | Stateful recurrent inference and `v3.2` temporal handling                        |
| `tools/inferno_rl/tests/test_observation_v32.py`  | Added v3.2 shape / geometry / blob forecast / temporal regression tests          |
| `tools/inferno_rl/docs/OBSERVATION_SPACE.md`      | Added full `v3.2` reference and parity notes                                     |
| `tools/inferno_rl/docs/RL_CHANGES.md`             | Added V3.2 structural change entry                                               |

## Metrics Log

| Ckpt  | Steps | Frontier | Promo% | EV   | Entropy | KL    | Grad | Ep Len | FPS  | Notes                                                                                                                                                                                         |
|-------|-------|----------|--------|------|---------|-------|------|--------|------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ~190  | 3.1M  | 46       | 21%    | 0.81 | 0.049   | 0.012 | 0.37 | 229    | 1713 | Frontier W45→W46. EV jumped 0.70→0.81. Max wave from W46=48.                                                                                                                                  |
| ~371  | 6.1M  | 48       | 40%    | 0.77 | 0.049   | 0.012 | 0.37 | 406    | 1610 | Frontier W46→W48. Max wave from W48=53. Ep len nearly doubled (deeper waves). FPS dipped (harder waves).                                                                                      |
| ~543  | 8.9M  | 59       | 26%    | 0.88 | 0.048   | 0.013 | 0.36 | 281    | 1526 | Frontier W48→W59 (+11 in 2.8M steps — massive jump). EV new best 0.88. Max wave from W59=61. Clip fraction high at 0.20.                                                                      |
| ~693  | 11.4M | 62       | 24%    | 0.84 | 0.047   | 0.014 | 0.39 | 245    | 1342 | Frontier W59→W62 (+3 in 2.5M steps). Max wave from W62=63. Clip fraction 0.21, KL 0.014 — both creeping up. FPS dropped to 1342 (deep waves).                                                 |
| ~838  | 13.7M | 63       | 15%    | 0.79 | 0.047   | 0.006 | 0.38 | 267    | 1403 | Frontier W62→W63. Max wave from W63=66 — agent reaches the end! KL dropped sharply 0.014→0.006, clip fraction 0.21→0.08 (policy stabilizing).                                                 |
| ~996  | 16.3M | 37 (P2)  | 100%   | 0.86 | 0.046   | 0.012 | 0.35 | 1002   | 1402 | Curriculum entered phase 2 — frontier reset to W37. 0 deaths. Return mean 0.61 (3x). EV held through phase reset (no tank/KL spike — first time; critic generalized across wave range).       |
| ~1181 | 19.3M | 48 (P2)  | 50%    | 0.74 | 0.045   | 0.013 | 0.33 | 478    | 1139 | P2 frontier W37→W48 (+11 in 3M steps). EV dipped 0.86→0.74 (re-learning harder waves). Max wave from W48=53. 13 deaths. FPS dropped to 1139 (deep episodes + eval running). Return mean 0.43. |
| ~1274 | 20.9M | 53 (P2)  | 83%    | 0.70 | 0.045   | 0.011 | 0.33 | 611    | 1380 | P2 frontier W48→W53 (+5 in 1.6M steps). Promo 83% (5/6), max streak 3. Only 8 deaths. Max wave from W53=57. EV still declining (0.74→0.70). FPS recovered to 1380 (eval done).                |
| ~1425 | 23.3M | 56 (P2)  | 42%    | 0.75 | 0.044   | 0.013 | 0.34 | 433    | 1379 | P2 frontier W53→W56 (+3 in 2.4M steps). EV recovering 0.70→0.75. Max wave from W56=60. Promo 42% (5/12). 13 deaths. Return mean 0.44 (stable). |

## Eval Results

### Broad Eval (`W49-66`, 100 seeds)

| Ckpt | Steps | Frontier | Clear | Death | Timeout | Mean Max Wave | Notes |
|------|-------|----------|-------|-------|---------|---------------|-------|
|      |       |          |       |       |         |               |       |

### Narrow Eval (`W55-66`, 100 seeds)

| Ckpt | Steps | Clear | Death | Timeout | Top Death Waves              | Notes                                                                                |
|------|-------|-------|-------|---------|------------------------------|--------------------------------------------------------------------------------------|
| 1100 | 18.0M | 16%   | 74%   | 10%     | W63=12, W59=10, W60=9, W55=9 | First eval. Mean wave 60.6, median 60. Deaths spread across W55-65, heaviest at W63. |

## Success Criteria

1. Frontier reaches at least W20 by 15M steps.
2. Frontier reaches at least W49 by 40M steps.
3. Manual review shows cleaner next-tick movement choices around the NE pillar than V38.
4. Once V39 reaches W49+, broad `W49-66` eval is at least competitive with the best available V38 checkpoint and ideally exceeds it.

## Failure / Stop Criteria

1. Frontier stalls below W20 after 15M steps.
2. Frontier stalls below W32 after 30M steps.
3. Forecast blocks appear unused in policy behavior during manual review.
4. Training destabilizes with sustained KL above `0.03`, collapsing EV, or obvious entropy collapse.

## Key Risks

1. Fresh-start cost may hide the real value of `v3.2` for a long time.
2. The new forecast features may be partially redundant with the residual LSTM, reducing net gain.
3. If forecast features are even slightly mismatched from simulator behavior, the policy could learn brittle heuristics.
4. The larger observation (`363` vs `295`) may slow early sample efficiency.

## Notes

- V39 is the natural follow-up to V38: same recurrent backbone, new observation family.
- The cleanest interpretation is "does V3.2 add value beyond V38's residual memory path?"
- Because V3.2 is forecast-heavy, manual replay review should focus on:
    - blob scan windows
    - next-tick pillar-side movement
    - inter-wave repositioning
    - avoiding fake-safe moves that still settle into LOS
- Policy export remains unsupported for recurrent checkpoints.
