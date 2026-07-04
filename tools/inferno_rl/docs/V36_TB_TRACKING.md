# V36 TB Tracking

## Goal

Replace long single-phase continuation runs with an **adaptive curriculum controller** that:

- keeps a live champion checkpoint
- evaluates broad performance during training
- switches training regime mid-run at safe rollout boundaries
- rolls back to the current champion when a regime regresses
- periodically runs short **opener-only** sessions to focus training on the highest-leverage part of each wave

V35 established that:

- **`S2-300`** is the current true champion
- later V35 training could produce short-lived narrow improvements
- but no checkpoint after `S2-300` beat it on the broader `W49-66` 500-seed benchmark

So V36 is designed to solve the main remaining problem:

- the model can improve
- but the training process keeps pushing past the best generalist checkpoint

## Base Checkpoint

- Base checkpoint: **V35 `S2-300`**
- Path: **`models/V35_stage2_backfill/inferno_gpu_w49-66_20260304_112258_300.pt`**
- Best broad benchmark so far: **best `W49-66` 500-seed result**

This is the seed champion for the first adaptive run.

## What Changed (V35 -> V36)

### 1. Adaptive Curriculum Controller

V36 adds a runtime controller that:

- runs periodic broad evals during training
- tracks the best-performing checkpoint as the current champion
- updates the champion only when broad eval truly improves
- reloads the champion if the current regime regresses
- switches regime automatically after regression, repeated plateau, or a fixed max window count

This is implemented in [adaptive_curriculum.py](../adaptive_curriculum.py).

### 2. Regime Cycle

V36 first cut uses a fixed repeating regime cycle:

1. `harden_full`
2. `backfill_full`
3. `backfill_opener`
4. back to `harden_full`

There is **no automatic single-wave drill** in V36 first cut.

### 3. Broad Eval Is the Control Metric

Runtime switching is driven by:

- **`W49-66`**
- **100 seeds**
- **full-clear rate**

This is now the primary control signal.

Narrower `W55-66` evals can still be used diagnostically, but they do not decide the champion.

### 4. Opener Episode Mode

V36 adds a short episode mode that ends early when the critical opening phase is resolved.

In `episode_mode=opener`, an episode ends when either:

- `50` ticks elapse, or
- all alive **magers** and **melees** are dead

Success requires:

- player is alive
- no wave timeout
- player HP is **strictly above 40**

Failure occurs on:

- death
- wave timeout
- hitting an opener boundary with HP `<= 40`

This keeps training focused on the part of the wave that matters most.

### 5. Opener Terminal Shaping

V36 keeps the V35 reward changes, and adds small opener-local terminal shaping in the env:

- `Opener Resolved Success = +2.0`
- `Opener Survive Window = +0.5`
- `Opener Failure = -1.0`

This makes true tactical opener resolution more valuable than merely surviving 50 ticks.

### 6. Global Backfill Wave Stats

V35’s `backfill_waves_mastered` style metrics were noisy because wave stats were effectively local to whichever env reported last.

V36 now maintains a **global merged wave-stats aggregate** in the training process and broadcasts it back to all workers during reconfigure.

This makes:

- `backfill` sampling more consistent
- `backfill_waves_mastered_global`
- `backfill_worst_wave_fail_rate_global`

more meaningful than before.

### 7. Runtime Worker Reconfiguration

V36 adds live worker reconfiguration so the trainer can switch:

- `phase`
- `episode_mode`
- opener thresholds
- shared backfill wave stats

without restarting the run.

## V36 Hypothesis

If V35’s main remaining issue was:

- each phase can improve the model
- but long uninterrupted exposure to one phase over-specializes the policy

then V36 should:

- preserve the best broad generalist more reliably
- stop walking past short-lived peaks
- use opener sessions to improve the highest-leverage subproblem
- outperform long static continuations on the broad benchmark

## Training Settings

| Setting              | Value                                             | Notes                                  |
|----------------------|---------------------------------------------------|----------------------------------------|
| warmstart            | V35 `S2-300`                                      | Seed champion                          |
| curriculum-mode      | `adaptive_v36`                                    | New adaptive controller                |
| control eval         | `W49-66`, 100 seeds                               | Broad runtime control metric           |
| eval interval        | every `50` rollouts                               | About `2.46M` steps per window         |
| regime cycle         | `harden_full -> backfill_full -> backfill_opener` | Repeats                                |
| harden max windows   | `3`                                               | About `7.4M` max before forced switch  |
| backfill max windows | `6`                                               | About `14.8M` max before forced switch |
| opener max windows   | `1`                                               | Short corrective burst only            |
| improve threshold    | `+0.5pp`                                          | Champion update threshold              |
| regress threshold    | `-2.0pp`                                          | Rollback threshold                     |
| plateau windows      | `2`                                               | Then switch and reload champion        |
| observation-version  | `v2`                                              | Same as V35                            |
| policy-arch          | `entity_pool_lstm`                                | Same as V35                            |
| lstm-hidden-size     | `256`                                             | Same as V35                            |
| lstm-seq-len         | `32`                                              | Same as V35                            |
| lstm-burn-in         | `0`                                               | Same as V35                            |
| actor/critic sizes   | `512,512 / 512,512`                               | Same as V35                            |
| n-envs               | `48`                                              | Same as V35                            |
| n-steps              | `1024`                                            | Same as V35                            |
| batch-size           | `2048`                                            | Same as V35                            |
| n-epochs             | `1`                                               | Conservative continuation              |
| lr                   | `3e-5`                                            | Conservative continuation              |
| target-kl            | `0.015`                                           | Tighter than late V35                  |
| entropy-start/end    | `0.02 / 0.002`                                    | Same schedule                          |
| gamma                | `0.995`                                           | Same as V35                            |
| gae-lambda           | `0.95`                                            | Same as V35                            |
| vf-coef              | `0.5`                                             | Same as V35                            |
| max-grad-norm        | `0.5`                                             | Same as V35                            |
| normalize-reward     | yes                                               | Same as V35                            |
| normalize-obs        | yes                                               | Same as V35                            |
| checkpoint-every     | `50`                                              | Align with control windows             |
| total budget         | `30M`                                             | First unattended adaptive run          |

## Run Command

### V36 Adaptive Run

```powershell
python -m tools.inferno_rl.train_gpu --load models/V35_stage2_backfill/inferno_gpu_w49-66_20260304_112258_300.pt --curriculum-mode adaptive_v36 --adaptive-eval-every 50 --adaptive-eval-episodes 100 --adaptive-eval-start-wave 49 --adaptive-eval-max-wave 66 --adaptive-harden-max-windows 3 --adaptive-backfill-max-windows 6 --adaptive-opener-max-windows 1 --adaptive-improve-threshold 0.5 --adaptive-regress-threshold 2.0 --adaptive-plateau-windows 2 --start-wave 49 --max-wave 66 --observation-version v2 --policy-arch entity_pool_lstm --lstm-hidden-size 256 --lstm-seq-len 32 --lstm-burn-in 0 --episode-mode full --opener-tick-limit 50 --opener-min-health 40 --n-envs 48 --n-steps 1024 --batch-size 2048 --n-epochs 1 --lr 3e-5 --target-kl 0.015 --entropy-start 0.02 --entropy-end 0.002 --gamma 0.995 --gae-lambda 0.95 --vf-coef 0.5 --max-grad-norm 0.5 --normalize-obs --normalize-reward --actor-sizes 512,512 --critic-sizes 512,512 --save-dir models/V36_adaptive --log-dir logs/V36_adaptive --checkpoint-every 50 --timesteps 30000000 --device cuda --log-reward-terms
```

## Adaptive Switch Rules

### Improvement

If broad full-clear rate is:

- more than `+0.5pp` above champion
- or tied on clear with better tie-breaks

then:

- update champion
- continue the current regime (unless max window count is reached)

### Plateau

If score stays within `±0.5pp` of the champion without improving:

- count a plateau window
- after `2` consecutive plateau windows:
    - reload champion
    - switch to the next regime

### Regression

If score drops more than `2.0pp` below champion:

- immediately reload champion
- switch to the next regime

### Forced Regime Limit

Even without regression:

- `harden_full` is capped at `3` windows
- `backfill_full` is capped at `6` windows
- `backfill_opener` is capped at `1` window

When the cap is reached:

- reload champion
- switch regime

This is the main guardrail against overshooting.

## New TensorBoard Metrics

### Adaptive Controller

- `adaptive/current_regime`
- `adaptive/current_score`
- `adaptive/current_death_rate`
- `adaptive/current_timeout_rate`
- `adaptive/current_mean_max_wave`
- `adaptive/champion_score`
- `adaptive/champion_death_rate`
- `adaptive/champion_timeout_rate`
- `adaptive/champion_mean_max_wave`
- `adaptive/regime_window`
- `adaptive/plateau_count`
- `adaptive/rollback_count`
- `adaptive/switch_reason`

### Opener Regime

- `rollout/opener_success_rate`
- `rollout/opener_failure_rate`
- `rollout/opener_resolved_rate`
- `rollout/opener_survive_window_rate`
- `rollout/opener_low_hp_fail_rate`
- `rollout/opener_mean_end_hp`
- `rollout/opener_mean_end_tick`
- `rollout/opener_mean_magers_remaining`
- `rollout/opener_mean_melees_remaining`

### Global Backfill Metrics

- `rollout/backfill_waves_mastered_global`
- `rollout/backfill_worst_wave_fail_rate_global`
- `rollout/phase_worst_wave_global`

Note:

- a wave only counts as mastered if fail rate `< 2%`
- and it has at least `25` total samples

## Metrics Log

| Window    | Steps  | Regime          | Broad Clear (100)              | Death             | Timeout | Mean Max Wave | Champion? | Switch Reason             | Notes                                                                                                                                                                                                                     |
|-----------|--------|-----------------|--------------------------------|-------------------|---------|---------------|-----------|---------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Seed      | 0      | harden_full     |                                |                   |         |               | yes       | seed                      | Seed champion from `S2-300`                                                                                                                                                                                               |
| 1         | ~2.46M | backfill_full   | 24% (current) / 33% (champion) | 69% / 64%         | 7% / 3% | 60.25 / 60.57 | no        | —                         | 1 rollback. EV 0.82, entropy -1.71, KL 0.008, grad 0.63. 5192 FPS. Worst wave: W52 (100% fail). TB steps are cumulative from V35 (~159.7M in TB).                                                                         |
| 2         | ~4.92M | backfill_full   | 33%                            | 63%               | 4%      | 60.3          | **yes**   | improved                  | Matched seed champion score. Controller updated champion. Ckpt: `_150.pt`.                                                                                                                                                |
| 3         | ~7.4M  | backfill_opener | 22% (current) / 33% (champion) | 77% / 63%         | 1% / 4% | 59.75 / 60.34 | no        | regression (2nd rollback) | Now in opener mode (ep_len ~50t). Opener: 97% success, 0% resolved, mean HP 88.4, 1.0 magers remaining. **EV -0.13 (watch)**, **grad 4.27 (watch)** — critic not adapted to opener reward scale. Entropy -1.89, KL 0.008. |
| 4-5 (mid) | ~10.9M | (back to full)  | — (mid-window, no eval)        | 6 deaths/rollout  | 0       | —             | —         | —                         | Post-opener recovery. EV 0.86 (recovered), grad 0.57 (recovered), entropy -1.88, KL 0.006. Ep_len ~392. Worst wave fail: 11.8% (W63). 5050 FPS.                                                                           |
| ~6 (mid)  | ~14.7M | (full)          | — (mid-window, no eval)        | 14 deaths/rollout | 0       | —             | —         | —                         | EV 0.85, entropy -1.87, KL 0.007, grad 0.58. Ep_len ~563. Deaths up (was 6). Worst wave fail: 14.4% (W63). 4987 FPS.                                                                                                      |
| 7         | ~17.3M | harden_full     | 24% (current) / 33% (champion) | 74% / 63%         | 2% / 4% | 60.1 / 60.34  | no        | regression (6th rollback) | **6 rollbacks total, 0 champion improvements since W2.** Cycled through full regime loop multiple times. Opener: 95% success, 0% resolved. EV 0.89, entropy -1.91, KL 0.007, grad 0.60. Approaching failure criterion #1. |

## Eval Results

### Broad Runtime Control Eval (`W49-66`, 100 seeds)

| Window / Ckpt | Steps  | Regime          | Clear | Death | Timeout | Mean Max Wave | Champion? | Notes                                                                                                               |
|---------------|--------|-----------------|-------|-------|---------|---------------|-----------|---------------------------------------------------------------------------------------------------------------------|
| Seed          | 0      | harden_full     |       |       |         |               | yes       | Initial seed champion                                                                                               |
| 50 (W1)       | ~2.46M | backfill_full   | 24%   | 69%   | 7%      | 60.25         | no        | Champion: 33% clear, 64% death, 3% timeout, 60.57 mean max wave. 1 rollback. TB step ~159.7M (cumulative from V35). |
| 100 (W2)      | ~4.92M | backfill_full   | 33%   | 63%   | 4%      | 60.3          | **yes**   | improved                                                                                                            | Matched seed score. New champion: `_150.pt`. |
| 150 (W3)      | ~7.4M  | backfill_opener | 22%   | 77%   | 1%      | 59.75         | no        | 2nd rollback. Champion now `_150.pt` (33%).                                                                         |
| ~350 (W7)     | ~17.3M | harden_full     | 24%   | 74%   | 2%      | 60.1          | no        | 6th rollback. Champion unchanged (33%). Full regime cycle completed with no improvement.                            |

### Optional Diagnostic Eval (`W55-66`, 100 seeds)

Use only if you want narrow-band diagnostic comparison against V35/V34. Do not use this table alone to select the champion.

| Window / Ckpt | Steps   | Clear | Death | Timeout | Top Death Waves | Notes |
|---------------|---------|-------|-------|---------|-----------------|-------|
| Seed          | 0       |       |       |         |                 |       |
| 100           | ~4.92M  |       |       |         |                 |       |
| 200           | ~9.83M  |       |       |         |                 |       |
| 300           | ~14.75M |       |       |         |                 |       |
| 400           | ~19.66M |       |       |         |                 |       |
| 500           | ~24.58M |       |       |         |                 |       |
| 600           | ~29.49M |       |       |         |                 |       |

## Success Criteria

V36 is a success if it does at least one of:

1. Improves on the current broad `W49-66` benchmark from the `S2-300` seed champion.
2. Matches the champion while reducing rollback frequency and keeping stable broad performance across multiple windows.
3. Produces a more stable broad generalist than late V35 continuations, even if peak narrow `W55-66` is unchanged.

## Failure / Stop Criteria

Stop the run early if any of these happen:

1. Multiple consecutive adaptive windows regress enough to trigger repeated rollbacks with no champion improvement.
2. Broad control eval never improves after a full regime cycle (`harden_full -> backfill_full -> backfill_opener`).
3. The opener regime clearly harms broad full-clear rate instead of improving later windows.
4. Training becomes unstable:
    - EV collapses and stays low
    - KL rises and stays near the cap
    - rollback count grows rapidly
    - no regime can hold broad performance near the seed champion

## Notes

- The current champion must always be chosen by the **broad** benchmark first.
- Narrow eval gains do not count as a true upgrade if broad `W49-66` regresses.
- If V36 works, this should become the default continuation strategy instead of long static phase runs.
