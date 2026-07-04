# V35 TB Tracking

## Goal

Continue from the best V34 checkpoint instead of training a fresh climb branch.

- Base checkpoint: **V34 ckpt 2800**
- Base steps: **138M**
- Base eval: **36% clear** on `W55-66` full eval (best V34 result)

V34 validated the new representation path (`observation v2` + `entity_pool_lstm`), but it also showed that the late `climb` curriculum
starts hurting transfer:

- rollout metrics stayed strong
- eval peaked at ckpt `2800`
- later checkpoints regressed

V35 is designed to preserve the good V34 model, fix the observed reward misalignment, and replace the over-specializing prestige climb loop
with a broader post-climb curriculum.

## What Changed (V34 -> V35)

### 1. Warmstart / Branch Strategy

- V34: fresh init
- V35: **continue from V34 ckpt 2800**

This is now a continuation branch, not a fresh ablation.

### 2. Representation / Network

Unchanged from V34:

- `observation-version = v2`
- `policy-arch = entity_pool_lstm`
- `lstm-hidden-size = 256`
- `lstm-seq-len = 32`
- `lstm-burn-in = 0`
- actor / critic heads `512,512`
- action space unchanged (single 43-way head)

V35 keeps the V34 representation stack because V34 is the first post-V31 branch that transferred meaningfully to eval.

### 3. Reward Changes

These are code-level reward changes in [rewards.py](../training/rewards.py). There are
no new CLI flags for them.

#### Mager kill-order shaping

Added:

- `MAGER_EARLY_KILL_BONUS_BASE = 0.6`
- `MAGER_EARLY_KILL_BONUS_PER_NPC = 0.15`
- `MAGER_DELAY_PENALTY = -0.05`

New behavior:

- If a mager dies while non-magers were still alive at step start, add:
    - `Early Mager Kill = 0.6 + 0.15 * non_mager_enemies_at_step_start`
- If the priority target is a safely focusable mager and the step makes no mager progress:
    - add `Mager Delay = -0.05`

#### Safe-focusable mager condition

The delay penalty only applies when all of these are true:

- `priority_target_entity_type == MAGER`
- `npcs_with_los_now <= 1`
- player has LOS to the priority mager at step start or step end

This is a practical heuristic, not a full tactical solver. It is intentionally conditional so the model is not punished for delaying the
mager when focusing it would require unsafe exposure.

#### Farmable shaping suppression while safely ignoring a mager

When the above safe-focus condition is true and the step makes no mager progress, V35 also suppresses:

- `Single-LOS Engagement`
- `NE Pillar Zone`

This removes part of the incentive to "milk" a safe wave while leaving the mager alive.

#### Stronger resurrection penalties

Adjusted:

- `MAGER_RESURRECTION_PENALTY: 0.4 -> 0.6`
- `MELEE_RESURRECTION_PENALTY: 0.2 -> 0.3`

Note: resurrection penalties were already active before; V35 increases their strength.

### 4. Curriculum Changes

The main curriculum change is **removing late-stage climb / prestige as the primary training loop**.

V34 issue:

- `climb` with prestige eventually optimized for repeated local wave windows
- when `min_waves_to_advance` increased, the model fit the currently drilled band
- broad transfer started to regress after the best checkpoint

V35 replaces that with staged post-climb training:

1. `harden`
    - uniform random starts across `49-66`
    - restores broad coverage and prevents frontier-window specialization
2. `backfill`
    - failure-weighted starts across `49-66`
    - focuses training on the still-leaking waves, but only after broad coverage is refreshed
3. `drill` (optional short burst)
    - only for the single worst eval wave
    - short targeted correction, not the main loop

### 5. Optimizer De-Aggression

To reduce late-run overcooking:

- `lr: 1e-4 -> 5e-5`
- `n-epochs: 2 -> 1`
- keep `target-kl = 0.02`

Rationale:

- V34 late training showed `early_stop` saturation, lower effective epochs, and rising brittleness
- V35 should preserve good behavior from ckpt `2800`, not rewrite the policy too aggressively

### 6. Checkpoint Granularity

- `checkpoint-every: 100 -> 50`

V34 peaked mid-run, so V35 should save more frequently.

## V35 Hypothesis

If V34’s main failure after ckpt `2800` was:

- reward misalignment (bad mager order)
- plus late curriculum specialization (climb prestige)
- plus overly aggressive continuation updates

then V35 should:

- preserve or improve the `36%` baseline from V34 ckpt `2800`
- reduce the tendency to leave the mager alive while farming other NPCs
- improve stability deeper into continuation training
- find a better late-band generalist fit than V34’s late climb continuation

## Training Settings

### V35 Stage 1: Harden

| Setting             | Value             | Notes                      |
|---------------------|-------------------|----------------------------|
| warmstart           | V34 ckpt 2800     | Continue from best V34     |
| phase               | harden            | Uniform random start wave  |
| start-wave          | 49                | Keep late-band focus       |
| max-wave            | 66                | Same target range          |
| observation-version | v2                | Same as V34                |
| policy-arch         | entity_pool_lstm  | Same as V34                |
| lstm-hidden-size    | 256               | Same as V34                |
| lstm-seq-len        | 32                | Same as V34                |
| lstm-burn-in        | 0                 | Same as V34                |
| actor/critic sizes  | 512,512 / 512,512 | Same as V34                |
| target-kl           | 0.02              | Keep early-stop guardrail  |
| n-envs              | 48                | Same as V34                |
| n-steps             | 1024              | Same as V34                |
| batch-size          | 2048              | Same as V34                |
| n-epochs            | 1                 | Reduced from V34           |
| lr                  | 5e-5              | Reduced from V34           |
| entropy-start/end   | 0.02 / 0.002      | Keep same schedule         |
| gamma               | 0.995             | Same as V34                |
| gae-lambda          | 0.95              | Same as V34                |
| vf-coef             | 0.5               | Same as V34                |
| max-grad-norm       | 0.5               | Same as V34                |
| normalize-reward    | yes               | Same as V34                |
| normalize-obs       | yes               | Same as V34                |
| checkpoint-every    | 50                | Finer checkpointing        |
| stage budget        | 20M               | Short bounded continuation |

### V35 Stage 2: Backfill

Same settings as Stage 1, except:

- `phase = backfill`
- load from the best Stage 1 checkpoint

This stage should only start after a Stage 1 eval pass confirms the branch is still healthy.

### V35 Stage 3: Optional Drill Burst

Use only if one specific wave remains clearly dominant in eval.

- `phase = drill`
- `start-wave = <worst wave>`
- `max-wave = <worst wave>`
- same reduced `lr = 5e-5`
- same `n-epochs = 1`
- short budget: `5M`

Do not use drill as the main training loop.

## Run Commands

Replace `<V34_CKPT_2800_PATH>` with the actual V34 checkpoint filename (the timestamped run file ending in `_2800.pt`).

### Stage 1: Harden

```powershell
python -m tools.inferno_rl.train_gpu --load models/v34/inferno_gpu_w49-66_20260303_220713_2800.pt --phase harden --start-wave 49 --max-wave 66 --observation-version v2 --policy-arch entity_pool_lstm --lstm-hidden-size 256 --lstm-seq-len 32 --lstm-burn-in 0 --n-envs 48 --n-steps 1024 --batch-size 2048 --n-epochs 1 --lr 5e-5 --target-kl 0.02 --entropy-start 0.02 --entropy-end 0.002 --gamma 0.995 --gae-lambda 0.95 --vf-coef 0.5 --max-grad-norm 0.5 --normalize-obs --normalize-reward --actor-sizes 512,512 --critic-sizes 512,512 --save-dir models/V35_stage1_harden --log-dir logs/V35_stage1_harden --checkpoint-every 50 --timesteps 20000000 --device cuda --log-reward-terms
```

### Stage 2: Backfill

Replace `<V35_STAGE1_BEST_PATH>` with the best Stage 1 checkpoint after eval.

```powershell
python -m tools.inferno_rl.train_gpu --load models/V35_stage1_harden/inferno_gpu_w49-66_20260304_091331_100.pt --phase backfill --start-wave 49 --max-wave 66 --observation-version v2 --policy-arch entity_pool_lstm --lstm-hidden-size 256 --lstm-seq-len 32 --lstm-burn-in 0 --n-envs 48 --n-steps 1024 --batch-size 2048 --n-epochs 1 --lr 5e-5 --target-kl 0.02 --entropy-start 0.02 --entropy-end 0.002 --gamma 0.995 --gae-lambda 0.95 --vf-coef 0.5 --max-grad-norm 0.5 --normalize-obs --normalize-reward --actor-sizes 512,512 --critic-sizes 512,512 --save-dir models/V35_stage2_backfill --log-dir logs/V35_stage2_backfill --checkpoint-every 50 --timesteps 20000000 --device cuda --log-reward-terms
```

### Stage 3: Optional Drill Burst

Replace `<V35_STAGE2_BEST_PATH>` with the best Stage 2 checkpoint and `<WORST_WAVE>` with the single worst eval wave.

```powershell
python -m tools.inferno_rl.train_gpu --load <V35_STAGE2_BEST_PATH> --phase drill --start-wave <WORST_WAVE> --max-wave <WORST_WAVE> --max-drill-retries 10 --observation-version v2 --policy-arch entity_pool_lstm --lstm-hidden-size 256 --lstm-seq-len 32 --lstm-burn-in 0 --n-envs 48 --n-steps 1024 --batch-size 2048 --n-epochs 1 --lr 5e-5 --target-kl 0.02 --entropy-start 0.02 --entropy-end 0.002 --gamma 0.995 --gae-lambda 0.95 --vf-coef 0.5 --max-grad-norm 0.5 --normalize-obs --normalize-reward --actor-sizes 512,512 --critic-sizes 512,512 --save-dir models/V35_stage3_drill --log-dir logs/V35_stage3_drill --checkpoint-every 50 --timesteps 5000000 --device cuda --log-reward-terms
```

### Stage 4: Broad Harden Refresh (Recommended Next Run)

Use the best broad-range Stage 2 checkpoint (`S2-300`) as the base and run a broad `49-66`
uniform harden pass to recover generality after the failure-weighted backfill specialization.

```powershell
python -m tools.inferno_rl.train_gpu --load models/V35_stage2_backfill/inferno_gpu_w49-66_20260304_112258_300.pt --phase harden --start-wave 49 --max-wave 66 --observation-version v2 --policy-arch entity_pool_lstm --lstm-hidden-size 256 --lstm-seq-len 32 --lstm-burn-in 0 --n-envs 48 --n-steps 1024 --batch-size 2048 --n-epochs 1 --lr 3e-5 --target-kl 0.015 --entropy-start 0.02 --entropy-end 0.002 --gamma 0.995 --gae-lambda 0.95 --vf-coef 0.5 --max-grad-norm 0.5 --normalize-obs --normalize-reward --actor-sizes 512,512 --critic-sizes 512,512 --save-dir models/V35_stage4_harden_refresh --log-dir logs/V35_stage4_harden_refresh --checkpoint-every 50 --timesteps 30000000 --device cuda --log-reward-terms
```

## Eval Cadence

Do not run V35 blind for long stretches.

### Stage 1

Evaluate `W55-66` full eval at:

- ckpt `50`
- ckpt `100`
- ckpt `200`
- end of Stage 1

### Stage 2

Evaluate `W55-66` full eval at:

- ckpt `50`
- ckpt `100`
- ckpt `200`
- end of Stage 2

### Stage 3 (if used)

Evaluate:

- the drilled wave directly
- then a full `W55-66` eval immediately after the short burst

The point of drill is local correction, not sacrificing the full-band model.

## Primary Success Criteria

V35 is a success if it does any of these relative to V34 ckpt `2800`:

1. Beats **36% clear** on `W55-66`.
2. Matches `36%` but shifts deaths later overall.
3. Holds similar clear rate while reducing the visible mager-last / point-farming behavior.

Secondary success:

- fewer timeouts
- lower death concentration on the earliest failing wave
- less regression after continuation training than V34 showed after ckpt `2800`

## Kill Criteria

Stop or branch-select immediately if any of these occur:

1. Stage 1 never matches the `36%` V34 baseline and is clearly worse by ckpt `100`.
2. The model still visibly farms other NPCs while leaving an easily hittable mager alive.
3. Eval improves locally but broad `W55-66` clear regresses after Stage 2.
4. Training re-enters the same overcooked regime seen in late V34:
    - `early_stop` near-saturated for long stretches
    - effective `epochs` collapsing toward `1.0`
    - rising average KL
    - entropy collapsing hard

## Metrics Log

### Stage 1: Harden

| Step                 | Deaths | Waves Comp | Mean Reward | EV   | KL    | Entropy | Grad Norm | FPS  | Notes                                                                                            |
|----------------------|--------|------------|-------------|------|-------|---------|-----------|------|--------------------------------------------------------------------------------------------------|
| 2.0M (140.7M total)  | 7      | 336        | 4.49        | 0.89 | 0.010 | -1.87   | 0.56      | 4263 | Early harden, EV 0.89 (strong), deaths 7 (low), entropy -1.87 stable from V34 base               |
| 7.5M (145.5M total)  | 15     | 349        | 3.56        | 0.89 | 0.008 | -1.97   | 0.55      | 4660 | Deaths 7→15, reward dipped 4.49→3.56. EV steady 0.89, KL 0.008 healthy, entropy -1.97 stable     |
| 10.7M (148.7M total) | 15     | 329        | 3.57        | 0.84 | 0.012 | -1.91   | 0.76      | 3974 | EV 0.89→0.84, grad norm 0.55→0.76 (rising). Waves comp 349→329 dipped. Entropy stable -1.91.     |
| 12.0M (150.0M total) | 15     | 352        | 4.31        | 0.87 | 0.010 | -1.86   | 0.67      | 4435 | EV recovered 0.84→0.87, grad norm 0.76→0.67 (settling), waves 329→352 bounced back.              |
| 24.3M (162.3M total) | 10     | 356        | 4.84        | 0.86 | 0.015 | -1.94   | 0.65      | 4300 | Run 2. Deaths 15→10, reward 4.31→4.84. EV 0.86 steady. Entropy -1.94 stable. KL 0.015 (healthy). |

### Stage 2: Backfill

From best Stage 1 checkpoint: ckpt 100 (142.5M total, 40% clear).

| Step                 | Deaths | Waves Comp | Mean Reward | EV   | KL    | Entropy | Grad Norm | FPS  | Notes                                                                                                                          |
|----------------------|--------|------------|-------------|------|-------|---------|-----------|------|--------------------------------------------------------------------------------------------------------------------------------|
| 3.4M (145.9M total)  | 17     | 324        | 4.25        | 0.82 | 0.014 | -2.00   | 0.68      | 3652 | Early backfill. EV 0.82 (dipped from 0.89 harden base — adapting to failure-weighted starts). Deaths 17, entropy -2.00 stable. |
| 13.4M (155.9M total) | 13     | 333        | 6.20        | 0.78 | 0.011 | -1.76   | 0.65      | 4216 | EV 0.82→0.78 **(watch — declining)**. Reward up 4.25→6.20. Deaths 17→13. Entropy -2.00→-1.76 (tightening). KL 0.011 healthy.   |
| 18.3M (160.8M total) | 11     | 352        | 7.09        | 0.78 | 0.009 | -1.50   | 0.61      | 3866 | EV stable 0.78. Reward 6.20→7.09. Deaths 13→11. Entropy -1.76→-1.50 (still tightening). KL 0.009 healthy.                      |
| 20.0M (162.5M total) | 10     | 358        | 7.10        | 0.78 | 0.010 | -1.36   | 0.63      | 4662 | End of S2 run. EV stable 0.78. Entropy -1.50→-1.36 (tightening continues). Deaths 10, reward 7.10 plateau.                     |

### Stage 3: Drill (Optional)

| Step | Drill Wave | Deaths | Mean Reward | EV | KL | Entropy | Grad Norm | FPS | Notes |
|------|------------|--------|-------------|----|----|---------|-----------|-----|-------|
|      |            |        |             |    |    |         |           |     |       |

### Stage 4: Broad Harden Refresh

From best Stage 2 broad-range checkpoint: S2-300 (157.3M total, 33.4% clear W49-66).
lr=3e-5, target-kl=0.015, harden W49-66.

| Step                 | Deaths | Waves Comp | Mean Reward | EV   | KL    | Entropy | Grad Norm | FPS  | Notes                                                                                                |
|----------------------|--------|------------|-------------|------|-------|---------|-----------|------|------------------------------------------------------------------------------------------------------|
| 3.0M (160.3M total)  | 7      | 342        | 4.49        | 0.87 | 0.010 | -1.90   | 0.63      | 4791 | Early S4. EV 0.87 (up from S2's 0.78 — critic adapting to uniform starts). Deaths 7, entropy stable. |
| 6.0M (163.3M total)  | 18     | 336        | 4.75        | 0.88 | 0.008 | -1.92   | 0.57      | 4893 | Deaths spiked 7→18 (transient). EV 0.88 strong. KL 0.008 healthy. Entropy -1.92 stable.              |
| 9.0M (166.3M total)  | 15     | 341        | 4.14        | 0.84 | 0.009 | -1.95   | 0.60      | 4643 | EV dipped 0.88→0.84. Reward 4.75→4.14. Deaths 18→15 (settling). Entropy -1.95.                       |
| 12.0M (169.3M total) | 9      | 341        | 4.95        | 0.88 | 0.018 | -1.94   | 0.47      | 4582 | EV recovered 0.84→0.88. Deaths 15→9. KL 0.018 (highest yet, watch). Grad 0.47 (low).                 |
| 15.0M (172.3M total) | 10     | 340        | 4.64        | 0.84 | 0.011 | -1.85   | 0.56      | 4717 | EV oscillating 0.84-0.88. Entropy -1.94→-1.85 (loosening slightly). Steady state.                    |
| 18.0M (175.3M total) | 9      | 356        | 4.46        | 0.91 | 0.013 | -1.75   | 0.60      | 4631 | **EV 0.91 (new S4 high).** Waves 356 (best). Deaths 9. Entropy -1.75 (loosening continues).          |
| 21.0M (178.3M total) | 6      | 350        | 4.26        | 0.90 | 0.008 | -1.73   | 0.62      | 4585 | EV 0.90 (strong). Deaths 6 (best). Entropy -1.73. Healthy across all metrics.                        |
| 23.1M (180.4M total) | 19     | 339        | —           | —    | —     | —       | —         | —    | Rollout only (mid-update). Deaths spiked 6→19 (transient noise).                                     |

## Eval Results (100 seeds)

### W55-66 Full Eval

| Checkpoint | Stage    | Steps              | Clear   | Death | Timeout | Top Death Waves              | Notes                                                                          |
|------------|----------|--------------------|---------|-------|---------|------------------------------|--------------------------------------------------------------------------------|
| 50         | harden   | 2.0M (140M total)  | 37%     | 62%   | 1%      | W65=12, W56=11, W63=9        | Beats V34 baseline (36%). W65 worst.                                           |
| 100        | harden   | 4.5M (143M total)  | **40%** | 60%   | 0%      | W63=15, W62=8, W60=6         | **Best.** 0 timeouts. W65 fixed (12→5). W63 spiked (9→15).                     |
| 200        | harden   | 9.5M (148M total)  | 24%     | 75%   | 1%      | W63=16, W65=13, W60=9        | **Regression.** 40%→24%. Deaths +15. W63/W65 worst. W64 spiked 4→7.            |
| 250        | harden   | 12.0M (150M total) | 37%     | 63%   | 0%      | W60=10, W63=9, W56=6         | Recovered from 200 dip. 0 timeouts. Deaths more evenly spread. W60 now worst.  |
| R2-250     | harden   | 24.2M (162M total) | 29%     | 71%   | 0%      | W56=11, W63=11, W65=10       | Regression from peak. 0 timeouts but deaths up. W56/W63/W65 all double-digit.  |
| S2-50      | backfill | 2.5M (145M total)  | 35%     | 64%   | 1%      | W56=12, W63=12, W62=10       | Matches V34 baseline. W56/W63 tied worst. 1 timeout W58.                       |
| S2-100     | backfill | 5.0M (148M total)  | 37%     | 62%   | 1%      | W56=12, W63=9, W61=6         | Near S1 peak (40%). Deaths more evenly spread. W64 spiked (1→5). 0 W66 deaths. |
| S2-150     | backfill | 7.4M (150M total)  | 38%     | 60%   | 2%      | W63=12, W65=11, W62=6        | Holding steady. 2 timeouts appeared (W57, W60). W56 improved (12→4).           |
| S2-200     | backfill | 9.9M (152M total)  | 41%     | 53%   | 6%      | W65=9, W62=7, W59=6          | Deaths 53% (best ever). But 6 timeouts (watch). W63 halved (12→5).             |
| S2-250     | backfill | 12.3M (155M total) | 41%     | 57%   | 2%      | W63=13, W62=8, W65=7         | Holds 41%. Timeouts 6→2 (fixed). W63 re-emerged (5→13). W55-59 very clean.     |
| S2-300     | backfill | 14.8M (157M total) | **46%** | 54%   | 0%      | W56=7, W61=7, W62=7, W65=7   | **New ATB.** 0 timeouts. Deaths even — no wave >7. W63 fixed (13→6).           |
| S2-350     | backfill | 17.2M (160M total) | 35%     | 65%   | 0%      | W63=14, W65=10, W56=8, W61=8 | Regression. W63 spiked back (6→14). W66 death appeared (1). 0 timeouts.        |
| S2-400     | backfill | 19.7M (162M total) | 43%     | 57%   | 0%      | W63=8, W65=8, W56=7, W60=6   | Recovered. 0 timeouts. Even spread (no wave >8).                               |
| S2-406     | backfill | 20.0M (163M total) | 43%     | 56%   | 1%      | W62=9, W56=7, W58=7, W64=6   | End of run. Holds 43%. W63 fixed (8→6). W58/W64 slightly up.                   |
| S4-50      | harden4  | 2.5M (160M total)  | 39%     | 59%   | 2%      | W65=10, W63=8, W55=7, W56=7  | Solid start. Deaths 59%. W65 worst. 2 timeouts.                                |
| S4-100     | harden4  | 4.9M (162M total)  | 37%     | 62%   | 1%      | W65=12, W56=10, W63=8        | Slight dip. W65 spiked (10→12). W56 up (7→10). 1 timeout.                      |
| S4-150     | harden4  | 7.4M (165M total)  | **47%** | 52%   | 1%      | W56=8, W65=7, W55=6          | **New ATB.** Deaths 52% (best). W63=3 (lowest ever). Most even spread.         |
| S4-200     | harden4  | 9.9M (167M total)  | 33%     | 65%   | 2%      | W65=11, W62=10, W64=8        | Regression dip. Deaths 65%. W62/W64 spiked. 2 timeouts.                        |
| S4-250     | harden4  | 12.3M (170M total) | 29%     | 71%   | 0%      | W60=12, W65=12, W56=10       | Steep regression. W60/W65 spiked. 0 timeouts.                                  |
| S4-350     | harden4  | 17.2M (175M total) | 33%     | 66%   | 1%      | W56=11, W63=10, W65=8        | Below S2 baseline. W56/W60 heavy. W66 death (1).                               |
| S4-450     | harden4  | 22.1M (179M total) | 42%     | 57%   | 1%      | W63=9, W61=9, W60=8          | Recovered. W60/W61 cluster. 0 W66 deaths.                                      |
| S4-550     | harden4  | 27.0M (184M total) | **44%** | 56%   | 0%      | W63=11, W65=10, W60=6        | **Ties ATB.** 0 timeouts. W55-59 very clean (20 deaths total).                 |
| S4-610     | harden4  | 30.0M (187M total) | **44%** | 56%   | 0%      | W56=10, W62=9, W65=8         | Holds 44%. 0 timeouts. W59=0 deaths. W62 spiked (4→9).                         |

### Drill-Wave Eval (Optional)
 
| Checkpoint | Drill Wave | Clear | Death | Timeout | Notes | 
|------------|------------|-------|-------|---------|-------|
|            |            |       |       |         |       |

### W49-66 Full Eval (500 seeds)

| Checkpoint | Stage    | Steps              | Clear     | Death | Timeout | Notes                                                          |
|------------|----------|--------------------|-----------|-------|---------|----------------------------------------------------------------|
| S2-300     | backfill | 14.8M (157M total) | **33.4%** | 63.8% | 2.8%    | Best on full range. Deaths spread evenly. 14 timeouts (watch). |
| S2-400     | backfill | 19.7M (162M total) | 29.4%     | 70.2% | 0.4%    | 4% lower clear. W63=54 deaths (dominant). Near-zero timeouts.  |
| S4-150     | harden4  | 7.4M (165M total)  | 28.6%     | 68.4% | 3.0%    | Below S2-300 despite W55-66 ATB. W63=42 dominant. 15 timeouts. |

#### Per-Wave Death Distribution (500 seeds, W49-66, deaths only)

| Wave | S2-300 | S2-400 | S4-150 |
|------|--------|--------|--------|
| 49   | 1      | 5      | 2      |
| 50   | 14     | 14     | 15     |
| 51   | 17     | 15     | 19     |
| 52   | 9      | 13     | 14     |
| 53   | 16     | 20     | 19     |
| 54   | 24     | 17     | 20     |
| 55   | 14     | 17     | 15     |
| 56   | 22     | 34     | 29     |
| 57   | 11     | 13     | 8      |
| 58   | 10     | 14     | 19     |
| 59   | 12     | 16     | 17     |
| 60   | 23     | 20     | 23     |
| 61   | 21     | 19     | 24     |
| 62   | 30     | 26     | 22     |
| 63   | 31     | **54** | **42** |
| 64   | 24     | 19     | 18     |
| 65   | 38     | 34     | 32     |
| 66   | 2      | 1      | 4      |

## Interpretation Notes

- Judge V35 against **V34 ckpt 2800**, not against the latest V34 checkpoint.
- The main point of V35 is to keep the V34 representation gains while removing the two biggest late-run problems:
    - mager-order reward misalignment
    - climb/prestige over-specialization
- If V35 still improves early but then regresses during continuation, the next likely bottleneck is not representation anymore; it is how
  training time is allocated across stages.
 
