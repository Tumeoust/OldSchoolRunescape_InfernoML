# V41 TB Tracking

## Goal

Test whether a **sweep curriculum** (all waves from the start, failure-rate-weighted sampling) combined with reward tuning
produces better coverage and faster learning than V40's climb curriculum (frontier-based advancement from W35).

## Base Checkpoint

- Base checkpoint: **none (fresh start)**
- Start point: **W1** (full wave range)

## What Changed (V40 -> V41)

### 1. Curriculum: Climb -> Sweep

V40 used a `climb` phase — forward curriculum starting at W35, advancing the frontier after consecutive completions.
V41 uses a new `sweep` phase — all waves (1-66) available from the start, sampled weighted by failure rate.

- **Warmup** (first 100 episodes per env): uniform random wave selection
- **After warmup**: waves sampled proportional to failure rate, with 0.02 floor so mastered waves still appear occasionally
- Each env independently samples its own wave (16 envs -> ~16 distinct waves per rollout)
- No frontier, no promotion — the failure-rate weighting naturally focuses training on hard waves

### 2. Reward Tuning

| Reward Signal                 | V40                | V41                                                                   | Rationale                                                        |
|-------------------------------|--------------------|-----------------------------------------------------------------------|------------------------------------------------------------------|
| `MELEE_PROXIMITY_PENALTY`     | -0.01 (melee only) | Renamed to `NPC_PROXIMITY_PENALTY` -0.01 (melee, mager, ranger, blob) | Penalize standing adjacent to any dangerous NPC, not just melees |
| `MULTI_LOS_PENALTY`           | -0.02              | -0.04                                                                 | Stronger signal to avoid exposing to multiple NPCs               |
| `SINGLE_LOS_ENGAGEMENT_BONUS` | 0.02               | 0.04                                                                  | Stronger signal to reward clean 1v1 engagement                   |
| `DAMAGE_NO_MOVE` penalty      | active             | **removed**                                                           | Was penalizing correct behavior (tanking hits while killing)     |

### 3. Target KL: 0.015 -> 0.03

Raised to allow larger policy updates per epoch, reducing the frequency of early-stopped PPO updates.

### 4. Epochs: 3 -> 5

More optimization passes per rollout to compensate for the wider wave distribution.

## V41 Hypothesis

Climb curriculum front-loads easy waves and gates progress behind consecutive completions. This wastes samples on
already-mastered waves and creates a bottleneck at each new frontier wave. Sweep curriculum should:

- Expose the policy to hard waves (W55-66) much earlier in training
- Naturally concentrate samples on failure-prone waves without manual frontier management
- Produce a more robust policy across all wave types since no wave is ever fully excluded
- Combined with stronger LOS signals and broader adjacency penalty, learn safer positioning faster

## Training Settings

| Setting             | Value                | Notes                                |
|---------------------|----------------------|--------------------------------------|
| warmstart           | `none (fresh)`       | Clean comparison vs V40              |
| curriculum-mode     | `static`             | Same as V40                          |
| phase               | `sweep`              | **New** — failure-weighted all waves |
| start-wave          | `1`                  | Full wave range                      |
| max-wave            | `66`                 | Full Inferno                         |
| observation-version | `v3.2`               | Same as V40 (317 dims)               |
| policy-arch         | `flat_lstm_residual` | Same as V40                          |
| lstm-hidden-size    | `128`                | Same as V40                          |
| lstm-seq-len        | `16`                 | Same as V40                          |
| lstm-burn-in        | `8`                  | Same as V40                          |
| actor/critic sizes  | `512,512 / 512,512`  | Same as V40                          |
| n-envs              | `16`                 | Same as V40                          |
| n-steps             | `128`                | Per-env steps per rollout            |
| batch-size          | `2048`               | Same as V40                          |
| n-epochs            | `5`                  | V40 was 3                            |
| lr                  | `2e-4`               | Same as V40                          |
| target-kl           | `0.03`               | V40 was 0.015                        |
| entropy-start/end   | `0.05 / 0.002`       | Same as V40                          |
| gamma               | `0.995`              | Same                                 |
| gae-lambda          | `0.95`               | Same                                 |
| vf-coef             | `0.5`                | Same                                 |
| max-grad-norm       | `0.5`                | Same                                 |
| normalize-reward    | yes                  | Same                                 |
| normalize-obs       | yes                  | Same                                 |
| checkpoint-every    | `100`                | Same                                 |
| total budget        | `200M`               | Same first-pass budget               |

## Run Command

```powershell
python -m tools.inferno_rl.train_gpu --curriculum-mode static --phase sweep --start-wave 1 --max-wave 66 --observation-version v3.2 --policy-arch flat_lstm_residual --lstm-hidden-size 128 --lstm-seq-len 16 --lstm-burn-in 8 --episode-mode full --n-envs 16 --n-steps 128 --batch-size 2048 --n-epochs 5 --lr 2e-4 --target-kl 0.03 --entropy-start 0.05 --entropy-end 0.002 --gamma 0.995 --gae-lambda 0.95 --vf-coef 0.5 --max-grad-norm 0.5 --normalize-obs --normalize-reward --actor-sizes 512,512 --critic-sizes 512,512 --save-dir models/V41_sweep --log-dir logs/V41_sweep --checkpoint-every 100 --timesteps 200000000 --device cuda
```

## Files Changed

| File                                       | Changes                                                                                                        |
|--------------------------------------------|----------------------------------------------------------------------------------------------------------------|
| `tools/inferno_rl/training/env.py`         | Add `"sweep"` phase: delegates to `_sample_backfill_wave()`, tracks wave stats                                 |
| `tools/inferno_rl/train_gpu.py`            | Add `"sweep"` to phase guards, CLI choices, constraint enforcement                                             |
| `tools/inferno_rl/training/rewards.py`     | `MELEE_PROXIMITY_PENALTY` -> `NPC_PROXIMITY_PENALTY` (all NPCs), doubled LOS signals, removed `DAMAGE_NO_MOVE` |
| `tools/inferno_rl/docs/V41_TB_TRACKING.md` | This file                                                                                                      |

## Metrics Log

| Ckpt   | Steps | Fail% | EV   | Entropy | KL    | Grad   | Ep Len | FPS  | Worst Wave | Notes                                                                                                                                   |
|--------|-------|-------|------|---------|-------|--------|--------|------|------------|-----------------------------------------------------------------------------------------------------------------------------------------|
| ~1342  | 2.7M  | 100%  | 0.97 | 0.049   | 0.004 | 0.72   | 35     | 1347 | W1→5       | Early training; EV strong at 0.97; from W1 only reaches W5; higher starts pass-through (W50→50, W60→60); no early stops                 |
| ~2327  | 4.8M  | 100%  | 0.85 | 0.049   | 0.004 | 0.32   | 95     | 1275 | W1→4       | EV dropped 0.97→0.85 (noise — see next row); ep len 35→95; W7→10; return mean 1.69; grad norm halved                                    |
| ~2775  | 5.7M  | 100%  | 0.96 | 0.049   | 0.003 | 0.36   | 66     | 1380 | W1→4       | EV recovered to 0.96 (0.85 was outlier); return mean up to 1.93; waves completed 10 (was 6); upper waves still pass-through             |
| ~9750  | 20.0M | 100%  | 0.99 | 0.045   | 0.003 | 1.21   | 102    | 1258 | W1→5       | Healthy; grad norm 1.21 (slight rise); value_loss 0.05; return mean 2.34; W1→5, W10→12, W15→16; running_reward_var 20.3                 |
| ~12209 | 25.0M | 100%  | 0.99 | 0.044   | 0.001 | 6.13   | 51     | 1524 | W1→?       | **Onset**: grad 1.2→6.1; value_loss 0.05→1.08; return mean 2.3→10.5; clip_frac dropped 0.027→0.008; KL halved; running_reward_var 20→23 |
| ~14659 | 30.0M | 100%  | 0.99 | 0.043   | 0.004 | 29.42  | 101    | 1432 | W1→2       | **Escalating**: grad 6→29; value_loss 1.1→1.7; return mean 10.5→11.6; W1→2 (was 5); running_reward_var 23→83; wave regression spreading |
| ~26691 | 54.7M | 100%  | 1.00 | 0.037   | 0.001 | 21.27  | 122    | 1254 | W1→1       | **GRAD EXPLOSION** grad 29→21 (clipped); value_loss 1.7→8.9; return mean 11.6→69.4; all waves pass-through; running_reward_var 83→1278  |
| ~26758 | 54.8M | 100%  | 1.00 | 0.037   | 0.001 | 332.08 | 96     | 1434 | W1→1       | **DIVERGING** grad 21→332; value_loss 8.9→16.9; clip_frac 0.005→0.002 (policy frozen); confirmed not noise — run needs to be killed     |
| ~26824 | 54.9M | 100%  | 0.99 | 0.037   | 0.001 | 179.38 | 62     | 1708 | W1→1       | Still diverging; value_loss 16.9→18.5; return mean 68→96; waves_completed=0; policy_grad_loss positive (0.001); terminal                |

## Eval Results

### Broad Eval (`W49-66`, 100 seeds)

| Ckpt | Steps | Clear | Death | Timeout | Mean Max Wave | Notes |
|------|-------|-------|-------|---------|---------------|-------|
|      |       |       |       |         |               |       |

## Success Criteria

1. Sweep phase produces a meaningful failure-rate distribution within 5M steps (not stuck on uniform).
2. Policy handles W55+ waves by 20M steps (V40 climb reached W60 frontier by 10M).
3. Broad eval (W49-66) matches or exceeds V40's best checkpoint.
4. No sustained entropy collapse or KL blowup from the wider wave distribution.

## Failure / Stop Criteria

1. Failure-rate weighting collapses to a few waves (degenerate sampling).
2. No improvement on W49-66 eval after 50M steps.
3. Training destabilizes: sustained KL > 0.05, EV collapse, or entropy collapse.

## Key Risks

1. Without frontier gating, the policy sees hard waves before it can handle easy ones — may slow early learning.
2. Failure-rate weighting could oscillate: policy learns a wave -> weight drops -> forgets it -> weight rises -> cycle.
3. 100-episode warmup per env (1600 total across 16 envs) may be insufficient for stable failure-rate estimates.
4. Higher target-kl (0.03) could cause policy instability if combined with the wider wave distribution.

## Postmortem

### Outcome

V41 diverged and should not be resumed from post-onset checkpoints.

Observed pattern:

- healthy through roughly `20M` steps
- onset around `25M` (`grad 1.2 -> 6.1`, `value_loss 0.05 -> 1.08`, `return_mean 2.3 -> 10.5`)
- full divergence by `54.8M` (`grad 332`, `value_loss 16.9`, `return_mean 69+`, all waves pass-through)

### Likely failure mode

Most likely cause was not PPO under-updating. The stronger explanation is reward farming plus critic target inflation:

- `KL` and `clip_frac` stayed low while `return_mean`, `value_loss`, and `running_reward_var` exploded
- this is consistent with the actor settling into a locally stable exploit while the critic chases inflated returns
- the most likely exploit path was stale dense shaping on easy waves (`Single-LOS Engagement` + `NE Pillar Zone`)

### Confirmed implementation issue

Sweep warmup was also implemented incorrectly:

- intended: `100 episodes per env`
- actual V41 code: failure-weighted sampling activated after roughly `100 episodes total`

This was a real bug, though probably not the primary cause of the late-run blow-up.

### Follow-up

Retry is tracked in [V42_TB_TRACKING.md](V42_TB_TRACKING.md).

V42 changes:

- gate dense shaping behind recent real engagement
- reduce `Single-LOS` bonus
- fix sweep warmup behavior
- resume from checkpoint `9900`
- use `n-epochs=3`, `lr=2e-4`, `target-kl=0.02`
- enable `--log-reward-terms`
