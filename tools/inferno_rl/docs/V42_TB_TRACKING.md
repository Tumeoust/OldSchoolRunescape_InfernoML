# V42 TB Tracking

## Goal

Retry V41 sweep from the last healthy checkpoint with the stale dense-shaping exploit removed.

V41 did not fail because the actor was under-updating. It failed because returns and value targets blew up while
policy KL stayed low. V42 keeps sweep, but de-risks the continuation:

- resume from the last clearly healthy checkpoint before onset
- reduce PPO optimizer aggression (`n-epochs 5 -> 3`, `target-kl 0.03 -> 0.02`)
- keep learning rate conservative (`2e-4`, not higher)
- log raw reward terms so any new reward farming shows up immediately

## Base Checkpoint

- Base checkpoint: `models/V41_sweep/inferno_gpu_w1-66_20260312_213010_9900.pt`
- Approx step count: `~20.3M`
- Reason: last checkpoint before the 25M-step onset (`grad 1.21`, `value_loss 0.05`, `return_mean 2.34`)

## What Changed (V41 -> V42)

### 1. Reward exploit fix

Dense shaping now requires recent real engagement.

- `Single-LOS Engagement` only pays when engagement is recent
- `NE Pillar Zone` only pays during active/recent combat, not stale tagged states
- `SINGLE_LOS_ENGAGEMENT_BONUS`: `0.04 -> 0.02`

Rationale:

- V41 likely found a local exploit on easy waves: tag once, hold 1-LOS / zone state, farm shaped reward
- This matches the TB signature: `KL` and `clip_frac` down while `return_mean`, `value_loss`, and `running_reward_var` explode

### 2. Sweep warmup fix

The sweep sampler warmup now matches the intended behavior.

- Warmup is `100 episodes per env`, not `100 episodes total across the worker`

Rationale:

- V41 switched into failure-weighted sampling far too early
- This was a real bug, even if it was not the main cause of the late divergence

### 3. PPO de-aggression

| Setting     | V41    | V42    | Rationale                                   |
|-------------|--------|--------|---------------------------------------------|
| `n-epochs`  | `5`    | `3`    | Fewer passes per rollout, less overcooking  |
| `target-kl` | `0.03` | `0.02` | Tighter actor update cap                    |
| `lr`        | `2e-4` | `2e-4` | Do not increase LR while retrying this run  |

## V42 Hypothesis

If the V41 divergence was caused by reward farming plus critic target inflation, then V42 should:

- keep `return_mean` in the same rough band as the healthy 20M V41 checkpoint instead of drifting upward
- avoid the `value_loss` step jump seen at 25M
- keep `running_reward_var` from re-accelerating
- preserve or improve wave coverage without regressing to `W1 -> 1/2`

If V42 still diverges with the exploit fix in place, the next suspect is continuation aggressiveness from the
checkpoint itself, and the follow-up move should be lowering `lr` to `1.5e-4`, not raising it.

## Training Settings

| Setting             | Value                | Notes                                      |
|---------------------|----------------------|--------------------------------------------|
| warmstart           | `V41 ckpt 9900`      | Resume from last healthy checkpoint        |
| curriculum-mode     | `static`             | Same as V41                                |
| phase               | `sweep`              | Same as V41                                |
| start-wave          | `1`                  | Full wave range                            |
| max-wave            | `66`                 | Full Inferno                               |
| observation-version | `v3.2`               | Same as V41                                |
| policy-arch         | `flat_lstm_residual` | Same as V41                                |
| lstm-hidden-size    | `128`                | Same as V41                                |
| lstm-seq-len        | `16`                 | Same as V41                                |
| lstm-burn-in        | `8`                  | Same as V41                                |
| actor/critic sizes  | `512,512 / 512,512`  | Same as V41                                |
| n-envs              | `16`                 | Same as V41                                |
| n-steps             | `128`                | Same as V41                                |
| batch-size          | `2048`               | Same as V41                                |
| n-epochs            | `3`                  | Reduced from 5                             |
| lr                  | `2e-4`               | Keep conservative for first retry          |
| target-kl           | `0.02`               | Reduced from 0.03                          |
| entropy-start/end   | `0.05 / 0.002`       | Same as V41                                |
| gamma               | `0.995`              | Same                                       |
| gae-lambda          | `0.95`               | Same                                       |
| vf-coef             | `0.5`                | Same                                       |
| max-grad-norm       | `0.5`                | Same                                       |
| normalize-reward    | yes                  | Same                                       |
| normalize-obs       | yes                  | Same                                       |
| log-reward-terms    | yes                  | New: required to catch reward farming      |
| checkpoint-every    | `100`                | Same as V41                                |
| total budget        | `200M`               | Same first-pass budget                     |

## Starting Command

```powershell
python -m tools.inferno_rl.train_gpu --load models/V41_sweep/inferno_gpu_w1-66_20260312_213010_9900.pt --curriculum-mode static --phase sweep --start-wave 1 --max-wave 66 --observation-version v3.2 --policy-arch flat_lstm_residual --lstm-hidden-size 128 --lstm-seq-len 16 --lstm-burn-in 8 --episode-mode full --n-envs 16 --n-steps 128 --batch-size 2048 --n-epochs 3 --lr 2e-4 --target-kl 0.02 --entropy-start 0.05 --entropy-end 0.002 --gamma 0.995 --gae-lambda 0.95 --vf-coef 0.5 --max-grad-norm 0.5 --normalize-obs --normalize-reward --actor-sizes 512,512 --critic-sizes 512,512 --save-dir models/V42_sweep_resume --log-dir logs/V42_sweep_resume --checkpoint-every 100 --timesteps 200000000 --device cuda --log-reward-terms
```

## Files Changed

| File                              | Changes                                                                       |
|-----------------------------------|-------------------------------------------------------------------------------|
| `tools/inferno_rl/training/rewards.py` | Gate dense shaping behind recent engagement, reduce `Single-LOS` bonus        |
| `tools/inferno_rl/training/env.py`     | Fix sweep warmup to be per-env rather than effectively global                 |
| `tools/inferno_rl/tests/test_reward_shaping.py` | Add regression tests for stale shaping exploit                         |
| `tools/inferno_rl/tests/test_sweep_sampling.py` | Add regression tests for sweep warmup / weighting behavior            |
| `tools/inferno_rl/docs/V42_TB_TRACKING.md`      | This file                                                              |

## Metrics Log

| Ckpt | Steps | Fail% | EV | Entropy | KL | Grad | Ep Len | FPS | Worst Wave | Notes |
|------|-------|-------|----|---------|----|------|--------|-----|------------|-------|
| ~10475 | 21.5M | 100% | 0.98 | 0.050 | 0.001 | 0.42 | 68 | 1149 | W40 | Fresh resume from V41 ckpt-9900; frontier=1; return_mean 1.72 (below V41 healthy 2.34); progress dies at ~W40 (W40+→stuck on start wave); W1-39 clears +3-5 waves past start; no exploit signals, no deaths |

## Reward-Term Watchlist

Track these in TensorBoard (`raw_reward_terms/*`) from the start:

- `Single-LOS Engagement`
- `NE Pillar Zone`
- `Stall Penalty`
- `Wave Complete`
- `Wave End HP Bonus`
- `Damage Dealt`
- `Damage Taken`

Red flags:

- `Single-LOS Engagement` or `NE Pillar Zone` rising sharply while waves completed fall
- total positive shaping growing faster than kill / completion rewards
- episode reward rising while eval behavior regresses

## Success Criteria

1. No repeat of the V41 25M-step pattern: no sharp `value_loss` jump and no runaway `return_mean`.
2. `running_reward_var` stays bounded instead of re-accelerating.
3. Sweep continues to improve coverage on early and mid waves without collapsing into easy-wave farming.
4. Raw reward terms remain proportionate to actual combat progress and wave completion.

## Failure / Stop Criteria

1. `value_loss` jumps `> 5x` from the checkpoint-9900 baseline and stays elevated.
2. `return_mean` climbs sharply while `KL` and `clip_frac` stay suppressed.
3. `running_reward_var` begins another sustained acceleration.
4. Raw reward-term logs show positive shaping dominating kill / completion terms.

## Notes

- Do not interpret low `KL` alone as proof the run is under-updating.
- In this codebase, `KL` is an actor metric; critic instability can still dominate total gradients.
- If V42 is stable but too slow, only then test a second retry with `lr=1.5e-4` or `2.5e-4` in a controlled branch.
