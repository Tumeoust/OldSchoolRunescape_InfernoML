# V43 TB Tracking

## Goal

Fresh retry from the last healthy V41 checkpoint, with both critical fixes now in place:

1. stale dense-shaping exploit removed
2. episodes persist across PPO rollouts instead of being hard-reset every `128` ticks

This is the first clean sweep continuation where TensorBoard rollout metrics should reflect real episode progression
rather than rollout-boundary resets.

## Base Checkpoint

- Base checkpoint: `models/V41_sweep/inferno_gpu_w1-66_20260312_213010_9900.pt`
- Approx step count: `~20.3M`
- Reason: last clearly healthy pre-onset V41 checkpoint

## Why V43 Exists

V42 started from the correct checkpoint and included the reward fix, but the run began before the rollout sampler bug was
fixed. That bug was severe:

- every rollout began with a hard `reset_async()` on all envs
- with `n-steps=128`, no env could continue beyond `128` ticks unless it terminated inside the slice
- sweep wave stats were therefore learned from a biased subset of starts
- deaths/timeouts/waves-completed in TB were not representing full episodes

Because the sampler semantics changed mid-run, V42 is not a clean experiment and should not be treated as one.

## What Changed (V42 -> V43)

### 1. Persistent episodes across rollouts

The sampler now carries per-env state across successive PPO rollout collections.

- no full reset at the start of every rollout
- envs only reset on real terminal conditions
- LSTM state persists with the env and only resets on terminal

Rationale:

- sweep/full-episode training only makes sense if a sampled start wave can continue until death, timeout, or inferno clear
- resetting every `128` ticks was fundamentally changing the task

### 2. TensorBoard cleanup

Removed redundant curriculum scalars:

- `rollout/curriculum_frontier_max`
- `rollout/curriculum_frontier_mean`
- `rollout/curriculum_phase_max`

These were not useful for sweep and were adding noise.

### 3. Keep the V42 retry settings

| Setting          | Value  | Notes                                  |
|------------------|--------|----------------------------------------|
| `n-epochs`       | `3`    | keep PPO de-aggressed                  |
| `lr`             | `2e-4` | do not raise LR yet                    |
| `target-kl`      | `0.02` | tighter than V41                       |
| reward fix       | on     | recent-engagement gate for dense shape |
| sweep warmup fix | on     | warmup is actually per env             |

## V43 Hypothesis

If the V41 failure was primarily:

- reward farming from stale dense shaping
- plus distorted sweep statistics from rollout-boundary resets

then V43 should:

- remain stable past the old `25M` onset window
- show deaths/timeouts that actually match full-episode sweep behavior
- produce sweep wave stats that reflect real outcomes rather than short-window censoring
- keep `running_reward_var` and `value_loss` bounded

## Training Settings

| Setting             | Value                | Notes                                        |
|---------------------|----------------------|----------------------------------------------|
| warmstart           | `V41 ckpt 9900`      | Resume from last healthy checkpoint          |
| curriculum-mode     | `static`             | Same as V41/V42                              |
| phase               | `sweep`              | Same as V41/V42                              |
| start-wave          | `1`                  | Full wave range                              |
| max-wave            | `66`                 | Full Inferno                                 |
| observation-version | `v3.2`               | Same                                         |
| policy-arch         | `flat_lstm_residual` | Same                                         |
| lstm-hidden-size    | `128`                | Same                                         |
| lstm-seq-len        | `16`                 | Same                                         |
| lstm-burn-in        | `8`                  | Same                                         |
| actor/critic sizes  | `512,512 / 512,512`  | Same                                         |
| n-envs              | `16`                 | Same                                         |
| n-steps             | `128`                | PPO rollout size only, no longer episode cap |
| batch-size          | `2048`               | Same                                         |
| n-epochs            | `3`                  | Reduced from V41                             |
| lr                  | `2e-4`               | Conservative retry                           |
| target-kl           | `0.02`               | Reduced from V41                             |
| entropy-start/end   | `0.05 / 0.002`       | Same                                         |
| gamma               | `0.995`              | Same                                         |
| gae-lambda          | `0.95`               | Same                                         |
| vf-coef             | `0.5`                | Same                                         |
| max-grad-norm       | `0.5`                | Same                                         |
| normalize-reward    | yes                  | Same                                         |
| normalize-obs       | yes                  | Same                                         |
| log-reward-terms    | yes                  | Required                                     |
| checkpoint-every    | `100`                | Same                                         |
| total budget        | `200M`               | Same                                         |

## Starting Command

```powershell
python -m tools.inferno_rl.train_gpu --load models/V41_sweep/inferno_gpu_w1-66_20260312_213010_9900.pt --curriculum-mode static --phase sweep --start-wave 1 --max-wave 66 --observation-version v3.2 --policy-arch flat_lstm_residual --lstm-hidden-size 128 --lstm-seq-len 16 --lstm-burn-in 8 --episode-mode full --n-envs 16 --n-steps 128 --batch-size 2048 --n-epochs 3 --lr 8e-4 --target-kl 0.02 --entropy-start 0.05 --entropy-end 0.002 --gamma 0.995 --gae-lambda 0.95 --vf-coef 0.5 --max-grad-norm 0.5 --normalize-obs --normalize-reward --actor-sizes 512,512 --critic-sizes 512,512 --save-dir models/V43_sweep_persistent --log-dir logs/V43_sweep_persistent --checkpoint-every 1000 --timesteps 200000000 --device cuda --log-reward-terms
```

### Continuation — 32 envs, 256 steps (from ~34.9M)

Changes from initial command:

- `--load` → V43 ckpt 7000 (~34.9M steps)
- `--n-envs 32` (was 16) — more wave coverage per rollout
- `--n-steps 256` (was 128) — longer trajectories per env
- `--batch-size 2048` (unchanged) — now 4 minibatches/epoch instead of 1
- `--n-epochs 3` (unchanged) — 12 gradient updates/rollout (was 3); acceptable given KL ≈ .0003

Rationale: low KL caused by multi-wave gradient interference. 4x more transitions per rollout
should give each wave better representation and reduce gradient cancellation.

```powershell
python -m tools.inferno_rl.train_gpu --load models/V43_sweep_persistent/inferno_gpu_w1-66_20260313_140455_7000.pt --curriculum-mode static --phase sweep --start-wave 1 --max-wave 66 --observation-version v3.2 --policy-arch flat_lstm_residual --lstm-hidden-size 128 --lstm-seq-len 16 --lstm-burn-in 8 --episode-mode full --n-envs 32 --n-steps 256 --batch-size 2048 --n-epochs 3 --lr 8e-4 --target-kl 0.02 --entropy-start 0.05 --entropy-end 0.002 --gamma 0.995 --gae-lambda 0.95 --vf-coef 0.5 --max-grad-norm 0.5 --normalize-obs --normalize-reward --actor-sizes 512,512 --critic-sizes 512,512 --save-dir models/V43_sweep_persistent --log-dir logs/V43_sweep_persistent --checkpoint-every 1000 --timesteps 200000000 --device cuda --log-reward-terms
```

### Continuation — below-normal priority (from ~45M)

Same settings as previous continuation, resumed from ckpt 1000 (~45M steps).
Process priority set to below-normal to allow concurrent CPU use.

```powershell
Start-Process -FilePath python -ArgumentList "-m tools.inferno_rl.train_gpu --load models/V43_sweep_persistent/inferno_gpu_w1-66_20260313_174633_1000.pt --curriculum-mode static --phase sweep --start-wave 1 --max-wave 66 --observation-version v3.2 --policy-arch flat_lstm_residual --lstm-hidden-size 128 --lstm-seq-len 16 --lstm-burn-in 8 --episode-mode full --n-envs 32 --n-steps 256 --batch-size 2048 --n-epochs 3 --lr 8e-4 --target-kl 0.02 --entropy-start 0.05 --entropy-end 0.002 --gamma 0.995 --gae-lambda 0.95 --vf-coef 0.5 --max-grad-norm 0.5 --normalize-obs --normalize-reward --actor-sizes 512,512 --critic-sizes 512,512 --save-dir models/V43_sweep_persistent --log-dir logs/V43_sweep_persistent --checkpoint-every 1000 --timesteps 200000000 --device cuda --log-reward-terms"; Get-Process python | ForEach-Object { $_.PriorityClass = 'BelowNormal' }
```

## Files Changed Since V41

| File                                                         | Changes                                                                |
|--------------------------------------------------------------|------------------------------------------------------------------------|
| `tools/inferno_rl/training/rewards.py`                       | Gate dense shaping behind recent engagement, reduce `Single-LOS` bonus |
| `tools/inferno_rl/training/env.py`                           | Fix sweep warmup to be per-env                                         |
| `tools/inferno_rl/rollout_sampler.py`                        | Persist episodes; filter full-buffer envs; drain pending at boundary   |
| `tools/inferno_rl/train_gpu.py`                              | Remove redundant frontier/phase TB scalars                             |
| `tools/inferno_rl/tests/test_reward_shaping.py`              | Reward exploit regression tests                                        |
| `tools/inferno_rl/tests/test_sweep_sampling.py`              | Sweep warmup regression tests                                          |
| `tools/inferno_rl/tests/test_rollout_sampler_persistence.py` | Rollout persistence regression test                                    |
| `tools/inferno_rl/docs/V43_TB_TRACKING.md`                   | This file                                                              |

## Metrics Log

Each row covers a ~30-minute trailing window. **Waves/Ep** = waves cleared per episode, averaged
across all sweep starts — the uniform progress metric regardless of start wave.

| Steps                                                                       | Eps | Deaths | Waves/Ep | EV   | KL     | VL    | Clip  | Ent   | Return | RVar  | Notes                                                                                      |
|-----------------------------------------------------------------------------|-----|--------|----------|------|--------|-------|-------|-------|--------|-------|--------------------------------------------------------------------------------------------|
| 20.71M                                                                      | 37  | 34     | 12.9     | .942 | .00100 | .0505 | .0021 | 2.870 | 2.36   | 20.11 | Baseline post-fix; KL low, returns stable                                                  |
| 22.2M                                                                       | 1   | 1      | 17.0     | .962 | .00060 | .0529 | .0003 | 2.759 | 2.80   | 19.67 | KL still dropping (.001→.0006); policy barely updating                                     |
| **New run — LR 8e-4 (4x), restart from same V41 ckpt**                      |     |        |          |      |        |       |       |       |        |       |
| 20.7M                                                                       | 2   | 2      | 9.0      | .938 | .00360 | .0739 | .0441 | 2.578 | 2.40   | 20.09 | KL 6x higher than 2e-4 (LR effect); VL initially elevated                                  |
| 22.3M                                                                       | 0   | 0      | 18.0     | .971 | .00170 | .0384 | .0124 | 2.735 | 1.91   | 19.53 | VL recovered well (.074→.038); KL healthy at .0017; wave perf similar to 2e-4              |
| 28.6M                                                                       | 0   | 0      | 10.0     | .968 | .00160 | .0327 | .0130 | 2.728 | 2.08   | 19.85 | Past 25M onset — stable; KL/Clip/VL all holding; no divergence                             |
| 30.7M                                                                       | 2   | 2      | 21.0     | .975 | .00050 | .0446 | .0031 | 2.793 | 2.89   | 19.38 | KL collapsed again (.0016→.0005); Clip dropped (.013→.003); VL up; return recovering       |
| 32.8M                                                                       | 0   | 0      | 14.0     | .942 | .00110 | .0419 | .0098 | 2.749 | 2.37   | 19.05 | KL rebounded (.0005→.0011); Clip recovered (.003→.010); grad_norm hitting max (.504)       |
| 34.9M                                                                       | 4   | 4      | 18.0     | .933 | .00030 | .0874 | .0020 | 2.746 | 2.31   | 18.72 | VL spiked 2x (.042→.087); KL collapsed (.0003); grad_norm 1.19 (2.4x max)                  |
| **Continuation — 32 envs, 256 steps (4x rollout), 12 grad updates/rollout** |     |        |          |      |        |       |       |       |        |       |
| 36.4M                                                                       | 6   | 6      | 65.0     | .953 | .00380 | .0578 | .0367 | 2.867 | 2.60   | 19.07 | KL 12x up (.0003→.0038); Clip 18x up (.002→.037); 12 updates/rollout working; FPS 2199     |
| 39.9M                                                                       | 2   | 2      | 58.0     | .935 | .00380 | .0474 | .0353 | 2.832 | 2.62   | 19.63 | KL/Clip holding steady (.0038/.035); VL down (.058→.047); grad_norm 1.04 (clipping active) |
| 43.3M                                                                       | 2   | 2      | 64.0     | .959 | .00470 | .0367 | .0389 | 2.828 | 2.58   | 19.93 | KL up (.0038→.0047); VL still improving (.047→.037); first early_stop_kl seen (.0215)      |
| 45.0M                                                                       | 6   | 6      | 53.0     | .917 | .00390 | .0548 | .0359 | 2.809 | 2.40   | 19.67 | KL eased (.0047→.0039); EV dropped (.959→.917); VL up (.037→.055); grad_norm .90; FPS 1896 |
| 48.5M                                                                       | 3   | 3      | 20.0     | .917 | .00420 | .0567 | .0362 | 2.869 | 2.55   | 20.27 | EV flat at .917 (not recovering); RVar slowly climbing (19.67→20.27); grad_norm 0.98 (clipping); stall_penalty elevated; waves/ep 20 (noisy, 3-ep window); max_wave_from_1=45 |
| 48.7M                                                                       | 2   | 2      | 28.0     | .951 | .00580 | .0345 | .0381 | 2.811 | 2.38   | 20.28 | EV recovered (.917→.951); VL dropped (.057→.034); KL up (.004→.006, early_stop_kl triggered mid-window at .025); only 238K new steps (below-normal priority heavy throttling) |
| 50.3M                                                                       | 2   | 2      | 31.0     | .921 | .00290 | .0448 | .0229 | 2.878 | 2.64   | 20.34 | grad_norm spike to 1.44 (2.9x max); ep_len jumped to 189 ticks; EV bouncing (.951→.921); KL eased (.006→.003); FPS recovered to 2039 |
| 53.4M                                                                       | 3   | 3      | 20.0     | .932 | .00560 | .0421 | .0430 | 2.781 | 2.66   | 20.56 | grad_norm recovered (1.44→0.98, spike transient); EV slowly improving (.921→.932); RVar creeping (20.34→20.56); KL/Clip up (.003→.006/.023→.043) |
| 55.7M                                                                       | 2   | 2      | 29.0     | .962 | .00360 | .0397 | .0315 | 2.802 | 2.67   | 20.67 | EV best yet in 32-env run (.932→.962); VL dropped to .040; FPS crashed to 54 (rollout_time=150s, system under load); early_stop_kl mid-window |
| 55.8M                                                                       | 6   | 6      | 10.5     | .932 | .00230 | .0682 | .0160 | 2.773 | 2.54   | 20.67 | grad_norm spiked to 2.40 (4.8x max); VL spiked .040→.068; EV dropped .962→.932; only 74K steps (FPS=39, rollout_time=212s, system severely throttled) |
| 58.7M                                                                       | 6   | 6      | 10.3     | .957 | .00460 | .0489 | .0426 | 2.814 | 2.57   | 20.83 | Recovery from throttling: FPS 2208, VL .068→.049, EV .932→.957; grad_norm still elevated (1.25); RVar slow creep (20.67→20.83); early_stop_kl mid-window |
| 62.0M                                                                       | 4   | 4      | 15.0     | .945 | .00350 | .0532 | .0293 | 2.792 | 2.53   | 21.07 | RVar crossed 21 for first time (rate accelerating); grad_norm fully normalized (.95); VL slight uptick (.049→.053); EV .957→.945; return stable |
| 65.4M                                                                       | 4   | 4      | 14.3     | .934 | .00510 | .0593 | .0430 | 2.693 | 2.55   | 21.22 | Slow deterioration pattern: RVar climbing (21.07→21.22), VL climbing (.053→.059), EV declining (.945→.934), grad_norm re-elevating (.95→1.11); return stable so far |
| 68.8M                                                                       | 3   | 3      | 21.3     | .930 | .00450 | .0495 | .0223 | 2.725 | 2.68   | 21.36 | Return_mean broke plateau (2.55→2.68, first significant move); grad_norm re-spiked (1.11→1.45); RVar continuing climb (21.22→21.36); VL eased (.059→.050); EV .930 |
| 72.2M                                                                       | 4   | 4      | 15.5     | .946 | .00460 | .0445 | .0337 | 2.727 | 2.71   | 21.41 | EV bounced (.930→.946); VL improved (.050→.044); return_mean 3rd consecutive rise (2.68→2.71); RVar rate slowed (+0.05 vs +0.14 prev); early_stop_kl mid-window; ep_len dropped 147→70 |
| 75.6M                                                                       | 2   | 2      | 34.5     | .958 | .00370 | .0331 | .0329 | 2.676 | 2.85   | 21.53 | EV/VL best yet (.958/.033) — value fit strong; return_mean new high (2.71→2.85, 4th rise); RVar 21.53 (+0.12); grad_norm 1.42; Mager_Priority elevated at 0.013/tick |
| 79.0M                                                                       | 7   | 6      | 9.3      | .928 | .00370 | .0648 | .0313 | 2.732 | 2.66   | 21.62 | EV/VL sharply reversed (.958→.928; .033→.065); long episodes back (189 ticks); return_mean down (2.85→2.66); RVar still climbing (21.62); phase_failure_rate 0.857 |
| 82.4M                                                                       | 5   | 5      | 10.6     | .934 | .00190 | .0707 | .0146 | 2.662 | 2.57   | 21.89 | RVar accelerating (+0.27, was +0.09); VL new high (.071); grad_norm 1.72; KL near-zero (.002) — policy barely moving; clip_fraction .015; smooth running_reward still rising (6.15) |
| 85.8M                                                                       | 1   | 1      | —        | .951 | .00410 | .0599 | .0204 | 2.712 | 2.92   | 21.97 | Recovery: EV .934→.951, KL .002→.004, VL .071→.060, grad_norm 1.72→1.30; early_stop_kl triggered; RVar 21.97 (+0.08); 1-ep window (anomalous waves_completed=68) |
| 86.1M                                                                       | 2   | 2      | 29       | .934 | .00320 | .0404 | .0247 | 2.67  | 2.91   | 21.98 | VL recovered (.060→.040, best since 75.6M); RVar nearly flat (+0.01, was +0.08); EV oscillating (.951→.934); return_mean holding at recent high (2.91); phase_failure_rate 1.0; ep_len 103; FPS 1903 |
| 89.1M                                                                       | 3   | 3      | 20.7     | .933 | .00320 | .0531 | .0218 | 2.57  | 2.88   | 22.10 | VL rebounded (.040→.053, ep_len 103→171, LSTM cycle); RVar crossed 22 (+0.12, resuming climb); max_wave_from_35=66 and max_wave_from_61=66 (inferno clears!); return_mean slightly down (2.91→2.88) |
| 92.4M                                                                       | 3   | 2      | 21       | .914 | .00320 | .0759 | .0191 | 2.55  | 3.05   | 22.20 | ⚠ VL near alarm (.053→.076); EV crashed to .914 (worst since 45M); grad_norm spiked 1.02→2.18; return_mean broke 3.0 barrier (new high); early_stop_kl .021; RVar +0.10; max_wave_from_64=66 |
| 95.7M                                                                       | 4   | 4      | 16.3     | .904 | .00290 | .0658 | .0264 | 2.67  | 2.98   | 22.29 | Partial stabilization: VL .076→.066, return_mean 3.05→2.98, grad_norm 2.18→1.81; BUT EV .914→.904 (worst in V43); ep_len 175; RVar +0.09; early_stop_kl .024 |
| 99.1M                                                                       | 1   | 1      | 61       | .947 | .00400 | .0491 | .0278 | 2.53  | 2.92   | 22.37 | Strong recovery: EV .904→.947, VL .066→.049, grad_norm 1.81→0.93 (short ep 104); LSTM oscillation cycle confirmed; max_wave_from_3=66, _45=66, _56=66 (3 inferno clears); RVar +0.08 |
| 102.3M                                                                      | 2   | 2      | 32       | .941 | .00300 | .0509 | .0222 | 2.57  | 2.93   | 22.45 | Mid-oscillation: ep_len 104→158, EV .947→.941, VL .049→.051; grad_norm 0.93→1.36; return_mean stable (2.93); RVar +0.08; max_wave_from_60=66; early_stop_kl .023 |
| 105.6M                                                                      | 4   | 4      | 15.5     | .947 | .00620 | .0481 | .0476 | 2.65  | 2.87   | 23.23 | ⚠ RVar jumped +0.78 (22.45→23.23, largest single-read increase); KL .003→.006, clip .022→.048 (big policy moves); EV held .947, VL .048; return_mean 2.87; max_wave_from_65=66; grad_norm normal (1.03) |
| 108.5M                                                                      | 1   | 1      | 74       | .971 | .00180 | .0331 | .0155 | 2.55  | 3.26   | 23.25 | New V43 highs: EV .971, VL .033, return_mean 3.26; 74 waves/ep; RVar spike was transient (23.23→23.25); KL .006→.002; grad_norm 1.94; FPS 1621 (hard single ep) |
| 110.6M                                                                      | 10  | 10     | 6.7      | .894 | .00390 | .0789 | .0307 | 2.56  | 2.66   | 23.30 | ⚠⚠ EV new V43 low (.894); VL near alarm (.079); early_stop_kl .081 (4× target, unprecedented); return_mean collapsed 3.26→2.66; 10 short eps (min 35t, one neg return); RVar barely moved (+0.05) |
| 113.4M                                                                      | 4   | 4      | 16       | .938 | .00380 | .0684 | .0148 | 2.49  | 2.94   | 23.41 | Recovering from alarm: EV .894→.938, VL .079→.068, return_mean 2.66→2.94; early_stop_kl .020 (normal); FPS 1178 (rollout_time 6.95s, harder waves); max_wave_from_65=66; RVar +0.11 |
| 116.2M                                                                      | 1   | 1      | 65       | .920 | .00170 | .0408 | .0141 | 2.48  | 2.92   | 23.48 | VL recovered .068→.041; 65 waves/ep (207 ticks); grad_norm 2.61 (pre-clip, new V43 high; KL=.002 so effective update small); max_wave_from_50=66, _61=66; FPS 1810; RVar +0.07 |
| 119.2M                                                                      | 1   | 1      | 69       | .927 | .00310 | .0476 | .0177 | 2.47  | 3.18   | 23.52 | return_mean back above 3.0 (2.92→3.18); EV recovering (.920→.927); grad_norm normalized (2.61→1.22); 69 waves/ep; Early_Mager_Kill ep_sum 17.4 (strongest); RVar +0.04; max_wave_from_59=66 |
| 121.9M                                                                      | 1   | 1      | 64       | .959 | .00290 | .0370 | .0258 | 2.51  | 2.88   | 23.58 | EV .927→.959 (near V43 peak); VL .037 (very healthy); 7 inferno clears (from_10,19,36,48,57,62,64=66); return_mean retreated (3.18→2.88); running_reward_mean 6.90; RVar +0.06 |
| 124.8M                                                                      | 2   | 2      | 26.5     | .884 | .00430 | .0504 | .0328 | 2.36  | 2.89   | 23.69 | EV new V43 low (.959→.884); VL .050 (manageable); min_ep_reward -1.0 (one near-total-loss ep); 3 inferno clears (from_38,50,64=66); grad_norm 0.95 (normal); RVar +0.11; early_stop_kl .020 |
| 127.5M                                                                      | 2   | 2      | 27.5     | .916 | .00280 | .0475 | .0256 | 2.43  | 2.91   | 23.72 | EV recovering from trough (.884→.916); 4 inferno clears (from_2,33,63,64=66); return_mean stable (2.91); RVar nearly flat (+0.03); grad_norm 1.67 (clipping); ep_len 29-252 (mixed) |
| 130.2M                                                                      | 2   | 0      | 40.5     | .970 | .00220 | .0275 | .0184 | 2.37  | 3.15   | 23.78 | EV .916→.970 (matches V43 peak); VL new V43 low (.0275); return_mean 3.15 (back above 3.0); 0 deaths; 4 inferno clears (from_54,59,64,66=66); RVar +0.06 (slow) |
| 133.0M                                                                      | 4   | 3      | 18.5     | .952 | .00550 | .0571 | .0266 | 2.41  | 3.25   | 23.84 | EV peak fading (.970→.952); VL doubled (.028→.057, long eps 201t — trough incoming); return_mean 3.25 (V43-high-tier, matches 108.5M peak); KL 2.5x jump; 2 inferno clears (from_60,65=66); RVar +0.06 |
| 136.1M                                                                      | 3   | 2      | 21.0     | .898 | .00420 | .0555 | .0251 | 2.34  | 3.04   | 23.94 | ⚠ EV crashed .952→.898 (near V43 low .884); Stall_Penalty ep_sum jumped 20x (-1→-27, policy stalling on hard waves); return_mean 3.04 (still above 3.0); 4 inferno clears (from_11,13,41,50=66); RVar +0.10 |
| 139.2M                                                                      | 4   | 4      | 15.8     | .943 | .00160 | .0691 | .0124 | 2.36  | 3.25   | 24.03 | EV .898→.943 (recovering); VL spiked .055→.069 (highest since 82.4M); grad_norm 2.20 + KL near-zero (.002) — large gradients heavily clipped; return_mean 3.25; 5 inferno clears (from_51,59,61,62,65=66); RVar crossed 24 (+0.09) |
| 141.8M                                                                      | 2   | 2      | 30.0     | .954 | .00210 | .0545 | .0128 | 2.33  | 3.41   | 24.26 | ★ return_mean NEW V43 HIGH (3.41, prev 3.26); EV .943→.954 (recovery); VL eased .069→.054; RVar +0.23 (largest jump since 105.6M); grad_norm 2.52 + KL near-zero; 3 inferno clears (from_40,46,47=66); FPS 1166 (hard eps) |
| 144.6M                                                                      | 0   | 0      | —        | .956 | .00260 | .0434 | .0164 | 2.35  | 2.65   | 27.60 | ⚠ RVar jumped +3.34 (24.26→27.60, largest ever — likely artifactual with 0 complete eps this window); return_mean 2.65 (bootstrapped partial eps); EV/VL healthy (.956/.043); grad_norm normalized 2.52→1.33; 3 inferno clears (from_4,31,53,62=66) |
| 146.5M                                                                      | 0   | 0      | —        | .982 | .00040 | .3875 | .0015 | 2.38  | 1.91   | 29.65 | ⚠⚠ ALARM: VL .387 (5× prev all-time high); grad_norm 5.96 (12× max_grad_norm); Stall_Penalty -369/ep (100× jump); mean_ep_reward -8.49 (strongly negative); RVar +2.05; 0 complete eps again; loss=+0.133 (positive!); KL near-zero (.0004) — does NOT match V41 tripwire (return dropped) — watch urgently |
| 148.9M                                                                      | 3   | 1      | 20.7     | .936 | .00480 | .0295 | .0216 | 2.33  | 2.66   | 29.71 | ✅ 146.5M alarm FULLY RESOLVED: VL .387→.030 (13× drop), Stall_Penalty -369→-3.2 (fully normalized), KL .0004→.005 (policy updating again), eps resumed (0→3); RVar +0.06 (essentially flat, confirms spike was transient); grad_norm 5.96→1.33; early_stop_kl .021; 1 inferno clear (from_25=66) |
| 151.3M                                                                      | 5   | 5      | 13.2     | .930 | .00500 | .0577 | .0221 | 2.29  | 2.91   | 30.11 | Return recovered 2.66→2.91 (+0.25); VL doubled .030→.058 + grad_norm 1.33→1.96 (longer eps 110→137t — oscillation trough incoming); 2 inferno clears (from_34,55=66); no early_stop |
| 153.7M                                                                      | 3   | 3      | 23.0     | .952 | .00150 | .0427 | .0110 | 2.26  | 2.99   | 30.08 | return_mean 2.91→2.99 (just below 3.0); EV .930→.952 (recovering); KL collapsed .005→.0015 + grad_norm 2.69 (5.4× max_grad_norm) — large gradients but policy barely updating; 5 inferno clears (from_7,22,34,43,48=66); FPS 1493→1015 (harder eps, rollout 8.1s) |
| **— RUN ENDED at ~153.7M —**                                                |     |        |          |      |        |       |       |       |        |       | Training stopped manually. Best checkpoint: ckpt_12000 (~144M, 28% clear rate W55–66). V43 peak return_mean 3.41 at 141.8M. |

## Eval Benchmarks

100 seeds (seeds 0–99), stochastic mode. Command: `python -m tools.inferno_rl.death_analysis --model <ckpt> --start-wave <N> --max-wave 66 --episodes 100`

### Bench 1 — ckpt_8000 (~108.5M steps): W49–66

| Wave | Deaths | Timeouts | Cum.Survival |
|------|--------|----------|--------------|
| 49   | 1      | 0        | 99%          |
| 50   | 1      | 0        | 98%          |
| 51   | 3      | 0        | 95%          |
| 52   | ~8     | 0        | ~87%         |
| 53   | ~3     | 0        | ~84%         |
| 54   | ~4     | 1        | ~79%         |
| 55   | ~7     | 0        | ~71%         |
| 56   | ~9     | 1        | ~61%         |
| 57   | ~1     | 0        | ~60%         |
| 58   | ~2     | 0        | ~58%         |
| 59   | ~5     | 0        | ~53%         |
| 60   | ~7     | 0        | **44%**      |
| 61   | 9      | 0        | 35%          |
| 62   | **16** | 0        | **19%**      |
| 63   | ~5     | 0        | ~14%         |
| 64   | ~3     | 0        | ~11%         |
| 65   | ~3     | 0        | ~8%          |
| 66   | 0      | 0        | **8% ✓**    |

**8/100 cleared (8%).** W62 dominant kill (16 deaths, ~54% per-wave death rate for those reaching it). W49–W55 was manageable; the cliff starts at W56 and peaks at W62.

W63-W65 per-wave deaths are approximate (raw episode log for seeds 93-99 not recovered). Cumulative checkpoints (95%/44%/19%/8%) are exact from full run.

---

### Bench 2 — ckpt_12000 (~144+M steps): W55–66

| Wave | Deaths | Timeouts | Per-Wave Survival | Cum.Survival |
|------|--------|----------|-------------------|--------------|
| 55   | 4      | 0        | 96.0%             | 96%          |
| 56   | 6      | 0        | 93.8%             | 90%          |
| 57   | 2      | 0        | 97.8%             | 88%          |
| 58   | 1      | 0        | 98.9%             | 87%          |
| 59   | 6      | 0        | 93.1%             | 81%          |
| 60   | **11** | 0        | 86.4%             | 70%          |
| 61   | 8      | 0        | 88.6%             | 62%          |
| 62   | **11** | 0        | 82.3%             | 51%          |
| 63   | **13** | 1        | 72.5%             | 37%          |
| 64   | 4      | 0        | 89.2%             | 33%          |
| 65   | 2      | 2        | 87.9%             | 29%          |
| 66   | 1      | 0        | 96.6%             | **28%**      |

**28/100 cleared (28%).** Kill wave hierarchy shifted: W63 is now top killer (13 deaths + 1 timeout). W60 and W62 are tied second (11 deaths each).

---

### Bench 1 vs Bench 2 — Per-Wave Survival Comparison (W55–66 aligned)

| Wave | Bench 1 Per-Wave Survival | Bench 2 Per-Wave Survival | Delta   |
|------|---------------------------|---------------------------|---------|
| 55   | ~90%                      | 96.0%                     | +6pp    |
| 56   | ~85%                      | 93.8%                     | +9pp    |
| 57   | ~98%                      | 97.8%                     | ~0      |
| 58   | ~97%                      | 98.9%                     | +2pp    |
| 59   | ~91%                      | 93.1%                     | +2pp    |
| 60   | ~84%                      | 86.4%                     | +2pp    |
| 61   | ~80%                      | 88.6%                     | +9pp    |
| 62   | ~54%                      | 82.3%                     | **+28pp** |
| 63   | ~74%                      | 72.5%                     | -2pp    |
| 64   | ~84%                      | 89.2%                     | +5pp    |
| 65   | ~73%                      | 87.9%                     | +15pp   |
| 66   | 100%                      | 96.6%                     | -3pp    |

Bench 1 per-wave survivals are approximate (reconstructed from cumulative checkpoints). Bench 2 are exact.

**Key findings:**
- W62 per-wave survival improved dramatically (+28pp): from ~54% to 82.3%. The blob wave is no longer the dominant wall.
- W55–W61 improved consistently across the board.
- W63 is the new bottleneck (~73% per-wave, unchanged). This is now where most runs end.
- W66 (Zuk) still rarely kills: 0 deaths in Bench 1, 1 in Bench 2. Once you reach Zuk, you clear.

## Reward-Term Watchlist

Track these from the start:

- `Single-LOS Engagement`
- `NE Pillar Zone`
- `Stall Penalty`
- `Wave Complete`
- `Wave End HP Bonus`
- `Damage Dealt`
- `Damage Taken`

Red flags:

- positive shaping terms climbing while kills/completions do not
- `return_mean` rising while `KL` stays tiny
- `running_reward_var` beginning another sustained climb

## Success Criteria

1. Stable beyond the old V41 onset window (`25M+` total steps).
2. Deaths/timeouts now behave plausibly for full sweep episodes.
3. `value_loss` and `running_reward_var` stay bounded.
4. Sweep stats and reward terms remain interpretable and aligned with actual combat outcomes.

## Failure / Stop Criteria

1. Another V41-style divergence: `value_loss`, `return_mean`, and `running_reward_var` all accelerating together.
2. Reward-term logs show shaping dominating progress rewards again.
3. Full-episode sweep behavior still looks implausible after the sampler fix.

## Notes

- `n-steps=128` is now only the PPO rollout chunk size. It is no longer an implicit episode cap.
- Compare V43 against V41 only after `25M+` total steps; before that, stability is necessary but not sufficient.
