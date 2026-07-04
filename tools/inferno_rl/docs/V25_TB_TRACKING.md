# V25 TensorBoard Tracking

Coordinated overhaul targeting three structural issues that capped V21 at W60-63: curriculum (no revisitation, single-wave
starts), reward balance (rush-and-die incentives), and model capacity (371K params). All three changed simultaneously —
acceptable because the curriculum and reward changes are independent of each other and the capacity increase is necessary
to absorb the new curriculum complexity.

**Run 1** (0–21.3M): `logs/V25_climb/`
**Phase**: Climb (W49→66, promote-after 5, multi-wave episodes, revisitation every 5 episodes)
**Load**: `bc_warmstart_v4.pt` with `--resize-lstm 256` (LSTM 128→256, actor [256,128]→[256,256], ~807K params)
**Why not V21?** V21's best checkpoint (`_222731_2200`, 408M steps) has `actor_input=186`, `lstm=None` — completely
different architecture (no LSTM, 186-dim MLP-only obs with different feature layout). Zero-padding into 262-dim LSTM
would not transfer any meaningful learned behavior. BC warmstart v4 (262-dim, LSTM 128, 95% action accuracy) is the
correct starting point.
**Changes**: See "All Changes: V24 → V25" section below.
**Stopped**: ~21.3M steps. Model not killing nibblers at all — NE pillar taking damage with no response. Need initial
barrage heuristic as curriculum scaffolding (same as V21 first 140M).

**Run 2** (21.3M–~40M): `logs/V25_climb/`
**Load**: `models/V25_climb/inferno_gpu_w49-66_20260226_140831_1300.pt`
**Changes**: Re-enabled `initial_barrage_heuristic` in env.py. Tick 1: wait, tick 2: ice barrage NE-targeting nibblers,
tick 3: switch to blowpipe. Same scaffolding V21 used for first 140M steps — model learns nibbler handling from
demonstration while PPO trains everything else. Plan to disable at ~100-150M when the model has enough value estimation
to maintain the behavior independently.

**Run 3** (~40M–~120M): `logs/V25_climb/`
**Load**: Latest Run 2 checkpoint.
**Changes**: Reward rebalance targeting Damage_Dealt dominance (27.2% of budget at 40M) and invisible blood barrage
signal (2.4% episode frequency). See "Reward Rebalance: Run 2 → Run 3" section below.
**Stopped**: ~120M steps. Frontier hit ceiling (66) at ~120M. Deaths plateaued at ~5-8/rollout for 60M steps (60M–120M).
Mastery-weighted random sampling not concentrating experience on death waves. Switching to drill curriculum.

**Run 4** (~120M–~138M): `logs/V25_drill/`
**Phase**: Drill (W49→66, retry-on-failure, max 10 retries per wave)
**Load**: `models/V25_climb/inferno_gpu_w49-66_20260226_220125_5000.pt`
**Changes**: Switched from climb to drill curriculum. Each env independently climbs from start_wave to max_wave,
retrying on the death wave until it clears (or hits max-drill-retries=10, then auto-advances). After clearing max_wave,
loops back to start_wave. This is automatic hard-example mining — the model spends the most time on waves it can't solve.
No reward, observation, action, or hyperparameter changes.

**Run 5** (~138M–~165M): `logs/V25_drill/`
**Phase**: Drill (W49→66, unchanged)
**Load**: `models/V25_drill/inferno_gpu_w49-66_20260227_105413_900.pt`
**Changes**: Added **hindsight death penalty** — retroactive decaying penalties on the N ticks before each death/timeout
terminal. After rollout collection but before GAE, scans buffer for episode terminals and walks backward, injecting
`-peak * decay^k` on steps `t-1, t-2, ..., t-window`. Addresses credit assignment bottleneck: when the model dies at
tick 200, the mistake happened 30-50 ticks earlier but GAE with gamma=0.99/lambda=0.97 dilutes the death signal across
the entire episode. The hindsight penalty directly assigns blame to recent actions.
No observation, action, reward constant, or hyperparameter changes — only the hindsight shaping mechanism is new.
**Expected**: Deaths should decrease faster than Run 4 baseline as the model gets sharper credit for pre-death mistakes.
Mean reward will decrease slightly (extra negative shaping). EV may dip temporarily as value function adapts to new
reward distribution. KL spike expected for 2-3M steps.
**Kill criteria**: Deaths increasing over 10M steps, or KL sustained >0.04 for 5M+ (penalty destabilizing updates).

**Run 6** (~165M–~189M): `logs/V25_drill/`
**Phase**: Drill (W49→66, unchanged)
**Load**: `models/V25_drill/inferno_gpu_w49-66_20260227_133128_1700.pt`
**Changes**: Removed `ATTACK_ON_COOLDOWN_BONUS` (+2.0/tick, redundant with auto-attack). Blood barrage high HP threshold
`>95` → `>=99`. Hindsight death penalty carried over from Run 5.
**Stopped**: ~189M steps. Deaths flat at 3. AoC removal did not break attack incentive — no regression, but no improvement
either. Deaths plateaued at 3/rollout for 50M+ steps (138M–189M).

**Run 7** (~189M–~271M): `logs/V25_drill/`
**Phase**: Drill (W54→63, retry-on-failure, max 10 retries)
**Load**: `models/V25_drill/inferno_gpu_w49-66_20260227_181752_1100.pt`
**Changes**: Two changes targeting the 50M-step death plateau (3/rollout flat since 138M):
1. **Narrowed wave range W49-66 → W54-63**. Per-wave death analysis at 188M showed blob waves W56 (14 deaths), W62 (15),
   W63 (8) account for 50% of all deaths. W49-53 and W64-66 have near-zero deaths — wasted compute. Narrowing to W54-63
   concentrates all experience on the blob-heavy death band.
2. **Lowered LR 1e-4 → 5e-5**. Model is at refinement stage after 189M steps. Lower LR allows finer policy adjustments
   for the specific blob-handling behavior gap without overshooting.
No reward, observation, action, or other hyperparameter changes. Hindsight death penalty carried over from Run 5.
**Stopped**: ~271M steps (82M into run). **FAILED.** Three concurrent red flags:
- **Entropy collapse**: -1.45 → -0.78 over 82M steps. Policy became overly deterministic.
- **KL sustained >0.03**: 0.022 → 0.034. Policy thrashing, not converging.
- **EV degraded**: 0.84 → 0.70. Value function quality dropped.
Deaths flat→regressed (3→4). Eval: 21% clear rate (down from 26% pre-run). W56/W62 improved (14→8, 15→6) but
out-of-range waves regressed hard (W58 2→7, W61 3→9, W64-66 2→8). Net negative — model lost generalization.
Narrowed wave range caused catastrophic forgetting of excluded waves while entropy decay was too aggressive
relative to the halved LR — model converged on a deterministic policy before exploring blob alternatives.
Best checkpoint remains pre-Run 7: `_20260227_181752_1100.pt` (188M, 26% clear).

### Start Command — Run 1 (PowerShell, stopped at ~21.3M)

```powershell
python -m tools.inferno_rl.train_gpu --load models/bc_warmstart_v4.pt --resize-lstm 256 --phase climb --start-wave 49 --max-wave 66 --promote-after 5 --save-dir models/V25_climb --log-dir logs/V25_climb --n-envs 16 --n-steps 1024 --batch-size 256 --n-epochs 3 --lr 1e-4 --entropy-start 0.015 --entropy-end 0.002 --gae-lambda 0.97 --vf-coef 0.75 --max-grad-norm 0.5 --normalize-obs --normalize-reward --checkpoint-every 100 --timesteps 100000000 --device cuda --log-reward-terms
```

### Resume Command — Run 2 (PowerShell, initial barrage heuristic enabled, stopped ~40M)

```powershell
python -m tools.inferno_rl.train_gpu --load models/V25_climb/inferno_gpu_w49-66_20260226_140831_1300.pt --phase climb --start-wave 49 --max-wave 66 --promote-after 5 --save-dir models/V25_climb --log-dir logs/V25_climb --n-envs 16 --n-steps 1024 --batch-size 256 --n-epochs 3 --lr 1e-4 --entropy-start 0.015 --entropy-end 0.002 --gae-lambda 0.97 --vf-coef 0.75 --max-grad-norm 0.5 --normalize-obs --normalize-reward --checkpoint-every 100 --timesteps 100000000 --device cuda --log-reward-terms
```

### Resume Command — Run 3 (PowerShell, reward rebalance applied, stopped ~120M)

```powershell
python -m tools.inferno_rl.train_gpu --load models/V25_climb/inferno_gpu_w49-66_20260226_170317_1500.pt --phase climb --start-wave 49 --max-wave 66 --promote-after 5 --save-dir models/V25_climb --log-dir logs/V25_climb --n-envs 16 --n-steps 1024 --batch-size 256 --n-epochs 3 --lr 1e-4 --entropy-start 0.015 --entropy-end 0.002 --gae-lambda 0.97 --vf-coef 0.75 --max-grad-norm 0.5 --normalize-obs --normalize-reward --checkpoint-every 100 --timesteps 100000000 --device cuda --log-reward-terms
```

### Start Command — Run 4 (PowerShell, drill curriculum, stopped ~138M)

```powershell
python -m tools.inferno_rl.train_gpu --load models/V25_climb/inferno_gpu_w49-66_20260226_220125_5000.pt --phase drill --start-wave 49 --max-wave 66 --max-drill-retries 10 --save-dir models/V25_drill --log-dir logs/V25_drill --n-envs 16 --n-steps 1024 --batch-size 256 --n-epochs 3 --lr 1e-4 --entropy-start 0.015 --entropy-end 0.002 --gae-lambda 0.97 --vf-coef 0.75 --max-grad-norm 0.5 --normalize-obs --normalize-reward --checkpoint-every 100 --timesteps 100000000 --device cuda --log-reward-terms
```

### Resume Command — Run 5 (PowerShell, hindsight death penalty)

```powershell
python -m tools.inferno_rl.train_gpu --load models/V25_drill/inferno_gpu_w49-66_20260227_105413_900.pt --phase drill --start-wave 49 --max-wave 66 --max-drill-retries 10 --save-dir models/V25_drill --log-dir logs/V25_drill --n-envs 16 --n-steps 1024 --batch-size 256 --n-epochs 3 --lr 1e-4 --entropy-start 0.015 --entropy-end 0.002 --gae-lambda 0.97 --vf-coef 0.75 --max-grad-norm 0.5 --normalize-obs --normalize-reward --checkpoint-every 100 --timesteps 100000000 --device cuda --log-reward-terms --hindsight-death-penalty 2.0 --hindsight-death-window 10 --hindsight-death-decay 0.8
```

### Resume Command — Run 6 (PowerShell, removed AoC bonus + blood barrage threshold fix, stopped ~189M)

```powershell
python -m tools.inferno_rl.train_gpu --load models/V25_drill/inferno_gpu_w49-66_20260227_133128_1700.pt --phase drill --start-wave 49 --max-wave 66 --max-drill-retries 10 --save-dir models/V25_drill --log-dir logs/V25_drill --n-envs 16 --n-steps 1024 --batch-size 256 --n-epochs 3 --lr 1e-4 --entropy-start 0.015 --entropy-end 0.002 --gae-lambda 0.97 --vf-coef 0.75 --max-grad-norm 0.5 --normalize-obs --normalize-reward --checkpoint-every 100 --timesteps 100000000 --device cuda --log-reward-terms --hindsight-death-penalty 2.0 --hindsight-death-window 10 --hindsight-death-decay 0.8
```

### Start Command — Run 7 (PowerShell, narrowed W54-63 + lr 5e-5, FAILED ~271M)

```powershell
python -m tools.inferno_rl.train_gpu --load models/V25_drill/inferno_gpu_w49-66_20260227_181752_1100.pt --phase drill --start-wave 54 --max-wave 63 --max-drill-retries 10 --save-dir models/V25_drill --log-dir logs/V25_drill --n-envs 16 --n-steps 1024 --batch-size 256 --n-epochs 3 --lr 5e-5 --entropy-start 0.015 --entropy-end 0.002 --gae-lambda 0.97 --vf-coef 0.75 --max-grad-norm 0.5 --normalize-obs --normalize-reward --checkpoint-every 100 --timesteps 100000000 --device cuda --log-reward-terms --hindsight-death-penalty 2.0 --hindsight-death-window 10 --hindsight-death-decay 0.8
```

---

## How to Read This Document

Each snapshot records the same metrics at the same step intervals. Comparing across snapshots reveals:

1. **Is the frontier advancing?** — The primary success signal. Should climb from 49 toward 66.
2. **Are deaths trending down?** — Indicates the policy is learning survival, not just exploration.
3. **Is EV stable >0.80?** — Value function is calibrated. If EV drops below 0.70, something is wrong.
4. **Is entropy collapsing?** — entropy_loss approaching 0 = policy becoming deterministic too early. Should decay slowly
   with the entropy coefficient schedule, not crash.
5. **Is mean_reward trending up?** — Noisy but should show upward movement over 5-10M steps.
6. **Is KL stable <0.02?** — Measures how much the policy changes per update. Healthy range is 0.005–0.02. Sustained
   > 0.03 means updates are too aggressive (policy thrashing). Spikes after reward changes are expected but should settle
   within 1-2M steps.
7. **Is grad_norm stable ~1.0–1.6?** — Gradient magnitude before clipping (max-grad-norm=0.5). Stable values mean the
   loss landscape is smooth. Sudden spikes (>3.0) or drops (<0.5) indicate the optimizer is struggling.
8. **Is FPS holding?** — Rollout collection speed. Drops indicate env bottlenecks.

### Early Warning Signs (kill the run)

- Entropy loss goes to 0 or positive → entropy collapse, policy is deterministic
- EV drops below 0.60 and doesn't recover within 2M steps → value function diverging
- Deaths increase monotonically over 5M+ steps → reward signal is counterproductive
- Frontier hasn't moved from W49 after 10M steps → agent isn't learning to clear waves at all
- KL sustained above 0.03 for 5M+ steps → policy is thrashing, not converging
- Grad norm spikes above 3.0 repeatedly → loss landscape unstable, check reward scaling

### V25-Specific Diagnostics

- **Multi-wave episode impact**: Watch `rollout/waves_completed` — should be higher than V24 at equivalent steps since
  each episode now spans up to 4 waves. Deaths per rollout may initially be higher (harder episodes).
- **Revisitation working**: Every 5th episode should be a refresh (different start wave). If frontier stalls but deaths
  are low, revisitation is consuming too many episodes — can increase to every 7 or 10.
- **LSTM capacity utilization**: After resize, first 5M steps will have high KL as the policy adjusts to the new
  dimensions. KL >0.04 is expected during this warmup. Should settle below 0.02 by 10M.
- **Survival reward balance**: With SURVIVAL_REWARD_PER_TICK=0.3, check that the model isn't learning to hide at 0-LOS.
  `raw_reward_terms/ep_mean_per_tick_mean/Survival_HP` should not dominate
  `raw_reward_terms/ep_mean_per_tick_mean/Single-LOS_Engagement`.
- **Single-LOS 3-tick threshold**: `raw_reward_terms/ep_sum_mean/Single-LOS_Engagement` will be lower than V24 initially
  (harder to earn). Should recover as the model learns sustained positioning.
- **Drill phase progress** (Run 4+): `rollout/drill_wave_mean` shows average drill wave across envs — should cluster
  around hard waves (63/65). `drill_wave_min`/`drill_wave_max` shows env spread. `drill_cycles` counts full clears
  (49→66). If `drill_wave_mean` is stuck at one value for 10M+ steps, the retry cap is too high or the model can't
  learn that wave composition at all.

---

## Manual Observations & Progress

| Steps | Phase | Frontier | f.mean | Mastery% | Deaths | Waves | Timeouts | MeanRwd | EV   | Entropy | KL    | Grad | FPS  | Note                                                                                                                                                        |
|-------|-------|----------|--------|----------|--------|-------|----------|---------|------|---------|-------|------|------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 18.9M | 1     | 55       | 54.4   | —        | 37     | 115   | 0        | 4.43    | 0.76 | -2.20   | 0.024 | 1.80 | 1785 | First snapshot. Frontier 49→55 from BC warmstart. KL 0.024 (expected — LSTM resize, fresh optimizer). EV 0.76 warming up. Reward positive, 0 timeouts.      |
|       |       |          |        |          |        |       |          |         |      |         |       |      |      | **--- Run 2: re-enabled initial barrage heuristic (resume from _1300.pt) ---**                                                                              |
| 39.5M | 1     | 56       | 55.1   | —        | 29     | 110   | 0        | 8.71    | 0.79 | -1.72   | 0.029 | 1.85 | 2317 | Run 2 +18M. Frontier 55→56. Deaths 37→29. MeanRwd 4.4→8.7. **KL 0.029 (watch)** — still drifting toward 0.03. EV 0.79 (improving). FPS recovered 1785→2317. |
|       |       |          |        |          |        |       |          |         |      |         |       |      |      | **--- Run 3: reward rebalance (halve damage_dealt, double att_cooldown/mager_priority/blood_barrage) ---**                                                  |
| 49.4M | 1     | 56       | 55.2   | —        | 24     | 105   | 0        | 8.09    | 0.79 | -1.88   | 0.028 | 1.59 | 2490 | Run 3 +9.9M post-rebalance. Deaths 29→24. Waves 110→105. MeanRwd 8.7→8.1 (expected — halved damage reward). **KL 0.028 (watch)** — settling but still near threshold. EV 0.79 stable. Frontier holding at 56. |
| 120.1M | 1     | 66       | 65.1   | 76.6%    | 8      | 97    | 0        | 11.20   | 0.73 | -1.04   | 0.024 | 1.82 | 2315 | **Frontier 56→66 — hit ceiling.** Deaths 24→8. MeanRwd 8.1→11.2. KL 0.024 (settled below 0.03). Entropy -1.88→-1.04 — compressing fast, monitor for collapse. EV 0.79→0.73 (watch). |

*Climb phase ended at 120M. Run 4+ uses drill curriculum — see Drill Phase table below.*

---

## Drill Phase Observations (Run 4+)

Drill curriculum resets frontier/mastery metrics — they are not meaningful during drill. Track drill-specific signals here.

| Steps | Drill Cycles | Wave Mean | Wave Min | Wave Max | Deaths | Waves | MeanRwd | EV   | Entropy | KL    | Grad | FPS  | Note |
|-------|-------------|-----------|----------|----------|--------|-------|---------|------|---------|-------|------|------|------|
| 138.1M | 54         | 52.8      | 49       | 62       | 3      | 106   | 12.06   | 0.73 | -1.50   | 0.022 | 1.44 | 2332 | Drill +18M. 54 full clears, deaths 8→3, MeanRwd 11.2→12.1. wave_mean=52.8 (below midpoint 57.5 — partly reset-cycle artifact, some early-wave deaths). Entropy recovered -1.04→-1.50. |
|        |            |           |          |          |        |       |         |      |         |       |      |      | **--- Run 5: hindsight death penalty (peak=2.0, window=10, decay=0.8) ---** |
| 149.2M | 30         | 53.7      | 49       | 65       | 3      | 110   | 0.13    | 0.78 | -1.46   | 0.017 | 1.39 | 2339 | Run 5 +11M. Deaths flat at 3. Waves 106→110. EV 0.73→0.78 (improved). KL 0.022→0.017 (no destabilization from hindsight penalty). MeanRwd 12.06→0.13 (new run dir = reward normalizer reset, not comparable). Drill cycles 54→30 (same — new run restarted counter). wave_max 62→65. |
| 157.5M | 72         | 52.8      | 49       | 63       | 3      | 107   | 8.17    | 0.72 | -1.47   | 0.026 | 1.62 | 2243 | Run 5 +19M. Deaths still flat at 3. **KL 0.017→0.026 (watch)** — approaching 0.03. EV 0.78→0.72 (dipping, value fn adapting to hindsight penalty shape). Grad 1.39→1.62. MeanRwd recovered to 8.17 (normalizer stabilized). 72 drill cycles. |
| 164.4M | 108        | 53.3      | 49       | 61       | 3      | 103   | 9.39    | 0.77 | -1.47   | 0.019 | 1.55 | 2242 | Run 5 +26M. Deaths flat at 3. KL 0.026→0.019 (settled, no longer near threshold). EV 0.72→0.77 (recovered). MeanRwd 8.17→9.39. 108 drill cycles (+36). wave_max dropped 63→61 — likely sampling noise. Stable run, no warnings. |
|        |            |           |          |          |        |       |         |      |         |       |      |      | **--- Run 6: removed AoC bonus + blood barrage threshold >=99 (resume from _1700.pt) ---** |
| 189.3M | 86         | 52.8      | 49       | 64       | 3      | 109   | 9.85    | 0.75 | -1.45   | 0.022 | 1.48 | 2354 | Run 6 +25M. Deaths flat at 3. Waves 103→109. MeanRwd 9.39→9.85. KL 0.019→0.022 (mild increase, well within range). EV 0.77→0.75 (minor dip, stable). wave_max 61→64. Drill cycles 86 (new run dir reset counter). AoC removal did not break attack incentive — no regression. |
|        |            |           |          |          |        |       |         |      |         |       |      |      | **--- Run 7: narrowed W54-63 + lr 1e-4→5e-5 (resume from _1100.pt @ 188M) ---** |
| 193.3M | 41         | 54.7      | 54       | 59       | 3      | 101   | 7.04    | 0.84 | -1.45   | 0.022 | 1.66 | 2399 | Run 7 +4M. Early snapshot. Drill range narrowed to 54-63. Deaths still 3. EV jumped 0.75→0.84. KL stable 0.022. No red flags yet. |
| 271.0M | 737        | 54.9      | 54       | 62       | 4      | 109   | 11.18   | 0.70 | -0.78   | 0.034 | 1.74 | 2726 | **Run 7 FINAL — FAILED.** +82M. Deaths 3→4 (regressed). **Entropy collapse -1.45→-0.78.** **KL 0.022→0.034 (>0.03 sustained).** EV 0.84→0.70 (degraded). Narrowed range + halved LR caused overfitting + premature policy determinism. Killed. |

### Per-Wave Death Analysis — Pre-Run 7 (188M) vs Post-Run 7 (271M)

100 episodes each, seeds 0-99, W49-66.

| Wave | Deaths @188M | Deaths @271M | Change |
|------|-------------|-------------|--------|
| 49   | 3           | 0           | -3     |
| 50   | 4           | 4           | =      |
| 51   | 1           | 2           | +1     |
| 52   | 2           | 1           | -1     |
| 53   | 2           | 2           | =      |
| 54   | 7           | 6           | -1     |
| 55   | 5           | 5           | =      |
| **56** | **14**   | **8**       | **-6** |
| 57   | 2           | 4           | +2     |
| **58** | **2**    | **7**       | **+5** |
| 59   | 2           | 4           | +2     |
| 60   | 2           | 4           | +2     |
| **61** | **3**    | **9**       | **+6** |
| **62** | **15**   | **6**       | **-9** |
| 63   | 8           | 9           | +1     |
| 64   | 0           | 2           | +2     |
| 65   | 1           | 3           | +2     |
| 66   | 1           | 3           | +2     |
| **Cleared** | **26** | **21**  | **-5** |

**In-range waves (W54-63) improved**: W56 14→8, W62 15→6. Drill concentrated experience on these.
**Out-of-range waves regressed**: W58 2→7, W61 3→9, W64-66 2→8. Model forgot how to handle compositions
it stopped seeing. Deaths spread evenly across all waves instead of concentrating on blob waves — classic
entropy collapse behavior (deterministic policy plays confidently but lacks adaptive flexibility).

**Conclusion**: Pre-Run 7 checkpoint (188M, `_1100.pt`) is the better model. Run 7 traded generalization
for marginal in-range improvement — net negative.

### V21 vs V25 Comparison (100 episodes, seeds 0-99, W49-66)

Cross-architecture comparison to benchmark V25 (262-dim LSTM, 807K params) against V21 (186-dim MLP-only,
371K params). V21 checkpoints evaluated using `death_analysis_v21.py` with the original 186-dim obs builder.

| Model | Steps | Clear Rate | W49-53 Deaths | W54-66 Deaths |
|-------|-------|-----------|---------------|---------------|
| **V21 @ 136M** | 136M | **58%** | 42 | **0** |
| **V21 @ 484M** | 484M | **56%** | 44 | **0** |
| V21 @ 408M | 408M | 32% | 67 | 1 |
| V25 @ 188M (best) | 188M | 26% | 14 | 60 |
| V25 @ 271M (Run 7) | 271M | 21% | 15 | 64 |

**Key findings:**
1. **V21 completely solved W55-66** — zero deaths across all three checkpoints (136M, 408M, 484M). The
   smaller MLP-only network learned W55-66 combat perfectly with fewer parameters and less training.
2. **V25 bleeds out on W54-66** — 60 deaths spread across these waves, with blob waves W56/W62/W63 as the
   worst offenders. The larger LSTM network has not learned what the smaller MLP learned.
3. **V21's only weakness is W49-53** — waves it was never trained on (trained W55-66 only). All 42-44 deaths
   fall in this range.
4. **V25 is better at W49-53** — only 14 deaths (trained W49-66). But this advantage is wiped out by its
   poor W54-66 survival.
5. **V21 @ 408M regressed** — 32% clear vs 58% at 136M and 56% at 484M. Mid-training regression phase,
   likely from reward changes documented in V21 TB tracking.

**Implications for V25:**
- The 262-dim obs with pillar-relative features and LSTM may be adding complexity without proportional
  benefit for combat waves. V21's simpler 186-dim obs was sufficient to solve W55-66.
- V25 has trained ~190M steps on W49-66 without reaching V21's W55-66 performance, which V21 achieved in
  ~136M steps on the same waves. The architectural changes may be net negative for combat learning.
- A productive path forward may be to return to V21's proven architecture with expanded wave range (W49-66),
  or to identify what specific V25 obs/reward changes degraded combat wave performance.

### Per-Wave Death Analysis (188M, 100 episodes, seeds 0-99, W49-66)

| Wave | Deaths | Survival | Cum.Survival |
|------|--------|----------|--------------|
| 49   | 3      | 97.0%    | 97.0%        |
| 50   | 4      | 95.9%    | 93.0%        |
| 51   | 1      | 98.9%    | 92.0%        |
| 52   | 2      | 97.8%    | 90.0%        |
| 53   | 2      | 97.8%    | 88.0%        |
| 54   | 7      | 92.0%    | 81.0%        |
| 55   | 5      | 93.8%    | 76.0%        |
| **56** | **14** | **81.6%** | **62.0%** |
| 57   | 2      | 96.8%    | 60.0%        |
| 58   | 2      | 96.7%    | 58.0%        |
| 59   | 2      | 96.6%    | 56.0%        |
| 60   | 2      | 96.4%    | 54.0%        |
| 61   | 3      | 94.4%    | 51.0%        |
| **62** | **15** | **70.6%** | **36.0%** |
| **63** | **8** | **77.8%** | **28.0%** |
| 64   | 0      | 100.0%   | 28.0%        |
| 65   | 1      | 96.4%    | 27.0%        |
| 66   | 1      | 96.3%    | 26.0%        |

**Cleared: 26/100 (26.0%). Died: 74/100 (74.0%).**
Blob waves W56 (14), W62 (15), W63 (8) = 37/74 deaths (50%). W57-61 (harder non-blob) only 2-3 each.

### Per-Start-Wave Performance (138.1M)

| Start Wave | Best Wave Reached |
|------------|-------------------|
| 49         | 56                |
| 50         | 54                |
| 51         | 56                |
| 52-53      | 59                |
| 54-55      | 60                |
| 56         | 63                |
| 57         | 61                |
| 58         | 62                |
| 59         | 59                |
| 60+        | 66                |

---

## All Changes: V24 → V25

Three coordinated changes. No observation or action space changes (still 262-dim obs, 43 actions).

### 1. Curriculum Overhaul (env.py)

| Setting             | V24                      | V25                             | Rationale                                           |
|---------------------|--------------------------|---------------------------------|-----------------------------------------------------|
| Climb episode start | `frontier` (single wave) | `max(start_wave, frontier - 3)` | Multi-wave episodes force survival across sequences |
| Climb refresh_every | 0 (disabled)             | 5 (every 5th episode)           | Prevents catastrophic forgetting of earlier waves   |
| Mastery weights     | W55-65, W63=6 (26%)      | W49-66, W63=5 (11%)             | Flatter distribution, full training range coverage  |
| max-wave            | 65                       | 66                              | Mastery weights now include W66                     |

### 2. Reward Rebalance (rewards.py)

| Setting                  | V24                 | V25                 | Rationale                                               |
|--------------------------|---------------------|---------------------|---------------------------------------------------------|
| WAVE_STALL_ESCALATION    | 3.0                 | 1.5                 | Halved time pressure — patient repositioning viable     |
| WAVE_TIMEOUT_PENALTY     | -150.0              | -120.0              | Equalized with death — dying is never reward-optimal    |
| SURVIVAL_REWARD_PER_TICK | 0.1                 | 0.3                 | 3x survival incentive, competes with engagement signals |
| Single-LOS threshold     | 2 consecutive ticks | 3 consecutive ticks | Requires sustained safe positioning, not oscillation    |

**Reward balance at tick 400 (full HP):**

- V24: time pressure = -(0.1 + 3.0 × 0.8) = -2.5/tick, survival = 0.1/tick, net = -2.4/tick
- V25: time pressure = -(0.1 + 1.5 × 0.8) = -1.3/tick, survival = 0.3/tick, net = -1.0/tick
- Single-LOS engagement (+2.5) now clearly net-positive even at tick 400

### 3. Model Capacity (train_gpu.py, ppo.py)

| Component   | V24        | V25        | Params            |
|-------------|------------|------------|-------------------|
| LSTM hidden | 128        | 256        | ~200K → ~530K     |
| Actor MLP   | [256, 128] | [256, 256] | ~72K → ~131K      |
| Critic MLP  | [256, 256] | [256, 256] | Unchanged (~131K) |
| **Total**   | **~371K**  | **~792K**  | **2.1x increase** |

Weight migration: `PPO.load_with_resize()` zero-pads old weights into new dimensions. Optimizer state discarded
(Adam moments have old dimensions). First few M steps will show elevated KL as optimizer rebuilds momentum.

### Hyperparameters (unchanged from V24 Run 3)

| Param            | Value |
|------------------|-------|
| n-envs           | 16    |
| n-steps          | 1024  |
| batch-size       | 256   |
| n-epochs         | 3     |
| lr               | 1e-4  |
| entropy-start    | 0.015 |
| entropy-end      | 0.002 |
| gae-lambda       | 0.97  |
| vf-coef          | 0.75  |
| max-grad-norm    | 0.5   |
| normalize-obs    | yes   |
| normalize-reward | yes   |
| promote-after    | 5     |
| lstm-seq-len     | 32    |

---

## Reward Rebalance: Run 2 → Run 3

Mid-training reward adjustment at ~40M steps, informed by V25 reward analysis (`docs/V25_REWARD_ANALYSIS_40M.md`).
Targets two critical issues: Damage_Dealt dominance (27.2% of budget, 3x next term) and invisible blood barrage signal
(2.4% episode frequency). No observation, action, or hyperparameter changes — only 4 reward constants modified.

### Constants Changed (rewards.py)

| Constant                         | Run 2 | Run 3 | Rationale                                                                  |
|----------------------------------|-------|-------|----------------------------------------------------------------------------|
| DAMAGE_DEALT_REWARD_PER_HP       | 0.5   | 0.25  | Halves episode damage budget +925→~462, parity with Single-LOS (+455)      |
| ATTACK_ON_COOLDOWN_BONUS         | 1.0   | 2.0   | Compensates for reduced damage signal — rewards attacking, not just damage |
| MAGER_PRIORITY_BONUS_PER_NPC     | 0.3   | 0.6   | Restores mager priority magnitude (formula uses DAMAGE_DEALT internally)   |
| BLOOD_BARRAGE_HEAL_REWARD_PER_HP | 2.0   | 2.5   | Break-even with damage penalty — healing incentive comes from Survival_HP  |

### Projected Budget Comparison

```
Term               Run 2     Run 3    Run 2%  Run 3%
Damage_Dealt      +925      +462     27.2%   16.2%   Halved
Single-LOS        +455      +455     13.4%   16.0%   Now co-dominant
NE_Pillar_Zone    +328      +328      9.6%   11.5%   More visible
Damage_Taken      -222      -222      6.5%    7.8%
Multi-LOS         -221      -221      6.5%    7.8%
Mager_Priority    +211      +211      6.2%    7.4%   Restored via 0.6
Wave_Complete     +179      +179      5.3%    6.3%
Att_on_Cooldown    +75      +149      2.2%    5.2%   Doubled
Death             -120      -120      3.5%    4.2%   Now 26% of damage (was 13%)
```

No single term exceeds 17%. Top 3 within 5% of each other. Death penalty is now 26% of Damage_Dealt (was 13%).

### Verification Checklist (first 5-10M steps after resume)

- [ ] **KL**: spike to 0.03-0.04 for 2-3M, then settle < 0.02. Sustained > 0.04 past 5M → revert
- [ ] **Deaths**: flat or decreasing. Monotonic increase over 5M → problem
- [ ] **Attack_on_Cooldown density**: should increase slightly (doubled bonus)
- [ ] **Single-LOS**: stable or increasing
- [ ] **Blood Barrage Heal**: new appearances within 5-10M via exploration
- [ ] **Frontier**: not stalled vs pre-change trajectory
- [ ] Export reward_terms.csv at 10M for comparison with 40M baseline analysis
