# V22 TensorBoard Tracking

Larger actor/critic networks: [256, 128] / [256, 256] → [512, 256] / [512, 256]. Fresh BC warmstart (required — checkpoint shape mismatch).
All other settings identical to V21 at ~366M (including movement penalty -0.2, LSTM 32, GAE 0.97).

**Run 1** (0–): `logs/V22_climb/`
**Phase**: Climb (W49→66, promote-after 5)
**Load**: `models/bc_warmstart_v22.pt` (fresh, 512/256 + LSTM 128)
**Change**: Actor [512, 256], Critic [512, 256] (up from [256, 128] / [256, 256])

### Start Command (PowerShell)

```powershell
python -m tools.inferno_rl.train_gpu --load models/bc_warmstart_v22.pt --phase climb --start-wave 49 --max-wave 66 --promote-after 5 --save-dir models/V22_climb --log-dir logs/V22_climb --n-envs 16 --n-steps 1024 --batch-size 256 --n-epochs 5 --lr 1e-4 --entropy-start 0.015 --entropy-end 0.002 --gae-lambda 0.97 --vf-coef 0.75 --max-grad-norm 0.5 --normalize-obs --normalize-reward --checkpoint-every 100 --timesteps 150000000 --device cuda --log-reward-terms
```

---

## How to Read This Document

Each snapshot records the same metrics at the same step intervals. Comparing across snapshots reveals:

1. **Is the frontier advancing?** — The primary success signal. Should climb from 49 toward 66.
2. **Are deaths trending down?** — Indicates the policy is learning survival, not just exploration.
3. **Is EV stable >0.80?** — Value function is calibrated. If EV drops below 0.70, something is wrong.
4. **Is entropy collapsing?** — entropy_loss approaching 0 = policy becoming deterministic too early. Should decay slowly with the entropy
   coefficient schedule, not crash.
5. **Is mean_reward trending up?** — Noisy but should show upward movement over 5-10M steps.
6. **Is KL stable <0.02?** — Measures how much the policy changes per update. Healthy range is 0.005–0.02. Sustained >0.03 means updates are
   too aggressive (policy thrashing). Spikes after reward changes are expected but should settle within 1-2M steps.
7. **Is grad_norm stable ~1.0–1.6?** — Gradient magnitude before clipping (max-grad-norm=0.5). Stable values mean the loss landscape is
   smooth. Sudden spikes (>3.0) or drops (<0.5) indicate the optimizer is struggling. Grad norm rising over time can signal reward scale
   issues.
8. **Is FPS holding?** — Rollout collection speed. Drops indicate env bottlenecks (complex waves, pathfinding edge cases). Useful for
   estimating time-to-completion. Larger networks may reduce FPS slightly — watch for >20% drop vs V21 (~2900).

### Early Warning Signs (kill the run)

- Entropy loss goes to 0 or positive → entropy collapse, policy is deterministic
- EV drops below 0.60 and doesn't recover within 2M steps → value function diverging
- Deaths increase monotonically over 5M+ steps → reward signal is counterproductive
- Frontier hasn't moved from W49 after 10M steps → agent isn't learning to clear waves at all
- KL sustained above 0.03 for 5M+ steps → policy is thrashing, not converging
- Grad norm spikes above 3.0 repeatedly → loss landscape unstable, check reward scaling

---

## Manual Observations & Progress

| Steps | Phase | Frontier | f.mean | Deaths | Waves | Timeouts | MeanRwd | EV   | Entropy | KL    | Grad | FPS  | Note                                                                                                                                                                                           |
|-------|-------|----------|--------|--------|-------|----------|---------|------|---------|-------|------|------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| —     | —     | —        | —      | —      | —     | —        | —       | —    | —       | —     | —    | —    | **Run 1 (discarded): MLP-only (no LSTM) — BC warmstart generated without lstm_hidden_size. 1M steps logged below for reference, then restarted with LSTM.**                                    |
| 1.0M  | 1     | 63       | 62.3   | 111    | 60    | 0        | -0.03   | 0.53 | -0.18   | 0.022 | 1.35 | 2961 | MLP-only run. Frontier 63 @ 1M. Deaths 111. FPS 2961 (no slowdown from larger nets).                                                                                                           |
| —     | —     | —        | —      | —      | —     | —        | —       | —    | —       | —     | —    | —    | **Run 2: Regenerated BC warmstart with --lstm-hidden-size 128 --lstm-seq-len 32. Full intended architecture (512,256 + LSTM 128).**                                                            |
| 1.0M  | 1     | 64       | 61.0   | 105    | 71    | 0        | -1.02   | 0.59 | -1.07   | 0.044 | 1.97 | 2220 | Fresh start (LSTM run). Frontier 64 @ 1M. Deaths 105. **KL 0.044 (above 0.030 -- watch, expected early-run transient).** Grad 1.97 (elevated). FPS 2220 (lower than MLP run -- LSTM overhead). |
| 2.3M  | 2     | 66       | 57.6   | 86     | 77    | 0        | -0.46   | 0.56 | -1.71   | 0.038 | 2.24 | 2100 | Auto-logged. **EV 0.56 (below 0.60 -- watch)** **KL 0.038 (above 0.030 -- watch)**                                                                                                             |
| 3.9M  | 2     | 63       | 55.5   | 86     | 79    | 0        | -0.41   | 0.70 | -2.23   | 0.049 | 1.54 | 2072 | Auto-logged. **KL 0.049 (above 0.030 -- watch)**                                                                                                                                               |
| 4.4M  | 2     | 63       | 55.9   | 85     | 75    | 0        | -0.35   | 0.71 | -2.28   | 0.052 | 1.47 | 2242 | Auto-logged. **KL 0.052 (above 0.030 -- watch)** |
| 5.3M  | 2     | 63       | 56.5   | 83     | 72    | 0        | -0.79   | 0.76 | -2.38   | 0.047 | 1.68 | 2205 | Auto-logged. **KL 0.047 (above 0.030 -- watch)** |
| 6.2M  | 2     | 63       | 56.8   | 83     | 70    | 0        | -0.69   | 0.63 | -2.35   | 0.042 | 1.57 | 2238 | Auto-logged. **KL 0.042 (above 0.030 -- watch)** |
| 7.1M  | 2     | 63       | 56.9   | 90     | 75    | 0        | -0.86   | 0.54 | -2.34   | 0.044 | 1.62 | 2243 | Auto-logged. **EV 0.54 (below 0.60 -- watch)** **KL 0.044 (above 0.030 -- watch)** |
| 8.0M  | 2     | 63       | 55.8   | 71     | 77    | 0        | -0.40   | 0.60 | -2.45   | 0.038 | 1.36 | 222  | Auto-logged. **EV 0.60 (below 0.60 -- watch)** **KL 0.038 (above 0.030 -- watch)** |
| 9.5M  | 2     | 63       | 56.2   | 78     | 69    | 0        | -0.27   | 0.63 | -2.55   | 0.045 | 1.43 | 2431 | Auto-logged. **KL 0.045 (above 0.030 -- watch)** |
| 10.5M | 2     | 63       | 55.8   | 75     | 73    | 0        | 0.05    | 0.62 | -2.47   | 0.045 | 1.47 | 2411 | Auto-logged. **KL 0.045 (above 0.030 -- watch)** |
| 11.5M | 2     | 63       | 56.0   | 78     | 71    | 0        | -0.08   | 0.66 | -2.46   | 0.041 | 1.45 | 2426 | Auto-logged. **KL 0.041 (above 0.030 -- watch)** |
| 12.1M | 2     | 63       | 56.2   | 82     | 64    | 0        | -0.02   | 0.69 | -2.47   | 0.046 | 1.42 | 2380 | Auto-logged. **KL 0.046 (above 0.030 -- watch)** |
| 13.2M | 2     | 63       | 56.3   | 76     | 74    | 0        | 0.14    | 0.59 | -2.49   | 0.038 | 1.36 | 2474 | Auto-logged. **EV 0.59 (below 0.60 -- watch)** **KL 0.038 (above 0.030 -- watch)** |
| 14.3M | 2     | 63       | 56.6   | 61     | 76    | 0        | -0.17   | 0.63 | -2.49   | 0.049 | 1.68 | 2478 | Auto-logged. **KL 0.049 (above 0.030 -- watch)** |
---

## All Changes: V21 Startup -> V22 Startup

V22 inherits every change made during V21's 372M-step run. Full changelog:

### Architecture

| Change         | V21 start                   | V22 start                  | When changed                              |
|----------------|-----------------------------|----------------------------|-------------------------------------------|
| Actor network  | [256, 128]                  | [512, 256]                 | V22 start                                 |
| Critic network | [256, 256]                  | [512, 256]                 | V22 start                                 |
| LSTM seq_len   | 8                           | 32                         | V21 @ 352M                                |
| GAE lambda     | 0.95                        | 0.97                       | V21 @ 352M                                |
| Start wave     | 55                          | 49                         | V21 @ 290M (was 35, adjusted to 49 later) |
| BC warmstart   | Fresh (entity ordering fix) | Fresh (larger nets + LSTM) | V22 start                                 |

### Reward Changes (accumulated during V21)

| Change                                    | V21 start | V22 start | When changed |
|-------------------------------------------|-----------|-----------|--------------|
| DAMAGE_PENALTY_PER_HP                     | -1.0      | -2.5      | V21 @ 191M   |
| Single-LOS engagement bonus               | 1.5       | 2.5       | V21 @ 342M   |
| Single-LOS requires attack within 5 ticks | No        | Yes       | V21 @ 191M   |
| Mager priority bonus per NPC              | 0.5       | 0.3       | V21 @ 290M   |
| Resurrection penalty                      | -20       | -12       | V21 @ 290M   |
| Melee resurrections exempt                | No        | Yes       | V21 @ 290M   |
| Bat kill reward                           | 5         | 8         | V21 @ 342M   |
| Weapon switch penalty                     | 0         | -0.5      | V21 @ 342M   |
| Movement penalty                          | 0         | -0.2/tick | V21 @ ~366M  |
| Pillar damage (non-NE)                    | Active    | Removed   | V21 @ 140M   |
| Initial barrage heuristic                 | Active    | Disabled  | V21 @ 140M   |

### Bug Fixes (during V21)

| Fix                                                                     | When       |
|-------------------------------------------------------------------------|------------|
| Attack_on_Cooldown requires target in range (was farmable out-of-range) | V21 @ 140M |

### Unchanged from V21

lr=1e-4, entropy 0.015->0.002, vf-coef=0.75, max-grad-norm=0.5, n-envs=16, n-steps=1024, batch=256, n-epochs=5, promote-after=5, LSTM
hidden=128, normalize-obs, normalize-reward.

**Note:** Because V22 is a fresh BC warmstart, it cannot be directly compared step-for-step with V21. V21 accumulated 372M steps of
learning; V22 starts from scratch with the same reward code but untrained weights. The comparison is about convergence speed and eventual
ceiling, not absolute performance at matched steps.
