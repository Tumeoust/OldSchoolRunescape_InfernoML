# V20 TensorBoard Tracking

Periodic snapshots to determine how many steps are needed to judge whether a reward function works.

**Run 1** (0–9.8M): `logs/V20_climb/inferno_gpu_w55-66_20260223_150905`
**Run 2** (9.8M–~31.3M): `logs/V20_climb/inferno_gpu_w55-66_20260223_171259`
**Run 3** (~31.3M–): `logs/V20_climb/inferno_gpu_w55-66_20260223_203444`
**Phase**: Climb (W55→66, promote-after 5)
**Load**: `models/bc_warmstart.pt` → resumed from `_600.pt` at 9.8M

---

## Snapshot @ 1.5M steps

| Metric                  |    0K |  250K |  500K |  750K | 1000K | 1250K | 1500K |
|-------------------------|------:|------:|------:|------:|------:|------:|------:|
| deaths/rollout          |    42 |    27 |    38 |    52 |    57 |    46 |    37 |
| waves_completed/rollout |    30 |    25 |    21 |    30 |    17 |    25 |    30 |
| wave_timeouts/rollout   |    23 |    26 |    24 |    31 |    32 |    19 |    13 |
| frontier_max            |    55 |    56 |    56 |    56 |    56 |    56 |    56 |
| mean_episode_reward     | -0.31 |  1.65 |  2.31 |  1.52 |  2.22 |  2.56 |  1.44 |
| mean_episode_length     |   258 |   362 |   363 |   299 |   320 |   354 |   293 |
| explained_variance      |  0.23 |  0.85 |  0.87 |  0.87 |  0.90 |  0.86 |  0.87 |
| entropy_loss            | -0.05 | -0.10 | -0.11 | -0.11 | -0.09 | -0.14 | -0.13 |
| grad_norm               |  1.06 |  0.94 |  0.99 |  0.99 |  0.81 |  1.01 |  1.07 |
| KL divergence           |  0.00 |  0.02 |  0.01 |  0.04 |  0.03 |  0.01 |  0.01 |
| max_wave_from_55        |    56 |    57 |    57 |    57 |    57 |    57 |    56 |
| max_wave_from_56        |    56 |     — |    57 |    58 |    58 |    58 |    58 |

**Assessment @ 1.5M:**

- BC warmstart transferred successfully — EV jumped from 0.23 to 0.85 in the first 250K steps, now stable at 0.85-0.90.
- Frontier advanced to W56 early, holding there. Agent reaches W57-58 from W55/W56 starts but hasn't locked 5 consecutive clears to promote
  further.
- Deaths are noisy (27-57 per rollout), no clear downward trend yet. Timeouts trending down (23→13) which is good — agent is committing to
  fights rather than stalling.
- Mean reward moved from -0.31 to ~1.5-2.5 range but oscillating. Not yet converging.
- Entropy loss stable around -0.10 to -0.14, no collapse. Grad norm stable ~1.0. KL low. Training is healthy.
- **Too early to judge reward function.** The policy is still in the BC→PPO transition phase where value function is calibrating. Need to
  see deaths trending down and frontier advancing past W56 before drawing conclusions.

---

## How to Read This Document

Each snapshot records the same metrics at the same step intervals. Comparing across snapshots reveals:

1. **Is the frontier advancing?** — The primary success signal. Should climb from 55 toward 66.
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
   estimating time-to-completion.

### Early Warning Signs (kill the run)

- Entropy loss goes to 0 or positive → entropy collapse, policy is deterministic
- EV drops below 0.60 and doesn't recover within 2M steps → value function diverging
- Deaths increase monotonically over 5M+ steps → reward signal is counterproductive
- Frontier hasn't moved from W55 after 10M steps → agent isn't learning to clear waves at all
- KL sustained above 0.03 for 5M+ steps → policy is thrashing, not converging
- Grad norm spikes above 3.0 repeatedly → loss landscape unstable, check reward scaling

### Minimum Steps to Judge a Reward Function

*To be determined* — this document tracks the trajectory to answer that question. Working hypothesis:

- **2-3M steps**: Can detect catastrophic failures (entropy collapse, EV divergence, reward going negative)
- **5-10M steps**: Can detect stagnation (frontier not moving, deaths not trending down)
- **15-20M steps**: Can judge whether the reward function produces real frontier advancement

---

## Manual Observations, Changes & Reward Experiments

| Steps | Frontier | Deaths | Waves | Timeouts | MeanRwd | EV    | Entropy | KL    | Grad | FPS  | Note                                                                                                                                                                                                                                                                             |
|-------|----------|--------|-------|----------|---------|-------|---------|-------|------|------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1.5M  | 56       | 37     | 30    | 13       | 1.44    | 0.87  | -0.13   | 0.01  | 1.07 | —    | BC transition complete, EV stabilized                                                                                                                                                                                                                                            |
| 2.1M  | 58       | 60     | 41    | 9        | 1.40    | 0.853 | -0.158  | —     | —    | —    | Fast frontier advance                                                                                                                                                                                                                                                            |
| 2.7M  | 60       | 62     | 40    | 5        | 0.72    | 0.837 | -0.207  | —     | —    | —    |                                                                                                                                                                                                                                                                                  |
| 3.3M  | 60       | 50     | 15    | 13       | -0.11   | 0.851 | -0.217  | —     | —    | —    | Low waves completed — harder frontier                                                                                                                                                                                                                                            |
| 4.0M  | 60       | 55     | 25    | 6        | 1.35    | 0.866 | -0.285  | —     | —    | —    |                                                                                                                                                                                                                                                                                  |
| 4.7M  | 61       | 57     | 27    | 9        | 1.09    | 0.891 | -0.309  | —     | —    | —    |                                                                                                                                                                                                                                                                                  |
| 5.4M  | 61       | 50     | 20    | 8        | 1.04    | 0.882 | -0.430  | —     | —    | —    | Entropy deepening normally                                                                                                                                                                                                                                                       |
| —     | —        | —      | —     | —        | —       | —     | —       | —     | —    | —    | **Old run ended ~9.8M (frontier 63, f.mean 60.84). Cython built.**                                                                                                                                                                                                               |
| —     | —        | —      | —     | —        | —       | —     | —       | —     | —    | —    | **REWARD CHANGE @ 9.8M: +NE_PILLAR_ZONE_BONUS 1.0/tick (in zone + 1+ LOS, or grace period), +NE_PILLAR_ZONE_PENALTY −0.5/tick (outside zone during combat). Goal: stop running south, stay near pillar.**                                                                        |
| 10.0M | 56       | 40     | 27    | 8        | 0.34    | 0.87  | -0.72   | —     | 1.61 | 4212 | Resume from ckpt 600. Frontier reset to 55, re-promoting.                                                                                                                                                                                                                        |
| 11.0M | 57       | 40     | 18    | 13       | 1.68    | 0.88  | -0.79   | —     | 1.56 | 3901 | Re-climbing. Entropy adjusting to new reward landscape.                                                                                                                                                                                                                          |
| 12.0M | 60       | 50     | 18    | 15       | 1.88    | 0.89  | -1.09   | —     | 1.58 | 3905 | Frontier W60 in ~2M steps (fast re-promotion). Entropy deepening.                                                                                                                                                                                                                |
| 13.3M | 61       | 47     | 23    | 8        | 1.93    | 0.92  | -0.96   | 0.015 | 1.38 | 3099 | EV 0.92 (highest yet). Training healthy.                                                                                                                                                                                                                                         |
| 18.0M | 62       | 36     | 23    | 8        | 3.01    | 0.74  | -1.13   | 0.011 | 1.85 | 3057 | Deaths 47→36 (best post-change). MeanRwd 3.0 (new high). EV dipped to 0.74 — watch next reading.                                                                                                                                                                                 |
| 20.0M | 63       | 27     | 28    | 5        | 3.19    | 0.74  | -1.15   | 0.013 | 2.13 | 3183 | Frontier 63. Deaths 36→27 (new low). Timeouts 8→5. MeanRwd 3.19 (new high). EV still 0.74 — stable not recovering. Grad norm rising (1.85→2.13).                                                                                                                                 |
| —     | —        | —      | —     | —        | —       | —     | —       | —     | —    | —    | **REWARD CHANGE @ 20M: Escalating time penalty replaces flat -0.1/tick. Formula: -(0.1 + 3.0 * ticks/500). At tick 250: -1.6/tick, at tick 400: -2.5/tick (break-even with +2.5 stalling). Goal: stop agent farming engagement rewards by leaving 1-2 NPCs alive indefinitely.** |
| 19.7M | 55       | 14     | 29    | 8        | 0.49    | 0.72  | -1.01   | 0.012 | 2.26 | 3135 | Resume from ckpt 600 with escalating time penalty. Frontier reset to 55. MeanRwd dropped (new penalty landscape). Deaths 27→14 (artifact of easy frontier waves).                                                                                                                |
| 20.0M | 57       | 22     | 36    | 7        | 1.45    | 0.76  | -0.98   | 0.008 | 1.78 | 3137 | Re-climbing. Timeouts already dropping. KL low — policy adapting smoothly.                                                                                                                                                                                                       |
| 20.5M | 59       | 28     | 47    | 4        | 4.67    | 0.67  | -0.99   | 0.008 | 1.68 | 3248 | Frontier 59 in <1M steps. Waves 47 (new high). Timeouts 4. MeanRwd 4.67 (new all-time high). EV dipped to 0.67.                                                                                                                                                                  |
| 21.0M | 62       | 26     | 54    | 0        | 3.95    | 0.69  | -1.16   | 0.009 | 1.80 | 3526 | Frontier 62. **Timeouts 0** (escalating penalty working). Waves 54 (massive jump). EV recovering.                                                                                                                                                                                |
| 21.3M | 62       | 35     | 54    | 0        | 3.87    | 0.63  | -1.19   | 0.010 | 1.99 | 3144 | Deaths ticking up as frontier hardens. EV 0.63 — value function struggling with new reward scale. Watch next 2M.                                                                                                                                                                 |
| 22.0M | 65       | 22     | 58    | 0        | 5.37    | 0.78  | -1.23   | 0.007 | 2.09 | —    | **Frontier 65** (f.mean 62.3). MeanRwd 5.37 (peak). Frontier approaching max_wave=66 ceiling.                                                                                                                                                                                    |
| 22.5M | 65       | 29     | 65    | 1        | 4.07    | 0.81  | -1.26   | 0.013 | 1.92 | —    | **Frontier hit ceiling → level-up triggered.** min_waves_to_advance 1→2, frontier reset to W55. f.mean 62.3→58.5. Not a regression — curriculum leveled up.                                                                                                                      |
| 23.0M | 63       | 18     | 74    | 0        | 4.53    | 0.78  | -1.20   | 0.009 | 1.81 | 3591 | Re-climbing with min_waves=2. f.mean 56.3. Deaths 18 (low). Waves 74 (high — clearing easy waves efficiently).                                                                                                                                                                   |
| 23.5M | 63       | 15     | 77    | 0        | 3.48    | 0.68  | -1.19   | 0.008 | 1.71 | —    | f.mean 56.1. Deaths 15 (new all-time low). Steady re-climb.                                                                                                                                                                                                                      |
| 25.0M | 60       | 22     | 74    | 0        | 6.16    | 0.68  | -1.25   | 0.008 | 1.71 | —    | f.mean 56.6. Re-climb slower than first pass (min_waves=2 is harder). MeanRwd 6.16.                                                                                                                                                                                              |
| 25.5M | 60       | 22     | 75    | 0        | 6.67    | 0.72  | -1.22   | 0.008 | 1.68 | —    | f.mean 57.1. Slow but steady. Timeouts still 0. Training healthy.                                                                                                                                                                                                                |
| 26.0M | 60       | 17     | 76    | 0        | 4.14    | 0.74  | -1.31   | 0.008 | 1.66 | 3850 | f.mean 57.7. Deaths 17. FPS 3850 (highest).                                                                                                                                                                                                                                      |
| 27.0M | 61       | 27     | 69    | 0        | 5.30    | 0.74  | -1.36   | 0.007 | 1.68 | 3541 | f.mean 58.3. Frontier slowly advancing.                                                                                                                                                                                                                                          |
| 28.0M | 63       | 21     | 63    | 0        | 5.52    | 0.75  | -1.39   | 0.008 | 1.63 | 3428 | f.mean 59.7. Approaching ceiling again with min_waves=2.                                                                                                                                                                                                                         |
| 29.0M | 66       | 20     | 67    | 0        | 6.35    | 0.74  | -1.30   | 0.009 | 1.55 | 3322 | **Frontier 66 → level-up #2: min_waves 2→3.** f.mean 60.5. Re-climb #2 from W55, now needs 3 consecutive waves per clear. 7M steps for second pass (vs 2M for first).                                                                                                            |
| 30.0M | 63       | 22     | 61    | 2        | 5.53    | 0.73  | -1.37   | 0.008 | 1.66 | 3536 | Re-climbing with min_waves=3. f.mean 60.3. 2 timeouts (first since 21M) — likely noise or tough wave composition.                                                                                                                                                                |
| 34.0M | 64       | 16     | 75    | 0        | 5.74    | 0.84  | -1.32   | 0.006 | 1.77 | 3552 | f.mean 59.9. Deaths 16 (tied all-time low). Timeouts 0. KL 0.006 (lowest). EV recovering 0.73→0.84.                                                                                                                                                                              |
| —     | —        | —      | —     | —        | —       | —     | —       | —     | —    | —    | **REWARD CHANGE @ 32,768M: Added oscillation gate to LOS reward.**                                                                                                                                                                                                               |
| 36.5M | 60       | 18     | 85    | 0        | 4.07    | 0.67  | -1.44   | 0.013 | 1.41 | 3079 | f.mean 56.36, min_waves=3. from_65/66 probes hit W66 @ ~35.9M (single-episode peaks, not level-up). Still climbing toward W66 ceiling. Deaths 18/20 eps. EV dipped to 0.67 (watch).                                                                                             |
| 39.6M | 63       | 22     | 75    | 0        | 2.16    | 0.82  | -1.48   | 0.024 | 1.46 | 3013 | f.mean 59.85, min_waves=3. EV recovered to 0.82. KL slightly elevated at 0.024 — watch next reading.                                                                                                                                                                             |
| 41.8M | 64       | 26     | 69    | 0        | 6.03    | 0.89  | -1.63   | 0.014 | 1.44 | 2829 | f.mean 60.6, min_waves=3. Frontier advanced 63→64. EV 0.89 (highest since 13.3M). KL normalized 0.024→0.014. Grad norm down 1.46→1.44. All metrics healthy.                                                                                                                      |
| 52.7M | 62       | 18     | 77    | 0        | 5.80    | 0.76  | -1.46   | 0.014 | 1.29 | 3088 | f.mean 58.5, min_waves=3. Frontier regressed 64→62, f.mean 60.6→58.5. But deaths 26→18, waves 69→77. EV dropped 0.89→0.76. Mixed signals — survival improving, curriculum metrics dipped.                                                                                        |

---

### Curriculum Level-Up Mechanic

When `frontier_max` reaches `max_wave` (66), the climb phase doesn't stop — it **resets frontier to `start_wave` (55)** and increments
`min_waves_to_advance` (1→2→3...). The agent must now clear N consecutive waves from the frontier start to count as a completion, 5 times in
a row (`promote_after=5`).

| Level-up | Step   | min_waves | Requirement per promotion                     |
|----------|--------|-----------|-----------------------------------------------|
| #1       | ~21.8M | 1→2       | Clear frontier + next wave, 5× consecutive    |
| #2       | ~29.0M | 2→3       | Clear frontier + next 2 waves, 5× consecutive |

First pass (min_waves=1): 55→66 in ~2M steps. Second pass (min_waves=2): 55→66 in ~6.5M steps. Third pass will be harder still.

---

