# V23 TensorBoard Tracking

Pillar-aware observation space (186 → 220 floats). Fresh BC warmstart from transformed V2 data. Adds player-relative
and entity-relative NE pillar position features — gives the policy explicit spatial awareness of the pillar without
needing to infer it from raw coordinates.

**Run 1** (0–): `logs/V23_climb/`
**Phase**: Climb (W49→65, promote-after 5) → mastery (W55-65, W63-focused)
**Load**: `models/bc_warmstart_v3.pt` (fresh, 220-dim obs, LSTM 128, seq-len 32)
**Changes**: Observation space 186 → 220 floats (pillar-relative features). Curriculum: removed level-up mechanic,
replaced with mastery sampling when frontier reaches max_wave. W66 skipped (trivial double-mager).

### Observation Changes

| Section           | Old (186)        | New (220)        | Detail                            |
|-------------------|------------------|------------------|-----------------------------------|
| Player pos        | [0:2]            | [0:2]            | Unchanged                         |
| Player pillar-rel | —                | [2:4]            | NEW: `(x - 18)/29`, `(y - 23)/30` |
| Player rest       | [2:8]            | [4:10]           | Shifted +2                        |
| Pillars           | [8:20]           | [10:22]          | Shifted +2                        |
| Entity slots      | [20:180] (16×10) | [22:214] (16×12) | +2 floats/slot: pillar-rel x, y   |
| Wave context      | [180:186]        | [214:220]        | Shifted +34                       |

**BC warmstart accuracy**: 95.1% (10 epochs, lr=1e-3, batch 512). High accuracy expected — LSTM + observation
features are additive, not restructured.

**Known limitation**: Entity sort order mismatch between old BC data (base_priority: BLOB before MELEE) and current
code (_get_threat_priority: MELEE before BLOB). Affects ~10% of compositions. Acceptable for warmstart — model
relearns during PPO. V21 demonstrated frontier 65 within 2.8M steps from a similar mismatch.

### Start Command (PowerShell)

```powershell
python -m tools.inferno_rl.train_gpu --load models/bc_warmstart_v3.pt --phase climb --start-wave 49 --max-wave 65 --promote-after 5 --save-dir models/V23_climb --log-dir logs/V23_climb --n-envs 24 --n-steps 1024 --batch-size 384 --n-epochs 5 --lr 1e-4 --entropy-start 0.015 --entropy-end 0.002 --gae-lambda 0.97 --vf-coef 0.75 --max-grad-norm 0.5 --normalize-obs --normalize-reward --checkpoint-every 100 --timesteps 100000000 --device cuda --log-reward-terms
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
   estimating time-to-completion.

### Early Warning Signs (kill the run)

- Entropy loss goes to 0 or positive → entropy collapse, policy is deterministic
- EV drops below 0.60 and doesn't recover within 2M steps → value function diverging
- Deaths increase monotonically over 5M+ steps → reward signal is counterproductive
- Frontier hasn't moved from W49 after 10M steps → agent isn't learning to clear waves at all
- KL sustained above 0.03 for 5M+ steps → policy is thrashing, not converging
- Grad norm spikes above 3.0 repeatedly → loss landscape unstable, check reward scaling

---

## Manual Observations & Progress

| Steps | Phase | Frontier | f.mean | Deaths | Waves | Timeouts | MeanRwd | EV | Entropy | KL | Grad | FPS | Note |
|-------|-------|----------|--------|--------|-------|----------|---------|----|---------|----|------|-----|------|
| 2.5M | 1 | 65 | 64.4 | 147 | 111 | 0 | 0.72 | 0.88 | -1.03 | 0.041 | 1.60 | 3271 | Frontier 65 in <2.5M. Mastery mode active. **KL 0.041 (>0.03 — watch).** |
| 5.0M | 1 | 65 | 64.8 | 145 | 99 | 0 | 0.15 | 0.90 | -1.76 | 0.038 | 1.24 | 3254 | Mastery mode. Deaths not improving. Entropy -1.76 (dropping fast). KL 0.038. |
| 7.5M | 1 | 65 | 64.8 | 149 | 98 | 0 | 0.01 | 0.69 | -2.00 | 0.048 | 1.32 | 3277 | **EV 0.69 (watch).** Entropy -2.00 (near collapse). KL 0.048. Deaths 149 (worst yet). |
| 10.0M | 1 | 65 | 65.0 | 131 | 106 | 0 | 0.31 | 0.85 | -1.95 | 0.051 | 1.21 | 3387 | Deaths 149→131 (first improvement). EV recovered 0.85. KL 0.051 (rising). |
| 12.5M | 1 | 65 | 65.0 | 122 | 106 | 0 | 0.28 | 0.88 | -1.85 | 0.050 | 1.12 | 3009 | Deaths 131→122 (trending down). KL still >0.03. Entropy -1.85. Run ongoing. |

---

## Comparison: V23 vs V21

V23 starts at W49 (vs V21's W55), has 220-dim obs (vs 186), 24 envs (vs 16), and batch 384 (vs 256). Not directly
step-comparable due to different wave ranges, but frontier progression speed is informative.

| Metric | V21 @ 2.8M | V23 @ 2.5M | V21 @ 5.3M | V23 @ 5.0M | Note |
|--------|-----------|-----------|-----------|-----------|------|
| Start wave | 55 | 49 | 55 | 49 | V23 climbs from further back |
| Frontier | 65 | 65 | 64 | 65 | Both reach 65 fast |
| Deaths | 90 | 147 | 65 | 145 | V23 deaths much higher — mastery mode W63 focus |
| Waves | 49 | 111 | 46 | 99 | V23 more episodes (24 envs vs 16) |
| MeanRwd | -0.50 | 0.72 | -0.26 | 0.15 | V23 reward near zero, not climbing |
| EV | 0.92 | 0.88 | 0.92 | 0.90 | Comparable |
| Entropy | -0.17 | -1.03 | -0.37 | -1.76 | **V23 entropy collapsing much faster** |
| KL | 0.013 | 0.041 | 0.023 | 0.038 | **V23 KL sustained >0.03** |
| Grad | 1.14 | 1.60 | 1.22 | 1.24 | Comparable |

### Assessment

**Frontier climb is excellent** — W49→65 in 2.5M steps, faster than V21 despite starting 6 waves lower. The 220-dim obs
+ 95% BC accuracy warmstart is clearly helping early learning.

**Mastery mode is the bottleneck.** Once frontier hits 65, V23 switches to W63-focused sampling. Deaths are 122-149 and
barely improving. V21 went through level-up resets that gave the agent easier waves between hard pushes; V23 mastery mode
is relentless W63 drilling.

**KL >0.03 sustained for 12.5M steps is a red flag.** V21 never exceeded 0.023 in the first 5M steps. Possible causes:
- 24 envs + batch 384 may have higher effective gradient variance than V21's 16/256
- Mastery mode's W63 concentration gives a narrower experience distribution, amplifying policy updates

**Entropy collapsing** — V23 entropy hits -2.00 at 7.5M, compared to V21's -0.37 at 5.3M. The policy is becoming
deterministic too fast, likely because mastery mode concentrates training on a few wave compositions.

**Recommendation:** This run should probably be stopped. The high KL + entropy collapse + flat deaths pattern matches
V21's failed experiments. The climb phase worked well, but mastery mode with 24 envs is not working. Consider: (1)
reducing n-envs back to 16, (2) using level-up resets instead of mastery mode, or (3) a wider wave range in mastery to
prevent distribution narrowing.

---

## All Changes: V21 (latest) → V23 Startup

V23 inherits all reward/bug changes accumulated during V21's 481M-step run. Structural changes: observation space
expansion, curriculum level-up replaced with mastery sampling, max_wave 66→65.

### Architecture

| Change          | V21 final                   | V23 start                           | When changed |
|-----------------|-----------------------------|-------------------------------------|--------------|
| Observation dim | 186                         | 220                                 | V23 start    |
| Max wave        | 66                          | 65                                  | V23 start    |
| Curriculum      | Level-up (min_waves++)      | Mastery sampling (W63-focused)      | V23 start    |
| n-envs          | 16                          | 24                                  | V23 start    |
| Batch size      | 256                         | 384 (keeps 64 minibatches/epoch)    | V23 start    |
| Actor network   | [256, 128]                  | [256, 128]                          | Unchanged    |
| Critic network  | [256, 256]                  | [256, 256]                          | Unchanged    |
| LSTM hidden     | 128                         | 128                                 | Unchanged    |
| LSTM seq_len    | 32                          | 32                                  | Unchanged    |
| GAE lambda      | 0.97                        | 0.97                                | Unchanged    |
| Start wave      | 49                          | 49                                  | Unchanged    |
| BC warmstart    | Fresh (entity ordering fix) | Fresh (220-dim obs, transformed v2) | V23 start    |

### Curriculum: Climb → Mastery

**Climb phase** (unchanged): Frontier starts at W49, advances by 1 after 5 consecutive completions (promote-after=5,
min_waves_to_advance=1). Agent always starts at the frontier wave.

**Mastery mode** (new, replaces level-up): When frontier reaches W65 (max_wave), the old code would increment
min_waves_to_advance and reset to W49. Now it switches to difficulty-weighted sampling over W55-65:

| Wave | Weight | % of episodes |
|------|--------|---------------|
| 63   | 6      | 26%           |
| 62   | 4      | 17%           |
| 61   | 3      | 13%           |
| 59-60| 2 each | 9% each       |
| 55-58, 64-65 | 1 each | 4% each |

This gives the agent concentrated W63 practice (the bottleneck composition) while maintaining exposure to the full
W55-65 range to prevent regression. TensorBoard metric `rollout/mastery_mode_pct` tracks the fraction of steps in
mastery mode.

### Reward Settings (inherited from V21, unchanged)

| Setting                                   | Value     |
|-------------------------------------------|-----------|
| DAMAGE_PENALTY_PER_HP                     | -2.5      |
| Single-LOS engagement bonus               | 2.5       |
| Single-LOS requires attack within 5 ticks | Yes       |
| Mager priority bonus per NPC              | 0.3       |
| Resurrection penalty                      | -12       |
| Melee resurrections exempt                | Yes       |
| Bat kill reward                           | 8         |
| Weapon switch penalty                     | -0.5      |
| Movement penalty                          | -0.2/tick |
| Pillar damage (non-NE)                    | Removed   |
| Initial barrage heuristic                 | Disabled  |
| Death penalty                             | -120      |
| Nibbler kill reward                       | 6         |

### Hyperparameters (unchanged)

lr=1e-4, entropy 0.015→0.002, vf-coef=0.75, max-grad-norm=0.5, n-envs=24 (up from 16), n-steps=1024,
batch=384 (up from 256, keeps 64 minibatches/epoch), n-epochs=5, promote-after=5, normalize-obs, normalize-reward.

### What to Watch For

- **Pillar-relative features should accelerate pillar-play learning** — the policy no longer needs to infer pillar
  position from raw coords + pillar state. Expect faster frontier advancement in early steps vs V21.
- **BC accuracy 95.1% vs V21's ~87%** — higher warmstart quality may mean fewer early deaths and faster curriculum
  progression, but could also mean the policy is more "locked in" to heuristic behavior. Watch entropy.
- **V22 comparison**: V22 used larger nets (512/256) and suffered persistent KL >0.03. V23 returns to V21's proven
  [256, 128] / [256, 256] architecture. KL should stay <0.02.
- **Mastery mode transition** — once frontier hits 65, `mastery_mode_pct` should ramp to 1.0. Watch that deaths don't
  spike (W63 is hard but the agent should have some W63 experience from the climb). If deaths spike >50 and don't
  recover within 5M steps, the W63 weight may be too high.
- **n-envs 24** — first time using 24 envs. Batch 384 keeps minibatches/epoch at 64 (same as V21's 16/256). KL should
  stay <0.02; if it drifts >0.03 the batch-size compensation isn't sufficient.

**Note:** Because V23 is a fresh BC warmstart with a different observation space, it cannot be directly compared
step-for-step with V21 or V22. The comparison is about convergence speed and eventual ceiling.
