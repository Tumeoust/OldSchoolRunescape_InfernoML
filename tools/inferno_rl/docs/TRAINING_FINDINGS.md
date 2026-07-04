# Inferno RL — Training Findings

Consolidated training knowledge from V9 through V41 (Feb 21 — Mar 12, 2026). Per-version details live in
`V*_TB_TRACKING.md` files.

**Scope**: Waves 1-66. Jad/Triple Jad/Zuk are hand-coded, not RL.

**Important caveat**: Every finding here reflects the specific model, observation space, architecture, curriculum, and
hyperparameters used at the time. Results are **directional, not ground truth** — something that failed in V21 with a
186-dim MLP might work in V44 with a different architecture. Without controlled A/B testing (infeasible on a single GPU),
we can't isolate variables. Treat entries as "this happened under these conditions", not "this will always happen". Only
consider a finding reliable if it's been reproduced across multiple versions with different configurations.

---

## 1. Version Summary

### Era 1: MLP Discovery (V9-V26, Feb 21-28)

186-dim observation, MLP [256,128] / [256,256], 371K params.

| Version | Steps | Waves          | Key Change                                  | Outcome                                                | 
|---------|-------|----------------|---------------------------------------------|--------------------------------------------------------|
| V9-V10  | ~160M | 35-66          | SB3 stack, initial rewards                  | Deleted. Stuck below W55.                              |
| V12     | ~220M | 55-64          | BC warmstart, found proven config (phase3)  | **Breakthrough.** Frontier 63.4, EV 0.85-0.90.         |
| V13     | 120M  | 55-64          | Reward simplification                       | **FAILED.** Entropy collapse. Dense shaping necessary. |
| V20     | ~43M  | 55-66          | Equipment system (BoFa/BP/Mage)             | Stopped: entity ordering bug.                          |
| V21     | 481M  | 55-66 -> 35-66 | Bug fix, longest run                        | **Peak: 30% clear W49-66** (R5 @ 290M).                |
| V22-V23 | ~26M  | 49-66          | Larger nets, more envs, pillar-relative obs | **FAILED.** KL > 0.03, entropy collapse.               |
| V24-V25 | 278M  | 49-66          | LSTM 256, 262-dim obs, 807K params          | Peak 26% clear. Simpler V21 won.                       |
| V26     | ~10M  | 49-66          | Resume V21 peak, full code revert           | Baseline re-established.                               |

### Era 2: Architecture Reset (V27-V30, Mar 1-4)

Addressed V27's gradient norm problem (~6.0) through reward scaling, aux head removal, PVP-style settings.

| Version | Steps  | Key Change                                        | Outcome                                    |
|---------|--------|---------------------------------------------------|--------------------------------------------|
| V27     | 8 runs | NE pillar zone reward variants                    | All failed. Grad norm ~6.0 from aux head.  |
| V28     | Fresh  | Remove aux head, rewards /5, 1 epoch, gamma 0.995 | Framework reset. Grad norms normalized.    |
| V29     | Fresh  | [512,512] nets, 48 envs, PVP-style settings       | KL stabilized with reward normalization.   |
| V30     | Fresh  | LSTM 256 + seq_len 10, nibblers in entity slots   | First LSTM+large-net combo with stable KL. |

### Era 3: Observation Redesign (V31-V41, Mar 5-12)

Major observation space redesign through V3.0 -> V3.1 -> V3.2 (317 dims). `flat_lstm_residual` architecture.

| Version | Obs Dims | Key Change                                                        | Outcome                                      |
|---------|----------|-------------------------------------------------------------------|----------------------------------------------|
| V31-V33 | 186->269 | V3.0 obs: safety map, typed entity slots, temporal features       | Foundation for forecast obs.                 |
| V34     | 269      | entity_pool_lstm architecture                                     | 36% clear W55-66 (best eval of era).         |
| V35     | 269      | Continue V34 ckpt, reward tuning                                  | Mager kill-order shaping.                    |
| V36-V37 | 295      | V3.1 obs: blob scan, queued prayer, nibbler target, dead pool     | Major observability increase.                |
| V38     | 295      | flat_lstm_residual (lightweight recurrent)                        | Architecture validated. Fresh start from W1. |
| V39     | 363->267 | V3.2 obs: forecast features, directional movement, threat horizon | Compact forecast-first schema.               |
| V40     | 317      | V3.2 enriched: pillar-relative, per-slot blob scan, mager alive   | Frontier W60 at 10M steps. **Current.**      |
| V41     | 317      | Sweep curriculum (failure-weighted all waves)                     | Testing broader coverage. **Current.**       |

---

## 2. Architecture Evolution

| Era     | Obs     | Model                                      | Params    | Best Result                   |
|---------|---------|--------------------------------------------|-----------|-------------------------------|
| V9-V26  | 186     | MLP [256,128]/[256,256]                    | 371K      | 30% clear W49-66 (V21 @ 290M) |
| V22-V25 | 186-262 | MLP [512,256] + LSTM 128-256               | 600K-807K | 26% clear (V25 @ 188M)        |
| V28-V29 | 186     | MLP [512,512]/[512,512]                    | ~1.2M     | Framework reset, no eval      |
| V30     | 186     | MLP [512,512] + LSTM 256                   | ~1.5M     | Stable training achieved      |
| V34     | 269     | entity_pool_lstm                           | ~1.5M     | 36% clear W55-66              |
| V38-V41 | 295-317 | flat_lstm_residual (128) + [512,512] heads | ~1.5M     | **Current architecture.**     |

**Key finding**: The `flat_lstm_residual` architecture — keep full observation visible to actor/critic, add small
LSTM memory path, concatenate before heads — is the current best. It avoids the failure mode of heavy recurrent
front-ends (entity_pool_lstm) while providing multi-tick memory.

---

## 3. Current Training Settings (V40/V41)

| Setting             | Value              | Proven By                 |
|---------------------|--------------------|---------------------------|
| observation-version | v3.2 (317 dims)    | V40                       |
| policy-arch         | flat_lstm_residual | V38-V41                   |
| lstm-hidden-size    | 128                | V38                       |
| lstm-seq-len        | 16                 | V38                       |
| lstm-burn-in        | 8                  | V38                       |
| actor/critic sizes  | 512,512 / 512,512  | V29+                      |
| n-envs              | 16                 | V12 (ceiling: 24/32 fail) |
| n-steps             | 1024               | V12                       |
| batch-size          | 2048               | V29+                      |
| n-epochs            | 3-5                | 3 for V40, 5 for V41      |
| lr                  | 2e-4               | V39+                      |
| target-kl           | 0.015-0.03         | V39/V41                   |
| entropy-start/end   | 0.05 / 0.002       | V39+                      |
| gamma               | 0.995              | V28+                      |
| gae-lambda          | 0.95               | V12, V26 revert           |
| vf-coef             | 0.5                | V28+                      |
| max-grad-norm       | 0.5                | V12                       |
| normalize-obs       | yes                | V12+                      |
| normalize-reward    | yes                | V29+                      |
| checkpoint-every    | 100                | Standard                  |

**Note**: 48-env runs (V29, V30, V40 phase 2) were tested and produced stable KL when combined with 1 epoch +
batch 4096 + target-kl 0.02. This is a viable alternative to 16-env for throughput.

---

## 4. Anti-Patterns (What Fails)

| Anti-Pattern                              | Where Tested | What Happened                                              |
|-------------------------------------------|--------------|------------------------------------------------------------|
| n-envs=32 (old settings)                  | V12 phase2/4 | Gradient variance too high, deaths regressed               |
| n-envs=24 with batch=384                  | V23          | KL >0.03 sustained, entropy collapse in 7.5M steps         |
| max-grad-norm=1.0                         | V12 phase4   | Updates 2x larger, frontier +1 but avg regressing          |
| lr=5e-5                                   | V25 Run 7    | Starved optimizer, entropy decayed faster than learning    |
| Remove dense shaping rewards              | V13          | Entropy collapse -0.5 -> -1.8, no per-tick gradient signal |
| Narrowing wave range                      | V25 Run 7    | Catastrophic forgetting of excluded waves, 26% -> 21%      |
| Large wave range expansion                | V21 Run 6    | W55-66 -> W35-66 crashed 30% -> 10% clear                  |
| Concentrated wave sampling                | V23          | 26% of episodes on W63 -> entropy collapse                 |
| Change rewards + warmstart simultaneously | V13          | Can't attribute failure to either change                   |
| Large obs + large nets + LSTM             | V22-V25      | More params, worse sample efficiency, lower ceiling        |
| Aux damage prediction head                | V27          | MSE loss dominated gradients at any coefficient            |
| Hindsight death penalty                   | V25 Run 5    | Neutral impact, GAE already propagates death signal        |

---

## 5. Lessons Learned

### Training Dynamics

1. **Dense shaping rewards are structurally necessary.** Without per-tick signals (LOS engagement, damage, stall penalty),
   the policy has no gradient to maintain coherent behavior between sparse terminal rewards.

2. **Smaller models learn faster on this problem.** V21 MLP (371K) beat V25 LSTM (807K) with fewer steps. The bottleneck
   is credit assignment and exploration, not model capacity.

3. **Observation quality matters more than observation size.** V3.2's forecast features (317 dims) outperform V2's raw
   entity dump (262 dims) because pre-computed tactical features reduce the policy's inference burden.

4. **The flat_lstm_residual architecture works because it doesn't gate the observation.** Keeping the full normalized obs
   visible to actor/critic avoids the information bottleneck of heavy recurrent front-ends.

### Hyperparameters

5. **max-grad-norm 0.5 is correct.** Raw grad norms are 1.4-1.6; clipping to 0.5 clips ~67% of updates but training is
   stable. 1.0 destabilizes.

6. **n-envs 16 is the safe ceiling.** 48 envs works with compensating settings (1 epoch, batch 4096, target-kl 0.02),
   but 24-32 with standard settings fails consistently.

7. **gae-lambda 0.95 > 0.97.** Higher lambda tested at V21 @ 352M, coincided with performance decline. Reverted and
   stayed at 0.95 since V26.

### Curriculum

8. **Climb curriculum is sample-efficient for initial learning.** Frontier-based advancement concentrates on the current
   skill boundary.

9. **Drill curriculum plateaus.** Retry-on-failure reduces deaths to a floor (~3/rollout) then stops improving. Remaining
   failures need reward/observation changes, not more exposure.

10. **Sweep curriculum (V41) is untested at scale.** Hypothesis: failure-weighted all-wave sampling avoids climb's wasted
    samples on mastered waves and prevents the frontier bottleneck.

### Rewards

11. **Reward magnitude matters.** V27's grad norm problem (~6.0) was caused by too-large reward signals. Dividing all
    rewards by 5 (V28) normalized gradients.

12. **Terminal penalties (death, timeout) should be 0.** Episode termination IS the penalty. The critic can't attribute a
    single terminal reward to the actions that caused the death. Set to 0 in V21 and kept since.

13. **Weapon encoding must match checkpoint exactly.** Mismatched observation encoding causes the model to spam weapon
    switch actions (V26 debugging incident).

---

## 6. TensorBoard Health Checklist

| # | Metric             | Healthy       | Warning          | Kill                       |
|---|--------------------|---------------|------------------|----------------------------|
| 1 | Frontier           | Advancing     | Stalled 10M+     | N/A                        |
| 2 | Deaths/rollout     | Trending down | Increasing 5M+   | Monotonic increase         |
| 3 | Explained Variance | > 0.80        | < 0.60 for 2M+   | Diverging                  |
| 4 | Entropy loss       | Slow decay    | Approaching 0    | Collapse (positive/zero)   |
| 5 | KL divergence      | 0.005-0.020   | > 0.03 sustained | > 0.05 sustained           |
| 6 | Grad norm          | 1.0-1.6       | > 3.0 sustained  | Rising trend + EV collapse |
| 7 | Mean reward        | Trending up   | Flat 10M+        | Decreasing                 |
| 8 | FPS                | Stable        | N/A              | N/A                        |

**Common transients (not alarming):** EV dip after curriculum reset (2-5M recovery), KL spike after reward change (1-2M
to settle), deaths spike after level-up, mean reward drop after reward constant change.

---

## 7. Eval Methodology

- **Seeds**: 100 deterministic episodes (seeds 0-99)
- **Wave ranges**: W49-66 (broad) or W55-66 (narrow)
- **Mode**: Deterministic (argmax actions)
- **Metrics**: Clear %, death %, timeout %, per-wave death counts

```bash
python -m tools.inferno_rl.death_analysis \
  --checkpoint models/<version>/<checkpoint>.pt \
  --start-wave 49 --max-wave 66 \
  --num-episodes 100 --deterministic
```

**Known issue**: Pre-V26 `death_analysis.py` counted wave timeouts as clears (fixed Feb 28). All V21 eval numbers
in `RL_TRAINING_HISTORY.md` were corrected.

---

## 8. Open Environment Issues

From the V39 training environment audit. Issues not yet addressed:

### Simulator Bugs

- **BUG-S1**: Player movement ignores dead pillar passability (`simulator.py:541-542`). `_execute_movement` and
  `_predict_position_after_action` don't pass `pillar_alive` to `is_valid_tile`, making dead pillars block movement
  actions but not attack drag. Fix: pass `self.state.pillar_alive`.

### Observation Gaps

- **OBS-1** [High]: Frozen timer completely hidden. Entity slots encode `stunned` as binary but `frozen` (0-32 ticks)
  is not exposed. Policy can't distinguish "frozen for 30 ticks" from "unfreezing next tick". Fix: expose
  `min(frozen / 32.0, 1.0)` and `min(stunned / 4.0, 1.0)` per entity slot (+10 dims).

- **OBS-2** [Medium]: Attack delay clipped to non-negative. For non-melee NPCs, negative delay (waiting to attack
  without LOS) could be useful signal. Fix: signed normalization [-1, 1].

- **OBS-5** [Medium]: Directional forecast misses blob scan initiation. Moving to a tile that gives an unscanned blob
  LOS triggers a scan — not reflected in forecast. Fix: add `scan_trigger_count` per direction (+8 dims).

- **OBS-6** [Medium]: Entity slot overflow (10 max). After blob splits, entity counts can exceed 10. Fix: add
  `overflow_count` and `nearest_overflow_distance` globals (+2 dims).

### Reward Issues

- **REW-1** [Bug]: `SURVIVAL_REWARD_PER_TICK = 0.005` is defined but never referenced in `_calculate_internal()`.
  Remove or implement.

- **REW-4** [Medium]: Damage-no-move penalty uses action-index check instead of `result.player_moved`. False positive
  on attack-drag movement. **Status: removed in V41** (DAMAGE_NO_MOVE disabled entirely).
