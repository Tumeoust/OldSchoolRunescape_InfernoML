# V45 TB Tracking

## Goal

Generalize the agent across gear setups and stat levels. V44 trained a single loadout (crystal armor + BoFa, 99/99/99/99)
to 89% W49-66 clear rate. V45 introduces 5 loadout variants randomized per episode, forcing the model to adapt its play
to weapon speed, attack range, defensive stats, and HP caps. Secondary: melee mechanics fixes improve simulation fidelity.

## Base Checkpoint

Warmstart from V44 best: `ckpt_1100` (~132.7M steps, 89% W49-66 clear rate). Input layer resized 363→371 (+8 loadout
features). ~96 hours of V44 training preserved.

## Why V45 Exists

V44 achieved strong performance but is brittle — it learned one gear setup with hardcoded 99 HP, speed 4, range 10. The
agent can't transfer to different weapons, stat levels, or gear without retraining from scratch. For the agent to
generalize across account builds, it needs to:

1. **Read its own stats.** The loadout observation block (8 dims) tells the model what weapon it's holding, how fast it
   attacks, how far it can shoot, how tanky it is, and whether blowpipe is available. Without these features the policy
   has no way to distinguish setups.

2. **Adapt combat behavior.** A 1-def ACB pure (85 HP, 98.5% mager melee accuracy) needs flawless prayer and
   safespotting. A max tbow (68 max hit vs magers) can play more aggressively. The model must learn these trade-offs.

3. **Handle variable action space.** CRYSTAL_NO_BP masks the blowpipe switch action. The model must learn to fight
   without blowpipe when it's unavailable.

## What Changed (V44 → V45)

### 1. Multi-loadout system

5 loadouts with distinct gear, stats, and combat tables:

| ID            | Weapon            | Stats (HP/R/M/D) | Speed | Range | BP  | Tbow | vs Mager (acc / max) |
|---------------|-------------------|------------------|-------|-------|-----|------|----------------------|
| BUDGET_RCB    | Rune crossbow     | 80/80/94/75      | 5     | 7     | Yes | No   | 0.651 / 29           |
| MID_ACB       | Armadyl crossbow  | 85/85/90/1       | 5     | 8     | Yes | No   | 0.666 / 31           |
| CRYSTAL_BP    | Bow of Faerdhinen | 90/90/94/85      | 4     | 10    | Yes | No   | 0.804 / 37           |
| CRYSTAL_NO_BP | Bow of Faerdhinen | 90/90/94/85      | 4     | 10    | No  | No   | 0.804 / 37           |
| MAX_TBOW      | Twisted bow       | 99/99/99/99      | 5     | 10    | Yes | Yes  | 0.814 / 68           |

MID_ACB is the outlier — 1-defence pure with 30 ranged defence. Mager melee lands 98.5% of the time. Forces the model to
learn that some loadouts simply cannot tank hits.

Tbow passive implemented: accuracy and damage scale with target magic level (OSRS formula). Capped at +40% accuracy,
+150% damage. vs Mager (magic 300): 1.40x accuracy, 2.15x damage.

Combat tables pre-computed per loadout at import time (5 loadouts x 3 presets x 9 NPC types = trivial).

### 2. Observation expansion (363 → 371)

8-dim loadout block appended at the end of the actor observation (positions 317-324), after static slots and before
the privileged block. This placement keeps positions 0-316 byte-identical to V44, so `load_with_resize` only needs
to handle an insertion at position 317.

| Index | Feature                      | Normalization |
|-------|------------------------------|---------------|
| 317   | has_blowpipe                 | binary (0/1)  |
| 318   | main weapon speed            | / 5.0         |
| 319   | main weapon range            | / 10.0        |
| 320   | ranged attack bonus          | / 200.0       |
| 321   | ranged strength bonus        | / 130.0       |
| 322   | magic attack bonus           | / 80.0        |
| 323   | ranged defence bonus         | / 180.0       |
| 324   | max health (hitpoints level) | / 99.0        |

HP normalization changed: `player_health / max_health` (was `/ 99.0`). 1.0 = full HP regardless of loadout.

Actor: 325 dims (was 317). Critic: 371 dims (was 363). Privileged block starts at 325 (was 317).

### 3. Action masking for blowpipe

`SWITCH_BLOWPIPE` (switch head index 1) masked when `state.has_blowpipe == False`. Simulator also guards the switch as
defense-in-depth (returns False / no-op).

### 4. Parameterized combat formulas

All combat roll functions accept `ranged_level`, `magic_level`, `defence_level` kwargs (default 99). Blood barrage heal
cap uses `state.max_health` instead of hardcoded 99. Blood barrage high-HP threshold scales: `> max_health - 4` (was
`> 95`). Wave completion HP bonus: `health / max_health` (was `/ 99.0`).

### 5. Melee mechanics fixes

[TODO: document melee fixes made by user]

## V45 Hypotheses

1. **Loadout features are learnable.** The 8-dim block provides sufficient signal for the policy to distinguish setups.
   Expect: within 20M steps of warmstart, the model should show loadout-dependent behavior (e.g., lower blood barrage
   usage on low-HP loadouts, more conservative positioning on 1-def).

2. **Crystal-only warmstart preserves V44 performance.** Phase 1 (crystal-only) should recover to ~85%+ clear rate
   within 5-10M steps despite the obs space change, since the loadout block values are constant and the rest of the
   observation is identical.

3. **Gradual loadout introduction prevents catastrophic forgetting.** Jumping to uniform 5-loadout from crystal-only
   would crash performance. Phased introduction (crystal → crystal+budget → all) should maintain >70% crystal clear rate
   while learning new loadouts.

4. **1-def ACB is the hardest loadout.** Expect this to be the last loadout to reach acceptable clear rates. The model
   needs near-perfect prayer to survive, which is a higher bar than the other loadouts.

5. **Tbow loadout should converge fastest** (after crystal). Higher damage = faster kills = shorter exposure = fewer
   things to learn. The model just needs to adjust to speed 5 instead of 4.

## Training Plan

### Phase 1: Crystal-only warmstart (recover V44 performance)

Warmstart from V44 ckpt_1100 with `--loadout CRYSTAL_BP`. Observation space is 371 but loadout block is constant
(identical to V44 crystal stats). Goal: recover to 85%+ W49-66 clear rate. This validates that the obs resize and melee
fixes don't break anything.

**Stop condition:** 3 consecutive eval windows at 80%+ clear rate, or 20M steps (whichever first).

### Phase 2: Crystal + close variants

Add CRYSTAL_NO_BP and MAX_TBOW:
`--loadout-weights '{"CRYSTAL_BP":0.5,"CRYSTAL_NO_BP":0.2,"MAX_TBOW":0.3}'`

These are the closest to the learned policy. CRYSTAL_NO_BP is identical stats, just no blowpipe. MAX_TBOW has same range
and similar stats, just different damage curve.

**Stop condition:** All 3 loadouts at 60%+ individual clear rate.

### Phase 3: Full diversity

Add budget and ACB:
`--loadout-weights '{"CRYSTAL_BP":0.3,"CRYSTAL_NO_BP":0.1,"MAX_TBOW":0.2,"BUDGET_RCB":0.2,"MID_ACB":0.2}'`

Or uniform: no flags (default uniform random across all 5).

**Stop condition:** Average clear rate across all loadouts >50%, no single loadout below 30%.

## Training Settings

| Setting             | Value                   | Notes                                         |
|---------------------|-------------------------|-----------------------------------------------|
| warmstart           | V44 ckpt_1100 (~132.7M) | Input layer resize 363→371                    |
| curriculum-mode     | static                  | Same as V44                                   |
| phase               | sweep                   | Same as V44                                   |
| start-wave          | 49                      | Same as V44 final                             |
| max-wave            | 66                      | Same                                          |
| observation-version | v3.2                    | Same schema, +8 loadout dims                  |
| policy-arch         | flat_lstm_residual      | Same                                          |
| lstm-hidden-size    | 128                     | Same                                          |
| lstm-seq-len        | 16                      | Same                                          |
| lstm-burn-in        | 8                       | Same                                          |
| actor/critic sizes  | 512,512 / 512,512       | Same                                          |
| n-envs              | 64                      | Same                                          |
| n-steps             | 256                     | Same                                          |
| batch-size          | 4096                    | Same                                          |
| n-epochs            | 2                       | Same as V44 final                             |
| lr                  | 1.5e-4                  | Same as V44 final                             |
| target-kl           | 0.02                    | Same                                          |
| entropy-start/end   | 0.05 / 0.002            | Same                                          |
| gamma               | 0.997                   | Same as V44 final                             |
| gae-lambda          | 0.95                    | Same                                          |
| vf-coef             | 0.25                    | Same as V44 final                             |
| max-grad-norm       | 1.0                     | Same as V44 final                             |
| normalize-reward    | yes                     | Same                                          |
| normalize-obs       | yes                     | Same                                          |
| loadout             | CRYSTAL_BP (phase 1)    | Single loadout for warmstart recovery         |
| reward-schedules    | V44 (all decayed)       | Schedules already at terminal values from V44 |
| log-reward-terms    | yes                     | Required                                      |
| checkpoint-every    | 100                     | Same                                          |

## Starting Command (Phase 1)

```powershell
Start-Process -FilePath python -ArgumentList "-m tools.inferno_rl.train_gpu --load models/V45/inferno_gpu_w49-66_20260318_163842_400.pt --curriculum-mode static --phase sweep --start-wave 49 --max-wave 66 --observation-version v3.2 --policy-arch flat_lstm_residual --lstm-hidden-size 128 --lstm-seq-len 16 --lstm-burn-in 8 --episode-mode full --n-envs 64 --n-steps 256 --batch-size 4096 --n-epochs 2 --lr 1.5e-4 --target-kl 0.02 --entropy-start 0.05 --entropy-end 0.002 --gamma 0.997 --gae-lambda 0.95 --vf-coef 0.25 --max-grad-norm 1.0 --normalize-obs --normalize-reward --actor-sizes 512,512 --critic-sizes 512,512 --save-dir models/V45 --log-dir logs/V45 --checkpoint-every 100 --timesteps 200000000 --device cuda --log-reward-terms --loadout CRYSTAL_BP"
```

## Starting Command (Phase 2)

```powershell
# Direct (run in activated venv terminal):
python -m tools.inferno_rl.train_gpu --load models/V45/inferno_gpu_w49-66_20260318_224659_800.pt --curriculum-mode static --phase sweep --start-wave 49 --max-wave 66 --observation-version v3.2 --policy-arch flat_lstm_residual --lstm-hidden-size 128 --lstm-seq-len 16 --lstm-burn-in 8 --episode-mode full --n-envs 64 --n-steps 256 --batch-size 4096 --n-epochs 2 --lr 1.5e-4 --target-kl 0.02 --entropy-start 0.05 --entropy-end 0.002 --gamma 0.997 --gae-lambda 0.95 --vf-coef 0.25 --max-grad-norm 1.0 --normalize-obs --normalize-reward --actor-sizes 512,512 --critic-sizes 512,512 --save-dir models/V45 --log-dir logs/V45 --checkpoint-every 100 --timesteps 200000000 --device cuda --log-reward-terms --loadout-weights '{\"CRYSTAL_BP\":0.5,\"CRYSTAL_NO_BP\":0.2,\"MAX_TBOW\":0.3}'

# Background (new window):
Start-Process -FilePath python -ArgumentList "-m tools.inferno_rl.train_gpu --load models/V45/inferno_gpu_w49-66_20260318_224659_800.pt --curriculum-mode static --phase sweep --start-wave 49 --max-wave 66 --observation-version v3.2 --policy-arch flat_lstm_residual --lstm-hidden-size 128 --lstm-seq-len 16 --lstm-burn-in 8 --episode-mode full --n-envs 64 --n-steps 256 --batch-size 4096 --n-epochs 2 --lr 1.5e-4 --target-kl 0.02 --entropy-start 0.05 --entropy-end 0.002 --gamma 0.997 --gae-lambda 0.95 --vf-coef 0.25 --max-grad-norm 1.0 --normalize-obs --normalize-reward --actor-sizes 512,512 --critic-sizes 512,512 --save-dir models/V45 --log-dir logs/V45 --checkpoint-every 100 --timesteps 200000000 --device cuda --log-reward-terms --loadout-weights '{\`""CRYSTAL_BP\`"":0.5,\`""CRYSTAL_NO_BP\`"":0.2,\`""MAX_TBOW\`"":0.3}'"
```

## Starting Command (Phase 3)

```powershell
# Direct (run in activated venv terminal):
python -m tools.inferno_rl.train_gpu --load models/V45/inferno_gpu_w49-66_20260319_094424_800.pt --curriculum-mode static --phase sweep --start-wave 49 --max-wave 66 --observation-version v3.2 --policy-arch flat_lstm_residual --lstm-hidden-size 128 --lstm-seq-len 16 --lstm-burn-in 8 --episode-mode full --n-envs 64 --n-steps 256 --batch-size 4096 --n-epochs 2 --lr 1.5e-4 --target-kl 0.02 --entropy-start 0.05 --entropy-end 0.002 --gamma 0.998 --gae-lambda 0.95 --vf-coef 0.25 --max-grad-norm 1.0 --normalize-obs --normalize-reward --actor-sizes 512,512 --critic-sizes 512,512 --save-dir models/V45 --log-dir logs/V45 --checkpoint-every 100 --timesteps 200000000 --device cuda --log-reward-terms --loadout-weights '{\"CRYSTAL_BP\":0.3,\"CRYSTAL_NO_BP\":0.1,\"MAX_TBOW\":0.2,\"BUDGET_RCB\":0.2,\"MID_ACB\":0.2}'

# Background (new window):
Start-Process -FilePath python -ArgumentList "-m tools.inferno_rl.train_gpu --load models/V45/inferno_gpu_w49-66_20260319_094424_800.pt --curriculum-mode static --phase sweep --start-wave 49 --max-wave 66 --observation-version v3.2 --policy-arch flat_lstm_residual --lstm-hidden-size 128 --lstm-seq-len 16 --lstm-burn-in 8 --episode-mode full --n-envs 64 --n-steps 256 --batch-size 4096 --n-epochs 2 --lr 1.5e-4 --target-kl 0.02 --entropy-start 0.05 --entropy-end 0.002 --gamma 0.998 --gae-lambda 0.95 --vf-coef 0.25 --max-grad-norm 1.0 --normalize-obs --normalize-reward --actor-sizes 512,512 --critic-sizes 512,512 --save-dir models/V45 --log-dir logs/V45 --checkpoint-every 100 --timesteps 200000000 --device cuda --log-reward-terms --loadout-weights '{\`""CRYSTAL_BP\`"":0.3,\`""CRYSTAL_NO_BP\`"":0.1,\`""MAX_TBOW\`"":0.2,\`""BUDGET_RCB\`"":0.2,\`""MID_ACB\`"":0.2}'"
```

## Metrics Log

| Steps  | Eps | Deaths | Timeout% | EV   | KL    | VL    | Clip  | Ent   | Return | RVar | Clr% | Notes                                                                                                                                                                                                                      |
|--------|-----|--------|----------|------|-------|-------|-------|-------|--------|------|------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 132.8M | 8   | 5      | 0%       | 0.89 | 0.001 | 0.066 | 0.006 | 0.050 | 3.38   | 22.6 | 38%  | First V45 rollouts (~0.1M new). EV retained from warmstart. max_w49=56. 437 FPS.                                                                                                                                           |
| 133.2M | 12  | 6      | 0%       | 0.91 | 0.001 | 0.052 | 0.003 | 0.050 | 2.90   | 22.6 | 50%  | EV rising (0.89→0.91). max_w49=61, full clear from w51. VL dropping. 420 FPS.                                                                                                                                              |
| 134.0M | 11  | 5      | 0%       | 0.86 | 0.001 | 0.066 | 0.004 | 0.050 | 2.93   | 22.5 | 55%  | ~1.3M new. EV dipped (0.91→0.86). max_w49=63, clears from w54/w59/w60. Ep len up (160). 427 FPS.                                                                                                                           |
| 134.8M | 11  | 4      | 0%       | 0.93 | 0.001 | 0.048 | 0.004 | 0.050 | 2.89   | 22.5 | 64%  | EV recovered strong (0.86→0.93). First full clear from w49! VL lowest yet (0.048). 449 FPS.                                                                                                                                |
| 135.6M | 16  | 6      | 0%       | 0.92 | 0.001 | 0.054 | 0.003 | 0.049 | 2.80   | 22.5 | 63%  | EV stable (0.92). Clr% holding ~63%. Damage_Taken spike (-0.028/tick). 473 FPS (+24).                                                                                                                                      |
| 136.5M | 6   | 0      | 0%       | 0.95 | 0.001 | 0.033 | 0.004 | 0.049 | 2.83   | 22.5 | 100% | **0 deaths this rollout (6/6 clear).** EV new high (0.95). VL=0.033, best yet. Dmg_Taken back to normal (-0.008). 455 FPS.                                                                                                 |
| 137.3M | 13  | 8      | 0%       | 0.86 | 0.001 | 0.083 | 0.004 | 0.049 | 2.94   | 22.4 | 38%  | Correction after 100% rollout — 8/13 deaths (38%). EV dipped (0.86), VL spiked (0.083). Full clears from w50-w54/w56/w57. 452 FPS.                                                                                         |
| 138.1M | 18  | 4      | 0%       | 0.94 | 0.001 | 0.046 | 0.004 | 0.049 | 2.76   | 22.4 | 78%  | Strong bounce: 14/18 clear (78%). EV back to 0.94. Full clear from w49 again. Kill sums climbing. 444 FPS.                                                                                                                 |
| 138.8M | 15  | 5      | 0%       | 0.93 | 0.001 | 0.054 | 0.005 | 0.049 | 3.12   | 22.4 | 67%  | Steady 67% (10/15). EV 0.93 stable. Return up to 3.12 (new high). FPS dipped to 379 (longer eps, 142 avg).                                                                                                                 |
| 139.6M | 12  | 4      | 0%       | 0.92 | 0.001 | 0.059 | 0.003 | 0.048 | 3.08   | 22.4 | 67%  | Holding 67% (8/12). Full clears from w49/w50/w51/w58. Kill_Zek=6.4, Dmg_Dealt=68.3 (both highs). 423 FPS.                                                                                                                  |
| 140.2M | 10  | 2      | 0%       | 0.95 | 0.001 | 0.036 | 0.004 | 0.048 | 2.88   | 22.4 | 80%  | **80% clear (2/10 deaths).** EV=0.95, VL=0.036. FPS dropped to 301 (train/time=0.9s, possible GPU contention).                                                                                                             |
| 153.9M | 8   | 3      | 0%       | 0.92 | 0.001 | 0.055 | 0.005 | 0.047 | 3.32   | 22.0 | 63%  | **Overnight: +13.7M steps.** Clr% 63% (5/8). Return 3.32 (new high). RVar drifting down (22.0). Kill sums strong. 399 FPS.                                                                                                 |
| 153.7M | 8   | 2      | 0%       | 0.93 | 0.001 | 0.051 | 0.004 | 0.050 | 3.17   | 22.0 | 75%  | **Phase 2 start** (3 loadouts). Clr% 75% (6/8). EV 0.93. Full clear from w49. Entropy reset to 0.050. 412 FPS.                                                                                                             |
| 154.3M | 12  | 4      | 0%       | 0.93 | 0.001 | 0.055 | 0.004 | 0.050 | 2.96   | 22.0 | 67%  | P2 +0.6M. Clr% 67% (8/12). EV stable. Clears from w52/w54/w56/w58-w66. No forgetting signal. 408 FPS.                                                                                                                      |
| 154.9M | 13  | 9      | 0%       | 0.88 | 0.001 | 0.075 | 0.004 | 0.049 | 3.09   | 22.0 | 31%  | Bad rollout: 9/13 deaths (31%). EV dipped to 0.88, VL spiked 0.075. Likely new-loadout episodes dying early. Dmg_Taken up (-0.014/t). 366 FPS.                                                                             |
| 155.7M | 13  | 2      | 0%       | 0.96 | 0.001 | 0.039 | 0.003 | 0.049 | 3.06   | 22.0 | 85%  | **Strong recovery: 85% (2/13 deaths).** EV 0.96 (new P2 high). VL=0.039. Waves_completed=89. Kill sums high. 423 FPS.                                                                                                      |
| 156.4M | 13  | 3      | 0%       | 0.94 | 0.001 | 0.049 | 0.005 | 0.049 | 2.90   | 22.0 | 77%  | Holding: 77% (3/13). EV 0.94. Ep len 166 (long runs). Dmg_Dealt=75.1, Kill_Zek=7.2, Wave_Complete=21.7 — all P2 highs. 362 FPS.                                                                                            |
| 157.0M | 4   | 1      | 0%       | 0.94 | 0.001 | 0.045 | 0.005 | 0.049 | 3.11   | 22.0 | 75%  | Small sample (4 eps, 1 death). Full clear from w49. Waves_completed=90. Stall_Penalty lowest yet (-0.006/t). 411 FPS.                                                                                                      |
| 157.7M | 10  | 2      | 0%       | 0.94 | 0.001 | 0.041 | 0.007 | 0.049 | 3.12   | 22.0 | 80%  | Solid 80% (2/10). EV 0.94. Clip ticked up (0.007) + KL 0.0013 — policy updating more actively. Ep len 168. max_ep_reward=5.02. 372 FPS.                                                                                    |
| 158.4M | 11  | 5      | 0%       | 0.91 | 0.001 | 0.062 | 0.009 | 0.049 | 3.19   | 22.0 | 55%  | Dip: 55% (5/11). Clip 0.009 (highest P2), KL 0.0013 sustained. Dmg_Taken spike (-0.021/t). Stall up (-0.003/t). FPS 302.                                                                                                   |
| 159.0M | 11  | 5      | 0%       | 0.92 | 0.001 | 0.061 | 0.005 | 0.048 | 3.13   | 22.0 | 55%  | Flat: 55% again (5/11). EV 0.92. Clip settled back (0.005). Kill sums lower — shorter eps dying earlier. Clears from w54/w57/w60-w62/w64-w66. 346 FPS.                                                                     |
| 159.6M | 14  | 1      | 0%       | 0.93 | 0.001 | 0.044 | 0.004 | 0.048 | 3.39   | 21.9 | 93%  | **93% clear (1/14 deaths).** Bounce from 55% dip. Return 3.39 (P2 high). Kill_Zek=6.9, Wave_Complete=19.6. Broad clears w50-w58/w61/w64-w66. 364 FPS.                                                                      |
| 160.3M | 13  | 7      | 0%       | 0.86 | 0.001 | 0.089 | 0.006 | 0.048 | 3.13   | 21.9 | 46%  | Swing back: 46% (7/13). EV 0.86, VL 0.089 (P2 high). Stall up (-0.002/t). Kill sums down. Clears from w50/w55/w60/w62/w64/w66. 372 FPS.                                                                                    |
| 160.9M | 11  | 5      | 0%       | 0.92 | 0.001 | 0.072 | 0.006 | 0.048 | 3.19   | 21.9 | 55%  | Mid-range: 55% (5/11). EV recovered (0.92). Stall back down (-0.001/t). Return stable at 3.19. Clears from w55/w57/w58/w60/w63/w64/w66. 382 FPS.                                                                           |
| 161.6M | 10  | 4      | 0%       | 0.91 | 0.001 | 0.051 | 0.006 | 0.048 | 3.03   | 21.9 | 60%  | 60% (4/10). EV 0.91. Wave_Complete=21, Kill_Zek=6.8, Dmg_Dealt=71.5 — good kill sums. BB_at_HighHP up (-0.002/t), agent barraging more aggressively. 349 FPS.                                                              |
| 162.3M | 8   | 3      | 0%       | 0.90 | 0.001 | 0.061 | 0.007 | 0.048 | 3.16   | 21.9 | 63%  | 63% (3/8). EV 0.90. Full clear from w49. min_ep_reward=1.18 (no negative eps — even deaths scored well). Mager_Priority up (0.004/t). 399 FPS.                                                                             |
| 163.1M | 10  | 6      | 0%       | 0.88 | 0.001 | 0.074 | 0.004 | 0.047 | 3.15   | 21.9 | 40%  | Worst P2 rollout: 40% (6/10). EV dipped 0.88, VL spiked 0.074. phase_fail=0.6. min_ep_reward=-0.30 (first negative). Ep len short (111, min=10 — early deaths). Dmg_Taken up (-0.012/t). FPS 396.                          |
| 164.9M | 9   | 4      | 0%       | 0.91 | 0.001 | 0.059 | 0.007 | 0.047 | 3.24   | 21.9 | 56%  | Recovery from 40%: 56% (5/9). EV back to 0.91, VL down (0.059). Return 3.24. Clip up (0.007). Kill sums high (Zek=8.0, Wave_Complete=24.7). FPS crashed to 160 (eval CPU contention). min_ep_len=3 (one very early death). |
| 165.3M | 10  | 3      | 0%       | 0.91 | 0.001 | 0.057 | 0.004 | 0.047 | 3.26   | 21.9 | 70%  | Continuing recovery: 70% (3/10). FPS back to 414 (evals stopped). Waves_completed=94 (P2 high). phase_fail=0.3. min_ep_len=1 (still occasional instant death). Kill_Zek=7.1, Return stable at 3.26.                        |
| 165.5M | 17  | 13     | 0%       | 0.83 | 0.001 | 0.127 | 0.007 | 0.050 | 3.18   | 21.9 | 24%  | **Phase 3 start** (5 loadouts, gamma 0.998). 13/17 deaths (76%). EV crashed 0.91→0.83, VL spiked 0.127 — new loadouts disrupting value estimates. max_w49=56 (down from 66). Entropy reset 0.050. 392 FPS.                  |
| 166.2M | 14  | 9      | 0%       | 0.86 | 0.001 | 0.113 | 0.009 | 0.050 | 3.32   | 21.9 | 36%  | P3 +0.7M. Recovering: 36% (5/14). EV 0.83→0.86, VL 0.127→0.113. max_w49=66 (full clear back). Kill sums doubling (Zek=4.2, Wave_Complete=13). Return up 3.32. 436 FPS.                                                     |
| 167.0M | 11  | 6      | 0%       | 0.85 | 0.001 | 0.097 | 0.005 | 0.050 | 3.28   | 21.9 | 45%  | P3 +1.5M. 45% (5/11). VL continuing to drop (0.097). Stall up (-0.005/t). Dmg_Taken improved (-0.015/t). max_w49=64. Clip settled (0.005). 441 FPS.                                                                         |
| 167.8M | 21  | 10     | 0%       | 0.89 | 0.001 | 0.148 | 0.004 | 0.049 | 3.52   | 22.0 | 52%  | P3 +2.3M. **52% (11/21).** EV jumped 0.85→0.89. Return 3.52 (P3 high). VL spiked 0.148. Dmg_Taken best P3 (-0.011/t). Waves_completed=86. Kill_Zek=4.5. 442 FPS.                                                           |
| 168.6M | 12  | 4      | 0%       | 0.93 | 0.001 | 0.095 | 0.005 | 0.049 | 3.04   | 22.0 | 67%  | P3 +3.1M. **67% (8/12).** EV surged 0.89→0.93. VL spike resolved (0.095). min_ep_reward=0.61 (no negative eps). max_w49=66 (full clear). Kill_Zek=4.7. 440 FPS.                                                             |
| 169.4M | 11  | 7      | 0%       | 0.92 | 0.002 | 0.113 | 0.009 | 0.049 | 3.18   | 22.0 | 36%  | Dip: 36% (4/11). KL doubled (0.0016), grad spiked (0.66), clip up (0.009) — policy updating harder. Dmg_Taken up (-0.016/t). max_w49=63. min_ep_reward=-1.19. 446 FPS.                                                      |
| 170.1M | 18  | 8      | 0%       | 0.92 | 0.001 | 0.117 | 0.004 | 0.049 | 3.23   | 22.0 | 56%  | Bounce: 56% (10/18). KL/grad settled (0.0006/0.44). min_ep_reward=0.17 (all positive). Ep len up (137). Dmg_Taken improved (-0.011/t). max_w49=64. 411 FPS.                                                                  |
| 171.0M | 11  | 7      | 0%       | 0.93 | 0.001 | 0.109 | 0.003 | 0.049 | 3.33   | 22.1 | 36%  | Dip: 36% (4/11). But EV up (0.93), Return up (3.33). Kill sums P3 highs (Zek=5.7, Wave_Complete=17.5, Dmg_Dealt=60.4) — clears were deep. max_w49=61. 463 FPS.                                                              |
| 171.7M | 16  | 5      | 0%       | 0.94 | 0.001 | 0.103 | 0.003 | 0.049 | 3.58   | 22.1 | 69%  | **69% (11/16).** EV 0.94, Return 3.58 — both P3 highs. Dmg_Taken best P3 (-0.008/t). Ep len 155 (deep runs). phase_fail=0.31. 448 FPS.                                                                                      |
| 172.5M | 14  | 7      | 0%       | 0.93 | 0.001 | 0.100 | 0.003 | 0.048 | 3.38   | 22.1 | 50%  | Mid: 50% (7/14). EV stable 0.93. VL at 0.100 (P3 low). Ep len dropped to 71 (early deaths on weak loadouts). max_w49=55. 432 FPS.                                                                                           |
| 173.3M | 15  | 8      | 0%       | 0.94 | 0.001 | 0.112 | 0.004 | 0.048 | 3.56   | 22.1 | 47%  | Flat: 47% (7/15). EV 0.94 (P3 high). Return 3.56 (matching P3 best). Stall improved (-0.0015/t). Ep len recovered (121). max_w49=55. 427 FPS.                                                                               |
| 174.0M | 19  | 16     | 0%       | 0.93 | 0.001 | 0.132 | 0.003 | 0.048 | 3.55   | 22.2 | 16%  | **Worst P3: 16% (3/19).** phase_fail=0.84. Grad spiked (0.72). Dmg_Taken worst P3 (-0.017/t). But Return held (3.55), EV stable (0.93), ep len 141 — runs went deep before dying. Likely bad loadout sample. 466 FPS.        |
| 174.8M | 12  | 4      | 0%       | 0.97 | 0.001 | 0.079 | 0.005 | 0.048 | 3.65   | 22.2 | 67%  | **Huge bounce: 67% (8/12).** EV 0.97 (all-time P3 high!). VL 0.079 (P3 low). Return 3.65 (P3 high). Kill sums massive (Zek=6.1, Wave_Complete=18.2, Dmg_Dealt=63.7). 435 FPS.                                               |
| 175.6M | 14  | 6      | 0%       | 0.95 | 0.001 | 0.083 | 0.005 | 0.048 | 3.72   | 22.2 | 57%  | Solid: 57% (8/14). Return 3.72 (new P3 high!). EV 0.95. VL stable (0.083). max_w49=66 (full clear). Stall lowest P3 (-0.001/t). BB_at_HighHP lowest (-0.0006/t). 434 FPS.                                                   |
| 176.4M | 8   | 5      | 0%       | 0.96 | 0.001 | 0.061 | 0.004 | 0.047 | 3.89   | 22.2 | 38%  | Small sample: 38% (3/8). But **Return 3.89** (new P3 high!), **VL 0.061** (P3 low!). Grad 0.33 (smoothest P3). Dmg_Taken spike (-0.039/t) from early deaths. max_w49=64. 450 FPS.                                           |
| 177.1M | 8   | 2      | 0%       | 0.96 | 0.001 | 0.064 | 0.003 | 0.047 | 3.64   | 22.2 | 75%  | **75% (6/8).** phase_fail=0.25 (P3 best!). min_ep_len=82 (no early deaths!). min_ep_reward=0.84 (all positive). VL 0.064. Kill_ImKot=3.2 (P3 high). Ep len 165. 446 FPS.                                                    |
| 177.9M | 4   | 2      | 0%       | 0.96 | 0.001 | 0.073 | 0.006 | 0.047 | 3.86   | 22.3 | 50%  | Tiny sample: 50% (2/4). Return 3.86 (near P3 high). max_ep_reward=5.66 (P3 best!). BB_Heal=24.7, Mager_Resurrection=3.9 (P3 highs). High variance (std=2.14). EV 0.96. 442 FPS.                                             |
| 178.7M | 7   | 3      | 0%       | 0.97 | 0.001 | 0.069 | 0.008 | 0.047 | 3.74   | 22.3 | 57%  | Steady: 57% (4/7). EV 0.97 (matching P3 best). VL 0.069. Clip up (0.008). Melee_Resurrection=0.73 (P3 high — learning melee mechanics). Kill_ImKot=2.9. 439 FPS.                                                            |
| 179.5M | 7   | 2      | 0%       | 0.96 | 0.001 | 0.058 | 0.006 | 0.047 | 3.76   | 22.3 | 71%  | **71% (5/7).** VL 0.058 (new P3 low!). Kill sums at P3 highs (Zek=6.3, Dmg_Dealt=66.1, Mager_Res=4.4). std_reward=0.50 (very consistent). Dmg_Taken excellent (-0.009/t). 436 FPS.                                          |

## Reward-Term Watchlist

All V44 schedules are at terminal values (decayed). Key terms to watch:

- `Wave Complete` / `Damage Dealt` — should remain stable during phase 1, may dip during phase 2-3 introduction
- `Blood Barrage at High HP` — should trigger less on low-HP loadouts (80/85 HP) since threshold scales
- `Stall Penalty` — watch for spikes when new loadouts introduced (model may stall while adapting)
- `Mager Priority` — may shift with tbow (kills magers much faster, less time for priority tracking)

## Success Criteria

1. Phase 1: recover 85%+ crystal W49-66 clear rate within 10M steps of warmstart.
2. Phase 2: all 3 loadouts (crystal, crystal_no_bp, max_tbow) at 60%+ within 30M steps.
3. Phase 3: average clear rate >50% across all 5 loadouts within 60M steps.
4. No single loadout below 20% clear rate after 60M steps.

## Failure / Stop Criteria

1. Phase 1 doesn't recover to 70%+ within 20M steps (obs resize or melee fixes broke something fundamental).
2. Catastrophic forgetting: crystal clear rate drops below 50% after introducing new loadouts.
3. 1-def ACB loadout stays below 10% after 80M total steps (may need curriculum adjustment or loadout-specific reward).
4. V41-style divergence: VL + RVar + return all accelerating together.

## Eval Benchmarks

### W49→66, 100 seeds, deterministic

| Checkpoint      | Steps  | Loadout       | Clear% | Death% | Mean Wave | Notes                                                                                                                       |
|-----------------|--------|---------------|--------|--------|-----------|-----------------------------------------------------------------------------------------------------------------------------|
| ckpt_800 (run2) | 152.4M | CRYSTAL_BP    | ~90%   | ~10%   | ~65.4     | 50/100 partial (stopped early). Deaths at W54/61/61/62/63. Phase 1 passed.                                                  |
| ckpt_600 (run3) | 162.2M | CRYSTAL_BP    | ~75%   | ~25%   | ~64.8     | 28/50 partial. Deaths at W53/56/56/62/63/63/65. Weaker than Phase 1 baseline — multi-loadout training pulling down crystal. |
| ckpt_600 (run3) | 162.2M | CRYSTAL_NO_BP | ~68%   | ~32%   | ~64.2     | 28/50 partial. Deaths at W55/56/60/60/61/63/63/63+. Above 60% P2 target. Deaths cluster W60-63.                             |
| ckpt_600 (run3) | 162.2M | MAX_TBOW      | ~91%   | ~9%    | ~65.4     | 32/50 partial. 25 straight clears before first death. Deaths at W55/59/60. Strongest loadout by far.                        |
| ckpt_800 (run4) | 178.6M | BUDGET_RCB    | 33%    | 67%    | 61.5      | 30/30 complete. 10 clears. Deaths spread W51-W64, median W62. Above 30% floor target.                                      |
| ckpt_800 (run4) | 178.6M | MID_ACB       | 7%     | 93%    | 59.6      | 30/30 complete. 2 clears only. Mean wave 59.6 — gets deep but 1-def can't survive. Below 30% target.                       |
| ckpt_800 (run4) | 178.6M | CRYSTAL_BP    | 90%    | 10%    | 65.7      | 30/30 complete. 27 clears. Deaths at W58/65/65. **Up from 75% at P2 eval** — no forgetting, improved.                       |
| ckpt_800 (run4) | 178.6M | CRYSTAL_NO_BP | ~100%  | ~0%    | ~66.0     | 5/30 partial (stopped early). 5/5 clears. Likely strongest loadout.                                                         |

## Files Changed (V44 → V45)

| File                                               | Change                                                                                   |
|----------------------------------------------------|------------------------------------------------------------------------------------------|
| `tools/inferno_rl/simulator/equipment.py`          | LoadoutId, PlayerLevels, Loadout, 5 loadout definitions, LOADOUTS, DEFAULT               |
| `tools/inferno_rl/simulator/combat.py`             | Parameterized levels, tbow passive, CombatTables, build_combat_tables, ALL_COMBAT_TABLES |
| `tools/inferno_rl/simulator/state.py`              | max_health, has_blowpipe, loadout_preset_stats fields                                    |
| `tools/inferno_rl/simulator/simulator.py`          | combat_tables field, set_loadout() method                                                |
| `tools/inferno_rl/simulator/player_actions.py`     | combat_tables passthrough, max_health heal caps, blowpipe guard                          |
| `tools/inferno_rl/simulator/npc_combat.py`         | combat_tables passthrough to roll_npc_damage                                             |
| `tools/inferno_rl/simulator/step_result.py`        | max_health field, scaled blood barrage threshold                                         |
| `tools/inferno_rl/simulator/__init__.py`           | New exports (LoadoutId, Loadout, LOADOUTS, CombatTables, etc.)                           |
| `tools/inferno_rl/training/observation_common.py`  | LOADOUT_BLOCK_SIZE, normalization constants, obs size 363→371                            |
| `tools/inferno_rl/training/observation_v3.py`      | _fill_loadout_block (8 dims), HP normalization by max_health                             |
| `tools/inferno_rl/training/actions.py`             | Blowpipe switch mask when has_blowpipe=False                                             |
| `tools/inferno_rl/training/rewards.py`             | Wave HP bonus uses result.max_health                                                     |
| `tools/inferno_rl/training/env.py`                 | fixed_loadout, loadout_weights, _sample_loadout(), loadout in info dict                  |
| `tools/inferno_rl/async_env/subprocess_vec_env.py` | Loadout params passthrough to worker processes                                           |
| `tools/inferno_rl/train_gpu.py`                    | --loadout, --loadout-weights CLI args                                                    |
| `tools/inferno_rl/tests/test_observation_v32.py`   | Updated size assertions for +8 loadout block                                             |
