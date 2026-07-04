# V46 TB Tracking

## Goal

Force the model to learn perfect positioning by training all loadouts at 1 defence with uniform defensive bonuses. V45
Phase 3 showed strong crystal/tbow performance (90%+) but MID_ACB (1-def pure) at only 7% — the model never learned to
avoid damage because higher-def loadouts let it tank hits. At 1-def, NPCs hit ~95%+ of the time, so the *only* way to
survive is never letting them attack. Higher defence at inference is a free bonus.

## Base Checkpoint

Warmstart from V45 best: `ckpt_800` (run4, ~178.6M steps). Obs size unchanged (325 actor / 371 total). Standard
`PPO.load()` — no resize needed. Loadout block index 6 (ranged_defence) zeroed to 0.0 (was already low-variance).

## Why V46 Exists

V45 trained 5 loadouts with realistic defence stats. The model learned 5 different survival strategies:
- Crystal/tbow (85-99 def): tank occasional hits, pray correctly most of the time
- MID_ACB (1 def): die

Defence in OSRS only affects NPC hit chance, not damage. A model trained at 1-def is forced to learn the *universally
optimal* playstyle: never let NPCs attack (perfect safespotting + prayer). Higher def at inference only helps — some
attacks that would have hit now miss. This collapses 5 playstyles into 1.

## What Changed (V45 → V46)

### 1. Defence=1 for all loadouts

| Loadout       | Before (HP/R/M/D) | After (HP/R/M/D) |
|---------------|--------------------|--------------------|
| BUDGET_RCB    | 80/80/94/75        | 80/80/94/**1**     |
| MID_ACB       | 85/85/90/1         | 85/85/90/1         |
| CRYSTAL_BP    | 90/90/94/85        | 90/90/94/**1**     |
| CRYSTAL_NO_BP | 90/90/94/85        | 90/90/94/**1**     |
| MAX_TBOW      | 99/99/99/99        | 99/99/99/**1**     |

HP and offensive levels unchanged.

### 2. Uniform defensive bonuses (30 all styles)

All `AggregateStats` across all loadouts and presets now have identical defensive bonuses:
`stab_defence=30, slash_defence=30, crush_defence=30, ranged_defence=30, magic_defence=30`

Added `_with_uniform_defence()` helper applied to:
- `compute_aggregate_stats()` return value (covers `PRESET_STATS`, `_CRYSTAL_PRESET_STATS`)
- All hardcoded `AggregateStats` in BUDGET_RCB, MID_ACB, MAX_TBOW loadout definitions

Loadouts now differ ONLY in offensive stats (weapon speed, range, DPS, blowpipe/tbow availability) and HP.

### 3. Observation: zeroed defence dim

Loadout block index 6 (`ranged_defence / MAX_RANGED_DEFENCE_BONUS`) → `0.0`. This dim is now uninformative (all loadouts
have identical defence). The dim is kept at 0.0 (not removed) for checkpoint compatibility. Weights connected to it will
decay to zero via weight decay.

### 4. No other changes

- **combat.py**: `build_combat_tables()` reads loadout preset_stats → automatically picks up new defence=1 / bonuses=30.
  NPC accuracy will be very high (~95%+ for all attack styles).
- Obs size unchanged: 325 actor, 371 total.
- No action space changes, no reward changes.

## V46 Hypotheses

1. **1-def forces universal safespotting.** With ~95% NPC accuracy across all loadouts, the model can't survive by
   tanking. It must learn to: (a) always pray correctly against rangers/magers, (b) never be in melee range unless
   stacking behind pillar, (c) position so only 0-1 NPCs have LOS. This is optimal play regardless of def level.

2. **Crystal/tbow performance will drop initially.** These loadouts previously tanked occasional hits. At 1-def they'll
   die to the same mistakes that were previously forgiven. Expect 20-40M steps to recover as the model learns stricter
   positioning.

3. **MID_ACB will improve fastest** (relative to V45). It was already at 1-def — the only change is uniform defence
   bonuses (up from 5-30 → uniform 30), which is a slight buff. More importantly, the other loadouts now train the same
   survival strategy MID_ACB needs.

4. **Loadout convergence will be tighter.** Without defence as a differentiator, the main axes of variation are weapon
   speed/range/DPS and HP. The model should develop one core movement policy with minor weapon-speed adaptations.

5. **Transfer to real accounts will be better.** A model that never relies on defence will perform at least as well (and
   likely better) on any account, since real defence can only help.

## Training Plan

Single phase — no curriculum needed since the model already knows all 5 loadouts from V45. The change is environmental
(NPC accuracy), not observational.

Loadout weights: uniform across all 5 (default).

**Stop condition:** Average clear rate across all loadouts >50%, no single loadout below 25%.

## Training Settings

| Setting             | Value                   | Notes                                         |
|---------------------|-------------------------|-----------------------------------------------|
| warmstart           | V45 ckpt_800 (~178.6M)  | Standard PPO.load(), no resize                |
| curriculum-mode     | static                  | Same as V45                                   |
| phase               | sweep                   | Same as V45                                   |
| start-wave          | 49                      | Same as V45                                   |
| max-wave            | 66                      | Same                                          |
| observation-version | v3.2                    | Same schema, index 6 zeroed                   |
| policy-arch         | flat_lstm_residual      | Same                                          |
| lstm-hidden-size    | 128                     | Same                                          |
| lstm-seq-len        | 16                      | Same                                          |
| lstm-burn-in        | 8                       | Same                                          |
| actor/critic sizes  | 512,512 / 512,512       | Same                                          |
| n-envs              | 64                      | Same                                          |
| n-steps             | 256                     | Same                                          |
| batch-size          | 4096                    | Same                                          |
| n-epochs            | 2                       | Same                                          |
| lr                  | 1.5e-4                  | Same                                          |
| target-kl           | 0.02                    | Same                                          |
| entropy-start/end   | 0.05 / 0.002            | Same                                          |
| gamma               | 0.998                   | Same as V45 P3                                |
| gae-lambda          | 0.95                    | Same                                          |
| vf-coef             | 0.25                    | Same                                          |
| max-grad-norm       | 1.0                     | Same                                          |
| normalize-reward    | yes                     | Same                                          |
| normalize-obs       | yes                     | Same                                          |
| loadout             | uniform (default)       | All 5 loadouts, equal weight                  |
| reward-schedules    | V44 (all decayed)       | Inherited, at terminal values                 |
| log-reward-terms    | yes                     | Required                                      |
| checkpoint-every    | 100                     | Same                                          |

## Starting Command

```powershell
# Direct (run in activated venv terminal):
python -m tools.inferno_rl.train_gpu --load models/V45/inferno_gpu_w49-66_20260319_220107_800.pt --curriculum-mode static --phase sweep --start-wave 49 --max-wave 66 --observation-version v3.2 --policy-arch flat_lstm_residual --lstm-hidden-size 128 --lstm-seq-len 16 --lstm-burn-in 8 --episode-mode full --n-envs 64 --n-steps 256 --batch-size 4096 --n-epochs 2 --lr 1.5e-4 --target-kl 0.02 --entropy-start 0.05 --entropy-end 0.002 --gamma 0.998 --gae-lambda 0.95 --vf-coef 0.25 --max-grad-norm 1.0 --normalize-obs --normalize-reward --actor-sizes 512,512 --critic-sizes 512,512 --save-dir models/V46 --log-dir logs/V46 --checkpoint-every 100 --timesteps 200000000 --device cuda --log-reward-terms

# Background (new window):
Start-Process -FilePath python -ArgumentList "-m tools.inferno_rl.train_gpu --load models/V45/inferno_gpu_w49-66_20260319_220107_800.pt --curriculum-mode static --phase sweep --start-wave 49 --max-wave 66 --observation-version v3.2 --policy-arch flat_lstm_residual --lstm-hidden-size 128 --lstm-seq-len 16 --lstm-burn-in 8 --episode-mode full --n-envs 64 --n-steps 256 --batch-size 4096 --n-epochs 2 --lr 1.5e-4 --target-kl 0.02 --entropy-start 0.05 --entropy-end 0.002 --gamma 0.998 --gae-lambda 0.95 --vf-coef 0.25 --max-grad-norm 1.0 --normalize-obs --normalize-reward --actor-sizes 512,512 --critic-sizes 512,512 --save-dir models/V46 --log-dir logs/V46 --checkpoint-every 100 --timesteps 200000000 --device cuda --log-reward-terms"
```

## Continuation Command

```powershell
# Resume from ckpt_400 (~185.5M steps):
$p = Start-Process -FilePath python -PassThru -ArgumentList "-m tools.inferno_rl.train_gpu --load models/V46/inferno_gpu_w49-66_20260320_162754_200.pt --curriculum-mode static --phase sweep --start-wave 49 --max-wave 66 --observation-version v3.2 --policy-arch flat_lstm_residual --lstm-hidden-size 128 --lstm-seq-len 16 --lstm-burn-in 8 --episode-mode full --n-envs 32 --n-steps 256 --batch-size 4096 --n-epochs 2 --lr 1.5e-4 --target-kl 0.02 --entropy-start 0.05 --entropy-end 0.002 --gamma 0.998 --gae-lambda 0.95 --vf-coef 0.25 --max-grad-norm 1.0 --normalize-obs --normalize-reward --actor-sizes 512,512 --critic-sizes 512,512 --save-dir models/V46 --log-dir logs/V46 --checkpoint-every 100 --timesteps 200000000 --device cuda --log-reward-terms"; $p.PriorityClass = "BelowNormal"
```

## Metrics Log

| Steps  | Eps | Deaths | Timeout% | EV   | KL    | VL    | Clip  | Ent   | Return | RVar | Clr% | Notes |
|--------|-----|--------|----------|------|-------|-------|-------|-------|--------|------|------|-------|
| 178.7M | 21  | 14     | 5%       | 0.94 | 0.001 | 0.12  | 0.004 | 0.050 | 3.34   | 22.3 | 29%  | First log, ~0.15M V46 steps. Max wave 66 (from w51/56/65/66). DmgTaken/tick -0.025. |
| 179.0M | 18  | 14     | 0%       | 0.87 | 0.001 | 0.15  | 0.006 | 0.050 | 3.00   | 22.3 | 22%  | EV dipped 0.94→0.87. Clr% down 29→22%. DmgTaken/tick improved -0.025→-0.019. Ep len up 112→144. Max wave 66 (w57/64/65/66). |
| 179.7M | 20  | 15     | 0%       | 0.94 | 0.001 | 0.11  | 0.004 | 0.050 | 2.96   | 22.3 | 25%  | EV recovered 0.87→0.94. Clr% up 22→25%. DmgTaken/tick spiked -0.019→-0.030 (noisy). Stall penalty doubled. Max wave 66 (w61-66). |
| 180.4M | 14  | 6      | 7%       | 0.93 | 0.001 | 0.09  | 0.004 | 0.050 | 3.07   | 22.3 | 50%  | Clr% jumped 25→50%. Deaths 15→6. DmgTaken/tick -0.030→-0.013 (big improvement). Grad norm 0.37→0.72. Kill_Zek doubled. Small sample (14 eps). |
| 181.1M | 20  | 11     | 0%       | 0.94 | 0.001 | 0.08  | 0.003 | 0.049 | 2.90   | 22.3 | 45%  | Clr% settled 50→45% (larger sample, 20 eps). DmgTaken/tick stable -0.015. Grad norm back to 0.41. Mager_Delay rising (-0.007/tick). Max wave 66 (w54/61-66). |
| 181.7M | 22  | 18     | 0%       | 0.92 | 0.001 | 0.13  | 0.004 | 0.049 | 3.06   | 22.4 | 18%  | **Clr% dropped 45→18%.** Deaths 11→18. DmgTaken/tick -0.015→-0.021. Ep len 119→93. VL spiked 0.08→0.13. Max wave 66 only from w64+. Likely noisy batch — recheck next log. |
| 182.5M | 13  | 9      | 0%       | 0.92 | 0.001 | 0.08  | 0.004 | 0.049 | 2.90   | 22.4 | 31%  | Clr% recovered 18→31% (confirms outlier). DmgTaken/tick back to -0.013. VL normalized 0.13→0.08. Ep len 93→132. FPS up 336→400. Max wave 66 (w51/60/64-66). |
| 183.2M | 17  | 10     | 0%       | 0.95 | 0.001 | 0.08  | 0.005 | 0.049 | 2.97   | 22.4 | 41%  | Clr% up 31→41%. EV best yet 0.95. Max w66 from many starts (w51/53/56-59). Mager_Delay rising -0.008/tick. NE_Pillar_Dmg spiked -0.0012. |
| 183.9M | 10  | 7      | 0%       | 0.91 | 0.001 | 0.10  | 0.005 | 0.049 | 3.12   | 22.4 | 30%  | Clr% 41→30% (tiny sample, 10 eps). Per-tick Wave_Complete & Early_Mager_Kill at highs. DmgTaken/tick stable -0.013. FPS 463. Stall_Penalty up -0.004. |
| 184.7M | 17  | 13     | 0%       | 0.92 | 0.001 | 0.10  | 0.004 | 0.049 | 3.14   | 22.4 | 24%  | Clr% 30→24%. Recent 4-log avg ~31%. DmgTaken/tick stable -0.014. Stall_Penalty normalized -0.002. Return up 3.14. Max w66 from w50/52/59/62. |
| 185.5M | 17  | 11     | 0%       | 0.94 | 0.001 | 0.11  | 0.004 | 0.048 | 3.19   | 22.4 | 35%  | Clr% up 24→35%. Return keeps climbing (3.19). DmgTaken/tick -0.019 (noisy batch). EV back to 0.94. Max w66 from w52/54/57/62-66. ~7M V46 steps. |
| 185.7M | 13  | 8      | 0%       | 0.94 | 0.001 | 0.07  | 0.006 | 0.048 | 2.99   | 22.4 | 38%  | Resumed from ckpt_400. Clr% 38%. VL lowest yet 0.07. Max w66 from w49/50/52/53/56. DmgTaken/tick -0.016. |
| 185.2M | 9   | 7      | 0%       | 0.94 | 0.001 | 0.07  | 0.005 | 0.050 | 3.48   | 22.4 | 22%  | New event file (continuation run). Ent reset to 0.050. New terms: Avoidable_LOS, C_Tile_Position. DmgTaken/tick -0.011 (best). Return 3.48 (best). Tiny sample (9 eps). |
| 187.7M | 8   | 7      | 0%       | 0.95 | 0.001 | 0.08  | 0.004 | 0.049 | 3.48   | 22.5 | 13%  | Clr% 13% (1/8 eps, noise). Return stable 3.48. DmgTaken/tick -0.012. EV 0.95. Max w66 from w51/62-66. +2.5M continuation steps. |
| 187.7M | 15  | 11     | 0%       | 0.93 | 0.001 | 0.12  | 0.005 | 0.049 | 3.47   | 22.5 | 27%  | Clr% up 13→27% (4/15 eps). Grad norm jumped 0.44→0.59. VL up 0.08→0.12. DmgTaken/tick -0.013. Max w66 from w54/60/63/65/66. |
| 188.3M | 13  | 7      | 8%       | 0.95 | 0.001 | 0.07  | 0.003 | 0.049 | 3.13   | 22.5 | 38%  | Clr% up 27→38% (5/13). EV 0.95, VL 0.07 (best). DmgTaken/tick spiked -0.030 (recurring noise). 1 timeout. Max w66 from w53/61-66. |
| 189.1M | 16  | 11     | 0%       | 0.94 | 0.001 | 0.09  | 0.004 | 0.049 | 3.32   | 22.5 | 31%  | Clr% 38→31% (5/16). DmgTaken/tick recovering -0.030→-0.021. Return up 3.13→3.32. ~10.5M V46 total, ~3.8M continuation. |
| 188.5M | 15  | 13     | 0%       | 0.93 | 0.001 | 0.11  | 0.003 | 0.050 | 3.48   | 22.5 | 13%  | New run from ckpt_200 (~188.5M, rolled back). Ent reset 0.050. Clr% 13% (2/15 eps, very early ~2.8M steps). New terms: Single-LOS_Engagement, Tile_A_Proximity. DmgTaken/tick -0.018. Max w66 from w60/65 only. Grad 0.51. |

## Reward-Term Watchlist

Key terms to watch for 1-def training:

- `Damage Taken` — expect massive spike initially. The critical signal: this MUST decrease over time as the model learns
  to avoid being hit entirely. If it plateaus high, the model is failing to learn safespotting.
- `Blood Barrage Heal` — should increase (more damage to heal) but eventually decrease as the model takes less damage.
- `Stall Penalty` — may spike as the model becomes more cautious about positioning. Acceptable if damage taken drops.
- `Wave Complete` / `Kill_Zek` — expect sharp drop initially, gradual recovery.
- `Mager Priority` — critical. At 1-def, letting a mager melee you is near-certain death. Must stay high.

## Success Criteria

1. Average clear rate across all 5 loadouts >50% within 60M steps of V46 training.
2. No single loadout below 25% clear rate after 60M steps.
3. MID_ACB exceeds V45's 7% clear rate within 20M steps.
4. Damage_Taken per-tick converges below V45 levels (model learns to avoid hits, not just tank them).

## Failure / Stop Criteria

1. Clear rate doesn't recover above 30% average within 30M steps (1-def may be too harsh for warmstart).
2. Damage_Taken per-tick stays above -0.020 after 40M steps (model isn't learning to avoid hits).
3. V41-style divergence: VL + RVar + return all accelerating together.
4. Crystal clear rate stays below 40% after 40M steps (regression too severe, model can't adapt).

## Files Changed (V45 → V46)

| File                                          | Change                                                       |
|-----------------------------------------------|--------------------------------------------------------------|
| `tools/inferno_rl/simulator/equipment.py`     | defence=1 all loadouts, `_with_uniform_defence()` helper, uniform 30 def bonuses |
| `tools/inferno_rl/training/observation_v3.py` | Loadout block index 6 zeroed (ranged_defence → 0.0)         |
