# V30 TensorBoard Tracking

LSTM architecture on top of V29's [512,512] MLP. Hypothesis: LSTM with seq_len=10 enables
multi-tick reasoning (melee dig setups, barrage timing, prayer switch sequences) that a
memoryless MLP cannot represent. V25 tested LSTM with seq_len=32 but 32 ticks was too long.

## What Changed (V29 → V30)

### Observation Space

**Nibblers added to entity slots.** Previously excluded from the 16 entity slots (only visible
as a scalar count in wave context). Now sorted by priority alongside all other entities. The
model can see nibbler positions, HP, and which direction they're heading — enabling pillar
triage and barrage grouping decisions.

- `get_sorted_entities()` no longer filters `EntityTypes.NIBBLER`
- Nibbler count in wave context (`obs[184]`) retained as redundant summary
- Observation size unchanged (186 dims) — nibblers fill existing slots
- Max NPCs in any wave: 9 (wave 62) — 16 slots is sufficient

### Architecture

| Setting          | V29         | V30             |
|------------------|-------------|-----------------|
| actor-sizes      | 512,512     | 512,512         |
| critic-sizes     | 512,512     | 512,512         |
| LSTM             | none (MLP)  | **256 hidden**  |
| lstm-seq-len     | n/a         | **10**          |
| params           | ~1.2M       | ~1.5M (est)     |

### Reward Changes

| Reward | V29 | V30 | Rationale |
|--------|-----|-----|-----------|
| MAGER_PRIORITY_BONUS_PER_NPC | 0.0 | **0.25** | Nudge mager-first kill order. With 3 non-magers + 40dmg tbow hit: bonus = 0.3. One 30HP hit taken = -1.5. Damage penalty clearly dominates. |

### Training Settings

| Setting          | V29       | V30         | Rationale                                   |
|------------------|-----------|-------------|---------------------------------------------|
| warmstart        | KL distil | **none**    | Obs space changed (nibblers), fresh init    |
| n-epochs         | 1         | **2**       | KL 0.026 with 1 epoch leaves headroom      |
| n-envs           | 48        | 48          | Same                                        |
| batch-size       | 2048      | 2048        | Same                                        |
| lr               | 1e-4      | 1e-4        | Same                                        |
| entropy-start    | 0.01      | **0.02**    | Higher for random init — needs exploration  |
| entropy-end      | 0.001     | **0.002**   | V12_phase3 proven floor for from-scratch    |
| normalize-reward | no        | yes         | Proven stable with 48 envs (V29 R3)        |
| normalize-obs    | yes       | yes         | Same                                        |
| n-steps          | 1024      | 1024        | Same                                        |
| gamma            | 0.995     | 0.995       | Same                                        |
| gae-lambda       | 0.95      | 0.95        | Same                                        |
| vf-coef          | 0.5       | 0.5         | Same                                        |
| max-grad-norm    | 0.5       | 0.5         | Same                                        |

### Run Command

```powershell
python -m tools.inferno_rl.train_gpu --lstm-hidden-size 256 --lstm-seq-len 10 --phase climb --start-wave 55 --max-wave 66 --promote-after 5 --min-waves-to-advance 1 --save-dir models/V30_climb --log-dir logs/V30_climb --n-envs 48 --n-steps 1024 --batch-size 2048 --n-epochs 2 --lr 1e-4 --entropy-start 0.02 --entropy-end 0.002 --gamma 0.995 --gae-lambda 0.95 --vf-coef 0.5 --max-grad-norm 0.5 --normalize-reward --normalize-obs --checkpoint-every 100 --timesteps 200000000 --actor-sizes 512,512 --critic-sizes 512,512 --device cuda --log-reward-terms
```

### Continuation Command (from ckpt 1300, ~58M steps)

Global prestige sync added. Resume from checkpoint with same settings.

```powershell
python -m tools.inferno_rl.train_gpu --load models/V30_climb/inferno_gpu_w55-66_20260302_104452_1300.pt --lstm-hidden-size 256 --lstm-seq-len 10 --phase climb --start-wave 55 --max-wave 66 --promote-after 5 --min-waves-to-advance 1 --save-dir models/V30_climb --log-dir logs/V30_climb --n-envs 48 --n-steps 1024 --batch-size 2048 --n-epochs 2 --lr 1e-4 --entropy-start 0.02 --entropy-end 0.002 --gamma 0.995 --gae-lambda 0.95 --vf-coef 0.5 --max-grad-norm 0.5 --normalize-reward --normalize-obs --checkpoint-every 100 --timesteps 200000000 --actor-sizes 512,512 --critic-sizes 512,512 --device cuda --log-reward-terms
```

### Risks

1. **LSTM slows FPS** — sequential forward (10 steps per tick during training) slower than MLP. Expect ~30-50% FPS drop.
2. **2 epochs may push KL** — if KL sustains >0.05 in first 20M steps, drop back to 1 epoch.
3. **No warmstart** — random init + curriculum means ~10-20M extra steps vs a good warmstart. Negligible in a 200M run.
4. **LSTM gradients noisier** — BPTT can amplify gradient variance. grad-norm 0.5 should clamp the worst updates.

## Metrics Log

| Step | Frontier | Deaths | Waves Comp | Mean Reward | EV | KL | Entropy | Grad Norm | FPS | Notes |
|------|----------|--------|------------|-------------|-----|------|---------|-----------|-----|-------|
| 1.1M | 55 | 464 | 53 | -0.09 | 0.75 | 0.008 | -3.20 | 0.20 | 6256 | Fresh init, frontier stuck at 55. EV 0.75 healthy. KL very low. No timeouts. |
| 4.5M | 58 | 172 | 288 | 1.50 | 0.71 | 0.020 | -2.80 | 0.25 | 6188 | Frontier 55→58 in 3.4M steps. Deaths dropped 464→172, waves 53→288. Reward turned positive. Fast climb. |
| 7.7M | 58 | 170 | 287 | 1.79 | 0.63 | 0.038 | -2.58 | 0.31 | 5788 | Frontier stalled at 58. **KL 0.038 (watch)**, EV dropping 0.71→0.63. Entropy still decreasing. |
| 11.0M | 59 | 167 | 299 | 2.04 | 0.65 | 0.023 | -2.53 | 0.29 | 5351 | KL settled 0.038→0.023, EV recovering 0.63→0.65. Frontier 58→59. Reward climbing steadily. |
| 14.3M | 59 | 140 | 291 | 2.47 | 0.66 | 0.023 | -2.60 | 0.31 | 5467 | Frontier still 59 (6.6M steps). Deaths 167→140, reward 2.04→2.47. EV 0.66 stable. Survival improving without frontier advance. |
| 23.4M | 61 | 79 | 277 | 3.03 | 0.63 | 0.037 | -2.30 | 0.41 | 6497 | Frontier 59→61. Deaths halved 140→79. Reward 2.47→3.03. **KL 0.037 (watch)**. EV 0.63 flat. |
| 29.0M | 62 | 79 | 298 | 3.44 | 0.68 | 0.030 | -2.34 | 0.46 | 6337 | Frontier 61→62. KL settled 0.037→0.030. EV recovering 0.63→0.68. Waves 277→298, deaths flat at 79 (better rate). |
| 54.1M | 66 | 52 | 272 | 4.51 | 0.67 | 0.026 | -2.19 | 0.46 | 6127 | **Frontier 62→66 — hit ceiling!** Deaths 79→52. Reward 3.44→4.51. All metrics healthy. Frontier 66 @ 54M vs V21 @ 159M, V25 @ 120M. |
| 57.6M | 66 | 61 | 286 | 4.44 | 0.67 | 0.028 | -2.12 | 0.47 | 5952 | **Phase 1→2 (prestige).** Frontier back to 66 already (fast re-climb with min_waves=2). Deaths 52→61 (expected). All metrics stable. |
| — | — | — | — | — | — | — | — | — | — | **CODE CHANGE @ ~58M: Global prestige sync.** When any env prestiges, all envs reset frontier to start_wave and sync min_waves_to_advance. Previously per-env (most envs never prestiged). Resuming from ckpt 1300. |
| 75.0M | 66 | 60 | 299 | 4.32 | 0.75 | 0.022 | -2.31 | 0.47 | 5697 | New run from ckpt 1300 (+17M). Frontier back at 66, f.mean 60.9. Phase 1 (no prestige yet in new run). EV 0.67→0.75 (recovered). KL 0.022 (healthy). |
| 89.1M | 63 | 42 | 311 | 3.61 | 0.67 | 0.021 | -2.39 | 0.78 | 5904 | **Phase 1→2 (global prestige).** Frontier 66→63 (reset to 55, re-climbing). Deaths 60→42. EV 0.75→0.67 (prestige dip, expected). Grad 0.47→0.78 (elevated). |
| 105.4M | 59 | 23 | 345 | 5.71 | 0.72 | 0.019 | -2.33 | 0.40 | 6114 | **Phase 2→3 (global prestige, +16M).** Frontier reset, re-climbing (59, f.mean 56.8). Deaths 42→23 — best yet. Waves 311→345. EV recovering. Grad norm back to normal. |
| 116.0M | 63 | 26 | 334 | 7.28 | 0.77 | 0.018 | -2.25 | 0.40 | 5862 | Phase 3 climbing. Frontier 59→63, f.mean 60.0. Deaths stable 23→26. Reward 5.71→7.28. EV 0.77 — best since restart. All metrics clean. |
| 128.9M | 57 | 15 | 360 | 5.11 | 0.73 | 0.024 | -2.22 | 0.39 | 5880 | **Phase 3→4 (prestige, +12.9M).** Frontier reset, re-climbing (57, f.mean 55.5). Deaths 26→15 — new best. Reward dip expected post-prestige. |
| 145.6M | 63 | 27 | 346 | 8.02 | 0.82 | 0.025 | -2.05 | 0.42 | 5247 | Phase 4 climbing. Frontier 57→63, f.mean 59.1. Deaths 15→27 (harder waves at frontier). EV 0.82 — best since global prestige sync. Entropy -2.05 (slight upward trend, watch). |
| 159.1M | 63 | 37 | 342 | 7.89 | 0.81 | 0.028 | -2.01 | 0.41 | 5520 | **Frontier stalled at 63 (13.5M steps).** Deaths 27→37 (rising). Reward/EV flat. Entropy -2.01 (upward trend: -2.25→-2.01). Phase 4 requires clearing W63-66 in one episode — hard bar. Early stall watch. |
| 166.9M | 64 | 52 | 346 | 6.05 | 0.79 | 0.031 | -1.89 | 0.48 | 5406 | Frontier 63→64 (+1 in 7.8M). **Deaths surging 37→52 (+40%).** Reward 7.89→6.05 (declining). Entropy -2.01→-1.89 (accelerating). **KL 0.031 (watch).** Regression pattern emerging: frontier advances but survival deteriorates. |

## Reward Term Breakdown (ep_sum_mean)

Tracked at key checkpoints to detect reward balance drift. Values are mean per-episode sums across envs.

### @ 77.3M steps (Phase 1, frontier 66)

| Term | Value | Notes |
|------|-------|-------|
| Damage_Dealt | +25.39 | Densest signal — per-tick attack feedback |
| Wave_Complete | +9.37 | ~2.5 waves cleared/ep |
| Blood_Barrage_Heal | +7.50 | Active healing |
| Wave_End_HP_Bonus | +4.29 | Healthy at wave end |
| Mager_Priority | +3.62 | Targeting magers first |
| NE_Pillar_Zone | +3.62 | Time in safe zone |
| Single-LOS_Engagement | +2.61 | 1v1 positioning |
| Kill_Nib | +2.50 | Nibbler kills |
| Kill_Zek | +1.92 | Mager kills |
| Kill_ImKot | +1.13 | Melee kills |
| Kill_Xil | +1.03 | Ranger kills |
| Mager_Resurrection | +0.92 | — |
| Kill_Ak | +0.83 | Bat kills |
| Kill_MejRah | +0.68 | Blob kills |
| Kill_AkRek-Mej | +0.57 | Small mage kills |
| Kill_AkRek-Xil | +0.57 | Small range kills |
| Kill_AkRek-Ket | +0.56 | Small melee kills |
| Melee_Resurrection | +0.23 | — |
| **Damage_Taken** | **-10.31** | Main punishment (~4 hits/ep) |
| Multi-LOS | -2.78 | Exposed to >1 NPC |
| NE_Pillar_Zone_Penalty | -2.65 | Time outside zone |
| Blood_Barrage_at_High_HP | -1.05 | Wasted blood barrage |
| Stall_Penalty | -0.86 | Idle ticks |
| Invalid_Action | -0.83 | Bad action attempts |
| NE_Pillar_Damage | -0.18 | Pillar took damage |
| Pillar_Damage | -0.12 | Non-NE pillar damage |

### @ 169.2M steps (Phase 4, frontier 64) — comparison

| Term | 77.3M | 169.2M | Delta | Notes |
|------|-------|--------|-------|-------|
| Damage_Dealt | +25.39 | +24.42 | -0.97 | Slight decrease |
| Wave_Complete | +9.37 | +9.10 | -0.27 | Stable |
| Blood_Barrage_Heal | +7.50 | +7.70 | +0.20 | Stable |
| Wave_End_HP_Bonus | +4.29 | +4.15 | -0.14 | Stable |
| Mager_Priority | +3.62 | +3.88 | +0.26 | Stable |
| NE_Pillar_Zone | +3.62 | +3.35 | **-0.27** | Less time in safe zone |
| Single-LOS_Engagement | +2.61 | +2.38 | **-0.23** | Worse 1v1 positioning |
| Kill_Nib | +2.50 | +2.49 | -0.01 | Stable |
| Kill_Zek | +1.92 | +1.93 | +0.01 | Stable |
| Kill_ImKot | +1.13 | +1.77 | **+0.64** | More melee kills (harder waves) |
| Kill_Xil | +1.03 | +0.97 | -0.06 | Stable |
| Mager_Resurrection | +0.92 | +0.49 | **-0.43** | Fewer resurrections seen (dying earlier?) |
| Kill_Ak | +0.83 | +0.52 | -0.31 | Fewer bat kills (wave composition) |
| Kill_MejRah | +0.68 | +0.59 | -0.09 | Stable |
| Kill_AkRek-Mej | +0.57 | +0.35 | -0.22 | Fewer small NPC kills (composition) |
| Kill_AkRek-Xil | +0.57 | +0.37 | -0.20 | " |
| Kill_AkRek-Ket | +0.56 | +0.36 | -0.20 | " |
| Melee_Resurrection | +0.23 | +0.36 | +0.13 | More resurrections |
| **Damage_Taken** | **-10.31** | **-10.46** | **-0.15** | Flat — not getting hit more |
| Multi-LOS | -2.78 | -2.77 | +0.01 | Flat |
| NE_Pillar_Zone_Penalty | -2.65 | -2.85 | **-0.20** | More time outside zone |
| Blood_Barrage_at_High_HP | -1.05 | -0.75 | +0.30 | Less waste (improved) |
| Stall_Penalty | -0.86 | -0.90 | -0.04 | Flat |
| Invalid_Action | -0.83 | -0.67 | +0.16 | Fewer invalid actions |
| NE_Pillar_Damage | -0.18 | -0.15 | +0.03 | Flat |
| Pillar_Damage | -0.12 | -0.14 | -0.02 | Flat |

## Eval Results (W55-66, 100 episodes, seed 0)

Eval bug fix: `initial_barrage_enabled` was `False` in eval.py (default). Training sets it
`True` — enables tick 1 stay, tick 2 barrage nibblers, tick 3 switch BoFa heuristic. Without
it, the model had to perform these actions itself despite never being trained to. Fixed in eval.py.

All results below include the fix.

### Checkpoint Sweep (Run 2: continuation from ckpt 1300 @ ~58M)

| Ckpt | ~Steps | Death% | Clear% | Mean Wave | Median | Notes |
|------|--------|--------|--------|-----------|--------|-------|
| R1 1100 | 54M | 95.0% | 5.0% | 59.2 | 59 | Pre-prestige, frontier just hit 66 |
| R1 1300 | 64M | 89.0% | 9.0% | 59.8 | 60 | Frontier 66, phase 1 peak |
| R2 1000 | 113M | 82.0% | 8.0% | 59.7 | 60 | Phase 3 |
| R2 1200 | 123M | 69.0% | 26.0% | 62.6 | 63 | Phase 3 climbing |
| **R2 1300** | **128M** | **53.0%** | **35.0%** | **62.9** | **64** | **Best model. Phase 3→4 prestige. Lowest death rate.** |
| R2 1400 | 133M | 61.0% | 36.0% | 62.5 | 63 | Similar clear% but +8% death vs 1300 |
| R2 1500 | 138M | 68.0% | 25.0% | 61.7 | 62 | Regression begins |
| R2 1600 | 143M | 60.0% | 30.0% | 63.0 | 63 | Noisy — variance in 100 eps |
| R2 1700 | 147M | 72.0% | 20.0% | 61.7 | 62 | Clear decline continues |
| R2 1800 | 152M | 74.0% | 19.0% | 62.7 | 63 | — |
| R2 2100 | 167M | 86.0% | 12.0% | 60.8 | 61 | Latest. Confirmed regression. |

### Key Findings

1. **Best checkpoint: R2 1300 (~128M steps)** — 35% clear, 53% death, median W64. This is the
   phase 3→4 prestige transition where training deaths hit all-time low (15/rollout).
2. **Peak window is narrow: ~123-143M steps.** Before 123M: model hasn't converged enough.
   After 143M: regression from continued training on harder curriculum (phase 4 min_waves=4).
3. **Regression confirmed:** clear rate drops 35% → 19% → 12% from ckpt 1300 to 2100 (+39M steps).
   The phase 4 curriculum pushes the model to attempt W63-66 clears it can't consistently achieve,
   degrading general survival skills.
4. **V30 best (35% W55-66) vs previous versions:** V21 best was 22% W49-66, V29 was 26.5% W49-66.
   Direct comparison requires matching bench range (W55-66 is easier than W49-66), but the gap is
   substantial — LSTM + nibblers in obs + [512,512] MLP are genuine improvements.
5. **Eval fix impact:** without `initial_barrage_enabled`, ckpt 1600 scored 13% clear (vs 30% with
   fix). All previous V21/V25/V29 evals did not have this bug since those versions used different
   eval paths or didn't rely on the heuristic.

---

## Harden Phase (from R2 ckpt 1300 @ 128M)

Consolidation run: uniform random wave sampling W55-66. No frontier, no prestige. Lower entropy
(policy is mature). Goal: solidify survival skills the climb phase developed without the destructive
pressure of an ever-harder curriculum.

### What Changed (Climb → Harden)

| Setting | Climb | Harden | Rationale |
|---------|-------|--------|-----------|
| phase | climb | **harden** | Uniform random W55-66, no curriculum progression |
| load | fresh init | **R2 ckpt 1300** | Best climb checkpoint (35% clear, 53% death) |
| entropy-start | 0.02 | **0.005** | Policy mature, less exploration needed |
| entropy-end | 0.002 | **0.001** | Tighter floor |
| timesteps | 200M | **50M** | Consolidation budget |
| promote-after | 5 | *(unused)* | Harden has no frontier |

### Run Command

```powershell
tools\inferno_rl\venv\Scripts\activate.ps1; python -m tools.inferno_rl.train_gpu --load models/V30_climb/inferno_gpu_w55-66_20260302_133854_1300.pt --lstm-hidden-size 256 --lstm-seq-len 10 --phase harden --start-wave 55 --max-wave 66 --save-dir models/V30_harden --log-dir logs/V30_harden --n-envs 48 --n-steps 1024 --batch-size 2048 --n-epochs 2 --lr 1e-4 --entropy-start 0.005 --entropy-end 0.001 --gamma 0.995 --gae-lambda 0.95 --vf-coef 0.5 --max-grad-norm 0.5 --normalize-reward --normalize-obs --checkpoint-every 100 --timesteps 50000000 --actor-sizes 512,512 --critic-sizes 512,512 --device cuda --log-reward-terms
```

### Metrics Log

| Step | Deaths | Waves Comp | Mean Reward | EV | KL | Entropy | Grad Norm | FPS | Notes |
|------|--------|------------|-------------|-----|------|---------|-----------|-----|-------|
| 131.0M | 33 | 352 | 5.28 | 0.87 | 0.031 (watch) | -1.90 | 0.43 | 5437 | ~3M into harden. EV 0.66→0.87, deaths 52→33. KL borderline. |
| 138.5M | 43 | 331 | 3.76 | 0.84 | 0.036 **(watch)** | -1.49 | 0.48 | 5215 | **~10.5M in. Regression: deaths 33→43, reward 5.28→3.76. KL 0.036 sustained >0.030. Entropy -1.90→-1.49 accelerating.** |
