# V31 TensorBoard Tracking

Fresh start. Two changes from V30: triple LSTM seq_len (10 → 30) and widen wave range
(W55-66 → W49-66).

## What Changed (V30 → V31)

### Architecture

| Setting      | V30        | V31        |
|--------------|------------|------------|
| actor-sizes  | 512,512    | 512,512    |
| critic-sizes | 512,512    | 512,512    |
| LSTM         | 256 hidden | 256 hidden |
| lstm-seq-len | 10         | **30**     |
| params       | ~1.5M      | ~1.5M      |
| obs dims     | 186        | 186        |

`lstm_seq_len` is a training-time parameter only (excluded from checkpoint compat checks).
Model weights load cleanly from V30 — no architecture mismatch.

### Wave Range

| Setting    | V30 | V31    |
|------------|-----|--------|
| start-wave | 55  | **49** |
| max-wave   | 66  | 66     |

Wider range adds W49-54 (bat/blob/ranger-only waves). V30 never trained on these — the
model will need to generalize or learn new behaviors for the easier waves. Wider range also
provides more exploration pressure and prevents overfitting to W55-66 (lesson from V25 R7
narrowing failure).

### Reward Structure

Full reward weights as of V31 start. Increased MULTI-LOS, added C tile bonus, added early nibbler engagement penalty.
Other minor changes, from this moment forwards each document holds its rewards for documentation purposes.

#### Terminal

| Reward               | Value | Notes                                         |
|----------------------|-------|-----------------------------------------------|
| DEATH_PENALTY        | 0.0   | Disabled — episode termination IS the penalty |
| WAVE_TIMEOUT_PENALTY | 0.0   | Disabled                                      |

#### Combat

| Reward                           | Value | Notes                                          |
|----------------------------------|-------|------------------------------------------------|
| DAMAGE_PENALTY_PER_HP            | -0.05 | Per HP taken                                   |
| DAMAGE_DEALT_REWARD_PER_HP       | 0.01  | Per HP dealt                                   |
| BLOOD_BARRAGE_HEAL_REWARD_PER_HP | 0.06  | Per HP healed                                  |
| BLOOD_BARRAGE_HIGH_HP_PENALTY    | -0.2  | Wasted barrage at high HP                      |
| SURVIVAL_REWARD_PER_TICK         | 0.005 | Proportional to HP ratio                       |
| DAMAGE_NO_MOVE_ENABLED           | True  | Re-applies damage penalty if no move after hit |

#### Kill Rewards

| Entity                              | Value    |
|-------------------------------------|----------|
| MAGER                               | 0.6      |
| MELEE                               | 0.35     |
| BAT                                 | 0.3      |
| BLOB_MAGE / BLOB_RANGE / BLOB_MELEE | 0.3 each |
| RANGER                              | 0.25     |
| BLOB                                | 0.2      |
| NIBBLER                             | 0.15     |
| JAD                                 | 8.0      |
| HEALER                              | 1.0      |
| ZUK                                 | 80.0     |
| ZUK_HEALER                          | 1.0      |

#### Wave Completion

| Reward                         | Value | Notes                         |
|--------------------------------|-------|-------------------------------|
| WAVE_COMPLETE_REWARD_BASE      | 2.0   | + 0.3 per wave cleared so far |
| WAVE_COMPLETE_REWARD_INCREMENT | 0.3   | Scales with progress          |
| WAVE_END_HP_BONUS              | 1.5   | Scales with HP% at wave end   |
| INFERNO_COMPLETE_REWARD        | 0.0   | Disabled                      |

#### Positioning

| Reward                      | Value | Notes                                     |
|-----------------------------|-------|-------------------------------------------|
| NE_PILLAR_ZONE_BONUS        | 0.008 | Per-tick, in zone + engaging              |
| NE_PILLAR_ZONE_PENALTY      | -0.02 | Per-tick, outside zone past grace         |
| MULTI_LOS_PENALTY           | -0.02 | Per-tick, 2+ non-nibbler NPCs have LOS    |
| SINGLE_LOS_ENGAGEMENT_BONUS | 0.02  | Per-tick, exactly 1 NPC has LOS           |
| TILE_A_MAX_REWARD           | 0.04  | Per-tick at Tile A (17,26), between waves |
| TILE_A_REWARD_RADIUS        | 5     | Tiles beyond which reward is 0            |
| C_TILE_ON_REWARD            | 0.5   | Per-tick on C tile (19,25), ticks 0-4     |
| C_TILE_ADJACENT_REWARD      | 0.25  | Per-tick Chebyshev dist 1 from C tile     |
| C_TILE_ACTIVE_TICKS         | 5     | Reward fires ticks 0-4                    |
| MELEE_PROXIMITY_PENALTY     | -0.01 | Per-tick near melee footprint             |

#### Stall / Engagement

| Reward                   | Value | Notes                             |
|--------------------------|-------|-----------------------------------|
| STALL_BASE_PENALTY       | -0.08 | After 15 ticks disengaged         |
| STALL_ESCALATION         | 0.04  | Additional per tick beyond window |
| STALL_WINDOW             | 15    | Ticks before penalty starts       |
| WAVE_START_GRACE_TICKS   | 17    | No stall/LOS penalties            |
| ATTACK_ON_COOLDOWN_BONUS | 0.0   | Disabled                          |

#### Penalties / Guardrails

| Reward                       | Value  | Notes                         |
|------------------------------|--------|-------------------------------|
| INVALID_ACTION_PENALTY       | -0.1   | Invalid movement/switch       |
| INVALID_ATTACK_PENALTY       | -0.05  | Invalid attack target         |
| PILLAR_DAMAGE_PENALTY_PER_HP | -0.004 | Per HP, all pillars           |
| NE_PILLAR_DAMAGE_MULTIPLIER  | 9.0    | NE pillar weighted 9x         |
| EARLY_NIBBLER_PENALTY        | -0.8   | Targeting nibblers ticks 3-10 |

#### Priority / Resurrection

| Reward                       | Value | Notes                                  |
|------------------------------|-------|----------------------------------------|
| MAGER_PRIORITY_BONUS_PER_NPC | 0.25  | Per non-mager alive when hitting mager |
| MAGER_RESURRECTION_PENALTY   | 0.4   | Per mager/ranger/blob resurrection     |
| MELEE_RESURRECTION_PENALTY   | 0.2   | Per melee resurrection                 |

### Training Settings

| Setting              | V30   | V31      | Rationale                                      |
|----------------------|-------|----------|------------------------------------------------|
| warmstart            | none  | **none** | Fresh init                                     |
| lstm-seq-len         | 10    | **30**   | Triple context window for multi-tick reasoning |
| start-wave           | 55    | **49**   | Wider range, more exploration pressure         |
| max-wave             | 66    | 66       | Same                                           |
| n-epochs             | 2     | 2        | Same                                           |
| n-envs               | 48    | 48       | Same                                           |
| batch-size           | 2048  | 2048     | Same                                           |
| lr                   | 1e-4  | 1e-4     | Same                                           |
| entropy-start        | 0.02  | 0.02     | Same                                           |
| entropy-end          | 0.002 | 0.002    | Same                                           |
| normalize-reward     | yes   | yes      | Same                                           |
| normalize-obs        | yes   | yes      | Same                                           |
| n-steps              | 1024  | 1024     | Same                                           |
| gamma                | 0.995 | 0.995    | Same                                           |
| gae-lambda           | 0.95  | 0.95     | Same                                           |
| vf-coef              | 0.5   | 0.5      | Same                                           |
| max-grad-norm        | 0.5   | 0.5      | Same                                           |
| phase                | climb | climb    | Same                                           |
| promote-after        | 5     | 5        | Same                                           |
| min-waves-to-advance | 1     | 1        | Same                                           |

### Run Command

```powershell
python -m tools.inferno_rl.train_gpu --lstm-hidden-size 256 --lstm-seq-len 32 --phase climb --start-wave 49 --max-wave 66 --promote-after 5 --min-waves-to-advance 1 --save-dir models/V31_climb --log-dir logs/V31_climb --n-envs 48 --n-steps 1024 --batch-size 2048 --n-epochs 2 --lr 1e-4 --entropy-start 0.02 --entropy-end 0.002 --gamma 0.995 --gae-lambda 0.95 --vf-coef 0.5 --max-grad-norm 0.5 --normalize-reward --normalize-obs --checkpoint-every 100 --timesteps 200000000 --actor-sizes 512,512 --critic-sizes 512,512 --device cuda --log-reward-terms
```

### Risks

1. **seq_len=32 slows training** — 3x more LSTM steps per forward pass during training. Expect significant FPS drop vs V30 (which already
   had ~10% drop from MLP-only V29).
2. **seq_len=32 may be too long** — MEMORY.md notes 17 ticks is the absolute max decision-relevant horizon. 30 ticks spans ~3 full NPC
   attack cycles of noise. V25 failed with seq_len=32 but that was a different setup (fresh init, different obs space). Resuming from a
   strong checkpoint may help.
3. **Wider wave range dilutes focus** — W49-54 are easier waves the model was never trained on. Curriculum will spend early steps on these,
   potentially slowing frontier progress on the harder W60+ waves.

## Metrics Log

| Step   | Frontier | Deaths | Waves Comp | Mean Reward | EV   | KL    | Entropy | Grad Norm | FPS  | Notes                                                                                                                                                                                                                                       |
|--------|----------|--------|------------|-------------|------|-------|---------|-----------|------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 2.6M   | 51       | 236    | 334        | 0.33        | 0.64 | 0.018 | -3.02   | 0.28      | 5490 | Fresh init, ~2.6M steps. Frontier 51 (easy waves). EV 0.64 **(watch)**. KL/entropy healthy. FPS 5490 (~10% slower than V30's 6256 at same step count — seq_len=30 cost).                                                                    |
| 7.8M   | 53       | 192    | 387        | 1.16        | 0.62 | 0.025 | -2.76   | 0.31      | 5014 | Frontier 51→53. Deaths 236→192, waves 334→387, reward 0.33→1.16. EV 0.62 **(watch)** — still below 0.70, flat from 2.6M. FPS dropped 5490→5014. Still in easy wave range.                                                                   |
| 17.7M  | 55       | 132    | 373        | 2.18        | 0.68 | 0.024 | -2.56   | 0.36      | 5609 | Frontier 53→55 — entering melee wave territory. Deaths 192→132, reward 1.16→2.18. EV 0.62→0.68 (recovering, approaching 0.70). KL stable 0.024. FPS recovered 5014→5609.                                                                    |
| 200.0M | 58       | 12     | 448        | 8.65        | 0.81 | 0.032 | -0.86   | 0.44      | 5405 | Phase 6, frontier 55→58, f.mean 53.3. Deaths 132→12, waves 373→448, reward 2.18→8.65. EV 0.68→0.81 (healthy). KL 0.032 **(watch)**. **Entropy -2.56→-0.86 — significant entropy rise, unusual.** FPS stable 5405.                           |
| —      | —        | —      | —          | —           | —    | —     | —       | —         | —    | **R2 start: resumed from R1 200M checkpoint. Curriculum state reset.**                                                                                                                                                                      |
| 237.7M | 51       | 6      | 454        | 10.88       | 0.89 | 0.015 | -1.92   | 0.42      | 5564 | R2 ~37.7M steps. Phase 3 (curriculum re-climbing from reset). Deaths 6, EV 0.89 (excellent). KL 0.015 — settled from R1's 0.032 (optimizer reset). Entropy -0.86→-1.92 (normalizing). Reward 8.65→10.88. Model re-climbing curriculum fast. |
| 259.3M | 63       | 14     | 421        | 6.10        | 0.87 | 0.012 | -1.89   | 0.43      | 4422 | R2 ~59.3M. Phase 3, frontier 51→63 (fast re-climb). Deaths 6→14, reward 10.88→6.10, waves 454→421 — expected as curriculum pushes harder waves. EV 0.87 (healthy). KL/entropy stable. FPS 5564→4422 (drop).                                 |

## Eval Results (100 seeds per start wave)

| Checkpoint | Steps | W49 Clear | W49 Death | W55 Clear | W55 Death | W63 Clear | W63 Death |
|------------|-------|-----------|-----------|-----------|-----------|-----------|-----------|
| R1-100     | 4.9M  | —         | —         | 0%        | 90%       | —         | —         |
| R1-200     | 9.8M  | —         | —         | 0%        | 76%       | —         | —         |
| R1-300     | 14.7M | —         | —         | 0%        | 48%       | —         | —         |
| R1-400     | 19.7M | —         | —         | 0%        | 47%       | —         | —         |
| R1-500     | 24.6M | —         | —         | 0%        | 79%       | —         | —         |
| R1-700     | 34.4M | —         | —         | 0%        | 66%       | —         | —         |
| R1-1000    | 49M   | 9%        | 90%       | 11%       | 88%       | 47%       | 51%       |
| R1-2000    | 98M   | 26%       | 73%       | 31%       | 68%       | 62%       | 37%       |
| R1-4069    | 200M  | 24%       | 74%       | 38%       | 60%       | **70%**   | 29%       |
| R2-700     | 234M  | **33%**   | **61%**   | **41%**   | **59%**   | 64%       | 34%       |
| R2-1200    | 259M  | 2%        | 91%       | 10%       | 83%       | 43%       | 51%       |
| R1-1000†   | 49M   | —         | —         | 0%        | 54%       | —         | —         |

†Re-eval with current eval code (2026-03-03). Previous R1-1000 eval showed 11% clear / 88% death —
discrepancy likely due to eval code changes between runs. Fresh result: 0% clear, 54% death, 46%
timeout, 88/100 episodes stuck on W55 (44 deaths + 44 timeouts). Used as V33 ckpt 1000 comparison
baseline.

**Key findings:**

- R2-700 is the best full-range model: 33% clear from W49 (new record), 41% from W55.
- R1-4069 peaks at W63 (70% clear) — late R1 training specialized on harder waves.
- R2's curriculum re-climb is broadening the policy: better W49/W55 at slight W63 cost (64% vs 70%).
- R1-4069 dipped at W49 (24%) vs R1-2000 (26%), confirming late-R1 over-specialization on hard waves.
- Clear monotonic improvement at W55 across all checkpoints (11% → 31% → 38% → 41%).
- R2 started regressing quickly

### Early W55 Death Distribution (R1-100 to R1-300)

| Ckpt   | Steps | W55  | W56  | W57 | W58 | W59 | W60 | Timeout |
|--------|-------|------|------|-----|-----|-----|-----|---------|
| R1-100 | 4.9M  | 43   | 41   | 3   | 3   | —   | —   | 10%     |
| R1-200 | 9.8M  | 37   | 33   | 5   | —   | 1   | —   | 24%     |
| R1-300 | 14.7M | 41   | 7    | —   | —   | —   | —   | 52%     |
| R1-400 | 19.7M | 42   | 4    | 1   | —   | —   | —   | 53%     |
| R1-500 | 24.6M | 50   | 22   | 12  | 6   | 7   | 2   | 21%     |
| R1-700 | 34.4M | 63   | 1    | 2   | —   | —   | —   | 34%     |

**Analysis:** 0% clear at all early checkpoints — expected at 5-35M steps. Timeout peaks
at ckpt 300-400 (52-53%), drops at ckpt 500 (21%) as model starts fighting, then rises
again at ckpt 700 (34%). Ckpt 700 regressed — 92/100 episodes stuck on W55 (63 deaths +
29 timeouts), vs ckpt 500 which spread deaths across W55-W61. Non-monotonic progress.
