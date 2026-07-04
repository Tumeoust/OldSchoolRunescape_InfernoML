# V52 TB Tracking

## Status

V52 re-introduces pillar preservation signal after eval data showed the model is destroying its own pillars. With
`pillar_damage_per_hp` zeroed since V51 start and nibbler penalties removed, the model learned that pillar damage has
zero consequence. Eval at 530.8M steps (W1-66, 50 eps) showed:

| Pillar | Avg HP% | Death Rate | Mean Death Wave |
|--------|---------|------------|-----------------|
| NW     | 1.4%    | 96%        | W37             |
| NE     | 30.4%   | 12%        | W35             |
| S      | 0.0%    | 100%       | W36             |

NW and S die in virtually every run by the mid-30s. The 12% NE pillar death rate tracks almost exactly with the 14.8%
overall player death rate — losing the NE pillar is effectively a death sentence. The root cause: once NW and S are
gone, all nibbler attacks funnel into the NE pillar. If spawn configs are bad, the model can't protect it 100% of the
time.

## V52 First Attempt (failed, wiped)

Ran 532M–602M steps with entropy 0.05→0.002 and heavy pillar death penalties (NW/S: -7.5, NE: -15.0). Pillar
preservation improved dramatically (all pillars ~4% death rate, 56-65% avg HP) but combat regressed badly — BUDGET_RCB
eval showed ~33% death rate at 599M steps vs V51's ~14% on uniform loadout. The model learned to protect pillars but
forgot how to fight. Wiped and restarting with gentler penalties.

## What Changed

### Pillar Death Event Penalties (new)

Instead of per-HP damage penalties (noisy, fires every tick), V52 uses one-time death event penalties. This gives a
clean discrete signal without per-tick noise — the model won't abandon a safespot to prevent 2 HP of chip damage, but
will learn that letting a pillar die has serious consequences.

- `pillar_death_penalty`: -2.5 (NW/S pillar death, ~12.5% of death penalty)
- `ne_pillar_death_penalty`: -5.0 (NE pillar death, ~25% of death penalty)

Gentler than the first attempt (-7.5/-15.0) to avoid destabilizing combat. The model has pillar HP in its observation
space, so it should learn the causal link: nibblers alive -> pillar HP dropping -> pillar death event -> penalty.

**Projected per-episode impact** (based on V51 eval pillar death rates):
- NW deaths (96% of eps): -2.5 * 0.96 = ~-2.4/ep
- S deaths (100% of eps): -2.5 * 1.0 = ~-2.5/ep
- NE deaths (12% of eps): -5.0 * 0.12 = ~-0.6/ep
- Total projected: ~-5.5/ep initially, should decrease as model learns to preserve pillars

### Entropy Bump

`entropy_coef`: 0.014 (V51 end) -> 0.02 (V52 start). Minimal bump to avoid the catastrophic regression seen with the
0.5 reset in the first attempt. The penalties change the reward landscape directly — the gradient should push toward
pillar preservation without needing heavy exploration.

### Wave Range

Start wave 31 (was 1 in V51's final phase). Sweep phase failure-weighted across W31-66. Concentrates training on the
nibbler-heavy waves where pillar preservation matters most, while keeping episodes shorter for faster gradient updates.

### Unchanged

- Observation space (602, v4)
- Model architecture (flat_lstm_residual, 256 LSTM, 512x512 actor/critic)
- Action space
- All other reward terms (V51 minimal config, mager rewards zeroed)
- `pillar_damage_per_hp` stays at 0.0 (death events replace continuous damage)

### Files Changed

| File | Changes |
|------|---------|
| `simulator/step_result.py` | Added `pillar_deaths` and `ne_pillar_died` fields to `StepResult`; computed from `pillar_hp_at_step_start` vs current HP in `_build_step_result()` |
| `training/rewards.py` | Added `pillar_death_penalty` (-2.5) and `ne_pillar_death_penalty` (-5.0) to `RewardConfig`; added "Pillar Death" and "NE Pillar Death" reward terms in `_calculate_internal()` |
| `train_gpu.py` | Added `--rw-pillar-death-penalty` and `--rw-ne-pillar-death-penalty` CLI args; wired to `_reward_config_from_args()` |
| `eval.py` | Added pillar HP% and death wave tracking per episode; pillar summary in `print_stats()` |

## V52 Start Command

Continue from V51's latest checkpoint (3300, 530.8M steps). Normalization stats reset (no `--resume-normalization`)
since reward distribution changes with new pillar death penalties.

```powershell
python -m tools.inferno_rl.train_gpu --load models/V51/inferno_gpu_w1-66_20260401_224846_3300.pt --curriculum-mode static --phase sweep --start-wave 31 --max-wave 66 --observation-version v4 --policy-arch flat_lstm_residual --lstm-hidden-size 256 --lstm-seq-len 32 --lstm-burn-in 16 --episode-mode full --n-envs 64 --n-steps 512 --batch-size 4096 --n-epochs 1 --lr 3e-4 --target-kl 0.02 --entropy-start 0.02 --entropy-end 0.002 --gamma 0.998 --gae-lambda 0.95 --vf-coef 0.25 --max-grad-norm 1.0 --normalize-obs --normalize-reward --actor-sizes 512,512 --critic-sizes 512,512 --save-dir models/V52 --log-dir logs/V52 --checkpoint-every 100 --timesteps 200000000 --device cuda --log-reward-terms --rw-mager-early-kill-base 0 --rw-mager-early-kill-per-npc 0 --rw-mager-priority-per-npc 0 --rw-pillar-death-penalty -2.5 --rw-ne-pillar-death-penalty -5.0
```

## Current Settings

| Setting             | Value                  | Notes |
|---------------------|------------------------|-------|
| base checkpoint     | V51 3300 (530.8M steps) | policy weights preserved, optimizer + normalization reset |
| observation-version | `v4`                   | 602 features, unchanged |
| policy-arch         | `flat_lstm_residual`   | unchanged |
| lstm-hidden-size    | `256`                  | unchanged |
| episode-mode        | `full`                 | per-wave attribution |
| start-wave          | `31`                   | sweep across W31-66 |
| phase               | `sweep`                | failure-weighted across W31-66 |
| n-steps             | `512`                  | unchanged |
| gamma               | `0.998`                | unchanged |
| entropy             | `0.02 -> 0.002`        | minimal bump from V51's 0.014 |
| reward shaping      | V51 minimal + pillar death events | ~10 active terms: 8 from V51 + 2 pillar death events |
| pillar death        | NW/S: -2.5, NE: -5.0  | gentle one-time event penalties |
| wave timeout        | 800 ticks, -15.0 penalty | unchanged from V51 |
| loadout             | uniform random (all 5) | unchanged |

## What to Watch

- **Phase_Fail%** — should stay near V51's terminal ~16-18% initially. Entropy bump is minimal (0.014→0.02), so no
  major disruption expected. Normalization reset may cause a temporary spike.
- **Pillar Death reward terms** — track "Pillar Death" and "NE Pillar Death" ep_sum. Initially will be moderate negatives
  (~-5.5/ep based on current pillar death rates). Success = these terms trending toward zero.
- **Return** — may dip slightly from normalization reset + new penalties. Should recover quickly.
- **RVar** will spike from normalization reset + new reward terms. Should settle within 5-10M steps.
- **Combat quality** — the key metric. With gentle entropy + gentle penalties, combat should not regress. If Phase_Fail%
  degrades beyond 25% and doesn't recover within 10M steps, penalties may be too strong.
- **Nibbler kill behavior** — the desired outcome. Watch for the model actively targeting nibblers instead of ignoring
  them. Can verify via eval with pillar tracking.
- **Eval cadence** — run W1-66 eval with pillar tracking every ~20M steps:
  `.venv/Scripts/python.exe -m tools.inferno_rl.eval --model <checkpoint> --start-waves 1 --episodes 50 --workers 6`

## Metrics Log

| Steps | Eps | Deaths | Timeout% | Phase_Fail% | EV | KL | VL | Clip | Ent | Return | RVar | Grad | Notes |
|-------|-----|--------|----------|-------------|----|----|----|----|-----|--------|------|------|-------|
| 535.0M | 11 | 1.3 | 0.0 | 12.4 | 0.979 | 0.0003 | 0.00 | 0.002 | 0.0198 | 1.09 | 504 | 0.36 | V52 start, norm reset |
| 540.0M | 11 | 1.1 | 0.0 | 10.1 | 0.981 | 0.0003 | 0.00 | 0.001 | 0.0194 | 1.07 | 501 | 0.35 | |
| 545.0M | 11 | 1.3 | 0.0 | 11.7 | 0.979 | 0.0003 | 0.00 | 0.001 | 0.0189 | 1.06 | 497 | 0.35 | |
| 550.0M | 11 | 1.3 | 0.0 | 11.8 | 0.979 | 0.0003 | 0.00 | 0.001 | 0.0185 | 1.06 | 493 | 0.35 | |
| 555.0M | 11 | 1.2 | 0.0 | 10.5 | 0.981 | 0.0003 | 0.00 | 0.001 | 0.0181 | 1.08 | 489 | 0.35 | |
| 560.0M | 11 | 1.1 | 0.0 | 10.7 | 0.982 | 0.0003 | 0.00 | 0.001 | 0.0176 | 1.09 | 485 | 0.35 | |
| 565.0M | 11 | 1.3 | 0.0 | 11.2 | 0.983 | 0.0003 | 0.00 | 0.001 | 0.0172 | 1.11 | 481 | 0.35 | |
| 570.0M | 11 | 1.3 | 0.0 | 12.0 | 0.980 | 0.0003 | 0.00 | 0.001 | 0.0167 | 1.10 | 478 | 0.35 | |
| 575.0M | 11 | 1.2 | 0.0 | 10.8 | 0.982 | 0.0003 | 0.00 | 0.001 | 0.0162 | 1.10 | 474 | 0.35 | |
| 580.0M | 11 | 1.2 | 0.0 | 10.3 | 0.982 | 0.0003 | 0.00 | 0.001 | 0.0158 | 1.10 | 471 | 0.35 | |
| 585.0M | 11 | 1.2 | 0.1 | 10.7 | 0.972 | 0.0003 | 0.02 | 0.001 | 0.0153 | 1.09 | 477 | 0.35 | |
| 590.0M | 11 | 1.0 | 0.0 | 9.3 | 0.980 | 0.0003 | 0.00 | 0.002 | 0.0149 | 1.12 | 476 | 0.35 | |
| 595.0M | 11 | 1.1 | 0.0 | 9.9 | 0.982 | 0.0003 | 0.00 | 0.002 | 0.0144 | 1.12 | 473 | 0.35 | |
| 600.0M | 11 | 1.2 | 0.0 | 11.8 | 0.979 | 0.0003 | 0.00 | 0.002 | 0.0140 | 1.12 | 469 | 0.36 | |
| 605.0M | 11 | 1.1 | 0.0 | 9.9 | 0.982 | 0.0003 | 0.00 | 0.002 | 0.0135 | 1.15 | 466 | 0.36 | |
| 610.0M | 11 | 1.0 | 0.0 | 9.4 | 0.982 | 0.0003 | 0.00 | 0.002 | 0.0131 | 1.15 | 463 | 0.35 | |
| 615.0M | 11 | 1.0 | 0.1 | 9.5 | 0.975 | 0.0003 | 0.00 | 0.002 | 0.0126 | 1.12 | 461 | 0.36 | |
| 620.0M | 11 | 1.1 | 0.0 | 9.8 | 0.982 | 0.0003 | 0.00 | 0.002 | 0.0122 | 1.14 | 460 | 0.37 | |
| 625.0M | 11 | 1.1 | 0.0 | 9.5 | 0.982 | 0.0003 | 0.00 | 0.002 | 0.0117 | 1.14 | 457 | 0.36 | |
| 630.0M | 11 | 1.0 | 0.0 | 9.0 | 0.982 | 0.0003 | 0.00 | 0.002 | 0.0113 | 1.17 | 454 | 0.36 | |
| 635.0M | 11 | 1.2 | 0.0 | 10.1 | 0.979 | 0.0003 | 0.00 | 0.002 | 0.0108 | 1.17 | 451 | 0.37 | |
| 640.0M | 11 | 1.0 | 0.0 | 9.5 | 0.977 | 0.0003 | 0.00 | 0.002 | 0.0104 | 1.17 | 448 | 0.36 | |
| 645.0M | 11 | 1.2 | 0.1 | 10.7 | 0.972 | 0.0003 | 0.01 | 0.002 | 0.0099 | 1.14 | 452 | 0.36 | |
| 650.0M | 11 | 1.1 | 0.0 | 9.7 | 0.982 | 0.0003 | 0.00 | 0.002 | 0.0095 | 1.18 | 450 | 0.36 | |
| 655.0M | 11 | 1.1 | 0.1 | 9.9 | 0.976 | 0.0003 | 0.01 | 0.002 | 0.0090 | 1.17 | 449 | 0.41 | |
| 660.0M | 11 | 0.9 | 0.0 | 8.0 | 0.980 | 0.0003 | 0.00 | 0.002 | 0.0086 | 1.17 | 457 | 0.37 | |
| 665.0M | 11 | 1.0 | 0.0 | 9.0 | 0.983 | 0.0004 | 0.00 | 0.002 | 0.0082 | 1.18 | 454 | 0.38 | |
| 678.5M | 13 | 7 | 0.0% | 55.2% | 0.95 | 0.001 | 0.01 | 0.005 | 0.020 | 1.19 | 445.2 | 0.44 | n=5; push-out retrain start; Phase_Fail% 55.2% (prev 9.0%); Deaths 7 (prev 1.0); EV 0.947 (prev 0.983); KL 0.001 (prev 0.0004) |
| 682.1M | 11 | 4 | 0.0% | 30.4% | 0.95 | 0.000 | 0.01 | 0.002 | 0.020 | 1.05 | 444.2 | 0.40 | n=109; Phase_Fail% 30.4% (prev 55.2%); Deaths 4 (prev 7) |
| 687.5M | 11 | 2 | 0.0% | 16.0% | 0.97 | 0.000 | 0.00 | 0.002 | 0.019 | 1.10 | 441.8 | 0.37 | n=163; Phase_Fail% 16.0% (prev 30.4%); Deaths 2 (prev 4); EV 0.97 (prev 0.95) |
| 692.8M | 11 | 2 | 0.0% | 13.9% | 0.98 | 0.000 | 0.00 | 0.002 | 0.019 | 1.13 | 438.9 | 0.36 | n=163; NE_Pillar_Death new tag (n=1, avg=-0.63); Phase_Fail% 13.9% (prev 16.0%) |
| 698.2M | 11 | 1 | 0.0% | 11.7% | 0.98 | 0.000 | 0.00 | 0.002 | 0.019 | 1.14 | 436.1 | 0.37 | n=165; Phase_Fail% 11.7% (prev 13.9%); Deaths 1 (prev 2) |
| 703.6M | 11 | 1 | 0.0% | 12.8% | 0.98 | 0.000 | 0.00 | 0.003 | 0.018 | 1.13 | 433.3 | 0.35 | n=164; NE_Pillar_Death absent (was n=1 prev) |
| 708.9M | 11 | 1 | 0.0% | 12.4% | 0.98 | 0.000 | 0.00 | 0.002 | 0.018 | 1.14 | 430.5 | 0.35 | n=162; NE_Pillar_Death back (n=3, avg=-0.42) |
| 714.3M | 11 | 1 | 0.0% | 11.9% | 0.98 | 0.000 | 0.00 | 0.002 | 0.017 | 1.15 | 427.8 | 0.35 | n=165 |
| 719.7M | 11 | 1 | 0.0% | 10.9% | 0.98 | 0.000 | 0.00 | 0.002 | 0.017 | 1.19 | 425.1 | 0.35 | n=163; Return 1.19 (prev 1.15) |
| 725.0M | 11 | 1 | 0.0% | 10.2% | 0.98 | 0.000 | 0.00 | 0.002 | 0.016 | 1.18 | 422.5 | 0.35 | n=162 |
| 730.4M | 11 | 1 | 0.0% | 10.7% | 0.98 | 0.000 | 0.00 | 0.002 | 0.016 | 1.18 | 419.9 | 0.35 | n=163 |
| 735.7M | 11 | 1 | 0.0% | 11.2% | 0.98 | 0.000 | 0.00 | 0.002 | 0.015 | 1.19 | 417.4 | 0.35 | n=161; NE_Pillar_Death absent (was n=3 prev) |
| 741.0M | 11 | 1 | 0.0% | 9.3% | 0.98 | 0.000 | 0.00 | 0.002 | 0.015 | 1.20 | 414.9 | 0.35 | n=162; Phase_Fail% 9.3% (prev 11.2%) |
| 746.4M | 11 | 1 | 0.0% | 11.2% | 0.98 | 0.000 | 0.00 | 0.002 | 0.014 | 1.21 | 412.5 | 0.35 | n=163; Phase_Fail% 11.2% (prev 9.3%); NE_Pillar_Death back (n=1) |
| 751.7M | 11 | 1 | 0.0% | 10.7% | 0.98 | 0.000 | 0.00 | 0.002 | 0.014 | 1.21 | 410.0 | 0.35 | n=161 |
| 757.0M | 11 | 1 | 0.0% | 11.7% | 0.98 | 0.000 | 0.00 | 0.002 | 0.013 | 1.21 | 407.6 | 0.36 | n=161 |
| 762.3M | 11 | 1 | 0.0% | 10.8% | 0.98 | 0.001 | 0.00 | 0.004 | 0.013 | 1.20 | 405.3 | 0.36 | n=162; KL 0.001 (prev 0.000); Clip 0.004 (prev 0.002) |
| 767.6M | 11 | 2 | 0.0% | 14.9% | 0.98 | 0.001 | 0.00 | 0.006 | 0.012 | 1.17 | 403.0 | 0.37 | n=162; Phase_Fail% 14.9% (prev 10.8%); Deaths 2 (prev 1); Clip 0.006 (prev 0.004); Return 1.17 (prev 1.20) |
| 773.0M | 11 | 2 | 0.0% | 14.5% | 0.98 | 0.001 | 0.00 | 0.004 | 0.012 | 1.17 | 400.7 | 0.36 | n=163; DmgTkn -9.93 (prev range -7.99 to -8.84); NE_Pillar_Death absent |

## Reward Terms Log

Averaged per-episode sum of each raw reward term. Total = sum of all terms.

| Steps | Death | WavComp | WaveHP | InfComp | DmgDealt | DmgTkn | LOSSep | Stall | BBHighHP | WpnSw | PilDeath | NEPilDeath | Total | Notes |
|-------|-------|---------|--------|---------|----------|--------|--------|-------|----------|-------|----------|------------|-------|-------|
| 535.0M | -3.29 | 108.04 | 50.28 | 13.14 | 30.47 | -8.23 | 9.06 | -0.78 | -0.60 | -2.46 | -3.53 | -0.41 | 191.68 | |
| 540.0M | -3.09 | 111.02 | 51.53 | 13.49 | 31.21 | -8.47 | 9.57 | -0.82 | -0.71 | -2.54 | -3.28 | -0.39 | 197.52 | |
| 545.0M | -3.31 | 110.26 | 50.84 | 13.25 | 31.12 | -8.53 | 9.38 | -0.71 | -0.51 | -2.55 | -3.01 | -0.48 | 195.75 | |
| 550.0M | -3.26 | 111.16 | 51.74 | 13.23 | 31.34 | -8.77 | 9.64 | -1.26 | -0.68 | -2.61 | -2.64 | -0.43 | 197.47 | |
| 555.0M | -2.85 | 108.46 | 49.85 | 13.42 | 30.56 | -8.73 | 9.22 | -1.05 | -0.50 | -2.65 | -2.37 | -0.56 | 192.82 | |
| 560.0M | -3.06 | 111.24 | 51.51 | 13.40 | 31.36 | -8.62 | 9.53 | -1.02 | -0.39 | -2.70 | -2.56 | -0.46 | 198.22 | |
| 565.0M | -3.25 | 107.70 | 50.04 | 13.33 | 30.35 | -8.47 | 8.90 | -0.65 | -0.47 | -2.62 | -2.09 | -0.50 | 192.29 | |
| 570.0M | -3.30 | 109.09 | 50.68 | 13.21 | 30.69 | -8.39 | 9.29 | -0.81 | -0.52 | -2.70 | -2.23 | -0.49 | 194.52 | |
| 575.0M | -2.97 | 109.86 | 50.96 | 13.37 | 31.00 | -8.56 | 9.32 | -1.08 | -0.86 | -2.52 | -1.81 | -0.49 | 196.22 | |
| 580.0M | -3.11 | 110.93 | 51.31 | 13.46 | 31.23 | -8.69 | 9.48 | -0.78 | -0.79 | -2.68 | -2.09 | -0.44 | 197.81 | |
| 585.0M | -2.95 | 110.99 | 51.20 | 13.39 | 31.30 | -8.61 | 9.59 | -8.04 | -0.66 | -2.62 | -2.04 | -0.45 | 191.09 | stall spike |
| 590.0M | -2.93 | 110.79 | 51.61 | 13.61 | 31.20 | -8.22 | 9.21 | -0.83 | -0.72 | -2.56 | -2.02 | -0.67 | 198.46 | |
| 595.0M | -2.90 | 108.75 | 50.28 | 13.51 | 30.60 | -7.97 | 9.22 | -0.63 | -0.68 | -2.26 | -2.01 | -0.33 | 195.59 | |
| 600.0M | -3.17 | 109.43 | 50.93 | 13.23 | 30.87 | -8.12 | 9.36 | -0.77 | -0.64 | -2.46 | -1.78 | -0.38 | 196.50 | |
| 605.0M | -3.23 | 111.04 | 51.75 | 13.51 | 31.40 | -8.31 | 9.19 | -0.65 | -0.64 | -2.45 | -1.49 | -0.42 | 199.72 | |
| 610.0M | -2.94 | 112.53 | 52.10 | 13.59 | 31.78 | -8.54 | 9.41 | -0.60 | -0.78 | -2.44 | -1.43 | - | 202.67 | |
| 615.0M | -2.98 | 109.64 | 50.75 | 13.58 | 30.83 | -8.05 | 9.47 | -3.96 | -0.62 | -2.45 | -1.80 | -0.54 | 193.88 | stall spike |
| 620.0M | -2.98 | 113.76 | 52.83 | 13.53 | 32.01 | -8.13 | 9.69 | -0.60 | -0.47 | -2.35 | -2.07 | -0.36 | 204.86 | |
| 625.0M | -2.79 | 108.14 | 50.07 | 13.58 | 30.40 | -7.72 | 9.44 | -0.59 | -0.54 | -2.24 | -1.78 | -0.69 | 195.28 | |
| 630.0M | -2.83 | 111.51 | 51.68 | 13.65 | 31.43 | -7.89 | 9.06 | -0.67 | -0.55 | -2.48 | -1.61 | -0.50 | 200.81 | |
| 635.0M | -2.94 | 111.38 | 51.55 | 13.49 | 31.48 | -7.83 | 9.23 | -0.52 | -0.42 | -2.16 | -1.64 | -0.36 | 201.27 | |
| 640.0M | -2.99 | 108.55 | 50.25 | 13.58 | 30.51 | -7.45 | 9.19 | -1.66 | -0.62 | -2.10 | -1.72 | -0.44 | 195.09 | |
| 645.0M | -3.05 | 108.47 | 50.35 | 13.40 | 30.57 | -7.83 | 9.38 | -6.07 | -0.56 | -2.10 | -1.62 | -0.44 | 190.49 | stall spike |
| 650.0M | -2.97 | 110.65 | 51.25 | 13.54 | 31.24 | -8.25 | 9.18 | -0.44 | -0.67 | -1.86 | -1.27 | - | 200.39 | |
| 655.0M | -2.88 | 110.89 | 51.40 | 13.52 | 31.27 | -7.69 | 9.11 | -9.17 | -0.50 | -2.07 | -1.49 | -0.42 | 191.97 | stall spike |
| 660.0M | -2.81 | 110.96 | 51.61 | 13.79 | 31.22 | -7.69 | 9.38 | -0.45 | -0.51 | -2.09 | -1.07 | -0.44 | 201.90 | |
| 665.0M | -2.77 | 110.40 | 51.16 | 13.63 | 31.14 | -7.86 | 9.08 | -0.49 | -0.40 | -2.02 | -1.28 | -0.56 | 200.02 | |
| 678.5M | -11.04 | 34.26 | 14.55 | 6.72 | 10.14 | -4.72 | 3.13 | -1.54 | -0.13 | -0.49 | -0.34 | - | 49.08 | push-out retrain start |
| 682.1M | -6.49 | 102.25 | 48.15 | 10.44 | 29.27 | -9.99 | 8.48 | -1.82 | -0.52 | -2.37 | -0.97 | - | 171.10 | |
| 687.5M | -3.89 | 109.26 | 51.02 | 12.60 | 30.83 | -9.69 | 9.22 | -1.44 | -0.85 | -2.53 | -1.57 | - | 187.43 | |
| 692.8M | -3.60 | 110.41 | 51.36 | 12.91 | 31.15 | -9.63 | 9.05 | -1.11 | -0.63 | -2.62 | -1.38 | -0.63 | 189.67 | |
| 698.2M | -3.30 | 108.98 | 50.32 | 13.24 | 30.72 | -9.11 | 8.64 | -0.89 | -0.52 | -2.87 | -1.39 | -0.36 | 188.25 | |
| 703.6M | -3.49 | 105.44 | 48.55 | 13.08 | 29.65 | -9.00 | 8.93 | -1.07 | -0.53 | -2.46 | -1.39 | - | 182.53 | |
| 708.9M | -3.17 | 105.37 | 48.79 | 13.14 | 29.65 | -9.02 | 8.78 | -0.60 | -0.47 | -2.51 | -1.55 | -0.42 | 182.87 | |
| 714.3M | -3.35 | 107.37 | 49.66 | 13.21 | 30.30 | -8.77 | 9.02 | -0.62 | -0.61 | -2.42 | -1.54 | -0.40 | 186.44 | |
| 719.7M | -3.00 | 108.12 | 49.99 | 13.37 | 30.53 | -8.38 | 9.13 | -0.80 | -0.56 | -2.34 | -1.52 | -0.51 | 188.69 | |
| 725.0M | -3.09 | 108.34 | 49.85 | 13.46 | 30.53 | -8.85 | 8.92 | -0.62 | -0.47 | -2.40 | -1.59 | -0.36 | 188.47 | |
| 730.4M | -3.03 | 106.59 | 49.37 | 13.39 | 29.93 | -8.22 | 8.61 | -1.35 | -0.60 | -2.45 | -1.59 | -0.35 | 185.31 | |
| 735.7M | -3.29 | 107.53 | 49.86 | 13.33 | 30.31 | -8.24 | 8.91 | -0.52 | -0.40 | -2.25 | -1.30 | - | 188.81 | |
| 741.0M | -2.92 | 108.60 | 50.54 | 13.60 | 30.65 | -8.15 | 9.27 | -0.70 | -0.62 | -2.38 | -1.38 | - | 191.10 | |
| 746.4M | -3.08 | 108.12 | 50.22 | 13.32 | 30.49 | -8.43 | 9.14 | -0.61 | -0.42 | -2.40 | -1.48 | -0.36 | 189.43 | |
| 751.7M | -3.08 | 108.98 | 50.49 | 13.40 | 30.67 | -8.25 | 9.33 | -0.59 | -0.39 | -2.44 | -1.51 | -0.63 | 190.78 | |
| 757.0M | -3.15 | 107.17 | 49.38 | 13.25 | 30.11 | -7.99 | 9.23 | -0.77 | -0.49 | -2.61 | -1.44 | -0.43 | 187.11 | |
| 762.3M | -3.20 | 108.02 | 48.95 | 13.38 | 30.37 | -8.19 | 8.89 | -1.30 | -0.28 | -2.54 | -1.24 | -0.42 | 187.23 | |
| 767.6M | -3.62 | 107.40 | 48.39 | 12.77 | 30.36 | -8.84 | 8.84 | -1.34 | -0.15 | -2.60 | -1.65 | -0.50 | 183.43 | |
| 773.0M | -3.76 | 106.81 | 48.35 | 12.83 | 30.18 | -9.93 | 8.68 | -1.50 | -0.13 | -2.14 | -1.19 | - | 182.51 | DmgTkn spike |

## Eval: V52 Checkpoint 4500 (678.3M steps)

W1-66, 50 seeds/loadout, real stats.

| Loadout | Clear% | Death% | Mean Wave |
|---------|--------|--------|-----------|
| BUDGET_RCB | 92.0% | 8.0% | 65.5 |
| MID_ACB | 86.0% | 12.0% | 64.6 |
| CRYSTAL_BP | 100.0% | 0.0% | 66.0 |
| CRYSTAL_NO_BP | 100.0% | 0.0% | 66.0 |
| MAX_TBOW | 96.0% | 4.0% | 65.4 |

### Pillar Stats

| Loadout | NW Avg HP | NW Death% (Wave) | NE Avg HP | NE Death% (Wave) | S Avg HP | S Death% (Wave) |
|---------|-----------|-------------------|-----------|-------------------|----------|-----------------|
| BUDGET_RCB | 14.5% | 66% (W50) | 44.8% | 6% (W65) | 3.3% | 90% (W49) |
| MID_ACB | 16.1% | 66% (W50) | 33.1% | 14% (W42) | 4.5% | 94% (W45) |
| CRYSTAL_BP | 31.9% | 28% (W58) | 59.4% | 0% | 10.5% | 68% (W49) |
| CRYSTAL_NO_BP | 29.2% | 32% (W55) | 54.8% | 0% | 6.7% | 76% (W49) |
| MAX_TBOW | 13.4% | 52% (W55) | 53.1% | 4% (W28) | 7.3% | 78% (W53) |

### vs V51 Baseline (530.8M, W1-66, 50 eps uniform)

V51 had ~14.8% overall death rate, pillar death rates: NW 96%, NE 12%, S 100%.

- **Combat improved**: overall death rate down from ~15% to 4.8% weighted avg (0% on Crystal loadouts)
- **NE pillar**: 12% → 0-6% on most loadouts (was the critical pillar — losing it was a death sentence)
- **NW/S pillars**: still die frequently but later (W50-58 vs W35-37) and less often (28-66% vs 96-100%)
- **MID_ACB weakest**: 12% death rate, 14% NE pillar loss — likely due to lower DPS struggling with nibblers

## Eval: V52 Checkpoint 4500 (678.3M steps) — Post Push-Out Fix

Same checkpoint, same eval params. Simulator now includes player push-out mechanic: when a player stands on an NPC and
attacks, they get pushed to the nearest walkable tile outside the NPC footprint (deterministic, Euclidean distance
tiebreak). Previously the player would get stuck indefinitely — no LOS, no attack drag path.

| Loadout | Clear% | Death% | Mean Wave |
|---------|--------|--------|-----------|
| BUDGET_RCB | 34.0% | 64.0% | 60.5 |
| MID_ACB | 44.0% | 50.0% | 60.3 |
| CRYSTAL_BP | 52.0% | 48.0% | 62.2 |
| CRYSTAL_NO_BP | 64.0% | 34.0% | 63.4 |
| MAX_TBOW | 54.0% | 20.0% | 62.4 |

### Pillar Stats

| Loadout | NW Avg HP | NW Death% (Wave) | NE Avg HP | NE Death% (Wave) | S Avg HP | S Death% (Wave) |
|---------|-----------|-------------------|-----------|-------------------|----------|-----------------|
| BUDGET_RCB | 18.1% | 56% (W51) | 53.8% | 0% | 3.8% | 92% (W47) |
| MID_ACB | 14.8% | 66% (W48) | 53.8% | 0% | 4.6% | 90% (W41) |
| CRYSTAL_BP | 35.8% | 24% (W57) | 63.2% | 0% | 14.4% | 68% (W50) |
| CRYSTAL_NO_BP | 29.7% | 20% (W58) | 62.6% | 0% | 4.9% | 82% (W52) |
| MAX_TBOW | 22.1% | 44% (W58) | 61.9% | 0% | 5.1% | 80% (W51) |

### vs Pre-Fix Eval

The push-out fix **eliminated NE pillar deaths** (0% across all loadouts, was 0-14% pre-fix) but **massively degraded
clear rates and increased deaths**:

| Loadout | Clear% Pre | Clear% Post | Death% Pre | Death% Post |
|---------|------------|-------------|------------|-------------|
| BUDGET_RCB | 92.0% | 34.0% | 8.0% | 64.0% |
| MID_ACB | 86.0% | 44.0% | 12.0% | 50.0% |
| CRYSTAL_BP | 100.0% | 52.0% | 0.0% | 48.0% |
| CRYSTAL_NO_BP | 100.0% | 64.0% | 0.0% | 34.0% |
| MAX_TBOW | 96.0% | 54.0% | 4.0% | 20.0% |

The model was trained without push-out, so it learned behaviors that relied on standing inside NPCs (possibly using
overlap positions as safe spots or attack positions). The sim change at inference time breaks those learned strategies.
Retraining with push-out enabled should recover performance.

## V52 Continuation — Push-Out Retrain

Continue from checkpoint 4500 (678.3M steps) with push-out mechanic now baked into the simulator. No reward changes —
only sim mechanics changed. Entropy bumped back to 0.02 (same as V52 start, which handled the pillar penalty reward
change without regression). No normalization reset since reward terms are unchanged.

```powershell
python -m tools.inferno_rl.train_gpu --load models/V52/inferno_gpu_w31-66_20260403_001816_4500.pt --curriculum-mode static --phase sweep --start-wave 31 --max-wave 66 --observation-version v4 --policy-arch flat_lstm_residual --lstm-hidden-size 256 --lstm-seq-len 32 --lstm-burn-in 16 --episode-mode full --n-envs 64 --n-steps 512 --batch-size 4096 --n-epochs 1 --lr 3e-4 --target-kl 0.02 --entropy-start 0.02 --entropy-end 0.002 --gamma 0.998 --gae-lambda 0.95 --vf-coef 0.25 --max-grad-norm 1.0 --normalize-obs --normalize-reward --actor-sizes 512,512 --critic-sizes 512,512 --save-dir models/V52 --log-dir logs/V52 --checkpoint-every 100 --timesteps 200000000 --device cuda --log-reward-terms --rw-mager-early-kill-base 0 --rw-mager-early-kill-per-npc 0 --rw-mager-priority-per-npc 0 --rw-pillar-death-penalty -2.5 --rw-ne-pillar-death-penalty -5.0
```

## Eval: V52 Push-Out Retrain Checkpoint 1900 (740.6M steps)

W1-66, 50 seeds/loadout, real stats. Best checkpoint from push-out retrain based on TB metrics (Phase_Fail% 9.3%,
Return 1.20) and 10-seed screening.

| Loadout | Clear% | Death% | Mean Wave |
|---------|--------|--------|-----------|
| BUDGET_RCB | 100.0% | 0.0% | 66.0 |
| MID_ACB | 98.0% | 2.0% | 65.7 |
| CRYSTAL_BP | 100.0% | 0.0% | 66.0 |
| CRYSTAL_NO_BP | 100.0% | 0.0% | 66.0 |
| MAX_TBOW | 98.0% | 2.0% | 65.9 |

### Pillar Stats

| Loadout | NW Avg HP | NW Death% (Wave) | NE Avg HP | NE Death% (Wave) | S Avg HP | S Death% (Wave) |
|---------|-----------|-------------------|-----------|-------------------|----------|-----------------|
| BUDGET_RCB | 19.8% | 54% (W54) | 57.4% | 0% | 5.6% | 84% (W50) |
| MID_ACB | 10.8% | 70% (W53) | 53.8% | 2% (W66) | 1.2% | 94% (W46) |
| CRYSTAL_BP | 36.1% | 12% (W59) | 66.0% | 0% | 12.3% | 66% (W50) |
| CRYSTAL_NO_BP | 40.8% | 14% (W59) | 59.6% | 0% | 15.5% | 58% (W52) |
| MAX_TBOW | 26.0% | 26% (W59) | 69.3% | 0% | 7.3% | 70% (W54) |

### vs Pre-Push-Out Eval (Checkpoint 4500, 678.3M)

- **Combat**: 0.8% weighted death rate (2 deaths in 250 eps) vs 4.8% pre-push-out — best ever
- **NE pillar**: essentially 0% death (single death on MID_ACB at W66) — unchanged from push-out fix baseline
- **NW pillar**: 12-70% death vs 28-66% pre-push-out; Crystal loadouts significantly better (12-14% vs 28-32%)
- **S pillar**: still weakest, 58-94% death, but death waves pushed later (W46-54 vs W45-53)
- **MID_ACB still weakest**: only loadout with player death + NE pillar death, 70% NW death rate
