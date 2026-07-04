# V50 TB Tracking

## Status

V50 addresses the reward signal gap identified in V49: death episodes earned 70% of clear reward (+76.73 vs +110.17 at
~96.6M steps). Dense shaping terms (damage dealt, LOS separation, healing) dominated total return ~5:1 over task rewards,
making death and clear episodes nearly indistinguishable from the model's perspective. V49 reached 47.7% Phase_Fail at
best (checkpoint 2900, ~96.6M steps) but progress had stalled.

V50 rebalances rewards to make survival and wave completion the dominant signals. Fresh optimizer and normalization stats
from V49 best policy weights.

## What Changed

### Rewards

**Terminal signals (new):**

- `DEATH_PENALTY`: 0.0 -> -20.0. Simple terminal penalty on the death tick. NOT the hindsight death penalty from V48_1
  (which retroactively modified past-tick rewards and caused passivity). Propagates backward through GAE (lambda=0.95)
  ~50 ticks, teaching the critic to assign lower value to states approaching death.

- `INFERNO_COMPLETE_REWARD`: 0.0 -> 15.0. Large terminal reward for clearing wave 66. The ultimate goal should be the
  single largest reward event.

**Wave completion (scaled by progress):**

- `WAVE_COMPLETE_REWARD_BASE`: 2.0 -> 3.0
- New `WAVE_PROGRESS_BONUS`: 5.0. Scales linearly with wave number within the training range:
  - Wave 49: 3.0 + 0.0 = 3.0
  - Wave 57: 3.0 + 2.35 = 5.35
  - Wave 66: 3.0 + 5.0 = 8.0
- Later (harder) waves are worth more. Clears complete more late waves, deaths miss them.
- `RewardConfig` now carries `start_wave` and `max_wave` for the scaling calculation.

- `WAVE_END_HP_BONUS`: 1.5 -> 3.0. Doubles the incentive to end each wave at high HP.

**Shaping reduction:**

- `DAMAGE_DEALT_REWARD_PER_HP`: 0.01 -> 0.006 (40% reduction). Was 42% of total reward and the single largest term.
  The model already knows how to deal damage after 96M steps. Reducing this makes task rewards (wave completion,
  survival) relatively more important.

- Implicit: `MAGER_PRIORITY_BONUS` is derived from `DAMAGE_DEALT_REWARD_PER_HP`, so it's also nerfed ~40%.
  `MAGER_EARLY_KILL_BONUS_BASE` (0.6) and `MAGER_EARLY_KILL_BONUS_PER_NPC` (0.15) are unaffected.

### Unchanged

- Observation space (602, v4)
- Model architecture (flat_lstm_residual, 256 LSTM, 512x512 actor/critic)
- Action space
- All other reward terms (LOS separation, stall, blood barrage, kill rewards, pillar damage, etc.)
- Curriculum, episode mode, phase sampling
- NPC melee adjacency (probabilistic from V48)

### Projected Impact

| Metric | V49 (~96.6M) | V50 Projected |
|--------|--------------|---------------|
| Death return | +76.73 | ~+40 |
| Clear return | +110.17 | ~+115 |
| Delta | +33.44 | ~+75 |
| Death/Clear ratio | 70% | ~35% |

## V50 Start Command

Fresh start from V49 best checkpoint (2900), new optimizer and normalization stats:

```powershell
python -m tools.inferno_rl.train_gpu --load models/V49/inferno_gpu_w49-66_20260330_221413_2900.pt --curriculum-mode static --phase sweep --start-wave 49 --max-wave 66 --observation-version v4 --policy-arch flat_lstm_residual --lstm-hidden-size 256 --lstm-seq-len 32 --lstm-burn-in 16 --episode-mode full --n-envs 64 --n-steps 512 --batch-size 4096 --n-epochs 1 --lr 1e-3 --target-kl 0.02 --entropy-start 0.05 --entropy-end 0.002 --gamma 0.998 --gae-lambda 0.95 --vf-coef 0.25 --max-grad-norm 1.0 --normalize-obs --normalize-reward --actor-sizes 512,512 --critic-sizes 512,512 --save-dir models/V50 --log-dir logs/V50 --checkpoint-every 100 --timesteps 200000000 --device cuda --log-reward-terms
```

## Current Settings

| Setting             | Value                  | Notes |
|---------------------|------------------------|-------|
| restart             | fresh start            | `--load models/V49/...2900.pt`, fresh optimizer/normalization |
| observation-version | `v4`                   | 602 features, unchanged |
| policy-arch         | `flat_lstm_residual`   | unchanged |
| lstm-hidden-size    | `256`                  | unchanged |
| episode-mode        | `full`                 | per-wave attribution |
| phase               | `sweep`                | failure-weighted across 49-66 |
| n-steps             | `512`                  | unchanged |
| gamma               | `0.998`                | unchanged |
| entropy             | `0.05 -> 0.002`        | unchanged |
| reward shaping      | V50 rebalance          | death penalty, wave scaling, reduced damage dealt |
| loadout             | uniform random (all 5) | unchanged |

## V50 Files Changed

| File | Changes |
|------|---------|
| `training/rewards.py` | `RewardConfig` +start_wave/max_wave; `DEATH_PENALTY` 0.0->-20.0; `DAMAGE_DEALT_REWARD_PER_HP` 0.01->0.006; `WAVE_COMPLETE_REWARD_BASE` 2.0->3.0; new `WAVE_PROGRESS_BONUS` 5.0; `WAVE_END_HP_BONUS` 1.5->3.0; `INFERNO_COMPLETE_REWARD` 0.0->15.0; wave completion logic uses progress scaling |
| `train_gpu.py` | All 5 `build_v44_reward_config()` calls pass `start_wave`/`max_wave` |
| `tests/test_reward_shaping.py` | +6 tests: death penalty, wave scaling (start/mid/max/clamp), inferno complete |

## What to Watch

- **Value function recalibration** — EV will drop initially as the critic adjusts to the new reward scale. Should
  recover within 5-10M steps. If EV stays below 0.8 after 15M steps, the death penalty may be too large.
- **Critic loss (VL)** — May spike early from the -20 death penalty. Should settle. If VL stays elevated (>0.1),
  consider reducing death penalty magnitude.
- **Death rate vs stall rate** — The death penalty should reduce deaths without increasing stalls. If Timeout% rises
  while Deaths drop, the model is learning passivity (same V48_1 failure mode). Watch the ratio.
- **Mager priority metrics** — The implicit 40% nerf to mager priority bonus could affect kill order. If MagPri drops
  significantly relative to other terms, may need to compensate with a higher `MAGER_PRIORITY_BONUS_PER_NPC`.
- **Reward normalization** — Running stats will need time to adapt to the new distribution. Early training may show
  unstable normalized returns. Should stabilize within 1-2M steps.
- **Return variance (RVar)** — Expected to increase initially due to the -20 death penalty creating bimodal returns.
  Should settle as the model learns to avoid death.

## Metrics Log

| Steps | Eps | Deaths | Timeout% | Phase_Fail% | EV | KL | VL | Clip | Ent | Return | RVar | Grad | Notes |
|-------|-----|--------|----------|-------------|----|----|----|----|-----|--------|------|------|-------|
| 97.3 | 24 | 14 | 3.8% | 62.5% | 0.90 | 0.002 | 0.04 | 0.014 | 0.050 | 1.64 | 200.2 | 0.35 | n=71; Fresh start from V49 checkpoint 2900 |
| 99.4 | 24 | 13 | 4.2% | 60.0% | 0.91 | 0.002 | 0.04 | 0.014 | 0.049 | 1.56 | 198.7 | 0.34 | n=63; Deaths 14→13; Timeout% 3.8→4.2%; Return 1.64→1.56 |
| 104.7 | 24 | 13 | 4.3% | 57.9% | 0.90 | 0.002 | 0.05 | 0.014 | 0.048 | 1.58 | 197.0 | 0.35 | n=161; Phase_Fail% 60.0→57.9%; Return 1.56→1.58 |
| 110.0 | 32 | 19 | 5.3% | 64.1% | 0.89 | 0.002 | 0.05 | 0.013 | 0.047 | 1.37 | 199.5 | 0.34 | n=161; Eps 24→32, Deaths 13→19, Timeout% 4.3→5.3%, Return 1.58→1.37; early_stop 0.006 |
| 115.3 | 31 | 17 | 6.0% | 60.6% | 0.90 | 0.002 | 0.04 | 0.012 | 0.046 | 1.42 | 197.3 | 0.35 | n=163; Eps 32→31, Deaths 19→17, Timeout% 5.3→6.0%; early_stop 0.006 |
| 120.6 | 28 | 14 | 5.5% | 54.0% | 0.91 | 0.002 | 0.04 | 0.012 | 0.045 | 1.53 | 195.1 | 0.36 | n=162; Phase_Fail% 60.6→54.0%; new tag Avoidable_Imminent |
| 125.8 | 28 | 14 | 4.0% | 53.7% | 0.91 | 0.002 | 0.04 | 0.013 | 0.043 | 1.66 | 192.6 | 0.37 | n=160; Timeout% 5.5→4.0%; Return 1.53→1.66 |
| 131.0 | 28 | 14 | 4.1% | 52.6% | 0.91 | 0.002 | 0.04 | 0.013 | 0.042 | 1.67 | 190.5 | 0.38 | n=157; Phase_Fail% 53.7→52.6% |

## Reward Terms Log (ep_sum_mean)

Averaged per-episode sum of each raw reward term. Kills = sum of all Kill_* terms. Total = sum of all terms.

| Steps | DmgDealt | LOSSep | WavComp | BBHeal | WaveHP | MagPri | EarlyMag | Kills | DmgTkn | Stall | MagDel | InvAct | NPCProx | Total |
|-------|----------|--------|---------|--------|--------|--------|----------|-------|--------|-------|--------|--------|---------|-------|
| 97.3 | 22.08 | 18.20 | 30.83 | 12.00 | 13.38 | 3.68 | 4.39 | 17.82 | -12.19 | -4.76 | -5.17 | -3.75 | -1.38 | 86.13 |
| 99.4 | 22.97 | 18.90 | 32.21 | 12.78 | 13.92 | 3.77 | 4.56 | 18.19 | -12.68 | -5.85 | -5.47 | -5.23 | -1.48 | 92.68 |
| 104.7 | 22.83 | 18.65 | 32.19 | 13.18 | 14.07 | 3.61 | 4.41 | 18.45 | -12.97 | -6.54 | -5.54 | -4.97 | -1.58 | 87.01 |
| 110.0 | 16.78 | 13.66 | 23.74 | 10.16 | 9.50 | 2.56 | 3.02 | 13.49 | -10.68 | -5.00 | -4.35 | -3.84 | -1.21 | 77.35 |
| 115.3 | 17.70 | 14.27 | 25.32 | 10.30 | 10.28 | 2.66 | 3.16 | 14.23 | -10.70 | -4.59 | -4.42 | -4.11 | -1.17 | 79.61 |
| 120.6 | 19.52 | 15.75 | 28.56 | 10.67 | 11.80 | 2.99 | 3.59 | 16.27 | -10.75 | -4.90 | -4.65 | -4.43 | -1.23 | 78.76 |
| 125.8 | 19.61 | 16.12 | 28.80 | 10.48 | 12.10 | 3.02 | 3.63 | 16.28 | -10.59 | -3.71 | -4.66 | -4.21 | -0.97 | 80.39 |
| 131.0 | 19.59 | 15.86 | 28.85 | 10.25 | 11.99 | 3.08 | 3.62 | 16.62 | -10.42 | -3.88 | -4.55 | -4.20 | -1.08 | 85.75 |
