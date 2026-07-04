# V28 TensorBoard Tracking

Fresh BC warmstart (`models/bc_warmstart.pt`). Three simultaneous changes targeting the grad norm
problem that plagued all 8 V27 runs (grad ~6.0 regardless of reward tuning):

1. **Remove auxiliary damage prediction head** — the aux MSE loss dominated gradient direction
2. **1 epoch per rollout** (was 5) — reduces off-policy staleness
3. **All rewards ÷5** — smaller magnitudes → smaller return variance → smaller gradients

## What Changed (V27 R8 → V28)

### Architecture

Removed `damage_head` from Critic. Policy forward returns 6-tuple (was 7). No aux loss in training
loop. Critic is now pure value-only: `hidden → ReLU → value_head`.

### Training Settings

| Setting    | V27 R8   | V28      | Rationale                                                |
|------------|----------|----------|----------------------------------------------------------|
| n-epochs   | 5        | **1**    | PVP reference: 1 epoch reduces off-policy staleness      |
| gamma      | 0.99     | **0.995**| Longer horizon — credit assignment over 200+ tick waves   |
| vf-coef    | 0.75     | **0.5**  | Standard value, less critic gradient weight               |
| aux-coef   | 0.03     | **removed** | Aux head removed entirely                             |
| entropy    | 0.03→0.015 | **0.03→0.015** | Same schedule                                    |
| lr         | 1e-4     | **1e-4** | Same                                                     |
| n-envs     | 16       | 16       | Same                                                     |
| batch-size | 256      | 256      | Same                                                     |
| n-steps    | 1024     | 1024     | Same                                                     |
| grad-norm  | 0.5      | 0.5      | Same                                                     |

### Reward Changes (uniform ÷5)

All reward values divided by 5, ratios preserved. Key values:

| Reward                    | V27 R8 | V28   |
|---------------------------|--------|-------|
| DAMAGE_PENALTY_PER_HP     | -1.5   | -0.3  |
| DAMAGE_DEALT_REWARD_PER_HP| 0.5    | 0.1   |
| BLOOD_BARRAGE_HEAL/HP     | 2.0    | 0.4   |
| SURVIVAL_REWARD_PER_TICK  | 0.1    | 0.02  |
| WAVE_END_HP_BONUS         | 40.0   | 8.0   |
| WAVE_COMPLETE_BASE        | 50.0   | 10.0  |
| KILL_REWARDS (mager)      | 14.0   | 2.8   |
| KILL_REWARDS (nibbler)    | 8.0    | 1.6   |
| STALL_BASE_PENALTY        | -2.0   | -0.4  |
| INVALID_ACTION_PENALTY    | -3.0   | -0.6  |
| MAGER_RESURRECTION_PEN    | 10.0   | 2.0   |

Unchanged (0): DEATH_PENALTY, WAVE_TIMEOUT_PENALTY, INFERNO_COMPLETE, ATTACK_ON_COOLDOWN, MAGER_PRIORITY.

### Checkpoint

- **File**: `models/bc_warmstart.pt` (same BC warmstart as V21/V27)
- **Architecture**: MLP (371K params) — actor [256,256], critic [256,256], obs=186, actions=43
- **Starting from**: Behavioral cloning policy with no RL training

### Training Command

```powershell
tools\inferno_rl\venv\Scripts\activate.ps1; python -m tools.inferno_rl.train_gpu --load models/bc_warmstart.pt --phase climb --start-wave 55 --max-wave 66 --promote-after 5 --min-waves-to-advance 1 --save-dir models/V28_climb --log-dir logs/V28_climb --n-envs 16 --n-steps 1024 --batch-size 256 --n-epochs 1 --lr 1e-4 --entropy-start 0.03 --entropy-end 0.015 --gamma 0.995 --gae-lambda 0.95 --vf-coef 0.5 --max-grad-norm 0.5 --normalize-obs --normalize-reward --checkpoint-every 100 --timesteps 140000000 --device cuda --log-reward-terms
```

---

## Early Warning Checks

| Signal       | Threshold                          | Action                                     |
|--------------|------------------------------------|--------------------------------------------|
| Grad norm    | > 3.0 sustained                   | Reward scale still too large — investigate |
| Entropy loss | ≥ −0.05 (approaching 0)           | Entropy collapse — kill run                |
| EV           | < 0.60 for 2M+ steps              | Value function diverging                   |
| Deaths       | Monotonically increasing 5M+      | Reward counterproductive                   |
| Frontier     | Unchanged from W55 after 10M steps| Stagnation                                 |
| KL           | > 0.030 sustained                 | Policy thrashing                           |

**Key metric to watch:** Grad norm should be 1.5–2.0 (V21 baseline). If it's still 5–6, the aux
head was not the root cause and further investigation is needed.

---

## Manual Observations & Progress

| Steps | Phase | Frontier | f.mean | Deaths | Waves | Timeouts | MeanRwd | EV | Entropy | KL | Grad | FPS | Note |
|-------|-------|----------|--------|--------|-------|----------|---------|----|---------|----|------|-----|------|
