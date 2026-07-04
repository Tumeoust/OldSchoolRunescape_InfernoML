# V32 TensorBoard Tracking

Training stack overhaul. Three new features: LSTM burn-in warmup, target KL early-stopping,
and pre-LSTM input encoder. Seq_len halved (30 → 16). Structured 4-stage training plan.

## What Changed (V31 → V32)

### Architecture

| Setting       | V31        | V32                             |
|---------------|------------|---------------------------------|
| actor-sizes   | 512,512    | 512,512                         |
| critic-sizes  | 512,512    | 512,512                         | 
| LSTM          | 256 hidden | 256 hidden                      |
| lstm-seq-len  | 30         | **16**                          |
| lstm-burn-in  | N/A        | **8** (seq//2)                  |
| input encoder | N/A        | **LayerNorm + residual linear** |
| params        | ~1.5M      | ~1.5M                           |
| obs dims      | 186        | 186                             |

`lstm_seq_len` and `lstm_burn_in` are training-time parameters only. The input encoder
(LayerNorm + residual linear block before LSTM) is new in the recurrent path but does not
break old checkpoint loading.

### Training Stack Changes

#### LSTM Burn-in (`ppo.py:191`, `buffer.py:12`)

Each training sequence window now gets a burn-in prefix (default: `seq_len // 2 = 8` ticks)
extracted from before the window start. The LSTM runs through burn-in observations with
`th.no_grad()` to reconstruct hidden state, then passes the warm state to the actual training
forward pass. This replaces cold-starting every chunk with zeros. Zero-padding is used when
the window starts at the buffer boundary, with `episode_start=True` marking the first real
step after padding.

#### Target KL Early-Stopping (`ppo.py:337`)

PPO now checks KL divergence per mini-batch. If `approx_kl_mean > target_kl`, the entire
epoch is stopped early: accumulated gradients are discarded (optimizer zeroed), and the epoch
loop breaks. Default `target_kl=0.02` for climb, `0.015` for fine-tuning stages.

#### Pre-LSTM Input Encoder (`policy.py:338`)

Lightweight LayerNorm + residual linear block before the LSTM in the recurrent path. Does not
affect checkpoint compatibility — old checkpoints load cleanly.

### Curriculum Changes

#### Weighted Frontier Sampling (`env.py:161`)

Climb phase now samples the frontier band with a weighted bias toward the current bottleneck
wave instead of uniform sampling over `[frontier-3, frontier]`.

#### Backfill Phase (New)

New curriculum phase `"backfill"` alongside climb/harden/drill. Uses failure-weighted wave
sampling: after a 100-episode warmup (uniform random), waves are sampled proportional to their
failure rate with a `0.02` floor weight so even mastered waves get occasional retesting.

### Wave Range

| Setting    | V31 | V32 |
|------------|-----|-----|
| start-wave | 49  | 49  |
| max-wave   | 66  | 66  |

Unchanged from V31.

### Reward Structure

Unchanged from V31. Full weights below for reference.

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

| Setting              | V31   | V32      | Rationale                                              |
|----------------------|-------|----------|--------------------------------------------------------|
| warmstart            | none  | **none** | Fresh init                                             |
| lstm-seq-len         | 30    | **16**   | 30 too long; 17 ticks is max decision-relevant horizon |
| lstm-burn-in         | N/A   | **8**    | Warm LSTM state; default seq_len // 2                  |
| target-kl            | N/A   | **0.02** | Early-stop epochs on KL breach                         |
| start-wave           | 49    | 49       | Same                                                   |
| max-wave             | 66    | 66       | Same                                                   |
| n-epochs             | 2     | 2        | Same                                                   |
| n-envs               | 48    | 48       | Same                                                   |
| batch-size           | 2048  | 2048     | Same                                                   |
| lr                   | 1e-4  | 1e-4     | Same                                                   |
| entropy-start        | 0.02  | 0.02     | Same                                                   |
| entropy-end          | 0.002 | 0.002    | Same                                                   |
| normalize-reward     | yes   | yes      | Same                                                   |
| normalize-obs        | yes   | yes      | Same                                                   |
| n-steps              | 1024  | 1024     | Same                                                   |
| gamma                | 0.995 | 0.995    | Same                                                   |
| gae-lambda           | 0.95  | 0.95     | Same                                                   |
| vf-coef              | 0.5   | 0.5      | Same                                                   |
| max-grad-norm        | 0.5   | 0.5      | Same                                                   |
| phase                | climb | climb    | Same (Stage 1)                                         |
| promote-after        | 5     | 5        | Same                                                   |
| min-waves-to-advance | 1     | 1        | Same                                                   |

### 4-Stage Training Plan

#### Stage 1: Climb The Hard Range

Goal: learn stable late-wave mechanics without overlong LSTM windows.

```powershell
python -m tools.inferno_rl.train_gpu `
  --phase climb --start-wave 49 --max-wave 66 `
  --lstm-hidden-size 256 --lstm-seq-len 16 --lstm-burn-in 8 `
  --n-envs 48 --n-steps 1024 --batch-size 2048 --n-epochs 2 `
  --lr 1e-4 --target-kl 0.02 `
  --entropy-start 0.02 --entropy-end 0.002 `
  --gamma 0.995 --gae-lambda 0.95 `
  --vf-coef 0.5 --max-grad-norm 0.5 `
  --normalize-obs --normalize-reward `
  --actor-sizes 512,512 --critic-sizes 512,512 `
  --save-dir models/V32_climb --log-dir logs/V32_climb `
  --checkpoint-every 100 --timesteps 200000000 --device cuda --log-reward-terms
```

#### Stage 2: Stabilize The Same Range

Goal: preserve the best climb behavior without frontier pressure.
Run from the best Stage 1 checkpoint.

```powershell
python -m tools.inferno_rl.train_gpu `
  --load <best_stage1_checkpoint> `
  --phase harden --start-wave 49 --max-wave 66 `
  --lstm-hidden-size 256 --lstm-seq-len 16 --lstm-burn-in 8 `
  --n-envs 48 --n-steps 1024 --batch-size 2048 --n-epochs 2 `
  --lr 5e-5 --target-kl 0.015 `
  --entropy-start 0.005 --entropy-end 0.001 `
  --gamma 0.995 --gae-lambda 0.95 `
  --vf-coef 0.5 --max-grad-norm 0.5 `
  --normalize-obs --normalize-reward `
  --actor-sizes 512,512 --critic-sizes 512,512 `
  --save-dir models/V32_harden --log-dir logs/V32_harden `
  --checkpoint-every 100 --timesteps 50000000 --device cuda --log-reward-terms
```

#### Stage 3: Backfill Earlier Waves In Bands

Goal: broaden coverage without drowning late-wave signal.
Run two backfill passes: first `35-66`, then `1-66`.
Keep the lower learning rate and KL cap from Stage 2.

```powershell
python -m tools.inferno_rl.train_gpu `
  --load <best_stage2_checkpoint> `
  --phase backfill --start-wave 35 --max-wave 66 `
  --lstm-hidden-size 256 --lstm-seq-len 16 --lstm-burn-in 8 `
  --n-envs 48 --n-steps 1024 --batch-size 2048 --n-epochs 2 `
  --lr 5e-5 --target-kl 0.015 `
  --entropy-start 0.004 --entropy-end 0.001 `
  --gamma 0.995 --gae-lambda 0.95 `
  --vf-coef 0.5 --max-grad-norm 0.5 `
  --normalize-obs --normalize-reward `
  --actor-sizes 512,512 --critic-sizes 512,512 `
  --save-dir models/V32_backfill_35_66 --log-dir logs/V32_backfill_35_66 `
  --checkpoint-every 100 --timesteps 50000000 --device cuda --log-reward-terms
```

Repeat with `--start-wave 1 --max-wave 66` once `35-66` is stable.

#### Stage 4: Drill The Remaining Failure Cases

Goal: repeatedly hit the waves that still fail after backfill.

```powershell
python -m tools.inferno_rl.train_gpu `
  --load <best_backfill_checkpoint> `
  --phase drill --start-wave 1 --max-wave 66 --max-drill-retries 6 `
  --lstm-hidden-size 256 --lstm-seq-len 16 --lstm-burn-in 8 `
  --n-envs 48 --n-steps 1024 --batch-size 2048 --n-epochs 2 `
  --lr 5e-5 --target-kl 0.015 `
  --entropy-start 0.003 --entropy-end 0.001 `
  --gamma 0.995 --gae-lambda 0.95 `
  --vf-coef 0.5 --max-grad-norm 0.5 `
  --normalize-obs --normalize-reward `
  --actor-sizes 512,512 --critic-sizes 512,512 `
  --save-dir models/V32_drill --log-dir logs/V32_drill `
  --checkpoint-every 100 --timesteps 50000000 --device cuda --log-reward-terms
```

#### Checkpoint Selection

Do not keep the latest checkpoint by default. Promote checkpoints by eval:

- 100+ seeds from `W1`, `W35`, `W49`, `W55`, and `W63`
- Choose the best checkpoint by the *worst* start-wave clear rate, not mean reward

### Test Coverage

- `test_lstm_policy.py:189` — burn-in tensor shapes in `generate_sequence_batches`
- `test_phase_curriculum.py:32` — backfill phase: warmup uniform, failure weighting, floor weight
- `test_gpu_ppo.py:1` (new) — target KL early-stopping: stale log-probs force high KL, verifies zero optimizer steps

### Risks

1. **Burn-in adds compute** — 8 extra ticks of no-grad LSTM forward per window. Should be
   minor vs the 3x reduction from seq_len 30→16.
2. **Target KL may be too aggressive** — 0.02 threshold could starve learning if the reward
   normalizer hasn't calibrated. V29 showed KL spikes to 0.13 in first 10M steps from
   uncalibrated normalizer. Monitor early-stop frequency.
3. **Input encoder is untested at scale** — LayerNorm + residual linear before LSTM. Small
   parameter addition but new component. Watch for gradient flow issues.
4. **Backfill phase untested at scale** — failure-weighted sampling is sound in theory
   but the 100-episode warmup threshold and 0.02 floor weight are guesses.

## Metrics Log

| Step  | Frontier | Deaths | Waves Comp | Mean Reward | EV   | KL    | Entropy | Grad Norm | FPS  | Notes                                                                                                                                                        |
|-------|----------|--------|------------|-------------|------|-------|---------|-----------|------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1.5M  | 50       | 363    | 151        | -0.14       | 0.80 | 0.005 | -3.30   | 0.32      | 4957 | First entry. Phase 1, frontier 49→50. EV 0.80 solid for early training. KL 0.005 well under target 0.02.                                                     |
| 5.0M  | 56       | 282    | 259        | 0.41        | 0.69 | 0.007 | -2.88   | 0.48      | 5273 | Frontier 50→56 (+6). Deaths 363→282, waves 151→259. Reward flipped positive. EV 0.69 (watch).                                                                |
| 11.5M | 64       | 275    | 246        | 0.61        | 0.68 | 0.009 | -2.63   | 0.57      | 5078 | Frontier 56→64 (+8). EV 0.68 still below 0.70 (watch) — 6.5M sustained. Grad norm 0.57 above clip=0.5.                                                       |
| 20.3M | 61       | 109    | 250        | 1.62        | 0.65 | 0.009 | -2.51   | 0.62      | 4849 | **Phase 1→2 (level-up #1).** Frontier reset 64→61, f.mean 60.8→56.5. Deaths 275→109 (-60%). Reward 0.61→1.62. EV 0.65 (watch) — 15M+ below 0.70.             |
| 24.9M | 63       | 93     | 215        | 0.47        | 0.77 | 0.010 | -2.45   | 0.64      | 4979 | Frontier 61→63, f.mean 56.5→59.0. Deaths 109→93. **EV recovered 0.65→0.77** — critic catching up post phase-up. Reward dipped 1.62→0.47 (harder curriculum). |
| 30.5M | 63       | 107    | 199        | 0.81        | 0.85 | 0.011 | -2.57   | 0.61      | 5688 | Frontier held 63, f.mean 59.0→60.6. EV 0.77→0.85 — strong. Reward 0.47→0.81. Deaths 93→107 (slight uptick, within noise). Consolidating phase 2.             |
| 46.9M | 60       | 63     | 288        | 3.05        | 0.73 | 0.012 | -2.21   | 0.62      | 4473 | **Phase 2→3 (level-up #2).** Frontier reset 63→60, f.mean 60.6→54.8. Deaths 107→63 (-41%). Waves 199→288. Reward 0.81→3.05. EV 0.85→0.73 (dip post phase-up, expected). |
| 54.5M | 62       | 56     | 317        | 3.68        | 0.80 | 0.011 | -2.15   | 0.64      | 5185 | Frontier 60→62, f.mean 54.8→57.5. Deaths 63→56. Waves 288→317. EV 0.73→0.80 (recovering post phase-up). All metrics trending well. Consolidating phase 3. |
| 68.6M | 50       | 43     | 385        | 5.03        | 0.69 | 0.012 | -2.15   | 0.72      | 5857 | **Phase 3→4 (level-up #3).** Frontier reset 62→50, f.mean 57.5→49.2. Deaths 56→43. Waves 317→385. Reward 3.68→5.03. EV 0.80→0.69 (watch) — dip post phase-up. Timeouts 0→1 (first appearance). |
| 79.2M | 55       | 71     | 367        | 2.61        | 0.57 | 0.017 | -1.42   | 0.67      | 5511 | Frontier 50→55, f.mean 49.2→53.0. **Deaths 43→71 (+65%). EV 0.57 (< 0.60, diverging).** Entropy -2.15→-1.42 (large jump toward 0). Reward 5.03→2.61. Phase 4 curriculum hitting hard. |
| 88.7M | 55       | 94     | 316        | 0.02        | 0.53 | 0.024 | -1.25   | 0.59      | 5341 | **Multiple kill criteria breached.** Deaths rising 20M: 43→71→94. **EV 0.53 (< 0.60 for 20M+, diverging).** Frontier stalled at 55 for ~10M. Entropy -2.15→-1.25 (drift toward 0). KL 0.024 (approaching 0.030). Reward 5.03→0.02. Phase 4 regression. |

## Eval Results (100 seeds per start wave)

| Checkpoint | Steps | W49 Clear | W49 Death | W55 Clear | W55 Death | W63 Clear | W63 Death |
|------------|-------|-----------|-----------|-----------|-----------|-----------|-----------|
| 800        | 39.3M | —         | —         | 0%        | 99%       | —         | —         |
| 900        | 44.2M | —         | —         | 0%        | 92%       | —         | —         |
| 1000       | 49.2M | —         | —         | 0%        | 98%       | —         | —         |
| 1200       | 59.0M | —         | —         | 0%        | 94%       | —         | —         |
| 1300       | 63.9M | —         | —         | 0%        | 89%       | —         | —         |

### W55 Death Distribution (ckpt 800–1300)

| Ckpt | W55  | W56  | W57 | W58 | W59 | W60 | Timeout |
|------|------|------|-----|-----|-----|-----|---------|
| 800  | 30   | 35   | 16  | 8   | 8   | 2   | 1%      |
| 900  | 24   | 38   | 7   | 12  | 9   | 2   | 8%      |
| 1000 | 17   | 59   | 15  | 3   | 4   | —   | 2%      |
| 1200 | 41   | 46   | 6   | 1   | —   | —   | 6%      |
| 1300 | 30   | 43   | 6   | 6   | 2   | 2   | 11%     |

**Analysis:** 0% clear across all checkpoints. W55-56 wall accounts for 75-92% of deaths.
Model regresses after ckpt 800 — max wave reached drops from W60 to W58 at ckpt 1200.
Timeouts climb 1%→11%, indicating learned stalling. Training frontier (55-63) does not
translate to sequential eval performance.
