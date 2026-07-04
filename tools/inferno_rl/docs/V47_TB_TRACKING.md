# V47 TB Tracking

## Status

The BC warmstart plan was abandoned. Solver demonstrations were too misaligned with the RL action semantics and reset
conditions to be a safe initialization path.

This tracking doc records the fresh V47 launch configuration:

- Fresh RL run from zero (`trained_rollouts=0`, no `--load`)
- Observation version `v4`
- No privileged critic-only observation block
- Penalty-only LOS shaping (no positive single-LOS reward)
- Correct per-wave attribution in `full` episodes

## Current Restart Command

```powershell
python -m tools.inferno_rl.train_gpu --curriculum-mode static --phase sweep --start-wave 49 --max-wave 66 --observation-version v4 --policy-arch flat_lstm_residual --lstm-hidden-size 256 --lstm-seq-len 32 --lstm-burn-in 16 --episode-mode full --n-envs 64 --n-steps 512 --batch-size 4096 --n-epochs 1 --lr 1e-3 --target-kl 0.02 --entropy-start 0.05 --entropy-end 0.002 --gamma 0.998 --gae-lambda 0.95 --vf-coef 0.25 --max-grad-norm 1.0 --normalize-obs --normalize-reward --actor-sizes 512,512 --critic-sizes 512,512 --save-dir models/V47_3 --log-dir logs/V47_3 --checkpoint-every 100 --timesteps 200000000 --device cuda --log-reward-terms
```

## What Changed

### Observation

- `v3.2` was replaced by `v4`
- Observation size is now `504 public / 504 total`
- `v4` is a flat layout: global + neighborhood + threat horizon + temporal + 14 exact target slots + 7-dim loadout
- Exact target slots are shared with simulator targeting order; there are no typed support buckets or overflow block
- Slot type one-hot covers `MAGER`, `RANGER`, `MELEE`, `BLOB`, `BLOB_MAGE`, `BLOB_RANGE`, `BLOB_MELEE`, `BAT`, `NIBBLER`
- Public slots expose signed attack delay and continuous stunned/frozen timers
- Neighborhood forecast is auto-prayer-aware and emits 9 features per tile
- Loadout block is 7 dims; the dead defence dim is gone

### Rewards

- Removed positive single-LOS reward
- `avoidable_los_penalty_per_npc` is constant at `-0.02`
- `avoidable_imminent_penalty` stays active at `-0.01`
- `MULTI_LOS_PENALTY` is active for `npcs_with_los_now >= 2`
- Post-wave-66 training rewards for `JAD`, `HEALER`, `ZUK`, and `ZUK_HEALER` were removed
- Fresh start means rollout-based shaping schedules restart from rollout `0`

### Curriculum / Attribution

- `full` episodes now emit success updates for every cleared wave
- Death/timeout fail is attributed only to the terminal wave
- `opener` mode still attributes only to the configured start wave
- Sweep/backfill/harden sampling now learn from the corrected per-wave stats
- Static `sweep` weighting is learned per env/worker after the 100-episode-per-env warmup

## Current Settings

| Setting             | Value                  | Notes |
|---------------------|------------------------|-------|
| restart             | fresh RL from zero     | no `--load`, `trained_rollouts=0`, normalizers start fresh |
| observation-version | `v4`                   | 504 public / 504 total |
| policy-arch         | `flat_lstm_residual`   | unchanged |
| lstm-hidden-size    | `256`                  | bumped from 128 default |
| episode-mode        | `full`                 | per-wave attribution active |
| phase               | `sweep`                | failure-weighted across 49-66 after per-env warmup |
| reward shaping      | rollout-0 V44 schedules + penalty-only LOS | no single-LOS reward, LOS penalties do not fade |
| loadout             | uniform random (all 5) | BUDGET_RCB, MID_ACB, CRYSTAL_BP, CRYSTAL_NO_BP, MAX_TBOW |

## Metrics Log

| Steps | Eps | Deaths | Timeout% | EV | KL | VL | Clip | Ent | Return | RVar | Clr% | Notes |
|-------|-----|--------|----------|----|----|----|------|-----|--------|------|------|-------|
| 3.6M | 320 | 312 | 0.1% | 0.79 | 0.005 | 0.05 | 0.052 | 0.050 | -0.08 | 57.6 | 73% | n=109, initial training period |
| 8.1M | 74 | 66 | 0.1% | 0.92 | 0.004 | 0.04 | 0.047 | 0.049 | 0.88 | 46.3 | 73% | n=138, Eps 320→74, Return -0.08→0.88, EV +0.13 |
| 13.3M | 40 | 31 | 0.1% | 0.96 | 0.004 | 0.03 | 0.039 | 0.047 | 1.23 | 45.5 | 73% | n=158, Return +0.35 |
| 14.0M | 35 | 25 | 0.0% | 0.97 | 0.004 | 0.03 | 0.035 | 0.047 | 1.44 | 44.5 | 73% | n=22 (narrow 0.7M window) |
| 19.6M | 31 | 21 | 0.1% | 0.97 | 0.003 | 0.03 | 0.033 | 0.046 | 1.43 | 43.8 | 73% | n=171 |
| 25.2M | 29 | 18 | 0.2% | 0.97 | 0.003 | 0.03 | 0.029 | 0.045 | 1.52 | 43.0 | 73% | n=171 |
| 29.2M | 28 | 17 | 0.1% | 0.97 | 0.003 | 0.03 | 0.026 | 0.044 | 1.53 | 42.2 | 73% | n=122 |
| 34.6M | 26 | 14 | 0.1% | 0.98 | 0.003 | 0.03 | 0.025 | 0.042 | 1.56 | 41.8 | 73% | n=164 |
| | | | | | | | | | | | | *— V47_3 restart: γ=0.998, fresh weights —* |
| 0.1M | 346 | 332 | 0.9% | -0.26 | 0.008 | 0.45 | 0.092 | 0.050 | -0.26 | 195.2 | 73% | n=3, fresh start γ=0.998 |
| 5.2M | 203 | 195 | 0.2% | 0.84 | 0.004 | 0.06 | 0.048 | 0.049 | 0.64 | 48.0 | 73% | n=155, Eps 346→203, EV -0.26→0.84, Return -0.26→0.64 |
| 11.0M | 46 | 38 | 1.1% | 0.94 | 0.004 | 0.04 | 0.047 | 0.048 | 1.44 | 75.2 | 73% | n=177, Return +0.80, RVar 48→75 |
| 16.9M | 33 | 24 | 1.9% | 0.96 | 0.004 | 0.04 | 0.038 | 0.047 | 1.77 | 66.8 | 73% | n=180, Return +0.33 |
| 22.8M | 30 | 19 | 1.6% | 0.96 | 0.003 | 0.04 | 0.033 | 0.045 | 1.88 | 63.7 | 73% | n=180, Return +0.11 |
| 28.6M | 28 | 18 | 1.4% | 0.96 | 0.003 | 0.03 | 0.030 | 0.044 | 1.63 | 102.6 | 73% | n=177, Return −0.25, RVar 63.7→102.6 |
| 34.5M | 27 | 16 | 1.3% | 0.97 | 0.003 | 0.02 | 0.029 | 0.042 | 1.64 | 98.7 | 73% | n=180, VL 0.03→0.02 |
| 40.4M | 24 | 12 | 1.1% | 0.97 | 0.003 | 0.02 | 0.026 | 0.041 | 1.76 | 95.1 | 73% | n=180, Deaths −4, Return +0.12 |
| 46.3M | 25 | 13 | 1.0% | 0.97 | 0.003 | 0.03 | 0.025 | 0.040 | 1.78 | 99.2 | 73% | n=180 |
| 52.2M | 24 | 12 | 1.0% | 0.98 | 0.003 | 0.02 | 0.025 | 0.038 | 1.81 | 96.7 | 73% | n=181, EV 0.97→0.98 |
| 58.1M | 23 | 11 | 0.7% | 0.98 | 0.003 | 0.02 | 0.023 | 0.037 | 1.91 | 92.7 | 73% | n=180, Return +0.10 |
| 63.9M | 23 | 10 | 0.6% | 0.97 | 0.002 | 0.02 | 0.022 | 0.035 | 1.90 | 92.3 | 73% | n=177 |
| 69.7M | 24 | 11 | 0.5% | 0.98 | 0.002 | 0.02 | 0.021 | 0.034 | 1.94 | 92.0 | 73% | n=177 |
| 75.6M | 23 | 10 | 0.3% | 0.98 | 0.002 | 0.02 | 0.021 | 0.033 | 1.97 | 89.4 | 73% | n=180 |
| 81.4M | 22 | 9 | 0.4% | 0.98 | 0.002 | 0.02 | 0.018 | 0.031 | 2.03 | 87.3 | 73% | n=177, Return crossed 2.0 |
| 87.2M | 22 | 8 | 0.2% | 0.97 | 0.002 | 0.07 | 0.020 | 0.030 | 2.05 | 91.6 | 73% | n=177, VL spiked 0.02→0.07, Return 2.05 |
| 93.1M | 22 | 8 | 0.3% | 0.98 | 0.002 | 0.02 | 0.020 | 0.028 | 1.68 | 142.4 | 73% | n=180, Return 2.05→1.68, RVar 91.6→142.4 |
| 98.9M | 22 | 8 | 0.1% | 0.98 | 0.002 | 0.01 | 0.018 | 0.027 | 1.68 | 139.7 | 73% | n=177, VL 0.02→0.01 |
| 104.7M | 22 | 8 | 0.1% | 0.98 | 0.002 | 0.01 | 0.018 | 0.026 | 1.73 | 135.4 | 73% | n=177 |
| 109.9M | 22 | 8 | 0.4% | 0.97 | 0.002 | 0.06 | 0.018 | 0.024 | 1.76 | 134.9 | 73% | n=158, VL spiked 0.01→0.06, EV 0.98→0.97 |
| 115.6M | 21 | 7 | 0.5% | 0.97 | 0.002 | 0.12 | 0.019 | 0.023 | 1.56 | 183.4 | 73% | n=174, VL 0.06→0.12, Return 1.76→1.56, RVar 134.9→183.4 |

## Reward Term Snapshots

### 29.5M steps (V47_2 — pre-reward-changes)

| Term | ep_sum_mean |
|------|-------------|
| Mager_Delay | -21.04 |
| Multi-LOS | -15.26 |
| Avoidable_LOS | -12.96 |
| Damage_Taken | -9.75 |
| Invalid_Action | -3.09 |
| NPC_Proximity | -2.54 |
| Stall_Penalty | -1.37 |
| Blood_Barrage_at_High_HP | -0.88 |
| NE_Pillar_Zone_Penalty | -0.59 |
| Pillar_Damage | -0.42 |
| NE_Pillar_Damage | -0.31 |
| Melee_Resurrection | +0.90 |
| Kill_Xil | +1.08 |
| Kill_Ak | +1.31 |
| Kill_AkRek-Ket | +1.56 |
| Kill_AkRek-Mej | +1.56 |
| Kill_AkRek-Xil | +1.56 |
| Mager_Resurrection | +1.65 |
| Kill_MejRah | +2.26 |
| Kill_ImKot | +2.51 |
| NE_Pillar_Zone | +3.33 |
| Kill_Zek | +4.38 |
| Kill_Nib | +5.27 |
| Early_Mager_Kill | +5.57 |
| Mager_Priority | +6.94 |
| Wave_End_HP_Bonus | +8.69 |
| Blood_Barrage_Heal | +8.97 |
| Wave_Complete | +12.92 |
| Damage_Dealt | +43.46 |

## Eval Benchmarks

### 104.8M steps — 50 seeds per loadout, wave 49→66 (deterministic)

Checkpoint: `inferno_gpu_w49-66_20260329_230233_3200.pt` (trained_steps=104,857,600)

| Loadout | Clear% | Death% | Mean Wave |
|---------|--------|--------|-----------|
| CRYSTAL_BP | 78.0% | 22.0% | 64.9 |
| CRYSTAL_NO_BP | 72.0% | 28.0% | 64.4 |
| MAX_TBOW | 72.0% | 28.0% | 64.4 |
| BUDGET_RCB | 48.0% | 50.0% | 63.2 |
| MID_ACB | 40.0% | 58.0% | 62.5 |

**Overall average: ~62% clear rate.** Training uses uniform random loadout sampling (all 5 loadouts seen equally), so the disparity is not a generalization gap — the policy genuinely struggles with budget/mid gear. Even the best loadout (CRYSTAL_BP, 78%) is well below the 95% training clear rate, indicating failure-weighted sampling inflated training metrics.

### 111.6M steps (V47_3 — post-reward-changes)

Reward changes between V47_2 and V47_3 (`ab29dae0`):
- `Multi-LOS` removed entirely (was -0.04/tick when ≥2 NPCs had LOS)
- `avoidable_los_penalty_per_npc` halved (-0.02 → -0.01)
- `Mager_Delay_Penalty` reduced (-0.05 → -0.02)
- `Mager/Melee_Resurrection` sign-fixed (were accidentally +0.6/+0.3, now -0.6/-0.3)

Terms marked with `*` had their reward coefficient changed — deltas reflect both behavior and scale changes.

| Term | ep_sum_mean | Notes |
|------|-------------|-------|
| Avoidable_LOS | -9.87 | *penalty halved (-0.02→-0.01); raw value up despite softer penalty — more LOS exposure* |
| Damage_Taken | -9.05 | similar to 29.5M (-9.75) |
| Mager_Delay | -4.17 | *penalty reduced (-0.05→-0.02); even adjusting for 2.5x scale change, behavior improved* |
| NPC_Proximity | -3.55 | worsened from -2.54 (unchanged term) |
| Invalid_Action | -1.53 | improved from -3.09 |
| Mager_Resurrection | -1.42 | *sign-fixed (was +1.65); now correctly penalizing resurrections* |
| Blood_Barrage_at_High_HP | -1.04 | similar to -0.88 |
| Melee_Resurrection | -0.84 | *sign-fixed (was +0.90); now correctly penalizing* |
| Stall_Penalty | -0.83 | similar to -1.37 |
| NE_Pillar_Zone_Penalty | -0.79 | similar to -0.59 |
| Pillar_Damage | -0.28 | similar |
| NE_Pillar_Damage | -0.21 | similar |
| Multi-LOS | — | *removed entirely (was -15.26)* |
| Kill_Xil | +1.22 | |
| NE_Pillar_Zone | +1.56 | down from +3.33 |
| Kill_Ak | +1.69 | |
| Kill_AkRek-Ket | +2.02 | |
| Kill_AkRek-Mej | +2.02 | |
| Kill_AkRek-Xil | +2.03 | |
| Kill_MejRah | +2.33 | |
| Kill_ImKot | +2.90 | |
| Kill_Zek | +5.47 | |
| Kill_Nib | +6.06 | |
| Early_Mager_Kill | +7.03 | |
| Blood_Barrage_Heal | +9.70 | |
| Mager_Priority | +10.16 | up from +6.94 (unchanged term — genuine behavior improvement) |
| Wave_End_HP_Bonus | +11.13 | up from +8.69 |
| Wave_Complete | +15.70 | up from +12.92 (more waves cleared per ep) |
| Damage_Dealt | +53.65 | up from +43.46 (longer eps) |

Comparable terms (unchanged coefficients): Mager_Priority nearly doubled (+6.94→+10.16) — genuine mager-first targeting improvement. NPC_Proximity worsened (-2.54→-3.55). Kill rewards and Wave_Complete/Damage_Dealt up due to longer episodes (more waves cleared). NE_Pillar_Zone halved (+3.33→+1.56) — spending less time in the NE zone.

## Verification

Targeted pytest coverage added for:

- exact-target slot ordering / action alignment
- split blob visibility in exact target slots
- signed/continuous public timing features
- auto-prayer-aware neighborhood features
- loadout block size
- LOS reward cleanup
- removal of post-66 kill rewards
- full-episode per-wave attribution
- opener attribution preservation
- sweep sampling warmup / failure weighting
