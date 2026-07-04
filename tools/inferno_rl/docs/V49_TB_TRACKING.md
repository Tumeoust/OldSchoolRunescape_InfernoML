# V49 TB Tracking

## Status

V49 addresses the observation gap identified in V48's "Future Ideas": the model lacks direct pillar-geometry features
and can only evaluate positions one step ahead. V48 achieved 63.3% clear rate at 59M steps (best checkpoint: 1800).
V48_1 regressed to 36.7% — the continuation run destabilized the policy. V49 is a fresh start with two new
observation features while reverting all V48_1 changes.

## What Changed

### Observation (542 → 602, +60 features)

**Per-slot additions (SLOT_CORE_SIZE 18 → 21):**

- `pillar_angular_separation` (offset 18) — Angle between player and NPC around NE pillar center (18, 23), normalized
  to [0, 1] where 0 = same face (dangerous, no pillar between them) and 1 = opposite face (safe-spotted). Directly
  encodes "how much pillar is between us" as a single feature — shortcutting the implicit geometry reconstruction the
  LSTM had to do from dx/dy coordinates.

- `pillar_angle_sin` (offset 19) — sin(atan2(npc_dy, npc_dx)) from NE pillar center to NPC closest point. Combined
  with cos, provides a continuous 2D encoding of the NPC's angular position around the pillar.

- `pillar_angle_cos` (offset 20) — cos of the same angle. Sin/cos encoding avoids the atan2 discontinuity at ±π and
  lets the network learn rotational relationships between NPC positions.

**Neighborhood additions (NEIGHBORHOOD_FEATURES 10 → 12):**

- `best_los_in_2_steps` (feature 10, per tile) — Minimum dangerous NPC LOS count reachable within 2 BFS steps from
  this neighborhood tile, normalized by total dangerous NPCs alive. Encodes "this tile is a good waypoint toward a
  safer position" — the model can now evaluate multi-step repositioning plans that the 1-step neighborhood couldn't
  capture.
  - BFS uses OSRS-accurate pathfinding: 8-directional with diagonal collision validation.
  - Dangerous types: MAGER, RANGER, MELEE, BLOB (same as LOS separation reward).
  - Shared LOS cache across all 9 neighborhood tiles avoids redundant ray traces.

- `steps_to_single_los` (feature 11, per tile) — Minimum steps from this tile to reach a position with exactly 1
  dangerous NPC LOS AND ability to attack the priority target. Capped at 3, normalized as steps/3.0 where 0 = already
  at ideal safe-spot and 1.0 = no ideal position reachable in 3 steps.
  - Encodes "how far is this tile from an ideal safe-spot?" — the core tactical question for pillar wraps.
  - Combined with `best_los_in_2_steps`, gives the model both "best achievable LOS" and "steps to get there."

| Block | V48 | V49 | Delta |
|-------|-----|-----|-------|
| Global | 51 | 51 | — |
| Neighborhood | 90 (9×10) | 108 (9×12) | +18 |
| Threat Horizon | 9 | 9 | — |
| Temporal | 7 | 7 | — |
| Exact Slots | 378 (14×27) | 420 (14×30) | +42 |
| Loadout | 7 | 7 | — |
| **Total** | **542** | **602** | **+60** |

### Rewards

**Stall penalties reverted to V48 values:**

- `STALL_BASE_PENALTY`: -0.12 → -0.08
- `STALL_ESCALATION`: 0.06 → 0.04

V48_1 elevated these to counteract hindsight death penalty encouraging passivity. V49 drops the death penalty entirely,
so the elevated stall serves no purpose.

### V48_1 CLI params dropped

All V48_1-specific CLI params revert to defaults (not passed in V49 command):
- `--hindsight-death-penalty` — defaults to 0.0
- `--hindsight-death-window` / `--hindsight-death-decay` — inactive when penalty is 0
- `--sweep-death-retries` — defaults to 0

### Unchanged

- Model architecture (auto-scales to new obs size via `get_observation_size()`)
- Action space
- Observation version string (stays `v4` — incompatible with V48 checkpoints due to size change)
- Curriculum, episode mode, phase sampling
- All other reward terms (kill rewards, wave complete, damage dealt, LOS separation, NPC proximity, etc.)
- NPC melee adjacency (stays probabilistic from V48)

## V49 Start Command

Fresh start from V48 best checkpoint, reverting all V48_1 CLI changes:

```powershell
python -m tools.inferno_rl.train_gpu --load models/V49/inferno_gpu_w49-66_20260330_221413_2900.pt --curriculum-mode static --phase sweep --start-wave 49 --max-wave 66 --observation-version v4 --policy-arch flat_lstm_residual --lstm-hidden-size 256 --lstm-seq-len 32 --lstm-burn-in 16 --episode-mode full --n-envs 64 --n-steps 512 --batch-size 4096 --n-epochs 1 --lr 1e-3 --target-kl 0.02 --entropy-start 0.05 --entropy-end 0.002 --gamma 0.998 --gae-lambda 0.95 --vf-coef 0.25 --max-grad-norm 1.0 --normalize-obs --normalize-reward --actor-sizes 512,512 --critic-sizes 512,512 --save-dir models/V49 --log-dir logs/V49 --checkpoint-every 100 --timesteps 200000000 --device cuda --log-reward-terms
```

Note: Loading from V48 checkpoint will fail if observation size mismatch prevents weight loading. If so, start fresh
without `--load` — the V48 checkpoint was trained with obs size 542, V49 uses 602.

## Current Settings

| Setting             | Value                  | Notes |
|---------------------|------------------------|-------|
| restart             | fresh start            | `--load models/V48/...1800.pt` (or fresh if size mismatch) |
| observation-version | `v4`                   | 602 public / 602 total |
| policy-arch         | `flat_lstm_residual`   | unchanged |
| lstm-hidden-size    | `256`                  | unchanged |
| episode-mode        | `full`                 | per-wave attribution |
| phase               | `sweep`                | failure-weighted across 49-66 |
| n-steps             | `512`                  | reverted from 1024 |
| gamma               | `0.998`                | reverted from 0.999 |
| entropy             | `0.05 → 0.002`        | reverted from 0.04 → 0.01 |
| reward shaping      | V44 schedules + LOS sep | stall reverted to V48 values |
| loadout             | uniform random (all 5) | unchanged |

## V49 Files Changed

| File | Changes |
|------|---------|
| `training/observation_common.py` | `SLOT_CORE_SIZE` 18→21, `NEIGHBORHOOD_FEATURES` 10→12 |
| `training/observation_v4.py` | `pillar_angular_separation`, `pillar_angle_sin`, `pillar_angle_cos` per slot; `best_los_in_2_steps`, `steps_to_single_los` per neighborhood tile |
| `simulator/forecast.py` | `NeighborhoodForecast` +2 fields, `_bfs_multistep_forecast()` BFS function, `_BFS_DANGEROUS_TYPES` constant, BFS integration in `forecast_neighborhood_safety()` |
| `training/rewards.py` | `STALL_BASE_PENALTY` -0.12→-0.08, `STALL_ESCALATION` 0.06→0.04 |
| `tests/test_observation_v32.py` | Observation size 542→602, angular separation tests, BFS feature range tests |

## What to Watch

- **BFS performance overhead** — If training throughput drops significantly (>30%), consider implementing the BFS in
  Cython (forecast_fast.pyx) or reducing max_depth from 2 to 1.
- **Angular separation signal** — Check if the model learns to keep high angular separation during combat. If angular
  separation stays low despite the LOS separation reward, the feature may need to be added to the reward signal directly.
- **Multi-step positioning** — Watch for improved pillar-wrap behavior. The model should learn to chain 2-3 movement
  steps to reach single-LOS positions, which was impossible to plan with 1-step-only neighborhood features.
- **Value function** — EV may drop initially while the network learns to use the 60 new features. Should recover
  within 5-10M steps.
- **Observation normalization** — The new features are pre-normalized ([0,1] or [-1,1]), so `--normalize-obs` running
  stats should adapt quickly.

## Metrics Log

| Steps | Eps | Deaths | Timeout% | Phase_Fail% | EV | KL | VL | Clip | Ent | Return | RVar | Grad | Notes |
|-------|-----|--------|----------|-------------|----|----|----|----|-----|--------|------|------|-------|
| 0.5M | 400 | 387 | 0.4% | 64.3% | 0.50 | 0.004 | 0.08 | 0.043 | 0.050 | -0.39 | 76.6 | 0.33 | n=14, early_stop=0.0 |
| 2.9M | 226 | 219 | 0.1% | 63.3% | 0.77 | 0.005 | 0.07 | 0.060 | 0.050 | 0.77 | 36.1 | 0.32 | n=73, early_stop=0.0; Timeout% 0.4→0.1, EV 0.50→0.77, Return -0.39→0.77, RVar 76.6→36.1, Phase_Fail% 64.3→63.3 |
| 8.4M | 54 | 46 | 2.6% | 61.7% | 0.87 | 0.005 | 0.04 | 0.048 | 0.049 | 1.13 | 258.0 | 0.33 | n=166, early_stop=0.0; Eps 226→54, Deaths 219→46, Timeout% 0.1→2.6, RVar 36.1→258.0, Return 0.77→1.13, Phase_Fail% 63.3→61.7 |
| 13.7M | 34 | 26 | 4.8% | 60.5% | 0.93 | 0.004 | 0.02 | 0.038 | 0.047 | 1.15 | 241.7 | 0.32 | n=162, early_stop=0.0; Eps 54→34, Deaths 46→26, Timeout% 2.6→4.8, EV 0.87→0.93, VL 0.04→0.015, RVar 258.0→241.7, Phase_Fail% 61.7→60.5 |
| 19.1M | 31 | 22 | 5.9% | 59.3% | 0.94 | 0.003 | 0.02 | 0.033 | 0.046 | 1.44 | 182.6 | 0.34 | n=164, early_stop=0.0; Eps 34→31, Deaths 26→22, Timeout% 4.8→5.9, Return 1.15→1.44, RVar 241.7→182.6, EV 0.93→0.94, Phase_Fail% 60.5→59.3 |
| 24.5M | 29 | 19 | 5.4% | 58.4% | 0.95 | 0.003 | 0.02 | 0.029 | 0.045 | 1.61 | 157.9 | 0.34 | n=165, early_stop=0.0; Eps 31→29, Deaths 22→19, Timeout% 5.9→5.4, Return 1.44→1.61, RVar 182.6→157.9, VL 0.02→0.024, Phase_Fail% 59.3→58.4 |
| 29.9M | 28 | 19 | 4.3% | 57.3% | 0.94 | 0.003 | 0.03 | 0.026 | 0.044 | 1.67 | 157.9 | 0.35 | n=164, early_stop=0.0; Timeout% 5.4→4.3, Return 1.61→1.67, EV 0.95→0.94, VL 0.02→0.035, Grad 0.34→0.35, Phase_Fail% 58.4→57.3 |
| 35.3M | 28 | 18 | 3.6% | 56.3% | 0.95 | 0.003 | 0.02 | 0.024 | 0.042 | 1.47 | 193.5 | 0.33 | n=163, early_stop=0.0; Return 1.67→1.47, Timeout% 4.3→3.6, RVar 157.9→193.5, EV 0.94→0.95, VL 0.03→0.017, Grad 0.35→0.33, Phase_Fail% 57.3→56.3 |
| 40.6M | 27 | 17 | 3.3% | 54.1% | 0.95 | 0.003 | 0.02 | 0.024 | 0.041 | 1.58 | 175.9 | 0.34 | n=163, early_stop=0.0; Eps 28→27, Deaths 18→17, Timeout% 3.6→3.3, Return 1.47→1.58, RVar 193.5→175.9, Grad 0.33→0.34, Phase_Fail% 56.3→54.1 |
| 46.0M | 27 | 16 | 3.8% | 54.1% | 0.95 | 0.002 | 0.02 | 0.020 | 0.040 | 1.70 | 163.2 | 0.33 | n=165, early_stop=0.0; Deaths 17→16, Timeout% 3.3→3.8, Return 1.58→1.70, RVar 175.9→163.2, KL 0.003→0.002, Ent 0.041→0.040, Clip 0.024→0.020 |
| 51.4M | 26 | 15 | 3.8% | 53.1% | 0.96 | 0.002 | 0.02 | 0.020 | 0.038 | 1.77 | 153.6 | 0.34 | n=165, early_stop=0.0; Eps 27→26, Deaths 16→15, Return 1.70→1.77, RVar 163.2→153.6, EV 0.95→0.96, Ent 0.040→0.038, Grad 0.33→0.34, Phase_Fail% 54.1→53.1 |
| 56.8M | 25 | 14 | 4.0% | 52.2% | 0.96 | 0.002 | 0.02 | 0.020 | 0.037 | 1.91 | 145.6 | 0.36 | n=165, early_stop=0.0; Eps 26→25, Deaths 15→14, Return 1.77→1.91, RVar 153.6→145.6, Timeout% 3.8→4.0, Grad 0.34→0.36, VL 0.02→0.024, Phase_Fail% 53.1→52.2 |
| 62.2M | 26 | 14 | 3.3% | 50.8% | 0.95 | 0.002 | 0.06 | 0.019 | 0.036 | 1.88 | 140.1 | 0.36 | n=163, **early_stop=0.0123**; Return 1.91→1.88, EV 0.96→0.95, VL 0.02→0.057 (high), Timeout% 4.0→3.3, RVar 145.6→140.1, Clip 0.020→0.019, Phase_Fail% 52.2→50.8 |
| 67.6M | 24 | 12 | 3.0% | 50.2% | 0.97 | 0.002 | 0.02 | 0.018 | 0.034 | 1.83 | 159.7 | 0.35 | n=163, early_stop=0.0; Eps 26→24, Deaths 14→12, Return 1.88→1.83, VL 0.06→0.020 (normalized), RVar 140.1→159.7, EV 0.95→0.97, Clip 0.019→0.018, Ent 0.036→0.034, Phase_Fail% 50.8→50.2 |
| 72.9M | 24 | 12 | 2.7% | 50.0% | 0.96 | 0.002 | 0.03 | 0.019 | 0.033 | 1.86 | 172.5 | 0.36 | n=163, **early_stop=0.0123**; Return 1.83→1.86, Timeout% 3.0→2.7, RVar 159.7→172.5, VL 0.02→0.031, Eps 24→24 (stable), Deaths 12→12 (stable), EV 0.97→0.96, Ent 0.034→0.033, Phase_Fail% 50.2→50.0 |
| 78.3M | 24 | 12 | 2.6% | 49.1% | 0.96 | 0.002 | 0.02 | 0.017 | 0.032 | 1.85 | 168.9 | 0.37 | n=165, early_stop=0.0; Timeout% 2.7→2.6, Return 1.86→1.85, RVar 172.5→168.9, Clip 0.019→0.017, Ent 0.033→0.032, Grad 0.36→0.37, Eps 24→24, Deaths 12→12 (both stable), Phase_Fail% 50.0→49.1 |
| 83.7M | 23 | 11 | 2.4% | 49.3% | 0.97 | 0.002 | 0.02 | 0.019 | 0.031 | 1.80 | 172.2 | 0.36 | n=164, **early_stop=0.0122**; Eps 24→23, Deaths 12→11, Timeout% 2.6→2.4, Return 1.85→1.80, RVar 168.9→172.2, EV 0.96→0.97, Clip 0.017→0.019, Grad 0.37→0.36, Ent 0.032→0.031, Phase_Fail% 49.1→49.3 |
| 91.6M | 24 | 12 | 2.1% | 49.5% | 0.96 | 0.002 | 0.03 | 0.017 | 0.029 | 1.74 | 193.6 | 0.36 | n=240, **early_stop=0.0125**; Eps 23→24, Deaths 11→12, Timeout% 2.4→2.1, Return 1.80→1.74, RVar 172.2→193.6, EV 0.97→0.96, VL 0.02→0.033, Ent 0.031→0.029, Phase_Fail% 49.3→49.5 |
| 91.7M | 22 | 12 | 4.7% | 57.5% | 0.94 | 0.002 | 0.02 | 0.016 | 0.028 | 1.70 | 205.1 | 0.34 | n=2-3 (small sample); Eps 24→22, Timeout% 2.1→4.7, Phase_Fail% 49.5→57.5 (jump), Return 1.74→1.70, RVar 193.6→205.1, early_stop=0.0 |
| 94.6M | 25 | 12 | 1.5% | 48.8% | 0.97 | 0.002 | 0.02 | 0.017 | 0.028 | 1.61 | 203.5 | 0.34 | n=87, early_stop=0.0; Eps 22→25, Timeout% 4.7→1.5, Phase_Fail% 57.5→48.8, EV 0.94→0.97, Return 1.70→1.61, RVar 205.1→203.5, VL 0.02→0.015, Clip 0.016→0.017 |
| 96.6M | 25 | 12 | 0.9% | 47.7% | 0.97 | 0.002 | 0.02 | 0.017 | 0.027 | 1.66 | 200.3 | 0.34 | n=62, early_stop=0.0; Timeout% 1.5→0.9, Phase_Fail% 48.8→47.7, Return 1.61→1.66, RVar 203.5→200.3, Ent 0.028→0.027 (stable overall) |

## Reward Terms Log (ep_sum_mean)

Averaged per-episode sum of each raw reward term. Kills = sum of all Kill_* terms. Total = sum of all terms.

| Steps | DmgDealt | LOSSep | WavComp | BBHeal | WaveHP | MagPri | EarlyMag | Kills | DmgTkn | Stall | MagDel | InvAct | NPCProx | Total |
|-------|----------|--------|---------|--------|--------|--------|----------|-------|--------|-------|--------|--------|---------|-------|
| 0.5M | +2.2 | +0.4 | +0.2 | +0.2 | +0.1 | +0.1 | +0.0 | +1.1 | -3.7 | -0.9 | -0.4 | -0.4 | -0.3 | -3.0 |
| 2.9M | +10.7 | +4.4 | +2.8 | +2.8 | +1.7 | +1.4 | +1.1 | +5.3 | -5.6 | -2.5 | -1.5 | -1.6 | -0.3 | +16.0 |
| 8.4M | +23.3 | +11.7 | +6.4 | +7.4 | +4.1 | +3.2 | +2.5 | +11.4 | -9.0 | -3.2 | -3.8 | -3.0 | -1.0 | +47.9 |
| 13.7M | +32.1 | +16.3 | +8.9 | +11.0 | +5.9 | +4.6 | +3.6 | +15.8 | -11.7 | -3.3 | -4.9 | -4.3 | -1.4 | +69.8 |
| 19.1M | +32.9 | +16.4 | +9.3 | +10.6 | +6.1 | +5.3 | +3.9 | +16.0 | -11.4 | -3.7 | -4.7 | -4.1 | -1.1 | +73.3 |
| 24.5M | +34.3 | +17.3 | +9.6 | +11.4 | +6.4 | +5.5 | +4.1 | +16.8 | -12.0 | -3.5 | -5.0 | -3.9 | -1.2 | +77.7 |
| 29.9M | +31.1 | +15.6 | +8.7 | +10.4 | +5.6 | +5.0 | +3.7 | +15.3 | -11.1 | -2.1 | -4.3 | -3.4 | -1.1 | +71.3 |
| 35.3M | +34.2 | +16.9 | +9.7 | +11.0 | +6.4 | +5.7 | +4.2 | +16.8 | -11.6 | -2.3 | -4.3 | -3.2 | -1.1 | +80.6 |
| 40.6M | +35.0 | +17.7 | +9.9 | +10.3 | +6.5 | +5.7 | +4.2 | +16.9 | -10.8 | -2.8 | -4.7 | -3.1 | -1.1 | +81.5 |
| 46.0M | +38.0 | +18.3 | +10.8 | +11.6 | +7.3 | +6.6 | +4.7 | +18.3 | -11.8 | -2.2 | -4.8 | -3.4 | -1.3 | +89.2 |
| 51.4M | +40.2 | +19.2 | +11.4 | +12.0 | +7.8 | +7.0 | +5.0 | +19.4 | -11.8 | -2.0 | -5.1 | -3.2 | -1.0 | +96.0 |
| 56.8M | +38.3 | +18.8 | +10.9 | +11.6 | +7.4 | +6.4 | +4.7 | +18.6 | -11.6 | -2.8 | -5.0 | -3.2 | -1.2 | +89.7 |
| 62.2M | +35.8 | +18.5 | +10.2 | +10.3 | +6.8 | +6.3 | +4.5 | +17.4 | -10.8 | -11.8 | -4.9 | -3.6 | -1.0 | +74.6 |
| 67.6M | +43.3 | +21.4 | +12.5 | +11.5 | +8.5 | +7.7 | +5.5 | +21.0 | -11.4 | -10.6 | -5.5 | -3.0 | -1.0 | +96.9 |
| 72.9M | +41.4 | +20.4 | +12.1 | +11.2 | +8.1 | +7.4 | +5.3 | +20.0 | -11.1 | -1.5 | -5.1 | -3.5 | -1.3 | +100.8 |
| 78.3M | +41.6 | +20.3 | +12.1 | +10.7 | +8.2 | +7.1 | +5.2 | +20.0 | -10.7 | -2.0 | -5.4 | -3.3 | -1.0 | +99.9 |
| 83.7M | +39.5 | +19.7 | +11.4 | +10.4 | +7.7 | +7.3 | +5.1 | +19.1 | -10.4 | -1.8 | -4.6 | -3.1 | -0.9 | +96.6 |
| 91.6M | +40.8 | +19.2 | +11.8 | +10.6 | +8.0 | +7.3 | +5.2 | +19.6 | -10.7 | -2.2 | -4.5 | -2.8 | -1.2 | +98.7 |

## Reward Terms Snapshot (~95.6M, mean of last 50 points)

All `raw_reward_terms/ep_sum_mean` tags, sorted by magnitude.

| # | Term | ep_sum_mean |
|---|------|-------------|
| 1 | Damage_Dealt | +41.14 |
| 2 | LOS_Separation | +19.38 |
| 3 | Wave_Complete | +11.91 |
| 4 | Damage_Taken | -11.19 |
| 5 | Blood_Barrage_Heal | +11.14 |
| 6 | Wave_End_HP_Bonus | +7.93 |
| 7 | Mager_Priority | +7.21 |
| 8 | Early_Mager_Kill | +5.20 |
| 9 | Mager_Delay | -4.83 |
| 10 | Kill_Nib | +4.82 |
| 11 | Kill_Zek | +4.12 |
| 12 | Invalid_Action | -2.94 |
| 13 | Stall_Penalty | -2.51 |
| 14 | Kill_ImKot | +2.07 |
| 15 | Kill_MejRah | +2.04 |
| 16 | Kill_AkRek-Xil | +1.50 |
| 17 | Kill_AkRek-Ket | +1.50 |
| 18 | Kill_AkRek-Mej | +1.50 |
| 19 | Mager_Resurrection | -1.49 |
| 20 | Kill_Ak | +1.26 |
| 21 | NE_Pillar_Zone | +1.17 |
| 22 | NPC_Proximity | -1.14 |
| 23 | Kill_Xil | +0.99 |
| 24 | Blood_Barrage_at_High_HP | -0.90 |
| 25 | Melee_Resurrection | -0.60 |
| 26 | NE_Pillar_Damage | -0.40 |
| 27 | NE_Pillar_Zone_Penalty | -0.33 |
| 28 | Pillar_Damage | -0.27 |
| | **Total** | **+98.29** |

## Death vs Clear Reward Breakdown (~95.6M)

Mean ep_sum_mean split by episode outcome. Death: 13.5 eps/rollout, Clear: 9.6 eps/rollout.

| # | Term | All | Death | Clear | Delta (C-D) |
|---|------|-----|-------|-------|-------------|
| 1 | Damage_Dealt | +38.34 | +31.85 | +47.61 | +15.76 |
| 2 | LOS_Separation | +18.87 | +17.03 | +20.92 | +3.89 |
| 3 | Damage_Taken | -11.58 | -11.98 | -11.23 | +0.75 |
| 4 | Blood_Barrage_Heal | +11.43 | +10.39 | +13.04 | +2.65 |
| 5 | Wave_Complete | +11.06 | +9.21 | +13.80 | +4.59 |
| 6 | Wave_End_HP_Bonus | +7.22 | +5.95 | +9.10 | +3.15 |
| 7 | Mager_Priority | +6.78 | +6.05 | +7.84 | +1.79 |
| 8 | Early_Mager_Kill | +4.85 | +4.35 | +5.58 | +1.22 |
| 9 | Mager_Delay | -4.82 | -3.92 | -5.86 | -1.95 |
| 10 | Kill_Nib | +4.54 | +4.12 | +5.18 | +1.06 |
| 11 | Kill_Zek | +3.82 | +2.78 | +5.27 | +2.49 |
| 12 | Invalid_Action | -3.20 | -2.83 | -3.69 | -0.85 |
| 13 | Stall_Penalty | -3.00 | -2.41 | -3.79 | -1.38 |
| 14 | Kill_MejRah | +1.97 | +2.13 | +1.83 | -0.30 |
| 15 | Kill_ImKot | +1.89 | +1.48 | +2.49 | +1.01 |
| 16 | Mager_Resurrection | -1.46 | -1.15 | -1.84 | -0.69 |
| 17 | Kill_AkRek-Xil | +1.41 | +1.31 | +1.55 | +0.24 |
| 18 | Kill_AkRek-Ket | +1.41 | +1.31 | +1.55 | +0.24 |
| 19 | Kill_AkRek-Mej | +1.40 | +1.30 | +1.55 | +0.25 |
| 20 | Kill_Ak | +1.18 | +1.11 | +1.29 | +0.18 |
| 21 | NPC_Proximity | -1.15 | -1.05 | -1.29 | -0.24 |
| 22 | NE_Pillar_Zone | +1.12 | +1.02 | +1.24 | +0.22 |
| 23 | Kill_Xil | +0.93 | +0.78 | +1.15 | +0.36 |
| 24 | Blood_Barrage_at_High_HP | -0.83 | -0.65 | -1.09 | -0.44 |
| 25 | Melee_Resurrection | -0.56 | -0.45 | -0.71 | -0.26 |
| 26 | NE_Pillar_Zone_Penalty | -0.44 | -0.32 | -0.61 | -0.29 |
| 27 | NE_Pillar_Damage | -0.37 | -0.38 | -0.36 | +0.01 |
| 28 | Pillar_Damage | -0.29 | -0.28 | -0.32 | -0.04 |
| | **Total** | **+90.52** | **+76.73** | **+110.17** | **+33.44** |

**Delta composition** — where the clear-vs-death signal comes from:

| Source | Delta | % of gap |
|--------|-------|----------|
| DmgDealt (shaping) | +15.76 | 47% |
| WavComp (task) | +4.59 | 14% |
| LOSSep (shaping) | +3.89 | 12% |
| WaveHP (task) | +3.15 | 9% |
| BBHeal (shaping) | +2.65 | 8% |
| Kill_Zek (shaping) | +2.49 | 7% |
| Everything else | +2.91 | 9% |

Death episodes earn 70% of clear reward (+76.73 / +110.17). Only 14% of the gap comes from
WavComp — the rest is shaping reward accumulated by surviving longer.

## Eval Benchmarks

### ~95M steps — 50 seeds per loadout, wave 49→66 (deterministic)

Checkpoint: `inferno_gpu_w49-66_20260330_221413_2900.pt` (trained_steps=95,027,200)

| Loadout | Clear% | Death% | Mean Wave |
|---------|--------|--------|-----------|
| MAX_TBOW | 64.0% | 34.0% | 63.3 |
| CRYSTAL_BP | 62.0% | 38.0% | 63.3 |
| CRYSTAL_NO_BP | 42.0% | 56.0% | 62.7 |
| BUDGET_RCB | 34.0% | 60.0% | 61.7 |
| MID_ACB | 26.0% | 72.0% | 60.8 |

**Overall average: ~45.6% clear rate.** Down from V47's ~62% at 104.8M steps. MAX_TBOW and CRYSTAL_BP are the strongest
loadouts; MID_ACB and BUDGET_RCB remain the weakest. The gap between best (64%) and worst (26%) is wider than V47's
(78% vs 40%), suggesting the new observation features haven't yet generalized evenly across gear configurations.
V49 is at fewer training steps (95M vs 105M) and has 60 additional observation features (602 vs 504) which require
more training to integrate — continued improvement is expected.
