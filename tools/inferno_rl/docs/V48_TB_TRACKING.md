# V48 TB Tracking

## Status

V48 addresses three behavioral problems observed in V47_3 replays at ~115M steps:

1. **Nibbler targeting spam** — model targets nibblers repeatedly without waiting for cooldown to resolve. Root cause:
   `attack_target` was never in the observation. Slots re-sort every tick by LOS/priority/distance/id, so the LSTM had
   to track entity identity across shuffling slot positions with no anchor. It couldn't tell if a previous attack action
   was already queued.

2. **Avoidable LOS exposure** — model stands in LOS of multiple NPCs when one step would break it. Root cause: the
   -0.01/NPC penalty was too weak and fundamentally flawed — it penalized LOS to the model's own target (which is
   required to attack). V48 replaces the penalty with a positive gradient bonus (`LOS_SEPARATION_BONUS = +0.025/tick`)
   that rewards using the pillar to block dangerous NPCs while maintaining offensive LOS to at least one.

3. **NPC proximity** — model stands next to magers/rangers without tactical reason. Root cause: the -0.01/tick boolean
   penalty cost the same for 1 adjacent NPC as 3, and fired every tick even when no NPCs were attacking. The model
   learned off-ticking (which requires understanding NPC attack timers and precise positioning), proving it had capacity
   — this was a signal problem, not a model size problem.

## What Changed

### Observation (504 → 542, +38 features)

**Per-slot additions (SLOT_CORE_SIZE 16 → 18):**

- `can_attack_now` (offset 16) — Boolean: `attack_cooldown == 0 AND distance <= attack_range AND player_has_LOS`.
  Composite feature that eliminates a 3-way conjunction across global (cooldown), slot (distance/LOS), and loadout
  (attack_range). Directly answers "can I shoot this thing right now?" Teaches the model that cooldown ticks are free
  for repositioning.

- `is_current_attack_target` (offset 17) — Boolean: this entity is `state.attack_target`. Gives the LSTM a stable
  anchor for identity tracking across slot re-sorts. Combined with `can_attack_now`, the model can now see: "I already
  have a target queued AND it will resolve when cooldown hits 0" — no more re-targeting spam.

**Global addition (GLOBAL_V4_SIZE 50 → 51):**

- `has_attack_target` (index 50) — Boolean: `attack_target is not None and not dead`. Quick "am I currently engaged?"
  signal without scanning 14 slots.

**Neighborhood addition (NEIGHBORHOOD_FEATURES 9 → 10):**

- `priority_target_attackable` (feature 9, per tile) — Boolean: can player attack the priority target from this tile?
  The 9-tile neighborhood forecast was entirely defensive (LOS counts, imminent threats, prayer, blob scans). It had
  zero offensive information. The model couldn't evaluate "if I step north to break LOS, can I still attack my target?"
  This fills the offensive gap.
  - Priority target: `attack_target` if alive, else first entity from `get_exact_target_slots()`.
  - Computed in `forecast_neighborhood_safety()` as a Python overlay — Cython backend unchanged.
 
| Block | V47 | V48 | Delta |
|-------|-----|-----|-------|
| Global | 50 | 51 | +1 |
| Neighborhood | 81 (9×9) | 90 (9×10) | +9 |
| Threat Horizon | 9 | 9 | — |
| Temporal | 7 | 7 | — |
| Exact Slots | 350 (14×25) | 378 (14×27) | +28 |
| Loadout | 7 | 7 | — |
| **Total** | **504** | **542** | **+38** |

### NPC Melee Adjacency — fixed from deterministic to probabilistic

- **Old:** When player is cardinally adjacent to a Mager/Ranger/Blob, the NPC always (100%) uses melee. Auto-prayer
  predicted this and queued Protect from Melee → player took 0 damage. The model learned adjacency was free.
- **New:** Adjacent non-melee NPCs have a 40% chance to melee (`NPC_ADJACENT_MELEE_CHANCE = 0.40`), 60% chance to use
  their primary style. Auto-prayer always predicts the primary style (melee is unpredictable RNG in OSRS).
- **Effect:** When adjacent to a Mager, auto-prayer keeps Protect from Magic. 60% of attacks are blocked (magic, prayed).
  40% of attacks are melee — unavoidable damage the player must accept. Combined with the proximity penalty, this
  teaches the model that adjacency is dangerous.
- **Mager diagonal melee:** Magers can melee from diagonal tiles (Chebyshev distance 1), not just cardinal. Other NPCs
  remain cardinal-only. LOS is already required for the mager to attack.
- **Scope:** Mager, Ranger, Blob, Blob-Mage, Blob-Range, Bat. Native melee NPCs (Melee, Blob-Melee, Healer) unchanged.
- **Files:** `npc_combat.py` (probabilistic roll), `forecast.py` + `forecast_fast.pyx` (removed adjacency→melee override
  from `_resolve_attack_style_for_state`)

### Rewards

**NPC Proximity — redesigned from scratch:**

- **Old:** `-0.01` per tick, boolean, any adjacent NPC (MELEE/MAGER/RANGER/BLOB). Same penalty for 1 NPC as 3. Fired
  every tick regardless of whether NPCs were attacking.
- **New:** `-0.40` per event, count-scaled, MAGER/RANGER/BLOB only. Fires only when an NPC attacks while the player is
  adjacent (gated on `attacked_this_tick`). MELEE excluded — you can't avoid being near them.
- **Math:** BoFA hit = +0.30 reward. One adjacent mager attack = -0.40. Net: -0.10 per attack cycle when adjacent.
  Two attacking NPCs: -0.80. Standing next to NPCs is now decisively worse than the damage dealt reward.
- **Field:** `player_adjacent_to_npc: bool` → `adjacent_attacking_npc_count: int` on `StepResult`.

**Avoidable LOS penalty removed:**

- `avoidable_los_penalty_per_npc` deleted entirely from `RewardConfig` and `build_v44_reward_config`.
- Rationale: attacking an NPC requires LOS to it, so `avoidable_extra_los` is virtually always > 0 during combat.
  The penalty was always-on noise that couldn't drive behavioral change — the model can't avoid LOS to its own target.

**LOS Separation bonus — gradient reward for pillar use:**

- **New:** `LOS_SEPARATION_BONUS = +0.025` per tick, scaled by fraction of dangerous NPCs blocked behind the pillar.
- **Types:** MAGER, RANGER, MELEE, BLOB. BAT, NIBBLER, and mini-blobs (BLOB_MAGE/RANGE/MELEE) excluded.
- **Condition:** fires when `dangerous_npcs_with_los >= 1` (maintaining offensive LOS) and past grace period.
- **Math:** `fraction = blocked / (dangerous_alive - 1)`. 3 NPCs alive, 1 with LOS = full reward (+0.025/tick).
  3 alive, 2 with LOS = half (+0.0125/tick). 1 alive, 1 with LOS = full (nothing to block = perfect).
- **Rationale:** replaces the removed avoidable_los penalty with a positive shaping signal. The model is rewarded for
  using the pillar to separate NPCs, not penalized for having LOS to its own target. Gradient provides learning signal
  even when perfect 1-LOS isn't achievable.
- **Fields:** `dangerous_npcs_alive: int`, `dangerous_npcs_with_los: int` on `StepResult`.

### Unchanged

- Model architecture (auto-scales to new obs size via `get_observation_size()`)
- Action space
- Observation version string (stays `v4` — incompatible with V47 checkpoints, fresh start required)
- Curriculum, episode mode, phase sampling
- All other reward terms (kill rewards, wave complete, damage dealt, stall, mager priority, etc.)

## V48 Restart Command (initial fresh start)

```powershell
python -m tools.inferno_rl.train_gpu --curriculum-mode static --phase sweep --start-wave 49 --max-wave 66 --observation-version v4 --policy-arch flat_lstm_residual --lstm-hidden-size 256 --lstm-seq-len 32 --lstm-burn-in 16 --episode-mode full --n-envs 64 --n-steps 512 --batch-size 4096 --n-epochs 1 --lr 1e-3 --target-kl 0.02 --entropy-start 0.05 --entropy-end 0.002 --gamma 0.998 --gae-lambda 0.95 --vf-coef 0.25 --max-grad-norm 1.0 --normalize-obs --normalize-reward --actor-sizes 512,512 --critic-sizes 512,512 --save-dir models/V48 --log-dir logs/V48 --checkpoint-every 100 --timesteps 200000000 --device cuda --log-reward-terms
```

## V48_1 Continuation Command

Continuation from V48 checkpoint at ~56M steps. Death curve plateaued at ~14-15 deaths/rollout. Changes target:
stronger death signal, more exploration, focused curriculum, and death diagnostics.

```powershell
python -m tools.inferno_rl.train_gpu --load models/V48/latest.pt --curriculum-mode static --phase sweep --start-wave 49 --max-wave 66 --observation-version v4 --policy-arch flat_lstm_residual --lstm-hidden-size 256 --lstm-seq-len 32 --lstm-burn-in 16 --episode-mode full --n-envs 64 --n-steps 1024 --batch-size 4096 --n-epochs 1 --lr 1e-3 --target-kl 0.02 --entropy-start 0.04 --entropy-end 0.01 --gamma 0.999 --gae-lambda 0.95 --vf-coef 0.25 --max-grad-norm 1.0 --normalize-obs --normalize-reward --actor-sizes 512,512 --critic-sizes 512,512 --save-dir models/V48_1 --log-dir logs/V48_1 --checkpoint-every 100 --timesteps 200000000 --device cuda --log-reward-terms --hindsight-death-penalty 0.3 --hindsight-death-window 15 --hindsight-death-decay 0.85 --sweep-death-retries 2
```

### V48_1 Changes

| Param | V48 | V48_1 | Rationale |
|-------|-----|-------|-----------|
| `--load` | _(none)_ | `models/V48/latest.pt` | continuation run, preserves 56M steps of learned behavior |
| `--n-steps` | 512 | 1024 | 2x spawn diversity per gradient update, CPU has headroom |
| `--gamma` | 0.998 | 0.999 | effective horizon 500→1000 ticks, better death penalty propagation |
| `--entropy-start` | 0.05 | 0.04 | bump above current 0.038 to re-explore |
| `--entropy-end` | 0.002 | 0.01 | raised floor keeps exploration alive in the tail |
| `--hindsight-death-penalty` | 0.0 | 0.3 | retroactive penalty on 15 ticks before death (decay 0.85/tick, ~-1.83 total) |
| `--sweep-death-retries` | _(none)_ | 2 | retry death wave 2x before resuming normal sweep sampling |
| `STALL_BASE_PENALTY` | -0.08 | -0.12 | 50% increase to counteract death penalty incentivizing passive play |
| `STALL_ESCALATION` | 0.04 | 0.06 | faster ramp: -0.12, -0.18, -0.24, -0.30/tick past window |

### V48_1 Code Changes

**Death diagnostics — TB logging by wave & loadout:**

- `OutcomeStatsCallback` now tracks per-death wave number and loadout name
- New TB scalars: `rollout/death_wave_mean`, `rollout/death_wave_min`, `deaths/by_loadout/{name}`
- `loadout` field added to terminal info dict in env

**Hindsight death penalty — bug fix:**

- `_apply_hindsight_death_penalty` in `buffer.py` now gates on `player_died` or `wave_timeout`
- Previously applied to ALL terminals including successful clears (wave 66 completion was penalized)

**Sweep death retries:**

- New `--sweep-death-retries N` flag. After death on wave X, the next N resets in that env start on wave X
- After retries exhaust, resumes normal failure-weighted sweep sampling

### V48_1 Files Changed

| File | Changes |
|------|---------|
| `train_gpu.py` | `OutcomeStatsCallback` death wave/loadout logging, `--sweep-death-retries` arg |
| `training/env.py` | `loadout` in terminal info, `sweep_death_retries` retry-on-death mechanism |
| `ppo/buffer.py` | Gate hindsight death penalty on `player_died`/`wave_timeout` (skip successful clears) |
| `training/rewards.py` | `STALL_BASE_PENALTY` -0.08→-0.12, `STALL_ESCALATION` 0.04→0.06 |
| `async_env/subprocess_vec_env.py` | Plumb `sweep_death_retries` through to worker envs |

## Current Settings

| Setting             | Value                  | Notes |
|---------------------|------------------------|-------|
| restart             | continuation from V48  | `--load models/V48/latest.pt` |
| observation-version | `v4`                   | 542 public / 542 total |
| policy-arch         | `flat_lstm_residual`   | unchanged |
| lstm-hidden-size    | `256`                  | unchanged |
| episode-mode        | `full`                 | per-wave attribution |
| phase               | `sweep`                | failure-weighted across 49-66, with death retries (2x) |
| n-steps             | `1024`                 | doubled from 512 for spawn diversity |
| gamma               | `0.999`                | increased from 0.998 |
| entropy             | `0.04 → 0.01`         | raised floor from 0.002, slight bump to re-explore |
| hindsight death     | `peak=0.3, window=15, decay=0.85` | ~-1.83 total spread across 15 ticks before death |
| reward shaping      | V44 schedules + LOS sep + hindsight death | all V48 rewards preserved |
| loadout             | uniform random (all 5) | BUDGET_RCB, MID_ACB, CRYSTAL_BP, CRYSTAL_NO_BP, MAX_TBOW |

## V48 Files Changed

| File | Changes |
|------|---------|
| `training/observation_common.py` | `SLOT_CORE_SIZE` 16→18, `GLOBAL_V4_SIZE` 50→51, `NEIGHBORHOOD_FEATURES` 9→10 |
| `training/observation_v4.py` | `can_attack_now` + `is_current_attack_target` per slot, `has_attack_target` global, `priority_target_attackable` per neighborhood tile |
| `simulator/forecast.py` | `priority_target_attackable` field on `RawTileThreatSummary` + `NeighborhoodForecast`, computed in `forecast_neighborhood_safety` |
| `simulator/step_result.py` | `player_adjacent_to_npc: bool` → `adjacent_attacking_npc_count: int`, gated on `attacked_this_tick`, MELEE excluded; added `dangerous_npcs_alive`, `dangerous_npcs_with_los` for LOS separation |
| `training/rewards.py` | `NPC_PROXIMITY_PENALTY` → `ADJACENT_NPC_ATTACK_PENALTY = -0.40` × count, `avoidable_los_penalty_per_npc` removed, `LOS_SEPARATION_BONUS = +0.025` added |
| `simulator/npc_combat.py` | Adjacency melee now probabilistic (40%) via `NPC_ADJACENT_MELEE_CHANCE`, applies to all non-melee NPCs |
| `simulator/forecast.py` | Removed adjacency→melee override from `_resolve_attack_style_for_state`; always predicts primary style |
| `simulator/forecast_fast.pyx` | Same removal in Cython version |
| `tests/test_observation_v32.py` | Observation size assertion 504 → 542 |

## What to Watch

- **Hindsight death penalty magnitude** — If deaths drop fast but return also drops significantly, reduce peak from
  0.3 to 0.15. If deaths don't budge after 10M steps, increase to 0.5.
- **Death wave distribution** — Check `rollout/death_wave_mean` and `rollout/death_wave_min`. If deaths cluster on 2-3
  waves, increase `--sweep-death-retries 3-4` or switch to focused drill on those waves.
- **Death loadout distribution** — Check `deaths/by_loadout/*`. If one loadout dominates deaths, consider excluding it
  or oversampling it.
- **Value function recalibration** — EV may drop temporarily after load due to gamma change (0.998→0.999). Should
  recover within 5-10M steps.
- **Entropy** — Should start at ~0.04 and decay to 0.01 floor. If policy locks in fast (entropy drops quickly),
  the raised floor should prevent premature convergence.
- **NPC_Proximity magnitude** — -0.40 per event is steep. If the model becomes overly avoidant of all NPCs (even when
  it needs to be close for barrage), consider reducing to -0.20 or excluding BLOBs.
- **LOS Separation** — watch per-tick magnitude. At +0.025 max, sustained safe-spotting of 20 ticks = +0.50. If the
  model becomes overly passive (farming the bonus instead of killing), reduce to 0.01.

## Future Ideas

**Pillar-relative angular features (V49 candidate)** — If the model still struggles with LOS positioning at 50-80M
steps despite the LOS separation bonus, add per-slot angular separation around the pillar. Current spatial encoding
(dx/dy to player and pillar) requires the network to implicitly reconstruct pillar LOS geometry from coordinates. A
direct angular feature would shortcut this:

- `pillar_angular_separation` — angle between player and NPC around pillar center, normalized to [0, 1] where 0 = same
  face (dangerous) and 1 = opposite face (safe-spotted). Encodes "how much pillar is between us" as a single feature.
- `pillar_angle_sin/cos` — NPC's angle around the pillar, encoded as sin/cos to avoid discontinuities. Combined with
  the player's pillar-relative position (already in globals), gives direct "same side?" reasoning.

Cost: +2-3 features per slot. Don't add preemptively — wait to see if V48's reward shaping is sufficient to teach
pillar geometry from existing features. The concern is conflating "needs better encoding" with "needs more training
time."

**Multi-step movement forecast (V49 candidate)** — The neighborhood features are one-step only (9 tiles around the
player). The model can evaluate "if I step north, what's the LOS situation?" but not "if I step north then east, can I
attack with single LOS?" Pillar wraps — the core repositioning maneuver — require 2-3 movement steps. Add a compressed
multi-step forecast:

- `best_los_in_2_steps` — minimum dangerous NPCs with LOS achievable within 2 movement steps, per neighborhood tile.
  Encodes "this tile is a good waypoint toward a safe position" without requiring the network to chain two one-step
  forecasts through the LSTM.
- `steps_to_attackable_single_los` — minimum steps from each neighborhood tile to reach a position where the player
  has LOS to exactly one dangerous NPC (ideal safe-spot). 0 = already there, 1-3 = reachable, capped at 3.

Cost: +1-2 features per neighborhood tile (+9-18 total obs). Compute cost is moderate — BFS/flood-fill over the 7x7
reachable grid per tick. Consider Cython if Python is too slow. Pairs naturally with the pillar angular features: angles
encode the geometry, multi-step forecast encodes the reachability.

## Metrics Log

| Steps | Eps | Deaths | Timeout% | EV | KL | VL | Clip | Ent | Return | RVar | Grad | Notes |
|-------|-----|--------|----------|----|----|----|------|-----|--------|------|------|-------|
| 5.7M | 171 | 163 | 0.4% | 0.79 | 0.005 | 0.07 | 0.054 | 0.049 | 0.99 | 60.7 | 0.32 | n=173, initial training period |
| 8.0M | 46 | 39 | 3.0% | 0.91 | 0.004 | 0.06 | 0.044 | 0.048 | 1.95 | 66.7 | 0.35 | n=69, Eps 171→46, Deaths 163→39, Return 0.99→1.95, EV +0.12 |
| 13.1M | 36 | 27 | 4.0% | 0.93 | 0.004 | 0.06 | 0.040 | 0.048 | 2.21 | 67.1 | 0.37 | n=157, Return +0.26, Deaths −12, EV +0.02 |
| 18.3M | 35 | 27 | 3.9% | 0.92 | 0.003 | 0.07 | 0.032 | 0.046 | 1.68 | 181.6 | 0.36 | n=159, Return −0.53, RVar 67.1→181.6, VL +0.01, Clip −0.008 |
| 25.8M | 28 | 18 | 4.9% | 0.95 | 0.003 | 0.01 | 0.028 | 0.045 | 1.28 | 237.4 | 0.31 | n=229, Eps 35→28, Deaths 27→18, Return 1.68→1.28, VL 0.07→0.01, RVar 181.6→237.4 |
| 31.5M | 28 | 18 | 4.1% | 0.95 | 0.003 | 0.02 | 0.025 | 0.043 | 1.46 | 197.9 | 0.33 | n=174, Timeout% 4.9→4.1, Return 1.28→1.46, RVar 237.4→197.9, VL 0.01→0.02, Ent 0.045→0.043, Clip 0.028→0.025, Grad 0.31→0.33 |
| 35.0M | 27 | 18 | 3.7% | 0.95 | 0.003 | 0.02 | 0.023 | 0.042 | 1.58 | 180.2 | 0.33 | n=105, Return 1.46→1.58, Timeout% 4.1→3.7, RVar 197.9→180.2, VL stable 0.02, Clip 0.025→0.023, Ent 0.043→0.042 |
| 40.0M | 27 | 16 | 3.5% | 0.96 | 0.002 | 0.02 | 0.022 | 0.041 | 1.63 | 169.0 | 0.32 | n=152, Deaths 18→16, Return 1.58→1.63, Timeout% 3.7→3.5, RVar 180.2→169.0, KL 0.003→0.002, EV 0.95→0.96, Ent 0.042→0.041, Clip 0.023→0.022 |
| 48.2M | 27 | 16 | 3.2% | 0.96 | 0.002 | 0.02 | 0.020 | 0.039 | 1.75 | 156.4 | 0.35 | n=251, Return 1.63→1.75, Timeout% 3.5→3.2, RVar 169.0→156.4, Clip 0.022→0.020, Ent 0.041→0.039, Grad 0.32→0.35, Deaths/EV/KL/VL stable |
| 53.0M | 26 | 14 | 3.4% | 0.96 | 0.002 | 0.02 | 0.020 | 0.038 | 1.84 | 147.6 | 0.36 | n=146, Deaths 16→14, Return 1.75→1.84, Eps 27→26, RVar 156.4→147.6, Ent 0.039→0.038, Grad 0.35→0.36 |
| 56.5M | 26 | 15 | 2.8% | 0.96 | 0.002 | 0.02 | 0.019 | 0.037 | 1.87 | 142.0 | 0.35 | n=106, Return 1.84→1.87, Timeout% 3.4→2.8, RVar 147.6→142.0, Clip 0.020→0.019, Ent 0.038→0.037, Deaths 14→15, EV/KL/VL/Eps stable |
| | | | | | | | | | | | | **V48_1 continuation starts here** — gamma 0.999, n-steps 1024, entropy 0.04→0.01, hindsight death 0.3, stall -0.12/0.06, +sweep retries 2 |
| 59.2M | 55 | 28 | 3.0% | 0.96 | 0.004 | 0.03 | 0.042 | 0.040 | 1.95 | 137.9 | 0.41 | n=3, Eps 26→55, Deaths 15→28, Return 1.87→1.95, KL 0.002→0.004, VL 0.02→0.03, Clip 0.019→0.042, Ent 0.037→0.040, RVar 142.0→137.9 |
| 63.1M | 65 | 29 | 7.5% | 0.97 | 0.004 | 0.04 | 0.036 | 0.040 | 2.11 | 141.4 | 0.40 | n=59, Return 1.95→2.11, Eps 55→65, Timeout% 3.0→7.5, VL 0.03→0.04, RVar 137.9→141.4, Clip 0.042→0.036, Grad 0.41→0.40, EV/KL/Ent stable |
| 68.7M | 68 | 29 | 7.3% | 0.97 | 0.004 | 0.04 | 0.035 | 0.039 | 2.11 | 147.6 | 0.39 | n=85, Return/Deaths/EV/KL/VL stable, Eps 65→68, Timeout% 7.5→7.3, RVar 141.4→147.6, Clip 0.036→0.035, Ent 0.040→0.039, Grad 0.40→0.39 |
| 74.1M | 73 | 30 | 6.8% | 0.97 | 0.004 | 0.03 | 0.035 | 0.038 | 1.84 | 151.2 | 0.36 | n=81, Return 2.11→1.84, VL 0.04→0.03, Eps 68→73, Timeout% 7.3→6.8, RVar 147.6→151.2, Deaths 29→30, Ent 0.039→0.038, Grad 0.39→0.36, EV/KL/Clip stable |
| 79.6M | 71 | 27 | 6.5% | 0.97 | 0.004 | 0.03 | 0.032 | 0.037 | 1.85 | 153.4 | 0.37 | n=83, Deaths 30→27, Return 1.84→1.85, Timeout% 6.8→6.5, RVar 151.2→153.4, Clip 0.035→0.032, Ent 0.038→0.037, Grad 0.36→0.37, Eps/EV/KL/VL stable |
| 85.3M | 71 | 29 | 7.0% | 0.97 | 0.004 | 0.04 | 0.031 | 0.037 | 1.65 | 204.4 | 0.37 | n=87, Return 1.85→1.65, RVar 153.4→204.4, Timeout% 6.5→7.0, train/early_stop 0.0115, VL 0.03→0.04, Deaths 27→29, Clip 0.032→0.031, Eps/EV/KL/Ent/Grad stable |
| 91.0M | 71 | 28 | 5.3% | 0.97 | 0.003 | 0.02 | 0.030 | 0.036 | 1.65 | 208.9 | 0.36 | n=87, Return/Eps stable, Deaths 29→28, Timeout% 7.0→5.3, VL 0.04→0.02, Clip 0.031→0.030, Ent 0.037→0.036, KL 0.004→0.003, Grad 0.37→0.36, RVar 204.4→208.9, train/early_stop 0.0115→0.0000, EV stable |
| 96.7M | 68 | 25 | 5.0% | 0.97 | 0.004 | 0.02 | 0.032 | 0.035 | 1.67 | 207.3 | 0.35 | n=87, Deaths 28→25, Return 1.65→1.67, Timeout% 5.3→5.0, train/early_stop 0.0230, Ent 0.036→0.035, Clip 0.030→0.032, Grad 0.36→0.35, Eps 71→68, KL 0.003→0.004, EV/VL/RVar stable |
| 102.5M | 67 | 24 | 4.9% | 0.98 | 0.004 | 0.02 | 0.032 | 0.034 | 1.73 | 206.7 | 0.36 | n=88, Return 1.67→1.73, Deaths 25→24, EV 0.97→0.98, Ent 0.035→0.034, Timeout% 5.0→4.9%, Grad 0.35→0.36, train/early_stop 0.0230→0.0000, Eps/KL/VL/Clip/RVar stable |
| 108.2M | 64 | 22 | 4.8% | 0.97 | 0.004 | 0.03 | 0.032 | 0.033 | 1.76 | 210.1 | 0.37 | n=86, Return 1.73→1.76, Deaths 24→22, Timeout% 4.9→4.8%, Early_stop 0.0, EV 0.98→0.97 (slight dip), other metrics stable |
| 110.2M | 66 | 23 | 4.3% | 0.98 | 0.004 | 0.02 | 0.030 | 0.033 | 1.80 | 211.1 | 0.36 | n=30, Return 1.76→1.80, Timeout% 4.8→4.3%, EV 0.97→0.98 (recovered), VL 0.03→0.02, Clip 0.032→0.030, train/early_stop 0.0333, Deaths 22→23, Eps/KL/Ent/RVar/Grad stable |