# V51 TB Tracking

## Status

V51 tests the "less is more" hypothesis for reward design. After 125M+ steps across V49-V50, the model has learned
combat, positioning, healing, and kill priority from dense shaping. But 28 reward terms make the critic's job hard
(RVar ~195, noisy value targets) and dilute the task signal — shaping was still ~50% of total return even after V50's
rebalance. V50 reached 53.7% Phase_Fail at best (125.8M steps) but hadn't beaten V49's 47.7%.

V51 strips to ~11 active reward terms. Task signals (wave completion, HP bonus) become dominant by default. All reward
parameters are now CLI-configurable via `--rw-*` flags — old values can be restored without code changes.

## What Changed

### Rewards — Minimal Config

**Kept (unchanged from V50):**

- `death_penalty`: -20.0. Terminal penalty on death tick.
- `wave_complete_base`: 3.0 + `wave_progress_bonus`: 5.0. Progress-scaled wave completion.
- `inferno_complete_reward`: 15.0. Terminal reward for clearing wave 66.
- `wave_end_hp_bonus`: 3.0. HP ratio bonus at wave end.
- `damage_taken_per_hp`: -0.05. Per-tick signal for avoiding damage.
- `blood_barrage_high_hp_penalty`: -0.2. Prevents wasting blood barrage at high HP.
- `mager_early_kill_base`: 0.6 + `mager_early_kill_per_npc`: 0.15. Kill order shaping.
- `mager_delay_penalty`: -0.02. Penalty for not progressing safely focusable mager.
- `mager_priority_per_npc`: 0.25. Multiplier on mager damage when other NPCs alive.
- `stall_base_penalty`: -0.08, `stall_escalation`: 0.04. Anti-passivity guardrail.

**Reduced:**

- `damage_dealt_per_hp`: 0.006 -> 0.003 (50% reduction). Still gives per-tick gradient for attacking, but no longer
  a dominant term. Was ~25% of V50 total return — now projected ~15%.

- `los_separation_bonus`: 0.025 -> 0.01 (60% reduction). Maintains weak gradient for pillar use while V49's angular
  separation obs features carry the load. If the model holds pillar behavior, this can be zeroed in a follow-up.

**Removed (zeroed):**

- `blood_barrage_heal_per_hp`: 0.06 -> 0.0. Was +10.5/ep. Model knows healing.
- `kill_reward_scale`: 1.0 -> 0.0. All 9 entity-type kill rewards zeroed. Was +16.3/ep total. Mager priority system
  handles kill order; damage dealt rewards combat progress. Individual per-kill bonuses are noise at this stage.
- `invalid_action_penalty`: -0.1 -> 0.0. Was -4.2/ep. Invalid rate already low.
- `invalid_attack_penalty`: -0.05 -> 0.0. Same reasoning.
- `adjacent_npc_attack_penalty`: -0.40 -> 0.0. Was -1.0/ep. Model learned adjacency avoidance.
- `pillar_damage_per_hp`: -0.01 -> 0.0. Was -0.7/ep total. Noise.
- `mager_resurrection_penalty`: -0.6 -> 0.0. Should emerge from death penalty + lost wave progress.
- `melee_resurrection_penalty`: -0.3 -> 0.0. Same.
- `ne_pillar_zone_bonus/penalty`: already near-zero from V44 schedule decay. Now 0.0.
- `avoidable_imminent_penalty`: -0.01 -> 0.0. Small, noisy.
- `tile_a_max_reward`, `c_tile_on/adjacent_reward`: already 0.0 from schedule decay.
- `attack_on_cooldown_bonus`: already 0.0.

### Architecture — CLI-Configurable Rewards

All reward magnitudes moved from class-level constants on `InfernoReward` into `RewardConfig` dataclass fields.
Every field exposed via `--rw-*` CLI args with minimal defaults. To restore V50 rewards:

```powershell
--rw-damage-dealt-per-hp 0.006 --rw-blood-barrage-heal-per-hp 0.06 --rw-kill-reward-scale 1.0 --rw-invalid-action-penalty -0.1 --rw-invalid-attack-penalty -0.05 --rw-adjacent-npc-attack-penalty -0.40 --rw-pillar-damage-per-hp -0.01 --rw-mager-resurrection-penalty -0.6 --rw-melee-resurrection-penalty -0.3 --rw-los-separation-bonus 0.025 --rw-avoidable-imminent-penalty -0.01
```

### Unchanged

- Observation space (602, v4)
- Model architecture (flat_lstm_residual, 256 LSTM, 512x512 actor/critic)
- Action space
- Curriculum, episode mode, phase sampling
- NPC melee adjacency (probabilistic from V48)

### Projected Impact

Using V50 @ 125.8M reward terms as baseline:

| Term | V50 ep_sum | V51 Projected | Change |
|------|-----------|--------------|--------|
| DmgDealt | 19.61 | ~10 | halved (0.006->0.003) |
| LOSSep | 16.12 | ~6.5 | reduced (0.025->0.01) |
| WavComp | 28.80 | ~29 | unchanged |
| WaveHP | 12.10 | ~12 | unchanged |
| EarlyMag | 3.63 | ~3.6 | unchanged |
| MagPri | 3.02 | ~1.5 | tracks DmgDealt halving |
| DmgTkn | -10.59 | ~-10.6 | unchanged |
| Stall | -3.71 | ~-3.7 | unchanged |
| MagDel | -4.66 | ~-4.7 | unchanged |
| BBHeal | 10.48 | 0 | removed |
| Kills | 16.28 | 0 | removed |
| InvAct | -4.21 | 0 | removed |
| NPCProx | -0.97 | 0 | removed |
| **Positive total** | ~113 | ~62 | -45% |
| **Task share** | ~36% | ~66% | +30pp |

Task signal share (WavComp + WaveHP) projected at ~66% of positive reward, up from ~36% in V50 and ~20% in V49.
Critic has fewer terms to predict — should reduce RVar and improve value function accuracy.

## V51 Start Command

Fresh start from V50 best checkpoint, new optimizer and normalization stats:

```powershell
python -m tools.inferno_rl.train_gpu --load models/V50/inferno_gpu_w49-66_20260331_080550_1100.pt --curriculum-mode static --phase sweep --start-wave 49 --max-wave 66 --observation-version v4 --policy-arch flat_lstm_residual --lstm-hidden-size 256 --lstm-seq-len 32 --lstm-burn-in 16 --episode-mode full --n-envs 64 --n-steps 512 --batch-size 4096 --n-epochs 1 --lr 1e-3 --target-kl 0.02 --entropy-start 0.05 --entropy-end 0.002 --gamma 0.998 --gae-lambda 0.95 --vf-coef 0.25 --max-grad-norm 1.0 --normalize-obs --normalize-reward --actor-sizes 512,512 --critic-sizes 512,512 --save-dir models/V51 --log-dir logs/V51 --checkpoint-every 100 --timesteps 200000000 --device cuda --log-reward-terms
```

No `--rw-*` flags needed — minimal defaults are the V51 config.

## V51 LR Reduction Command (193.3M steps)

Resume from latest checkpoint with LR 3e-4 (down from 1e-3). Optimizer state and normalization stats preserved via
`--load` — `learn()` overrides LR from CLI on every training step (`ppo.py:276-277`).

```powershell
python -m tools.inferno_rl.train_gpu --load models/V51/inferno_gpu_w49-66_20260331_120107_1900.pt --curriculum-mode static --phase sweep --start-wave 49 --max-wave 66 --observation-version v4 --policy-arch flat_lstm_residual --lstm-hidden-size 256 --lstm-seq-len 32 --lstm-burn-in 16 --episode-mode full --n-envs 64 --n-steps 512 --batch-size 4096 --n-epochs 1 --lr 3e-4 --target-kl 0.02 --entropy-start 0.05 --entropy-end 0.002 --gamma 0.998 --gae-lambda 0.95 --vf-coef 0.25 --max-grad-norm 1.0 --normalize-obs --normalize-reward --actor-sizes 512,512 --critic-sizes 512,512 --save-dir models/V51 --log-dir logs/V51 --checkpoint-every 100 --timesteps 300000000 --device cuda --log-reward-terms
```

## V51 Anti-Stall Patch

Addresses stall-to-timeout exploit observed in W63 BUDGET_RCB eval (seed 5: legitimate slow play, seed 40: deliberate
stalling). Model attacks once every 15 ticks to reset stall counter, then waits for free timeout.

**Changes:**

1. `MAX_TICKS_PER_WAVE`: 500 → 800. Budget setups need more time on hard waves (seed 5 couldn't finish W63 in 500
   ticks while healing at low HP).
2. `wave_timeout_penalty`: 0.0 → -15.0. Timeout is no longer free. Less than death (-20) so model still prefers
   timeout over guaranteed death, but stalling is now costly.
3. **Per-wave stall doubling**: Each time the model enters a stall period (ticks_since_engagement > 15) and resets by
   attacking, then stalls again on the same wave, the penalty multiplier doubles. 1st period = 1x, 2nd = 2x, 3rd = 4x.
   Resets per wave. This makes the "attack once every 15 ticks" exploit exponentially expensive.

**Files changed:** `simulator/step_result.py` (MAX_TICKS), `training/rewards.py` (timeout default, stall doubling),
`train_gpu.py` (CLI default), `tests/test_reward_shaping.py` (+5 tests).

**What to watch:**
- Timeout% should drop near zero once the model learns timeout costs -15.
- Stall penalty ep_sum may increase short-term as the doubling kicks in, then decrease as the model learns to fight.
- If legitimate slow play on hard waves gets punished (stall doubling on waves where repositioning takes >15 ticks
  repeatedly), the STALL_WINDOW may need increasing. Current: 15 ticks.

**Resume command** (from latest checkpoint, `--entropy-start 0.046` to avoid schedule reset):

```powershell
python -m tools.inferno_rl.train_gpu --load models/V51/inferno_gpu_w49-66_20260331_181911_600.pt --curriculum-mode static --phase sweep --start-wave 49 --max-wave 66 --observation-version v4 --policy-arch flat_lstm_residual --lstm-hidden-size 256 --lstm-seq-len 32 --lstm-burn-in 16 --episode-mode full --n-envs 64 --n-steps 512 --batch-size 4096 --n-epochs 1 --lr 3e-4 --target-kl 0.02 --entropy-start 0.046 --entropy-end 0.002 --gamma 0.998 --gae-lambda 0.95 --vf-coef 0.25 --max-grad-norm 1.0 --normalize-obs --normalize-reward --actor-sizes 512,512 --critic-sizes 512,512 --save-dir models/V51 --log-dir logs/V51 --checkpoint-every 100 --timesteps 300000000 --device cuda --log-reward-terms
```

## V51 Reward Cleanup (~323M steps)

Three reward changes applied together before resuming from checkpoint 3400.

### 1. Disable Stall Doubling

The per-wave stall doubling mechanism (introduced in the anti-stall patch) was the root cause of recurring instability
events at 252.8M, 295.7M, and 322.8M. The exponential multiplier (1x → 2x → 4x → ...) created fat-tailed reward
variance the critic couldn't model, permanently inflating RVar from ~165 to 745+ and halving effective Return.

The timeout penalty (-15.0) alone already solved stalling — Timeout% has been 0% since 220M. The doubling was all
downside.

- Removed `_stall_trigger_count` tracking and per-wave doubling multiplier from `InfernoReward`.
- Stall penalty is now flat: `base_penalty - (stall_ticks - 1) * escalation` with no multiplier.

### 2. Zero Mager Delay Penalty

`mager_delay_penalty`: -0.02 → 0.0. The penalty fired every tick the model wasn't dealing damage to a safely focusable
mager — but weapon cooldown is 4 ticks, so the model ate -0.02 on 3 out of every 4 ticks even when perfectly executing
the safespot. At -0.06/cycle vs LOS sep's +0.01/tick, the delay penalty actively discouraged pillar repositioning since
the cost of "not attacking" during movement outweighed the separation bonus. The model has already learned mager
priority — remaining signals (mager_priority_per_npc, early_mager_kill) reward the outcome without penalizing cooldown
ticks.

### 3. Weapon Switch Penalty (new)

`weapon_switch_penalty`: -0.005. Per-switch penalty to discourage rapid back-and-forth weapon toggling observed in some
episodes. Calibrated against DPS math on the tightest case (CRYSTAL_BP loadout, Bowfa→BP vs melee NPC):

| Loadout | Switch | DPS/tick | Round-trip cost | Ticks to recoup |
|---------|--------|----------|-----------------|-----------------|
| CRYSTAL_BP | Bowfa→BP | 3.13→3.38 (+8%) | 0.01 | ~13 ticks |
| MAX_TBOW | Tbow→BP | 2.49→6.46 (+160%) | 0.01 | ~1 tick |

At -0.005, switching to BP for a single melee kill (~22 ticks) is net positive even on the tightest loadout. Pointless
toggling every tick costs -0.005/tick (~1-2% of per-tick reward) — enough to discourage without preventing strategic
switches.

### Files Changed

`training/rewards.py` (removed doubling logic + state, zeroed mager_delay_penalty, added weapon_switch_penalty),
`train_gpu.py` (added `--rw-weapon-switch-penalty` CLI arg),
`tests/test_reward_shaping.py` (replaced 3 doubling tests with 2 flat-penalty tests, updated mager delay test; 19
total).

### What to Watch

- **RVar** should start declining from 745 toward pre-crisis baseline (~160-175). May take 20-30M steps since running
  normalization stats are contaminated.
- **Stall penalty** ep_sum should stabilize in the -1 to -3 range with no more explosions to -9 or -16.
- **VL spikes** should stop recurring. If another instability event occurs, the cause is elsewhere.
- **MagDel** column will drop to 0 in reward terms log. Watch that mager kill order doesn't degrade — if magers start
  dying late, early_mager_kill + mager_priority may not be sufficient. Emergency fix: `--rw-mager-delay-penalty -0.01`.
- **LOS separation** behavior may improve now that the delay penalty no longer discourages repositioning. Watch LOSSep
  ep_sum for an uptick.
- **Weapon Switch** ep_sum should appear as a small negative (-1 to -3 range). If it's near zero, the model stopped
  switching entirely (bad) — lower to -0.002. If it's > -5, the model is still toggling excessively — raise to -0.01.

**Resume command** (from latest checkpoint, `--entropy-start 0.029` to match schedule position):

```powershell
python -m tools.inferno_rl.train_gpu --load models/V51/inferno_gpu_w49-66_20260331_215430_3400.pt --curriculum-mode static --phase sweep --start-wave 49 --max-wave 66 --observation-version v4 --policy-arch flat_lstm_residual --lstm-hidden-size 256 --lstm-seq-len 32 --lstm-burn-in 16 --episode-mode full --n-envs 64 --n-steps 512 --batch-size 4096 --n-epochs 1 --lr 3e-4 --target-kl 0.02 --entropy-start 0.029 --entropy-end 0.002 --gamma 0.998 --gae-lambda 0.95 --vf-coef 0.25 --max-grad-norm 1.0 --normalize-obs --normalize-reward --actor-sizes 512,512 --critic-sizes 512,512 --save-dir models/V51 --log-dir logs/V51 --checkpoint-every 100 --timesteps 400000000 --device cuda --log-reward-terms
```

## V51 Simulator Fixes (~346M steps)

Mechanics accuracy fixes to LOS, pathfinding, and melee dig applied to the simulator.
Training paused at checkpoint 700 (~346.6M steps) to apply fixes. Reward normalization stats reset (`--load` without
`--resume-normalization`) since the simulator changes alter reward distributions.

### 1. LOS Ray Direction Fix

Ray now traces NPC closest tile → player, matching RuneLite's `WorldArea.hasLineOfSightTo`. Was player → NPC tile. The
fixed-point Bresenham algorithm is not direction-symmetric, so this produced incorrect results at specific tile
alignments (especially near pillar corners). Both `player_has_los_to_npc` and `can_attack_entity` now route through the
same NPC→player ray path as `npc_has_los_to_player`.

### 2. Pathfinding Cardinal Priority Fix

When diagonal movement is blocked, X cardinal is now always tried before Y (was axis-gap-dependent — larger gap tried
first). Y-only movement is also blocked at Chebyshev distance ≤ 1, matching RuneLite's `WorldArea` guard. This affects
NPC pathing around pillars and in tight spaces where the old behavior could choose the wrong cardinal direction.

### 3. Melee Dig Mechanic Fix

Changed from "all size² overlap positions sorted by Manhattan distance" to the fixed 4-position priority from osrs-sdk:
1. SW of player (player at NE corner of NPC footprint)
2. On player (player at SW corner)
3. West of player (player at east edge)
4. South of player (player at north edge)
5. Fallback: slight SW offset, then spiral outward

The old approach sometimes picked wrong positions for 4×4 Jal-ImKot when multiple overlap tiles had equal Manhattan
distance.

### 4. LOS Separation Reward — Skip at 1 Dangerous NPC

`los_separation_bonus` now skips when only 1 dangerous NPC remains. Previously the reward was always computed, but with
a single NPC there's nothing to separate from — the bonus was just noise. (Committed as `a08df704`.)

### Files Changed

**Simulator:** `geometry.py`, `geometry.pyx` (LOS ray direction), `pathfinding.py`, `pathfinding.pyx` (cardinal
priority, Chebyshev guard), `npc_movement.py` (dig 4-position priority).

**Tests:** `test_simulator_parity.py` (+5 tests: X-first cardinal, Chebyshev guard, Y at distance >1, dig priority
order).

### What to Watch

- **Phase_Fail%** may spike short-term as the model adapts to corrected NPC behavior (different movement paths around
  pillars, different dig landing spots). Should recover within 10-20M steps if the model's policy generalizes.
- **LOSSep** ep_sum may shift — corrected ray direction changes which tiles count as in/out of LOS near pillar corners.
- **Stall penalty** — if corrected pathfinding makes NPCs reach the player faster, engagement windows change.
- **Deaths** — if dig positioning changes meaningfully, prayer-switch-on-dig timing could be affected.

**Resume command** (from latest checkpoint, `--entropy-start 0.028` to match schedule position):

```powershell
python -m tools.inferno_rl.train_gpu --load models/V51/inferno_gpu_w49-66_20260401_141420_800.pt --curriculum-mode static --phase sweep --start-wave 49 --max-wave 66 --observation-version v4 --policy-arch flat_lstm_residual --lstm-hidden-size 256 --lstm-seq-len 32 --lstm-burn-in 16 --episode-mode full --n-envs 64 --n-steps 512 --batch-size 4096 --n-epochs 1 --lr 3e-4 --target-kl 0.02 --entropy-start 0.028 --entropy-end 0.002 --gamma 0.998 --gae-lambda 0.95 --vf-coef 0.25 --max-grad-norm 1.0 --normalize-obs --normalize-reward --actor-sizes 512,512 --critic-sizes 512,512 --save-dir models/V51 --log-dir logs/V51 --checkpoint-every 100 --timesteps 500000000 --device cuda --log-reward-terms
```

## V51 W1-66 Expansion + Mager Reward Zeroing (~421.7M steps)

Two changes applied together before resuming from checkpoint 1500 (~421.7M steps).

### 1. Wave Range Expansion: 49-66 -> 1-66

Training now covers the full Inferno (waves 1-66). The model needs to learn pillar preservation across
the full gauntlet, not just the final 18 waves where pillars start at full HP. The model already kills
nibblers and preserves the NE pillar — no pillar damage penalty re-enabled.

**Sweep sampler behavior with W1-66:**
- Warmup (first 100 eps/env): uniform random across 1-66
- After warmup: failure-weighted. W1-48 will have near-zero fail rate (weight 0.02 floor each),
  W49-66 will dominate. Estimated ~20% episodes on easy waves, ~80% on hard waves.

**wave_progress_bonus rescaling** (formula: `3.0 + 5.0 * (wave - start_wave) / (max_wave - start_wave)`):

| Wave | Old (W49-66) | New (W1-66) |
|------|-------------|-------------|
| 1    | N/A         | 3.00        |
| 35   | N/A         | 5.62        |
| 49   | 3.00        | 6.69        |
| 66   | 8.00        | 8.00        |

Hard waves now have higher absolute completion rewards. Normalization stats reset since reward
distributions change significantly.

### 2. Mager Rewards Zeroed

- `mager_early_kill_base`: 0.6 -> 0.0
- `mager_early_kill_per_npc`: 0.15 -> 0.0
- `mager_priority_per_npc`: 0.25 -> 0.0

Removes ~10.7/ep (MagPri ~3.27 + EarlyMag ~7.46 at 421.7M). Active reward terms reduced from ~11 to
~8. The model has deeply learned mager priority over 421M steps. Remaining implicit signals: death
penalty (-20), wave completion (3-8 per wave), damage taken penalty (-0.05/hp).

### What to Watch

- **Phase_Fail%** — Will spike initially as sweep warmup sends episodes to unfamiliar W1-48. Should
  recover within 20-30M steps as failure-weighting concentrates on hard waves.
- **WavComp ep_sum** — Will increase since episodes starting at W1 clear more waves. Absolute value
  less meaningful; track Phase_Fail% instead.
- **Mager kill order** — With EarlyMag and MagPri zeroed, watch that magers don't start dying late. If
  Phase_Fail% on W57+ (triple mager waves) degrades significantly, emergency fix:
  `--rw-mager-early-kill-base 0.3 --rw-mager-priority-per-npc 0.1`.
- **Pillar HP** — Model should maintain NE pillar through W1-48 nibbler waves without explicit reward.
  If pillars are dying and causing downstream deaths: `--rw-pillar-damage-per-hp -0.02`.
- **RVar** — Normalization reset + distribution change will cause a temporary spike. Should settle
  within 5-10M steps.
- **Episode length** — Episodes starting at W1 are much longer. Monitor throughput (steps/second).

**Resume command** (from latest checkpoint, `--entropy-start 0.026` to match schedule position):

```powershell
python -m tools.inferno_rl.train_gpu --load models/V51/inferno_gpu_w49-66_20260401_165645_1500.pt --curriculum-mode static --phase sweep --start-wave 1 --max-wave 66 --observation-version v4 --policy-arch flat_lstm_residual --lstm-hidden-size 256 --lstm-seq-len 32 --lstm-burn-in 16 --episode-mode full --n-envs 64 --n-steps 512 --batch-size 4096 --n-epochs 1 --lr 3e-4 --target-kl 0.02 --entropy-start 0.026 --entropy-end 0.002 --gamma 0.998 --gae-lambda 0.95 --vf-coef 0.25 --max-grad-norm 1.0 --normalize-obs --normalize-reward --actor-sizes 512,512 --critic-sizes 512,512 --save-dir models/V51 --log-dir logs/V51 --checkpoint-every 100 --timesteps 200000000 --device cuda --log-reward-terms --rw-mager-early-kill-base 0 --rw-mager-early-kill-per-npc 0 --rw-mager-priority-per-npc 0
```

## Current Settings

| Setting             | Value                  | Notes |
|---------------------|------------------------|-------|
| restart             | resumed @ 193.3M       | `--load models/V51/...1900.pt`, optimizer+normalization preserved |
| observation-version | `v4`                   | 602 features, unchanged |
| policy-arch         | `flat_lstm_residual`   | unchanged |
| lstm-hidden-size    | `256`                  | unchanged |
| episode-mode        | `full`                 | per-wave attribution |
| start-wave          | `1`                    | was 49; full Inferno coverage |
| phase               | `sweep`                | failure-weighted across 1-66 |
| n-steps             | `512`                  | unchanged |
| gamma               | `0.998`                | unchanged |
| entropy             | `0.05 -> 0.002`        | unchanged |
| reward shaping      | V51 minimal, mager zeroed | ~8 active terms + timeout penalty (-15), no mager rewards, no stall doubling, +weapon switch (-0.005) |
| wave timeout        | 800 ticks, -15.0 penalty | was 500 ticks, 0.0 penalty |
| loadout             | uniform random (all 5) | unchanged |

## V51 Files Changed

| File | Changes |
|------|---------|
| `training/rewards.py` | `RewardConfig` expanded to 33 fields (was 9); all reward magnitudes moved from class constants to config; `InfernoReward` reads all values from `self.config`; `_BASE_KILL_REWARDS` dict + `kill_reward_scale`; `build_v44_reward_config()` returns full-rewards config for backward compat |
| `train_gpu.py` | 31 `--rw-*` CLI args in "Reward Configuration" group; `_reward_config_from_args()` helper; `reward_config` param on `train()`; all 6 reconfigure sites use static config dict |
| `tests/test_reward_shaping.py` | Updated for new defaults; +3 tests: minimal defaults, kill scaling, config round-trip (15 total) |

## What to Watch

- **Value function recalibration** — EV will drop initially as the critic adjusts to the new (simpler) reward scale.
  Should recover faster than V50 because the target is lower-variance. If EV stays below 0.8 after 10M steps,
  something is wrong.
- **RVar** — Should decrease from V50's ~195. Fewer terms = lower-variance value targets. If RVar doesn't drop within
  5M steps, the remaining terms may still be too noisy.
- **Pillar wrapping behavior** — LOSSep reduced 60%. If the model stops using the pillar (deaths spike on multi-NPC
  waves), the obs features aren't carrying the load. Emergency fix: `--rw-los-separation-bonus 0.025`.
- **Kill order** — Kill rewards removed but mager priority system kept. If mager kill order degrades (magers dying
  late), the per-kill rewards were providing more signal than expected. Fix: `--rw-kill-reward-scale 0.5`.
- **Passivity** — Removing invalid action penalty + NPC proximity penalty could theoretically enable more passive play.
  Stall penalty should prevent this. If Timeout% rises >8%, check stall metrics.
- **Reward normalization** — Running stats will adapt quickly since the reward distribution is simpler (fewer modes).
  Should stabilize within 1M steps.

## Metrics Log

| Steps | Eps | Deaths | Timeout% | Phase_Fail% | EV | KL | VL | Clip | Ent | Return | RVar | Grad | Notes |
|-------|-----|--------|----------|-------------|----|----|----|----|-----|--------|------|------|-------|
| 133.1M | 24 | 13 | 5.5% | 59.1% | 0.93 | 0.001 | 0.02 | 0.011 | 0.050 | 1.00 | 189.4 | 0.34 | n=62; new reward terms: Death (-10.66), Inferno_Complete (6.14) |
| 136.3M | 22 | 11 | 6.6% | 56.7% | 0.94 | 0.001 | 0.02 | 0.011 | 0.049 | 0.89 | 187.8 | 0.34 | n=98-99; Return down 0.11, Phase_Fail% down 2.4pp, DmgDealt up 1.17 |
| 138.8M | 22 | 11 | 4.6% | 56.1% | 0.93 | 0.001 | 0.02 | 0.010 | 0.049 | 0.84 | 186.2 | 0.33 | n=75-76; Return down 0.05, Timeout% down 2.0pp |
| 141.6M | 24 | 12 | 5.1% | 54.4% | 0.94 | 0.001 | 0.02 | 0.010 | 0.048 | 0.91 | 185.0 | 0.34 | n=84-85; Return up 0.07, Phase_Fail% down 1.7pp, Timeout% up 0.5pp |
| 146.6M | 26 | 11 | 5.5% | 48.9% | 0.94 | 0.001 | 0.02 | 0.011 | 0.047 | 1.00 | 182.8 | 0.34 | n=151-152; Return up 0.09, Phase_Fail% down 5.5pp, DmgDealt down 0.88, train/early_stop avg 0.007 |
| 151.6M | 25 | 10 | 4.6% | 44.8% | 0.94 | 0.001 | 0.03 | 0.011 | 0.046 | 1.09 | 180.6 | 0.35 | n=152-153; Return up 0.09, Phase_Fail% down 4.1pp, VL up 0.01 |
| 156.8M | 25 | 10 | 5.0% | 44.4% | 0.94 | 0.001 | 0.02 | 0.011 | 0.045 | 1.13 | 180.4 | 0.36 | n=157-158; Return up 0.04, VL down 0.01, Timeout% up 0.4pp |
| 162.0M | 26 | 11 | 5.2% | 46.8% | 0.93 | 0.001 | 0.03 | 0.010 | 0.043 | 1.04 | 179.4 | 0.35 | n=159-160; Return down 0.09, Phase_Fail% up 2.4pp, VL up 0.01, train/early_stop avg 0.0063 |
| 165.3M | 24 | 9 | 3.5% | 39.5% | 0.94 | 0.001 | 0.02 | 0.011 | 0.042 | 1.17 | 178.2 | 0.36 | n=100-101; Phase_Fail% down 7.3pp (46.8→39.5), Deaths down 2.2, Timeout% down 1.7pp, Return up 0.13, Inferno_Complete 9.07 (was 6.14), Death -7.21 (was -10.66) |
| 167.3M | 24 | 9 | 3.2% | 38.6% | 0.94 | 0.001 | 0.02 | 0.011 | 0.041 | 1.18 | 177.0 | 0.38 | n=62-63; Return up 0.01, Phase_Fail% down 0.9pp, Timeout% down 0.3pp, Grad up 0.02, Stall penalty -4.00 (down 0.92 vs prior) |
| 172.2M | 25 | 9 | 2.8% | 38.6% | 0.94 | 0.001 | 0.02 | 0.010 | 0.041 | 1.19 | 175.9 | 0.36 | n=149-150; Return up 0.01, Timeout% down 0.4pp, Clip down 0.001, Grad down 0.02, RVar down 1.1 |
| 177.0M | 24 | 9 | 2.7% | 38.3% | 0.94 | 0.001 | 0.02 | 0.011 | 0.040 | 1.23 | 173.6 | 0.36 | n=145-146; Return up 0.04, Phase_Fail% down 0.3pp, Timeout% down 0.1pp, RVar down 2.3, Ent down 0.001 |
| 182.3M | 25 | 9 | 2.0% | 38.3% | 0.93 | 0.001 | 0.03 | 0.011 | 0.038 | 1.24 | 173.4 | 0.38 | n=162-163; Return up 0.01, Timeout% down 0.7pp, VL up 0.01, Grad up 0.02 |
| 187.3M | 24 | 7 | 3.0% | 34.3% | 0.94 | 0.001 | 0.02 | 0.010 | 0.037 | 1.27 | 172.5 | 0.37 | n=152-153; Return up 0.03, Phase_Fail% down 4.0pp (38.3→34.3), Deaths down 2, Timeout% up 1.0pp, VL down 0.01 |
| 192.5M | 25 | 10 | 2.0% | 39.8% | 0.93 | 0.001 | 0.03 | 0.010 | 0.036 | 1.20 | 174.8 | 0.36 | n=160-161; Return down 0.07, Phase_Fail% up 5.5pp (34.3→39.8), Deaths up 3, Timeout% down 1.0pp, VL up 0.01, train/early_stop avg 0.0063 |
| 194.6M | 25 | 9 | 1.4% | 37.8% | 0.94 | 0.001 | 0.02 | 0.011 | 0.035 | 1.22 | 173.7 | 0.36 | n=64-65; Return up 0.02, Phase_Fail% down 2.0pp (39.8→37.8), Timeout% down 0.6pp, VL down 0.01 |
| 210.6M | 20 | 6 | 2.5% | 33.5% | 0.95 | 0.000 | 0.02 | 0.001 | 0.049 | 1.27 | 170.7 | 0.34 | n=523-526; resumed @ 193.3M with LR 3e-4; Return up 0.05, Phase_Fail% down 4.3pp (37.8→33.5), Deaths down 3, Timeout% up 1.1pp, KL down 0.001, Clip down 0.010 (significant), Ent up 0.014 |
| 211.6M | 22 | 6 | 2.5% | 30.9% | 0.95 | 0.000 | 0.02 | 0.001 | 0.047 | 1.31 | 168.3 | 0.35 | n=28-29; Return up 0.04, Phase_Fail% down 2.6pp (33.5→30.9), VL down 0.01, RVar down 2.4, Deaths stable at 6 |
| 215.6M | 22 | 6 | 2.0% | 28.1% | 0.95 | 0.000 | 0.02 | 0.001 | 0.047 | 1.34 | 167.2 | 0.36 | n=123; Return up 0.03, Phase_Fail% down 2.8pp (30.9→28.1), Timeout% down 0.5pp, RVar down 1.1 |
| 220.9M | 18 | 5 | 0.0% | 28.9% | 0.96 | 0.000 | 0.02 | 0.001 | 0.045 | 1.36 | 165.4 | 0.35 | n=161-162; Return up 0.02, Phase_Fail% up 0.8pp (28.1→28.9), Deaths down 1, Timeout% down 2.0pp (2.0→0.0), RVar down 1.8 |
| 225.2M | 18 | 4 | 0.0% | 25.1% | 0.96 | 0.000 | 0.01 | 0.001 | 0.044 | 1.44 | 163.4 | 0.37 | n=130-131; Return up 0.08, Phase_Fail% down 3.8pp (28.9→25.1), Deaths down 1, Timeout% stable, RVar down 2.0, VL down 0.01, Inferno_Complete up 0.60 (10.66→11.24) |
| 225.7M | 20 | 4 | 0.0% | 23.4% | 0.96 | 0.000 | 0.01 | 0.001 | 0.044 | 1.41 | 162.4 | 0.35 | n=15-16; Return down 0.03, Phase_Fail% down 1.7pp (25.1→23.4), DmgDealt down 1.52, Inferno_Complete up 0.24 (11.24→11.48), Death -4.69 (up 0.40) |
| 230.7M | 21 | 5 | 0.0% | 24.0% | 0.96 | 0.000 | 0.02 | 0.001 | 0.044 | 1.46 | 161.7 | 0.37 | n=151-152; Return up 0.05, Phase_Fail% up 0.6pp (23.4→24.0), Deaths up 1, Timeout% stable, RVar down 0.7, EarlyMag down 0.46, DmgDealt down 0.97, Stall penalty down 0.31 (-3.01→-2.70), VL up 0.01 |
| 231.1M | 20 | 5 | 0.0% | 21.5% | 0.96 | 0.000 | 0.01 | 0.001 | 0.043 | 1.46 | 160.7 | 0.36 | n=13-14; Phase_Fail% down 2.5pp (24.0→21.5), VL down 0.01, RVar down 1.0, Return stable, Deaths stable, Timeout% stable, EarlyMag up 0.12 (5.68→5.79), Inferno_Complete up 0.37 (11.41→11.77) |
| 236.1M | 21 | 5 | 0.0% | 23.4% | 0.95 | 0.000 | 0.07 | 0.001 | 0.043 | 1.47 | 161.1 | 0.38 | n=151-152; VL up 0.06 (0.01→0.07, significant), Phase_Fail% up 1.9pp (21.5→23.4%), EV down 0.01 (0.96→0.95), Return up 0.01 (1.46→1.47), Deaths stable, Timeout% stable, Stall penalty -3.56 (up from -2.72), BBHighHP -0.77 (down 0.06 from -0.83) |
| 236.6M | 21 | 4 | 0.0% | 21.8% | 0.96 | 0.000 | 0.01 | 0.001 | 0.043 | 1.37 | 172.7 | 0.37 | n=13-14; Return down 0.10 (1.47→1.37), Phase_Fail% down 1.6pp (23.4→21.8%), RVar up 11.6 to 172.7 (significant), VL down 0.06 (0.07→0.01, significant), EV up 0.01 (0.95→0.96), Deaths down 1 to 4, Stall penalty -2.76 (up from -3.56), Death -4.36 (down 0.40 from -4.74), BBHighHP -0.53 (up 0.24 from -0.77) |
| 241.5M | 20 | 5 | 0.0% | 21.8% | 0.96 | 0.000 | 0.02 | 0.001 | 0.042 | 1.45 | 171.7 | 0.38 | n=149-150; Return up 0.08 (1.37→1.45, best since 225.2M), Phase_Fail% stable at 21.8%, RVar down 1.0, VL up 0.01, Deaths up 1 to 5, Timeout% stable, EarlyMag up 0.12 (5.72→5.86), Inferno_Complete stable (5.72→11.74), Stall penalty -2.53 (down 0.23 from -2.76) |
| 242.0M | 22 | 4 | 0.0% | 16.4% | 0.96 | 0.000 | 0.01 | 0.001 | 0.042 | 1.51 | 170.7 | 0.39 | n=15-16; Phase_Fail% down 5.4pp (21.8→16.4%, best yet, significant), Return up 0.06 (1.45→1.51, best since 225.2M at 1.44), Deaths down 1 to 4, VL down 0.01 (0.02→0.01), RVar down 1.0, Inferno_Complete up 0.80 (11.74→12.54), Death -3.50 (up 0.94 from -4.44) |
| 246.9M | 21 | 5 | 0.0% | 21.6% | 0.95 | 0.000 | 0.02 | 0.001 | 0.041 | 1.48 | 169.6 | 0.38 | n=150-151; Return down 0.03 (1.51→1.48), Phase_Fail% up 5.2pp (16.4→21.6%), Deaths up 1 to 5, VL up 0.01, RVar down 1.1, EV down 0.01, Inferno_Complete down 0.78 (12.54→11.76), Stall penalty -2.15 (up 0.35 from -2.50), BBHighHP -0.70 (up 0.06 from -0.76) |
| 252.8M | 21 | 5 | 0.0% | 22.3% | 0.94 | 0.000 | 0.23 | 0.001 | 0.041 | 0.85 | 538.1 | 0.44 | **CRITICAL INSTABILITY**: n=181-182; VL=0.23 (12x elevated, normal 0.01-0.02), RVar=538.1 (3.2x elevated, normal 160-175), Stall=-16.12 (8x worse than baseline -2), Grad=0.44 (elevated), Return=0.85 (halved from 1.4-1.5), Wave_Timeout penalty introduced (-0.88), EV recovered to 0.94, Phase_Fail% up 0.7pp. Training rewards or loss function may be corrupted. Recommend checkpoint rollback to 242.0M and investigation. |
| 257.8M | 20 | 4 | 0.0% | 18.6% | 0.96 | 0.000 | 0.00 | 0.001 | 0.040 | 0.85 | 552.2 | 0.37 | n=152-153; **RECOVERY**: VL down 0.23→0.004 (58x drop, recovered), Stall up -16.12→-2.03 (normalized), Phase_Fail% down 3.7pp (22.3→18.6%), Deaths down 1 (5→4), EV up 0.02, Grad down 0.07, DmgDealt up 0.60, Inferno_Complete up 0.56, WavComp up 2.11, WaveHP up 1.37. RVar elevated (552 vs baseline 160-175) but stable. Return still at 0.85 (depressed from 1.4-1.5). Training stability restored. |
| 263.2M | 21 | 4 | 0.0% | 18.6% | 0.94 | 0.000 | 0.07 | 0.001 | 0.039 | 0.81 | 611.3 | 0.45 | n=165-166; VL up to 0.07 (from 0.004, elevated but below crisis 0.23), RVar up to 611.3 (from 552), Stall penalty down to -8.77 (from -2.03, degraded), Grad up to 0.45 (from 0.37), Return down to 0.814 (from 0.85), Phase_Fail% stable at 18.6%, Deaths stable at 4. Wave_Timeout penalty reappeared (1 episode, -0.83). Metrics show partial re-degradation but far below crisis levels. Training oscillating. |
| 268.9M | 20 | 4 | 0.0% | 18.6% | 0.96 | 0.000 | 0.01 | 0.001 | 0.038 | 0.79 | 635.3 | 0.37 | n=172-173; Return down 0.02 to 0.79, VL down 0.06 to 0.01, Stall penalty up 6.84 to -1.93, RVar up 24 to 635.3, Grad down 0.08, EV up 0.02 |
| 269.1M | 18 | 4 | 0.0% | 22.4% | 0.95 | 0.000 | 0.00 | 0.001 | 0.038 | 0.82 | 629.1 | 0.38 | n=6-7; Return up 0.03, Phase_Fail% up 3.8pp, VL down 0.01, RVar down 6.2, Stall penalty up 0.53 |
| 274.0M | 20 | 3 | 0.0% | 17.0% | 0.96 | 0.000 | 0.00 | 0.001 | 0.037 | 0.82 | 624.2 | 0.37 | n=150-151; Phase_Fail% down 5.4pp (22.4→17.0%), Deaths down 1 (4→3), RVar down 4.9, EV up 0.01, Grad down 0.01 |
| 274.5M | 22 | 4 | 0.0% | 17.9% | 0.93 | 0.000 | 0.01 | 0.001 | 0.037 | 0.82 | 619.3 | 0.36 | n=16-17; EV down 0.03 (0.96→0.93, significant), VL up 0.01, Phase_Fail% up 0.9pp, Deaths up 1 to 4, Eps up 2 to 22, Stall penalty down 2.53 to -4.57 (significant degradation), Reward total down 7.31 to 77.11 |
| 279.4M | 21 | 4 | 0.0% | 18.9% | 0.96 | 0.000 | 0.00 | 0.001 | 0.037 | 0.81 | 614.1 | 0.37 | n=150-151; EV up 0.03 (0.93→0.96, significant recovery), Phase_Fail% up 1.0pp (17.9→18.9%), Return down 0.01, RVar down 5.2, Stall penalty up 2.79 to -1.78 (normalized), Reward total up 4.55 to 81.66 |
| 279.9M | 20 | 3 | 0.0% | 16.1% | 0.95 | 0.000 | 0.00 | 0.001 | 0.036 | 0.80 | 608.9 | 0.36 | n=16-17; Phase_Fail% down 2.8pp (18.9→16.1%), Deaths down 1 to 3, Return down 0.01, RVar down 5.2, EV down 0.01, Stall penalty down 0.44 |
| 284.9M | 21 | 3 | 0.0% | 15.5% | 0.96 | 0.000 | 0.00 | 0.001 | 0.036 | 0.84 | 603.9 | 0.37 | n=151-152; Return up 0.04, Phase_Fail% down 0.6pp, RVar down 5.0, EV up 0.01, Grad up 0.01 |
| 285.3M | 20 | 3 | 0.0% | 16.7% | 0.97 | 0.000 | 0.00 | 0.001 | 0.035 | 0.86 | 598.8 | 0.38 | n=13-14; Return up 0.02, Phase_Fail% up 1.2pp, RVar down 5.1, EV up 0.01, Grad up 0.01 |
| 290.3M | 21 | 3 | 0.0% | 16.4% | 0.96 | 0.000 | 0.00 | 0.001 | 0.035 | 0.86 | 594.0 | 0.36 | n=151-152; Return stable, Phase_Fail% down 0.3pp (16.7→16.4%), RVar down 4.8, EV down 0.01, Grad down 0.02 |
| 290.8M | 22 | 3 | 0.0% | 13.2% | 0.96 | 0.000 | 0.00 | 0.001 | 0.035 | 0.92 | 589.1 | 0.37 | n=13-14; Phase_Fail% down 3.2pp (16.4→13.2%, significant), Return up 0.06 (0.86→0.92, highest since 285.3M), RVar down 4.9, Deaths stable at 3, Stall penalty up 0.53 to -0.89, Inferno_Complete up 0.48 to 13.02 |
| 295.7M | 21 | 3 | 0.0% | 15.4% | 0.94 | 0.000 | 0.11 | 0.001 | 0.034 | 0.84 | 627.0 | 0.36 | **INSTABILITY EVENT**: n=148-149; VL spike to 0.11 (11x increase from 0.01 baseline, matches crisis threshold), Stall penalty down to -9.27 (10.18 worse, exceeds prior -16.12), RVar up 37.9 to 627 (4.3x elevated), Return down 0.08 (0.92→0.84), Phase_Fail% up 2.2pp (13.2→15.4%), EV down 0.02, Wave_Timeout penalty appeared (1 ep, -0.54). Similar pattern to 252.8M crisis. |
| 296.2M | 21 | 4 | 0.0% | 18.2% | 0.95 | 0.000 | 0.00 | 0.001 | 0.034 | 0.74 | 743.2 | 0.35 | n=13-14; **ONGOING DEGRADATION**: VL recovered from 0.11 to 0.004, but RVar peaked at 743.2 (5.2x baseline), Return crashed to 0.74 (lowest since start of session), Phase_Fail% up to 18.2%, Stall normalized to -1.38, Deaths up 1 to 4, EV up 0.01. Training continues unstable post-crisis. |
| 301.1M | 21 | 3 | 0.0% | 15.2% | 0.96 | 0.000 | 0.00 | 0.001 | 0.033 | 0.77 | 737.2 | 0.36 | n=149-150; Return up 0.03 (0.74→0.77), Phase_Fail% down 3.0pp (18.2→15.2%), RVar down 6.0 (still 4.1x baseline), Deaths down 1 to 3, EV up 0.01, Grad up 0.01, VL stable at 0.00. Partial recovery from crisis. |
| 301.6M | 21 | 3 | 0.0% | 13.1% | 0.97 | 0.000 | 0.00 | 0.001 | 0.033 | 0.82 | 731.4 | 0.37 | n=14-15; Return up 0.05 (0.77→0.82, highest since 290.8M at 0.92), Phase_Fail% down 2.1pp (15.2→13.1%), RVar down 5.8 (still 4.1x baseline), Deaths stable at 3, EV up 0.01, Grad up 0.01, VL down 0.0005, Inferno_Complete up 0.29 to 13.04 |
| 306.5M | 21 | 3 | 0.0% | 15.3% | 0.96 | 0.000 | 0.00 | 0.001 | 0.033 | 0.79 | 725.5 | 0.38 | n=148-149; Return down 0.03 (0.82→0.79), Phase_Fail% up 2.2pp (13.1→15.3%), RVar down 5.9 (still 4.1x baseline), EV down 0.01, VL stable, Grad up 0.01, Reward total down 12.24 to 86.08 (drop from 301.6M peak) |
| 307.0M | 21 | 3 | 0.0% | 15.0% | 0.96 | 0.000 | 0.00 | 0.001 | 0.032 | 0.80 | 719.8 | 0.38 | n=15-16; Return up 0.01, Phase_Fail% down 0.3pp, RVar down 5.7, Stall penalty up 0.45 to -0.72, Reward total up 11.13 to 97.21 (strong recovery to near 301.6M peak) |
| 311.9M | 21 | 3 | 0.0% | 15.3% | 0.96 | 0.000 | 0.00 | 0.001 | 0.032 | 0.81 | 714.3 | 0.38 | n=150-151; Return up 0.01, Phase_Fail% up 0.3pp, RVar down 5.5, Stall penalty degraded 0.62 to -1.34, Reward total down 0.36 to 97.84 (stable near 301.6M peak) |
| 312.4M | 21 | 4 | 0.0% | 19.6% | 0.95 | 0.000 | 0.00 | 0.001 | 0.032 | 0.78 | 708.8 | 0.39 | n=15-16; Return down 0.03, Phase_Fail% up 4.3pp (15.3→19.6%, significant), Deaths up 1 to 4, EV down 0.01, Grad up 0.01, RVar down 5.5, Stall penalty down 0.64 to -1.97, Reward total down 5.65 to 92.19 |
| 317.3M | 21 | 3 | 0.0% | 14.4% | 0.96 | 0.000 | 0.00 | 0.001 | 0.031 | 0.82 | 703.4 | 0.39 | n=150-151; **RECOVERY**: Phase_Fail% down 5.2pp (19.6→14.4%, significant), Return up 0.04 (0.78→0.82), Deaths down 1 to 3, RVar down 5.4, Stall penalty up 0.61 to -1.36, EV up 0.01, Reward total up 5.01 to 97.20 (recovery from 312.4M degradation) |
| 317.8M | 21 | 4 | 0.0% | 16.2% | 0.95 | 0.000 | 0.00 | 0.001 | 0.031 | 0.79 | 698.1 | 0.39 | n=15-16; Return down 0.03, Phase_Fail% up 1.8pp (14.4→16.2%), Deaths up 1 to 4, EV down 0.01, VL up 0.0034, Clip up 0.0003, Grad down 0.01, RVar down 5.3, Stall penalty down 0.73 to -2.09 (degraded), Reward total down 1.29 to 95.91 |
| 322.8M | 21 | 3 | 0.0% | 16.1% | 0.96 | 0.000 | 0.01 | 0.001 | 0.030 | 0.79 | 722.4 | 0.39 | **INSTABILITY EVENT**: n=151-152; VL spike to 0.0089 (8.9x from 0.001, matches crisis), Stall penalty crashed to -5.27 (2.5x worse than -2.09), RVar up 24.3 to 722.4 (4.1x baseline), Wave_Timeout penalty reappeared (-0.71), Return stable at 0.79, Phase_Fail% stable at 16.1%, Ent down to 0.030. Pattern matches prior crisis events at 252.8M and 295.7M. |
| 323.3M | 21 | 3 | 0.0% | 14.7% | 0.97 | 0.000 | 0.00 | 0.001 | 0.030 | 0.82 | 745.2 | 0.38 | n=13-14; **RECOVERY (partial)**: Return up 0.03 to 0.82, Phase_Fail% down 1.4pp (16.1→14.7%), VL recovered from 0.01 to 0.003, Stall penalty normalized from -5.27 to -0.98 (major recovery), RVar up 22.8 to 745.2 (still 4.2x baseline), EV up 0.01 to 0.97, Reward total down 4.35 to 86.68, Deaths stable at 3 |
| 327.2M | 21 | 3 | 0.0% | 13.2% | 0.96 | 0.000 | 0.00 | 0.001 | 0.030 | 0.81 | 740.7 | 0.37 | n=118-119; Return down 0.01, Phase_Fail% down 1.5pp (14.7→13.2%), RVar down 4.5, EV down 0.01, Grad down 0.01, Deaths stable, Ent stable, Reward total up 13.11 to 99.79 |
| 329.5M | 18 | 3 | 0.0% | 16.2% | 0.97 | 0.000 | 0.00 | 0.001 | 0.029 | 0.81 | 734.7 | 0.37 | n=71-72; Phase_Fail% up 3.0pp (13.2→16.2%, significant), Eps down 3 (21→18), Return stable at 0.81, RVar down 6.0, EV up 0.01 |
| 335.3M | 19 | 3 | 0.0% | 14.1% | 0.97 | 0.000 | 0.00 | 0.001 | 0.029 | 0.82 | 726.7 | 0.37 | n=176-177; Phase_Fail% down 2.1pp (16.2→14.1%), Return up 0.01 (0.81→0.82), RVar down 8.0, Eps up 1 (18→19), Deaths stable at 3, EV stable at 0.97 |
| 339.1M | 19 | 3 | 0.0% | 14.2% | 0.96 | 0.000 | 0.00 | 0.001 | 0.028 | 0.82 | 717.4 | 0.36 | n=117-118; Phase_Fail% stable at 14.2%, Return stable at 0.82, RVar down 9.3, EV down 0.01, Ent down 0.001, Grad down 0.01 |
| 339.5M | 19 | 3 | 0.0% | 14.5% | 0.96 | 0.000 | 0.00 | 0.001 | 0.028 | 0.81 | 713.5 | 0.37 | n=11-12; Phase_Fail% up 0.3pp (14.2→14.5%), Return down 0.01 (0.82→0.81), RVar down 3.9, EarlyMag down 0.53, DmgDealt down 0.81, Stall penalty down 0.03 |
| 346.6M | 20 | 3 | 0.0% | 14.1% | 0.96 | 0.000 | 0.00 | 0.001 | 0.028 | 0.83 | 714.2 | 0.38 | n=216-217; Phase_Fail% down 0.4pp (14.5→14.1%), Return up 0.02 (0.81→0.83), Eps up 1 (19→20), RVar up 0.7, Grad up 0.01, VL up 0.0044 (elevated from 0.00) |
| 353.9M | 18 | 3 | 0.0% | 14.9% | 0.97 | 0.000 | 0.00 | 0.001 | 0.028 | 0.81 | 704.8 | 0.36 | n=199; Phase_Fail% up 0.8pp (14.1→14.9%), Return down 0.02 (0.83→0.81), Eps down 2 (20→18), RVar down 9.4, EV up 0.01 (0.96→0.97), Grad down 0.02 (0.38→0.36), Deaths stable at 3 |
| 359.9M | 19 | 3 | 0.0% | 13.4% | 0.96 | 0.000 | 0.00 | 0.001 | 0.028 | 0.83 | 694.3 | 0.36 | n=182-183; Phase_Fail% down 1.5pp (14.9→13.4%), Return up 0.02 (0.81→0.83), Eps up 1 (18→19), RVar down 10.5 (704.8→694.3), EV down 0.01 (0.97→0.96) |
| 365.2M | 20 | 2 | 0.0% | 11.9% | 0.97 | 0.000 | 0.00 | 0.001 | 0.027 | 0.84 | 685.5 | 0.36 | n=160-161; Phase_Fail% down 1.5pp (13.4→11.9%), Deaths down 1 to 2, Return up 0.01 (0.83→0.84), Eps up 1 (19→20), RVar down 8.8 (694.3→685.5), EV up 0.01 (0.96→0.97), Ent down 0.001 (0.028→0.027) |
| 370.1M | 20 | 3 | 0.0% | 12.6% | 0.96 | 0.000 | 0.00 | 0.001 | 0.027 | 0.85 | 676.9 | 0.36 | n=147-148; Phase_Fail% down 2.2pp (14.8→12.6%), Return up 0.03 (0.82→0.85), EV down 0.02 (0.98→0.96), RVar down 4.0 (680.9→676.9) |
| 374.1M | 17 | 2 | 0.0% | 11.7% | 0.96 | 0.000 | 0.00 | 0.002 | 0.028 | 0.85 | 667.2 | 0.35 | n=18; new event file (20260401_165645); Phase_Fail% down 6.4pp (18.1→11.7%), Deaths down 1 to 2, Return up 0.04 (0.81→0.85), Clip up 0.001 (0.001→0.002), Ent up 0.001, RVar down 5.8 |
| 379.4M | 18 | 2 | 0.0% | 12.3% | 0.96 | 0.000 | 0.00 | 0.001 | 0.028 | 0.86 | 662.5 | 0.36 | n=155-156; Phase_Fail% down 2.1pp (14.4→12.3%), Return up 0.03 (0.83→0.86), RVar down 4.1 (666.6→662.5), Grad up 0.01 (0.35→0.36) |
| 379.6M | 19 | 2 | 0.0% | 10.7% | 0.97 | 0.000 | 0.00 | 0.001 | 0.028 | 0.87 | 658.3 | 0.34 | n=6-7 (low confidence); Phase_Fail% down 1.6pp (12.3→10.7%), Return up 0.01 (0.86→0.87), EV up 0.01 (0.96→0.97), RVar down 4.2, Grad down 0.02 (0.36→0.34) |
| 384.6M | 19 | 2 | 0.0% | 12.9% | 0.96 | 0.000 | 0.00 | 0.001 | 0.028 | 0.87 | 664.0 | 0.36 | n=152-154; Phase_Fail% up 2.2pp (10.7→12.9%), VL up 0.0049 (1.9x vs prior 0.0026), Stall penalty down to -2.77 (from -0.98), RVar up 5.7 (658.3→664.0), EV down 0.01 (0.97→0.96), Grad up 0.02 (0.34→0.36), Wave_Timeout reappeared (n=1, -0.68) |
| 390.1M | 20 | 3 | 0.0% | 13.0% | 0.96 | 0.000 | 0.00 | 0.001 | 0.027 | 0.86 | 656.7 | 0.36 | n=158-159; Phase_Fail% up 2.3pp (10.7→13.0%), Deaths up 1 to 3, Return down 0.01 (0.87→0.86), RVar down 4.1 (660.8→656.7), EV down 0.01 (0.97→0.96) |
| 395.4M | 20 | 2 | 0.0% | 10.0% | 0.97 | 0.000 | 0.00 | 0.001 | 0.027 | 0.89 | 648.9 | 0.36 | n=159-160; Phase_Fail% down 4.3pp (14.3→10.0%, new low), Deaths down 1 to 2, Eps down 4 (24→20, normalized), Return up 0.01 (0.88→0.89), EV up 0.01 (0.96→0.97), RVar down 3.8, Grad up 0.02 (0.34→0.36) |
| 401.2M | 20 | 2 | 0.0% | 9.8% | 0.96 | 0.000 | 0.00 | 0.001 | 0.027 | 0.90 | 641.2 | 0.36 | n=177-178; Return up 0.01 (0.89→0.90), Phase_Fail% down 0.2pp (10.0→9.8%, new low), EV down 0.01 (0.97→0.96), RVar down 7.7 (648.9→641.2) |
| 406.3M | 20 | 2 | 0.0% | 10.4% | 0.97 | 0.000 | 0.00 | 0.001 | 0.026 | 0.90 | 633.7 | 0.37 | n=156-157; Phase_Fail% up 0.6pp (9.8→10.4%), EV up 0.01 (0.96→0.97), Ent down 0.001 (0.027→0.026), RVar down 7.5 (641.2→633.7), Grad up 0.01 (0.36→0.37) |
| 411.7M | 20 | 2 | 0.0% | 10.6% | 0.97 | 0.000 | 0.00 | 0.001 | 0.026 | 0.91 | 626.6 | 0.37 | n=165-166; Return up 0.01 (0.90→0.91), Phase_Fail% up 0.2pp (10.4→10.6%), RVar down 7.1 (633.7→626.6) |
| 415.4M | 19 | 2 | 0.0% | 11.2% | 0.97 | 0.000 | 0.00 | 0.001 | 0.026 | 0.91 | 620.5 | 0.37 | n=113-114; Phase_Fail% up 0.6pp (10.6→11.2%), Eps down 1 (20→19), RVar down 6.1 (626.6→620.5) |
| 416.2M | 20 | 2 | 0.0% | 8.0% | 0.96 | 0.000 | 0.00 | 0.001 | 0.026 | 0.93 | 617.7 | 0.35 | n=24-25 (low confidence); Phase_Fail% down 3.2pp (11.2→8.0%), Return up 0.02 (0.91→0.93), EV down 0.01 (0.97→0.96), Grad down 0.02 (0.37→0.35), VL up to 0.0040 (from 0.0029) |
| 417.0M | 21 | 2 | 0.0% | 11.0% | 0.96 | 0.000 | 0.00 | 0.001 | 0.026 | 0.93 | 616.7 | 0.36 | n=24-25 (low confidence); Phase_Fail% up 3.0pp (8.0→11.0%), Eps up 1 (20→21), Grad up 0.01 (0.35→0.36), RVar down 1.0 |
| 421.7M | 19 | 2 | 0.0% | 10.2% | 0.97 | 0.000 | 0.00 | 0.001 | 0.026 | 0.91 | 613.2 | 0.36 | n=142-143; Phase_Fail% down 0.8pp (11.0→10.2%), Return down 0.02 (0.93→0.91), EV up 0.01 (0.96→0.97), Eps down 2 (21→19), RVar down 3.5 (616.7→613.2) |
| 425.1M | 7 | 2 | 0.0% | 27.8% | 0.98 | 0.000 | 0.00 | 0.001 | 0.026 | 1.05 | 607.5 | 0.31 | n=69-72; **new event file: w1-66 (was w49-66)** — Eps 7 (was 19), Phase_Fail% 27.8% (was 10.2%), Return 1.05 (was 0.91), EV up 0.01 (0.97→0.98), Grad down 0.05 (0.36→0.31), RVar down 5.7; cross-scope deltas not comparable |
| 430.3M | 7 | 2 | 0.0% | 23.8% | 0.98 | 0.000 | 0.00 | 0.001 | 0.025 | 1.09 | 604.3 | 0.30 | n=159-160; Phase_Fail% down 4.0pp (27.8→23.8%), Return up 0.04 (1.05→1.09), VL up to 0.0049 (from 0.0029), Ent down 0.001 (0.026→0.025), Grad down 0.01 (0.31→0.30), RVar down 3.2 |
| 435.6M | 7 | 1 | 0.0% | 20.9% | 0.99 | 0.000 | 0.00 | 0.001 | 0.025 | 1.15 | 598.4 | 0.31 | n=161-162; Phase_Fail% down 2.9pp (23.8→20.9%), Deaths down 1 (2→1), Return up 0.06 (1.09→1.15), EV up 0.01 (0.98→0.99), VL down to 0.0025 (from 0.0049), RVar down 5.9 (604.3→598.4) |
| 440.8M | 7 | 1 | 0.0% | 17.0% | 0.99 | 0.000 | 0.00 | 0.001 | 0.024 | 1.15 | 592.3 | 0.31 | n=158-159; Phase_Fail% down 3.9pp (20.9→17.0%), Ent down 0.001 (0.025→0.024), RVar down 6.1 (598.4→592.3) |
| 446.0M | 7 | 1 | 0.0% | 16.6% | 0.99 | 0.000 | 0.00 | 0.001 | 0.024 | 1.18 | 586.4 | 0.32 | n=158-159; Return up 0.03 (1.15→1.18), Phase_Fail% down 0.4pp (17.0→16.6%), RVar down 5.9 (592.3→586.4), Grad up 0.01 (0.31→0.32) |
| 451.2M | 7 | 1 | 0.0% | 17.0% | 0.99 | 0.000 | 0.00 | 0.001 | 0.023 | 1.18 | 580.7 | 0.32 | n=159-160; Phase_Fail% up 0.4pp (16.6→17.0%), Ent down 0.001 (0.024→0.023), RVar down 5.7 (586.4→580.7) |
| 456.4M | 7 | 1 | 0.0% | 20.0% | 0.99 | 0.000 | 0.00 | 0.001 | 0.022 | 1.19 | 575.1 | 0.32 | n=159-160; Phase_Fail% up 3.0pp (17.0→20.0%), Return up 0.01 (1.18→1.19), Ent down 0.001 (0.023→0.022), RVar down 5.6 (580.7→575.1) |
| 461.6M | 7 | 1 | 0.0% | 17.0% | 0.99 | 0.000 | 0.00 | 0.001 | 0.022 | 1.19 | 569.7 | 0.33 | n=159-160; Phase_Fail% down 3.0pp (20.0→17.0%), RVar down 5.4 (575.1→569.7), Grad up 0.01 (0.32→0.33) |
| 466.8M | 7 | 1 | 0.0% | 15.9% | 0.99 | 0.000 | 0.00 | 0.001 | 0.021 | 1.21 | 564.4 | 0.32 | n=159-160; Phase_Fail% down 1.1pp (17.0→15.9%), Return up 0.02 (1.19→1.21), Ent down 0.001 (0.022→0.021), Grad down 0.01 (0.33→0.32), RVar down 5.3 (569.7→564.4) |
| 472.0M | 7 | 1 | 0.0% | 15.4% | 0.99 | 0.000 | 0.00 | 0.001 | 0.020 | 1.21 | 559.3 | 0.32 | n=158-159; Phase_Fail% down 0.5pp (15.9→15.4%), Ent down 0.001 (0.021→0.020), RVar down 5.1 (564.4→559.3) |
| 477.2M | 7 | 1 | 0.0% | 19.0% | 0.99 | 0.000 | 0.00 | 0.001 | 0.020 | 1.23 | 554.2 | 0.33 | n=158-159; Phase_Fail% up 3.6pp (15.4→19.0%), Return up 0.02 (1.21→1.23), Grad up 0.01 (0.32→0.33), RVar down 5.1 (559.3→554.2) |
| 482.4M | 7 | 1 | 0.0% | 15.8% | 0.99 | 0.000 | 0.00 | 0.001 | 0.019 | 1.25 | 549.3 | 0.33 | n=158-159; Phase_Fail% down 3.2pp (19.0→15.8%), Return up 0.02 (1.23→1.25), Ent down 0.001 (0.020→0.019), RVar down 4.9 (554.2→549.3) |
| 487.6M | 7 | 1 | 0.0% | 16.3% | 0.99 | 0.000 | 0.00 | 0.002 | 0.019 | 1.24 | 544.5 | 0.33 | n=157-158; Phase_Fail% up 0.5pp (15.8→16.3%), Return down 0.01 (1.25→1.24), Clip up to 0.002 (raw 0.0015, was 0.0013), RVar down 4.8 (549.3→544.5) |
| 492.7M | 7 | 1 | 0.0% | 17.8% | 0.99 | 0.000 | 0.00 | 0.001 | 0.018 | 1.26 | 539.7 | 0.34 | n=156-157; Phase_Fail% up 1.5pp (16.3→17.8%), Return up 0.02 (1.24→1.26), Ent down 0.001 (0.019→0.018), Clip down to 0.001 (raw 0.0013, was 0.0015), Grad up 0.01 (0.33→0.34), RVar down 4.8 (544.5→539.7) |
| 497.9M | 8 | 1 | 0.0% | 15.6% | 0.99 | 0.000 | 0.00 | 0.002 | 0.017 | 1.27 | 535.1 | 0.34 | n=158-159; Phase_Fail% down 2.2pp (17.8→15.6%), Eps up 1 (7→8), Return up 0.01 (1.26→1.27), Ent down 0.001 (0.018→0.017), Clip up to 0.002 (raw 0.0015, was 0.0013), RVar down 4.6 (539.7→535.1) |
| 503.1M | 7 | 1 | 0.0% | 15.4% | 0.99 | 0.000 | 0.00 | 0.001 | 0.017 | 1.26 | 530.6 | 0.34 | n=157-159; Eps down 1 (8→7), Phase_Fail% down 0.2pp (15.6→15.4%), Return down 0.01 (1.27→1.26), Clip down to 0.001 (raw 0.0014 vs 0.0015), RVar down 4.5 (535.1→530.6) |
| 508.3M | 7 | 1 | 0.0% | 14.5% | 0.99 | 0.000 | 0.00 | 0.001 | 0.016 | 1.28 | 526.2 | 0.34 | n=157-158; Phase_Fail% down 0.9pp (15.4→14.5%), Return up 0.02 (1.26→1.28), Ent down 0.001 (0.017→0.016), RVar down 4.4 (530.6→526.2) |
| 513.6M | 7 | 1 | 0.0% | 13.5% | 0.99 | 0.000 | 0.00 | 0.002 | 0.015 | 1.31 | 521.8 | 0.33 | n=159-161; Phase_Fail% down 1.0pp (14.5→13.5%), Return up 0.03 (1.28→1.31), Ent down 0.001 (0.016→0.015), Clip up to 0.002 (raw 0.0016, was 0.0014), RVar down 4.4 (526.2→521.8), Grad down 0.01 (0.34→0.33) |
| 518.6M | 7 | 1 | 0.0% | 16.0% | 0.99 | 0.000 | 0.00 | 0.002 | 0.015 | 1.31 | 517.6 | 0.33 | n=153-154; Phase_Fail% up 2.5pp (13.5→16.0%), RVar down 4.2 (521.8→517.6) |
| 523.8M | 7 | 1 | 0.0% | 16.7% | 0.99 | 0.000 | 0.00 | 0.002 | 0.014 | 1.31 | 513.6 | 0.34 | n=156-158; Ent down 0.001 (0.015→0.014), RVar down 4.0 (517.6→513.6), Grad up 0.01 (0.33→0.34) |
| 528.9M | 8 | 1 | 0.0% | 15.7% | 0.99 | 0.000 | 0.00 | 0.002 | 0.014 | 1.33 | 509.6 | 0.35 | n=155-156; Eps up 1 (7→8), Phase_Fail% down 1.0pp (16.7→15.7%), Return up 0.02 (1.31→1.33), RVar down 4.0 (513.6→509.6), Grad up 0.01 (0.34→0.35) |


## Reward Terms Log (ep_sum_mean)

Averaged per-episode sum of each raw reward term. Total = sum of all terms.

| Steps | DmgDealt | LOSSep | WavComp | WaveHP | MagPri | EarlyMag | DmgTkn | Stall | MagDel | BBHighHP | WpnSwitch | Total |
|-------|----------|--------|---------|--------|--------|----------|--------|-------|--------|----------|-----------|-------|
| 133.1M | 11.03 | 7.00 | 30.64 | 12.79 | 1.71 | 4.15 | -11.10 | -4.45 | -5.13 | -0.69 | - | 42.43 |
| 136.3M | 12.20 | 7.77 | 34.10 | 15.03 | 1.96 | 4.74 | -12.01 | -4.39 | -5.29 | -1.30 | - | 52.81 |
| 138.8M | 12.40 | 7.68 | 34.76 | 15.14 | 1.97 | 4.79 | -12.43 | -5.40 | -5.19 | -0.98 | - | 49.04 |
| 141.6M | 11.50 | 6.96 | 32.84 | 14.05 | 1.80 | 4.29 | -11.06 | -4.27 | -4.74 | -0.93 | - | 50.44 |
| 146.6M | 10.64 | 6.43 | 31.55 | 12.80 | 1.70 | 3.98 | -9.83 | -3.63 | -4.38 | -0.87 | - | 48.39 |
| 151.6M | 11.11 | 6.63 | 33.34 | 13.59 | 1.78 | 4.21 | -9.70 | -4.32 | -4.58 | -1.01 | - | 51.05 |
| 156.8M | 10.98 | 6.54 | 33.04 | 13.57 | 1.80 | 4.20 | -9.18 | -3.86 | -4.50 | -0.93 | - | 51.66 |
| 162.0M | 10.54 | 6.23 | 31.69 | 12.91 | 1.71 | 4.00 | -9.36 | -4.26 | -4.27 | -0.97 | - | 48.22 |
| 165.3M | 11.68 | 6.77 | 35.68 | 14.83 | 1.94 | 4.55 | -9.23 | -3.08 | -4.56 | -1.11 | - | 59.33 |
| 167.3M | 11.82 | 6.85 | 36.06 | 15.14 | 1.95 | 4.57 | -8.93 | -4.00 | -4.64 | -1.14 | - | 57.68 |
| 172.2M | 11.37 | 6.50 | 35.03 | 14.51 | 1.87 | 4.41 | -8.84 | -3.40 | -4.42 | -1.00 | - | 56.13 |
| 177.0M | 11.58 | 6.70 | 35.64 | 14.71 | 1.91 | 4.51 | -9.11 | -3.11 | -4.56 | -0.99 | - | 57.28 |
| 182.3M | 11.56 | 6.51 | 35.74 | 15.02 | 1.97 | 4.59 | -8.72 | -3.27 | -4.32 | -1.10 | - | 56.98 |
| 187.3M | 11.99 | 6.91 | 37.26 | 16.01 | 2.06 | 4.81 | -8.52 | -3.53 | -4.64 | -1.08 | - | 61.27 |
| 192.5M | 11.38 | 6.29 | 35.01 | 14.98 | 1.92 | 4.49 | -8.85 | -3.75 | -4.11 | -0.83 | - | 56.53 |
| 194.6M | 11.45 | 6.38 | 35.71 | 15.16 | 1.95 | 4.59 | -8.73 | -3.14 | -4.16 | -0.69 | - | 57.62 |
| 210.6M | 14.01 | 7.96 | 42.62 | 18.80 | 2.43 | 5.75 | -9.97 | -3.70 | -5.47 | -0.89 | - | 71.54 |
| 211.6M | 12.70 | 7.14 | 39.61 | 17.01 | 2.21 | 5.14 | -8.93 | -2.92 | -4.93 | -0.93 | - | 66.10 |
| 215.6M | 13.06 | 7.32 | 40.76 | 17.37 | 2.27 | 5.28 | -8.88 | -2.86 | -4.90 | -0.76 | - | 68.66 |
| 220.9M | 15.99 | 9.08 | 48.53 | 22.14 | 2.84 | 6.76 | -10.52 | -3.34 | -5.70 | -1.01 | - | 89.57 |
| 225.2M | 16.26 | 9.22 | 49.86 | 22.61 | 2.91 | 6.90 | -10.04 | -3.00 | -5.82 | -0.92 | - | 87.97 |
| 225.7M | 14.74 | 8.44 | 46.26 | 20.29 | 2.63 | 6.15 | -9.62 | -3.01 | -5.50 | -0.96 | - | 78.42 |
| 230.7M | 13.77 | 7.50 | 43.38 | 18.86 | 2.43 | 5.68 | -8.59 | -2.70 | -4.85 | -0.82 | - | 74.56 |
| 231.1M | 14.06 | 7.71 | 44.54 | 19.41 | 2.46 | 5.79 | -8.64 | -2.72 | -4.91 | -0.83 | - | 77.07 |
| 236.1M | 13.75 | 7.34 | 43.41 | 18.78 | 2.44 | 5.68 | -8.59 | -3.56 | -4.82 | -0.77 | - | 80.40 |
| 236.6M | 13.70 | 7.45 | 43.67 | 18.52 | 2.46 | 5.72 | -8.50 | -2.76 | -4.75 | -0.53 | - | 74.98 |
| 241.5M | 14.20 | 7.64 | 44.72 | 19.66 | 2.51 | 5.86 | -8.42 | -2.53 | -4.91 | -0.86 | - | 85.17 |
| 242.0M | 14.01 | 7.48 | 44.74 | 19.52 | 2.54 | 5.84 | -7.77 | -2.50 | -4.90 | -0.76 | - | 87.24 |
| 246.9M | 14.03 | 7.58 | 44.51 | 19.37 | 2.56 | 5.91 | -8.14 | -2.15 | -4.85 | -0.70 | - | 85.50 |
| 252.8M | 13.69 | 7.39 | 43.39 | 18.70 | 2.44 | 5.66 | -8.36 | -16.12 | -4.85 | -0.59 | - | 67.66 |
| 257.8M | 14.29 | 7.65 | 45.50 | 20.07 | 2.60 | 6.05 | -7.60 | -2.03 | -4.72 | -0.71 | - | 89.54 |
| 263.2M | 14.21 | 7.44 | 45.34 | 19.83 | 2.59 | 5.99 | -7.62 | -8.77 | -4.65 | -0.68 | - | 80.91 |
| 268.9M | 14.47 | 7.73 | 46.08 | 20.37 | 2.64 | 6.14 | -7.91 | -1.93 | -4.71 | -0.77 | - | 82.11 |
| 269.1M | 15.44 | 8.13 | 48.42 | 22.17 | 2.95 | 6.84 | -8.38 | -1.40 | -4.80 | -0.60 | - | 90.76 |
| 274.0M | 14.67 | 7.67 | 46.88 | 20.78 | 2.71 | 6.24 | -7.32 | -2.04 | -4.45 | -0.72 | - | 84.42 |
| 274.5M | 14.03 | 7.30 | 45.08 | 19.88 | 2.59 | 5.97 | -7.33 | -4.57 | -4.30 | -0.54 | - | 77.11 |
| 279.4M | 14.24 | 7.42 | 45.61 | 20.06 | 2.66 | 6.09 | -7.62 | -1.78 | -4.37 | -0.65 | - | 81.66 |
| 279.9M | 14.48 | 7.87 | 46.44 | 20.50 | 2.72 | 6.23 | -8.01 | -2.22 | -4.42 | -0.59 | - | 83.00 |
| 284.9M | 14.10 | 7.38 | 45.52 | 19.95 | 2.61 | 6.01 | -6.90 | -1.54 | -4.16 | -0.63 | - | 82.34 |
| 285.3M | 15.11 | 7.92 | 48.41 | 21.51 | 2.84 | 6.54 | -7.33 | -1.85 | -4.45 | -0.68 | - | 88.02 |
| 290.3M | 14.64 | 7.54 | 46.96 | 20.64 | 2.72 | 6.26 | -7.25 | -1.42 | -4.29 | -0.58 | - | 85.22 |
| 290.8M | 14.34 | 7.15 | 46.59 | 20.35 | 2.64 | 6.10 | -7.08 | -0.89 | -4.03 | -0.48 | - | 94.07 |
| 295.7M | 14.61 | 7.49 | 47.01 | 20.80 | 2.72 | 6.28 | -7.27 | -9.27 | -4.12 | -0.57 | - | 73.84 |
| 296.2M | 14.47 | 7.39 | 46.37 | 20.46 | 2.64 | 6.16 | -7.81 | -1.38 | -4.13 | -0.68 | - | 83.49 |
| 301.1M | 14.56 | 7.55 | 47.01 | 20.70 | 2.75 | 6.32 | -7.28 | -1.32 | -4.17 | -0.63 | - | 85.49 |
| 301.6M | 14.85 | 7.37 | 47.91 | 21.11 | 2.78 | 6.38 | -6.67 | -1.11 | -3.86 | -0.46 | - | 98.32 |
| 306.5M | 14.57 | 7.37 | 47.05 | 20.87 | 2.74 | 6.30 | -7.08 | -1.17 | -4.02 | -0.55 | - | 86.08 |
| 307.0M | 14.58 | 7.67 | 47.30 | 20.80 | 2.81 | 6.35 | -6.62 | -0.72 | -4.05 | -0.47 | - | 97.21 |
| 311.9M | 14.90 | 7.58 | 47.84 | 21.40 | 2.79 | 6.44 | -6.80 | -1.34 | -3.98 | -0.53 | - | 97.84 |
| 312.4M | 14.47 | 7.58 | 46.32 | 20.48 | 2.69 | 6.24 | -7.36 | -1.97 | -4.01 | -0.39 | - | 92.19 |
| 317.3M | 14.72 | 7.38 | 47.49 | 21.03 | 2.74 | 6.34 | -6.67 | -1.36 | -3.87 | -0.48 | - | 97.20 |
| 317.8M | 14.56 | 7.63 | 47.10 | 20.65 | 2.69 | 6.28 | -7.04 | -2.09 | -4.05 | -0.52 | - | 95.91 |
| 322.8M | 14.59 | 7.55 | 46.87 | 20.89 | 2.72 | 6.29 | -6.63 | -5.27 | -4.03 | -0.49 | - | 91.03 |
| 323.3M | 14.97 | 7.34 | 48.28 | 21.57 | 2.79 | 6.51 | -6.68 | -0.98 | -3.76 | -0.42 | - | 86.68 |
| 327.2M | 14.90 | 7.48 | 48.23 | 21.45 | 2.80 | 6.47 | -6.41 | -1.17 | -3.83 | -0.42 | - | 99.79 |
| 329.5M | 16.99 | 8.59 | 54.04 | 24.92 | 3.33 | 7.68 | -7.18 | -1.06 | -4.30 | -0.61 | -1.79 | 109.87 |
| 335.3M | 17.05 | 8.61 | 54.29 | 24.88 | 3.32 | 7.67 | -7.27 | -0.94 | -4.24 | -0.63 | -1.68 | 101.06 |
| 339.1M | 16.36 | 8.09 | 52.32 | 23.83 | 3.13 | 7.25 | -6.95 | -0.90 | -4.10 | -0.55 | -1.51 | 96.96 |
| 346.6M | 15.62 | 7.78 | 50.28 | 22.74 | 2.98 | 6.88 | -6.36 | -2.88 | -3.69 | -0.56 | -1.50 | 91.29 |
| 353.9M | 17.28 | 6.11 | 54.73 | 25.26 | 3.32 | 7.74 | -7.01 | -1.28 | -3.94 | -0.53 | -1.64 | 100.04 |
| 359.9M | 17.10 | 5.84 | 54.43 | 25.12 | 3.31 | 7.67 | -6.72 | -1.76 | -3.85 | -0.63 | -1.52 | 108.33 |
| 365.2M | 16.01 | 5.49 | 51.63 | 23.44 | 3.07 | 7.12 | -6.35 | -0.92 | -3.67 | -0.43 | -1.47 | 104.60 |
| 370.1M | 15.72 | 5.42 | 50.93 | 22.86 | 3.07 | 7.02 | -6.22 | -0.91 | -3.51 | -0.43 | -1.42 | 102.96 |
| 374.1M | 15.74 | 5.23 | 50.67 | 23.16 | 3.06 | 6.96 | -6.15 | -0.96 | -3.52 | -0.74 | -1.66 | 102.04 |
| 379.4M | 17.51 | 6.09 | 55.66 | 25.68 | 3.43 | 7.91 | -6.85 | -0.95 | -3.69 | -0.55 | -1.70 | 112.97 |
| 384.6M | 17.51 | 5.94 | 55.78 | 25.51 | 3.43 | 7.91 | -6.80 | -2.77 | -3.71 | -0.40 | -1.68 | 110.27 |
| 390.2M | 15.68 | 4.96 | 50.03 | 22.53 | 3.00 | 6.89 | -5.51 | -0.85 | -3.24 | -0.35 | -1.33 | 101.78 |
| 395.4M | 16.61 | 5.70 | 53.52 | 24.39 | 3.21 | 7.42 | -6.11 | -0.95 | -3.30 | -0.42 | -1.51 | 109.83 |
| 401.2M | 16.61 | 5.76 | 53.54 | 24.48 | 3.21 | 7.40 | -5.94 | -1.06 | -3.42 | -0.51 | -1.52 | 109.79 |
| 406.3M | 16.30 | 5.69 | 52.87 | 24.11 | 3.24 | 7.37 | -5.76 | -0.87 | -3.24 | -0.51 | -1.58 | 108.65 |
| 411.7M | 16.17 | 5.50 | 52.50 | 23.80 | 3.21 | 7.28 | -5.74 | -0.75 | -3.27 | -0.44 | -1.48 | 107.76 |
| 415.4M | 16.76 | 5.84 | 54.32 | 24.90 | 3.37 | 7.64 | -6.11 | -1.00 | -3.43 | -0.32 | -1.60 | 111.26 |
| 416.2M | 16.38 | 5.64 | 53.43 | 24.48 | 3.24 | 7.41 | -5.59 | -1.78 | -3.31 | -0.25 | -1.50 | 109.96 |
| 417.0M | 15.93 | 5.44 | 51.82 | 23.78 | 3.15 | 7.17 | -5.71 | -0.79 | -3.26 | -0.43 | -1.50 | 106.32 |
| 421.7M | 16.47 | 5.71 | 53.54 | 24.62 | 3.27 | 7.46 | -5.94 | -0.79 | -3.46 | -0.44 | -1.63 | 109.94 |
| 425.1M | 43.96 | 11.74 | 182.31 | 83.20 | - | - | -14.66 | -2.70 | -10.00 | -0.88 | -3.88 | 293.14 |
| 430.3M | 47.22 | 12.72 | 195.42 | 90.09 | - | - | -14.68 | -6.28 | -10.92 | -1.30 | -4.37 | 313.64 |
| 435.6M | 49.32 | 13.14 | 204.84 | 95.08 | - | - | -14.41 | -2.22 | -10.89 | -1.32 | -4.54 | 335.68 |
| 440.8M | 48.03 | 13.28 | 198.87 | 91.64 | - | - | -14.32 | -1.57 | -10.53 | -1.11 | -4.49 | 327.50 |
| 446.0M | 48.73 | 13.48 | 201.98 | 93.75 | - | - | -13.99 | -1.92 | -10.38 | -1.44 | -4.41 | 333.54 |
| 451.2M | 48.29 | 13.30 | 200.62 | 93.04 | - | - | -12.84 | -1.37 | -10.10 | -1.43 | -4.25 | 333.07 |
| 456.4M | 47.70 | 13.32 | 198.92 | 93.03 | - | - | -12.60 | -1.89 | -10.33 | -1.34 | -4.35 | 329.16 |
| 461.6M | 45.88 | 12.47 | 190.63 | 88.35 | - | - | -12.27 | -1.67 | -9.59 | -1.26 | -3.98 | 316.51 |
| 466.8M | 46.39 | 12.16 | 192.82 | 89.67 | - | - | -12.06 | -2.00 | -9.28 | -1.13 | -3.87 | 320.81 |
| 472.0M | 46.79 | 12.42 | 194.60 | 90.21 | - | - | -12.62 | -1.37 | -9.44 | -1.13 | -3.89 | 323.94 |
| 477.2M | 47.14 | 12.49 | 196.78 | 91.51 | - | - | -12.28 | -1.31 | -9.66 | -1.18 | -3.85 | 326.81 |
| 482.4M | 47.72 | 12.80 | 200.30 | 93.60 | - | - | -12.48 | -1.39 | -10.14 | -0.89 | -3.94 | 333.55 |
| 487.6M | 46.76 | 12.29 | 195.71 | 90.89 | - | - | -12.29 | -1.03 | -9.35 | -0.69 | -3.73 | 326.34 |
| 492.7M | 47.08 | 12.09 | 196.71 | 90.89 | - | - | -12.09 | -1.31 | -9.22 | -0.69 | -3.97 | 326.88 |
| 497.9M | 45.84 | 11.95 | 191.50 | 87.99 | - | - | -11.25 | -1.31 | -8.87 | -0.81 | -3.67 | 319.62 |
| 503.1M | 46.61 | 11.97 | 195.01 | 90.23 | - | - | -11.04 | -1.20 | -8.82 | -0.67 | -3.85 | 326.44 |
| 508.3M | 46.42 | 12.09 | 194.23 | 90.38 | - | - | -11.10 | -1.14 | -8.91 | -0.77 | -3.67 | 325.84 |
| 513.6M | 47.50 | 12.13 | 198.93 | 92.64 | - | - | -10.83 | -1.13 | -8.84 | -1.26 | -3.69 | 334.39 |
| 518.6M | 47.81 | 12.15 | 199.88 | 92.89 | - | - | -11.07 | -1.39 | -8.84 | -1.26 | -3.57 | 334.68 |
| 523.8M | 47.06 | 12.16 | 197.33 | 91.84 | - | - | -11.29 | -0.85 | -8.53 | -0.95 | -3.76 | 330.87 |
| 528.9M | 46.28 | 11.98 | 193.62 | 90.42 | - | - | -10.57 | -0.82 | -8.47 | -1.16 | -3.82 | 325.72 |

## Eval Results

### 167.1M checkpoint — W49-66, 50 seeds/loadout

Model: `models/V51/inferno_gpu_w49-66_20260331_120107_1100.pt` (167,116,800 trained steps)

| Loadout | Clear% | Death% | Mean Wave | Median |
|---------|--------|--------|-----------|--------|
| BUDGET_RCB | 46.0% | 44.0% | 62.5 | 64 |
| MID_ACB | 32.0% | 58.0% | 62.1 | 62 |
| CRYSTAL_BP | 66.0% | 22.0% | 63.5 | 66 |
| CRYSTAL_NO_BP | 72.0% | 14.0% | 64.2 | 66 |
| MAX_TBOW | 68.0% | 22.0% | 64.1 | 66 |
| **Overall (250 eps)** | **56.8%** | **32.0%** | **63.3** | — |

Notes:
- CRYSTAL_NO_BP best performer — removing BP option may reduce suboptimal weapon swap decisions.
- MID_ACB weakest — lower combat levels (85 ranged, Eagle Eye) cause most deaths in W54-63 range.
- Top 3 loadouts (CRYSTAL_BP/NO_BP/MAX_TBOW) cluster at 66-72% clear, 14-22% death.
- Phase_Fail% in TB (38.6%) tracks eval death% across loadouts as expected — TB uses uniform loadout sampling.

### 180.2M checkpoint — W49-66, 50 seeds/loadout, real stats

Model: `models/V51/inferno_gpu_w49-66_20260331_120107_1500.pt` (180,224,000 trained steps)

Uses `--real-stats` flag: real defence levels (= ranged level) and actual equipment defence bonuses
instead of training stats (1 def, uniform 30 equipment defence).

| Loadout | Clear% | Death% | Mean Wave | Notes |
|---------|--------|--------|-----------|-------|
| BUDGET_RCB | 72.0% | 18.0% | 64.2 | God d'hide, def=80 |
| MID_ACB | 78.0% | 14.0% | 64.8 | God d'hide, def=85 |
| CRYSTAL_BP | 92.0% | 6.0% | 65.6 | Crystal armour, def=90 |
| CRYSTAL_NO_BP | 84.0% | 14.0% | 65.1 | Crystal armour, def=90 |
| MAX_TBOW | 98.0% | 0.0% | 65.7 | Masori (f), def=99 |
| **Overall (250 eps)** | **84.8%** | **10.4%** | **65.1** | — |

Notes:
- Massive uplift from real defence: overall 56.8% → 84.8% clear, 32.0% → 10.4% death.
- MAX_TBOW: 98% clear, 0 deaths (1 timeout). 99 def + masori makes the model nearly unkillable.
- CRYSTAL_BP jumps from 66% → 92% clear — 90 def + crystal armour ranged_defence=152 vs training's 30.
- Budget/Mid benefit heavily from god d'hide defence (120 ranged def vs 30) and real def levels (80/85 vs 1).
- Real-stats eval defined in `simulator/eval_loadouts.py`, activated via `--real-stats` on eval.py/death_analysis.py.

### 324.4M checkpoint — W49-66, 50 seeds/loadout, real stats

Model: `models/V51/inferno_gpu_w49-66_20260331_215430_3400.pt` (324,403,200 trained steps)

| Loadout | Clear% | Death% | Mean Wave |
|---------|--------|--------|-----------|
| BUDGET_RCB | 86.0% | 12.0% | 65.1 |
| MID_ACB | 92.0% | 8.0% | 65.4 |
| CRYSTAL_BP | 98.0% | 2.0% | 65.9 |
| CRYSTAL_NO_BP | 100.0% | 0.0% | 66.0 |
| MAX_TBOW | 100.0% | 0.0% | 66.0 |
| **Overall (250 eps)** | **95.2%** | **4.4%** | **65.7** | 

Notes:
- Overall 84.8% → 95.2% clear, 10.4% → 4.4% death vs 180.2M checkpoint.
- CRYSTAL_NO_BP and MAX_TBOW both flawless at 100% clear, 0 deaths.
- CRYSTAL_BP near-perfect at 98% (1 death at W62, seed 25).
- BUDGET_RCB deaths scattered across W52-65 (6 deaths + 1 timeout at W56).
- MID_ACB deaths at W51, W60, W61, W62 — lower combat levels still weakest link.

### 528.9M checkpoint — W49-66, 50 seeds/loadout, real stats

Model: `models/V51/inferno_gpu_w1-66_20260401_224846_3300.pt` (528,875,520 trained steps)

| Loadout | Clear% | Death% | Mean Wave |
|---------|--------|--------|-----------|
| BUDGET_RCB | 100.0% | 0.0% | 66.0 |
| MID_ACB | 96.0% | 4.0% | 65.8 |
| CRYSTAL_BP | 96.0% | 4.0% | 65.8 |
| CRYSTAL_NO_BP | 100.0% | 0.0% | 66.0 |
| MAX_TBOW | 100.0% | 0.0% | 66.0 |
| **Overall (250 eps)** | **98.4%** | **1.6%** | **65.9** |

Notes:
- Overall 95.2% -> 98.4% clear, 4.4% -> 1.6% death vs 324.4M checkpoint.
- BUDGET_RCB flawless (was 86.0%) — biggest single-loadout improvement (+14pp).
- CRYSTAL_NO_BP and MAX_TBOW remain flawless at 100%.
- MID_ACB and CRYSTAL_BP each had 2 deaths (4.0%) — within noise at n=50.
- Trained on W1-66 since 421.7M but hard-wave performance still improved.

### 528.9M checkpoint — W1-66, 50 seeds/loadout, real stats

Model: `models/V51/inferno_gpu_w1-66_20260401_224846_3300.pt` (528,875,520 trained steps)

First full-Inferno eval (starting from wave 1).

| Loadout | Clear% | Death% | Mean Wave |
|---------|--------|--------|-----------|
| BUDGET_RCB | 86.0% | 14.0% | 65.1 |
| MID_ACB | 72.0% | 28.0% | 63.5 |
| CRYSTAL_BP | 88.0% | 12.0% | 64.6 |
| CRYSTAL_NO_BP | 82.0% | 18.0% | 64.3 |
| MAX_TBOW | 98.0% | 2.0% | 65.7 |
| **Overall (250 eps)** | **85.2%** | **14.8%** | **64.6** |

Notes:
- 13.2pp drop from W49-66 (98.4%) to W1-66 (85.2%) — risk compounds over 66 waves.
- MID_ACB weakest at 72% clear — 85 ranged + Eagle Eye over 66 waves compounds badly.
- CRYSTAL_NO_BP drops more than CRYSTAL_BP (82% vs 88%) — no BP means slower melee/nibbler kills, more exposure over 48 extra waves.
- MAX_TBOW barely affected (100% -> 98%) — 99 def + masori + tbow DPS absorbs the extra waves.
- Per-wave death rate ~0.24% (W1-66) vs ~0.09% (W49-66) — model is dying on W1-48 waves too.
- Phase_Fail% plateau at ~15% in TB matches the 14.8% W1-66 death rate.

## Observations

### Target slot instability from player movement

The exact targeting system sorts targets by LOS → imminence → type priority → distance → entity ID. Because
distance is a sort key, player movement can reshuffle which NPC occupies which slot between ticks. Observed case:
model chose ATK_T2 (Ranger), player moved, next tick ATK_T2 resolved to a Blob because the distance change
reordered the slots.

This is likely not a major issue — the observation fully encodes each slot's contents (type, HP, distance, delay),
so the model sees a different state when slots shuffle and can learn the indirect mapping. But it does mean the
model must learn "find the slot containing the NPC I want" rather than "always pick the same slot for the same
NPC," which adds learning difficulty. Worth monitoring if the model shows inconsistent target focus behavior.
