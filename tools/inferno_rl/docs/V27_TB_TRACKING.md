# V27 TensorBoard Tracking

Fresh BC warmstart (`models/bc_warmstart.pt`) — same warmstart V21 used. Reward rework focused on
damage avoidance as the primary learning signal. Starting from scratch rather than resuming V21
because the reward landscape is fundamentally different (DAMAGE_PEN -7.5 vs -1.0).

## What Changed (V26 → V27)

### Reward Rework

Removed dense shaping rewards and sparse completion bonus. Increased damage penalty 7.5× to make
damage avoidance the dominant signal.

| Reward                        | V26 (= V21 original) | V27         | Rationale                                                                                                                                                                                                     |
|-------------------------------|----------------------|-------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `DAMAGE_PENALTY_PER_HP`       | -1.0                 | **-1.5**    | 1.5× V21. Reduced from -3.0 (Run 4 failed — grad norm flatlined ~5-6, EV collapsed 0.96→0.55, high-variance damage swings drowned all other signals). Ranger hit = -37.5, mager hit = -60. |
| `ATTACK_ON_COOLDOWN_BONUS`    | 1.0                  | **0.0**     | Remove — proxy reward that lets model farm bonuses without improving survival.                                                                                                                                |
| `SURVIVAL_REWARD_PER_TICK`    | 0.1                  | **0.0**     | Remove — negligible signal (0.01–0.1/tick), noise floor.                                                                                                                                                      |
| `INFERNO_COMPLETE_REWARD`     | 2000.0               | **0.0**     | Remove — extremely sparse, most episodes never see it. Wave completion rewards remain.                                                                                                                        |
| `TIME_PENALTY_PER_TICK`       | -0.1 (flat)          | **removed** | Replaced by escalating stall penalty.                                                                                                                                                                         |
| `STALL_BASE_PENALTY`          | —                    | **-2.0**    | New: escalating penalty when not engaged for 15+ ticks. -2.0, -2.5, -3.0, ... per tick (+0.5/tick). Reduced from -10 base/-1.0 escalation (Run 5: stall penalty -198/ep, 48% of all negative reward, drove grad norms to 6.0). |
| `SINGLE_LOS_ENGAGEMENT_BONUS` | 0.0                  | 0.0         | Already disabled in V26.                                                                                                                                                                                      |

### Unchanged Rewards

| Reward                             | Value                                    | Notes                                                                                                         |
|------------------------------------|------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| `DEATH_PENALTY`                    | ~~-120.0~~ **0.0**                       | Removed — episode termination IS the penalty. One-shot terminal signal can't be attributed to causal actions. |
| `WAVE_TIMEOUT_PENALTY`             | ~~-150.0~~ **0.0**                       | Removed — same reasoning. Stall penalty handles anti-hiding pressure.                                         |
| `DAMAGE_DEALT_REWARD_PER_HP`       | 0.5                                      | Counterbalances damage avoidance — must kill to progress                                                      |
| `BLOOD_BARRAGE_HEAL_REWARD_PER_HP` | 2.0                                      | Healing is now 2.0/HP gained vs 1.5/HP lost — mild incentive to heal                                          |
| `WAVE_END_HP_BONUS`                | 40.0                                     | End wave at full HP                                                                                           |
| `WAVE_COMPLETE_REWARD_BASE`        | 50.0                                     | +10 per wave cleared                                                                                          |
| `KILL_REWARDS`                     | 14/7/6/5/6 (mager/ranger/melee/blob/nib) | Kill progression                                                                                              |
| `MAGER_PRIORITY_BONUS_PER_NPC`     | 0.5                                      | Kill magers first                                                                                             |
| `MAGER_RESURRECTION_PENALTY`       | -20.0                                    | Prevent resurrections                                                                                         |
| `PILLAR_DAMAGE_PENALTY_PER_HP`     | -0.3 (NE 3×)                             | Protect pillars                                                                                               |
| `TIME_PENALTY_PER_TICK`            | -0.1                                     | Prevents stalling                                                                                             |
| `BLOOD_BARRAGE_HIGH_HP_PENALTY`    | -5.0                                     | Don't waste blood barrage at high HP                                                                          |
| `INVALID_ACTION_PENALTY`           | -3.0                                     | Guardrail                                                                                                     |
| `TILE_A_MAX_REWARD`                | 1.0                                      | Between-wave positioning                                                                                      |

### Design Intent

The V21 reward system had ~15 active reward signals pulling in different directions. The model could
satisfy shaping rewards (ATTACK_ON_COOLDOWN, SINGLE_LOS) without actually improving survival. V27
simplifies to: **don't take damage, kill things, complete waves.**

At -1.5/HP, damage avoidance is still 50% stronger than V21 (-1.0) but no longer dominates the reward
landscape. A ranger hit (-37.5) is comparable to a wave completion reward (50-60), keeping signals
balanced rather than one term drowning everything else.

**Insight from Runs 1-5:** High DAMAGE_PEN creates high-variance returns that destabilize the critic.
Damage taken per episode swings ±400 points at -3.0/HP (good episode: 20 HP taken = -60, bad: 150 HP
= -450). This variance dwarfs all other reward terms combined, causing: (1) grad norms to flatline at
5-6× the clip value (90% of gradient discarded every step), (2) EV collapse as the critic can't learn
a stable value function when one term randomly dominates returns. DAMAGE_DEALT (0.5/HP) is large but
low-variance (similar damage dealt each wave), so it's not the problem.

### Auxiliary Damage Prediction Head

Added an auxiliary `damage_head` to the Critic that predicts future incoming damage over the next K
ticks (default 5). Shares the same hidden layers as the value head — only the output projection differs.

```
Critic hidden (64, 64, ReLU)
    ├── value head  → nn.Linear(64, 1) → V(s)     [existing]
    └── damage head → nn.Linear(64, 1) → D(s)     [new]
```

**Purpose:** Credit assignment for pillar play. Damage taken on tick T results from a positioning
decision on tick T-3, but the critic only learns this temporal link indirectly through rewards. The
aux head forces the shared critic layers to learn "this position is dangerous" as an explicit feature.

**Loss:** `aux_loss = MSE(damage_pred, future_damage)` where `future_damage[t] = sum(damage_taken[t+1..t+K])`
within episode boundaries. Added to total loss: `loss += aux_coef * aux_loss`.

**Backward compatible:** `aux_coef=0.0` (default) means the head exists but contributes zero gradient.
Old checkpoints load with `strict=False` — damage head weights initialize randomly.

### Code

Architecture change: added auxiliary damage prediction head to Critic. Same 186-dim MLP, same obs/action space.
Policy forward now returns 7-tuple (added `damage_pred` at position 5).

### Checkpoint

- **File**: `models/bc_warmstart.pt` (same BC warmstart used for V21)
- **Architecture**: V21 MLP (371K params) — actor [256,128], critic [256,256], obs=186, actions=43
- **Starting from**: Behavioral cloning policy with no RL training

### Training Command

```powershell
tools\inferno_rl\venv\Scripts\activate.ps1; python -m tools.inferno_rl.train_gpu --load models/bc_warmstart.pt --phase climb --start-wave 55 --max-wave 66 --promote-after 5 --min-waves-to-advance 1 --save-dir models/V27_climb --log-dir logs/V27_climb --n-envs 16 --n-steps 1024 --batch-size 256 --n-epochs 5 --lr 1.5e-4 --entropy-start 0.015 --entropy-end 0.002 --gae-lambda 0.95 --vf-coef 0.75 --max-grad-norm 0.5 --normalize-obs --normalize-reward --checkpoint-every 100 --timesteps 140000000 --device cuda --log-reward-terms --aux-damage-coef 0.03 --aux-damage-lookahead 8
# DAMAGE_PEN: -7.5 (Runs 1-3) → -3.0 (Run 4) → -1.5 (Run 5)
```

### Hyperparameters

Proven V12-phase3 config exactly. Runs 1-3 failed — reverting all experimental settings:

| Setting              | Value         | vs V26 | Rationale                                                                |
|----------------------|---------------|--------|--------------------------------------------------------------------------|
| lr                   | **1.5e-4**    | 1e-4   | Half-step bump from proven 1e-4. 3e-4 too aggressive; 1e-4 may be too slow with aux head adding gradient signal |
| n-steps              | **1024**      | 1024   | Reverted from 2048. Longer rollouts caused KL spikes to 0.27             |
| start-wave           | **55**        | 49     | Skip easy waves, focus on waves where pillar play matters                |
| normalize-reward     | **on**        | on     | Prevents gradient explosion from damage penalty magnitudes               |
| entropy              | 0.015 → 0.002 | same   |                                                                          |
| gae-lambda           | 0.95          | same   |                                                                          |
| vf-coef              | 0.75          | same   |                                                                          |
| max-grad-norm        | 0.5           | same   | Protective clipping                                                      |
| n-envs               | 16            | same   |                                                                          |
| batch-size           | 256           | same   | 64 gradient steps/epoch                                                  |
| n-epochs             | 5             | same   | Total = 320 gradient steps/rollout                                       |
| promote-after        | 5             | same   |                                                                          |
| min-waves-to-advance | **1**         | same   | Match V21 — clear 1 wave from frontier per episode to count              |
| phase                | climb         | same   |                                                                          |
| checkpoint-every     | 100           | same   | ~1.64M steps/checkpoint                                                  |
| aux-damage-coef      | **0.03**      | new    | Auxiliary damage prediction loss coefficient (reduced from 0.5 — grad norm 74× caused PPO signal drowning) |
| aux-damage-lookahead | **5**         | new    | Ticks ahead to sum damage for aux target (~1 attack cycle)               |

---

## Early Warning Checks

| Signal          | Threshold                           | Action                                      |
|-----------------|-------------------------------------|---------------------------------------------|
| Entropy loss    | ≥ −0.05 (approaching 0)             | Entropy collapse — kill run                 |
| EV              | < 0.60 for 2M+ steps                | Value function diverging                    |
| Deaths          | Monotonically increasing 5M+        | Reward counterproductive                    |
| Frontier        | Unchanged from W55 after 10M steps  | Stagnation                                  |
| KL              | > 0.030 sustained                   | Policy thrashing                            |
| Grad norm       | > 3.0 repeated spikes               | Loss landscape instability                  |
| **Timeouts**    | **> 10% sustained**                 | **Damage penalty too high — model hiding**  |
| Aux damage loss | Flat at 0 after 5M+                 | Head not learning — check lookahead/coef    |
| Value loss      | > 2× baseline after adding aux head | Aux head destabilizing critic — reduce coef |

---

## Manual Observations & Progress

| Steps | Phase | Frontier | f.mean | Deaths | Waves | Timeouts | MeanRwd | EV | Entropy | KL | Grad | FPS | Note |
|-------|-------|----------|--------|--------|-------|----------|---------|----|---------|----|------|-----|------|
| 2.2M | 1 | 57 | 56.3 | 65 | 78 | 0 | -0.84 | 0.92 | -0.14 | 0.013 | 74.07 | 5163 | First entry. Frontier W57 from BC start W55. EV excellent. **Grad 74 — 50× normal (~1.5 in V21); aux head likely dominating gradient direction. With clip=0.5 updates are bounded but PPO/value signal drowned out. Consider reducing aux-coef.** |
| — | — | — | — | — | — | — | — | — | — | — | — | — | *New run: aux-coef 0.5 → 0.03, DAMAGE_PEN -3.0* |
| 1.3M | 1 | 57 | 56.3 | 66 | 79 | 0 | -0.76 | 0.94 | -0.19 | 0.233 | 3.44 | 5278 | Grad norm fixed (74→3.4). **KL 0.233 — 8× threshold (watch).** Early transient likely — first updates from BC warmstart. EV excellent 0.94. |
| 3.4M | 1 | 57 | 56.5 | 76 | 78 | 0 | -0.54 | 0.98 | -0.36 | 0.040 | 5.77 | 5653 | KL 0.233→0.040 (settling). **KL still above 0.030 (watch).** **Grad 5.77 — above 3.0, rising from 3.4 (watch).** Deaths 66→76, waves flat 78. Frontier stuck W57. EV 0.98 excellent. |
| 5.1M | 1 | 57 | 56.6 | 71 | 79 | 0 | -0.96 | 0.96 | -0.48 | 0.019 | 6.00 | 5670 | KL settled below threshold (0.019). Deaths 76→71 (improving). **Grad 6.0 — persisting above 3.0 at 5M+ (watch).** MeanRwd regressed -0.54→-0.96. Frontier still W57 — approaching 10M stagnation check. |
| 8.1M | 1 | 57 | 56.9 | 74 | 85 | 0 | 0.07 | 0.55 | -0.81 | 0.019 | 5.06 | 4952 | **EV 0.96→0.55 — below 0.60 threshold, critic degrading.** Grad flatlined ~5-6 (clipped every step). Waves 79→85 improving. MeanRwd turned positive. Frontier still W57 — 2M from stagnation flag. |
| — | — | — | — | — | — | — | — | — | — | — | — | — | *New run: DAMAGE_PEN -3.0 → -1.5. Grad norm flatlined ~5-6 and EV collapsed — high-variance damage term dominated returns.* |
| 1.3M | 1 | 57 | 56.2 | 72 | 78 | 0 | -0.27 | 0.95 | -0.16 | 0.029 | 3.03 | 5508 | Grad norm 3.0 — massively improved (was 5-6 at -3.0). KL 0.029 just under threshold. EV 0.95 healthy. Early snapshot — on track. |
| 4.7M | 1 | 57 | 56.4 | 64 | 83 | 0 | -0.66 | 0.94 | -0.37 | 0.031 | 4.82 | 5591 | Deaths 72→64, waves 78→83 — learning. **Grad crept 3.0→4.8 (watch).** KL 0.031 at threshold. EV 0.94 stable — no collapse like -3.0 run. Frontier still W57. |
| 8.1M | 1 | 57 | 56.7 | 73 | 79 | 0 | -0.12 | 0.94 | -0.73 | 0.015 | 6.02 | 5130 | Deaths/waves regressed (64→73, 83→79). **Grad 3.0→4.8→6.0 — monotonically increasing, now same as -3.0 run.** EV 0.94 still stable (key difference). KL settled 0.015. Frontier W57 — 2M from stagnation flag. |
| — | — | — | — | — | — | — | — | — | — | — | — | — | *New run: STALL_PENALTY -10 base/-1.0 escalation → -2.0 base/-0.5 escalation. Stall was -198/ep (48% of neg reward), primary grad norm driver.* |
| 1.0M | 1 | 57 | 56.3 | 60 | 79 | 0 | -0.66 | 0.96 | -0.17 | 0.063 | 3.34 | 4201 | Grad 3.3 — early but lower than -1.5 run start (3.0→4.8→6.0 trajectory). KL 0.063 (early transient, expect settling). EV 0.96. Deaths 60 — lowest first reading across runs. |
| 3.0M | 1 | 57 | 56.8 | 60 | 85 | 0 | 0.05 | 0.96 | -0.30 | 0.018 | 4.47 | 5570 | Grad 3.3→4.5 (prev run was 3.0→4.8 at same point). KL settled 0.018. Deaths flat 60, waves 79→85. f.mean 56.8 — ahead of prev run (56.4 at 4.7M). MeanRwd turned positive. EV stable. |
| 79.5M | 1 | 62 | 60.2 | 20 | 81 | 0 | 2.62 | 0.63 | -1.83 | 0.013 | 4.20 | 5972 | **Frontier 57→62, deaths 60→20 — strong learning.** Grad stabilized ~4.2 (stall fix worked — prev runs hit 6.0). KL 0.013 healthy. **EV 0.63 (watch)** — declining but above 0.60 kill. Entropy -1.83 deepening. |
| — | — | — | — | — | — | — | — | — | — | — | — | — | *New run (R7): Re-added NE pillar zone reward. Zone expanded: 5×5→7×7 ring (61 tiles total), added west strip x[11-14] y[22-24]. Zone bonus +1.0/tick (in zone + engaging or grace), penalty -0.5/tick (outside zone past grace). Model was wandering freely without zone shaping.* |
| — | — | — | — | — | — | — | — | — | — | — | — | — | *New run (R8): Fresh BC warmstart. Re-enabled dense shaping (Multi-LOS, Single-LOS, NE zone, survival) at low values. Entropy doubled (0.03→0.015 flat-ish). lr back to proven 1e-4. See below.* |
| 1.9M | 1 | 57 | 56.6 | 61 | 86 | 0 | 0.12 | 0.93 | -0.30 | 0.015 | 2.77 | 5541 | R8 first entry. Frontier W57 from BC start. EV 0.93 excellent. **Grad 2.77 — first V27 run below 3.0!** Dense shaping + lower stall keeping gradients healthy. KL 0.015 clean. Deaths 61 (typical BC start). |
| 9.3M | 1 | 57 | 56.9 | 79 | 64 | 0 | -0.05 | 0.91 | -1.66 | 0.016 | 5.98 | 3766 | Deaths 61→79, waves 86→64 — regressing. **Grad 2.77→5.98 — same ~6 plateau as all prior V27 runs.** Dense shaping only briefly suppressed grad norms. EV 0.91 stable. Frontier W57 at 9.3M — 0.7M from stagnation flag. Entropy deepened -0.30→-1.66 (fast). |

---

## Run 8 — Fresh BC Warmstart with Low-Value Dense Shaping

**Load**: `models/bc_warmstart.pt` (fresh — not continuing from R7)
**Wave range**: W55–66, climb, promote-after 5, min-waves 1

### Reward Changes (R7 → R8)

**Re-enabled dense shaping at low values:**
- `MULTI_LOS_PENALTY`: 0.0 → **-0.25**/tick (was -1.5 in V21)
- `SINGLE_LOS_ENGAGEMENT_BONUS`: 0.0 → **0.25**/tick (was 2.5 in V21)
- `NE_PILLAR_ZONE_BONUS`: 1.0 → **0.2**/tick (5× reduction)
- `SURVIVAL_REWARD_PER_TICK`: 0.0 → **0.1** × (hp/99) (was 0.5 in V21)

**Other reward changes:**
- `MAGER_PRIORITY_BONUS_PER_NPC`: 0.5 → **0.0** (disabled)
- `MAGER_RESURRECTION_PENALTY`: 20.0 → **10.0** (halved)
- `MELEE_RESURRECTION_PENALTY`: (new) **5.0** (half of mager/ranger/blob)
- Bat resurrection: **0.0** (exempt — bats are low-threat)
- `PILLAR_DAMAGE_PENALTY_PER_HP`: -0.3 → **-0.1**
- `NE_PILLAR_DAMAGE_MULTIPLIER`: 3× → **9×** (net NE pillar: -0.3×3 = -0.9 → -0.1×9 = -0.9, same effective NE penalty, but non-NE pillars now 3× cheaper)
- Kill RANGER: 7.0 → **6.0**
- Kill MELEE: 6.0 → **9.0** (melees are high-threat, fast, hard to deal with)
- Kill NIBBLER: 6.0 → **8.0**

### Hyperparameter Changes (R7 → R8)

| Setting | R7 | R8 | Rationale |
|---------|-----|----|-----------|
| lr | 1.5e-4 | **1e-4** | Back to proven V12-phase3 value |
| entropy-start | 0.015 | **0.03** | Doubled — more exploration from BC start |
| entropy-end | 0.002 | **0.015** | Near-flat schedule (0.03→0.015 over 140M) |
| aux-damage-lookahead | 5 | **8** | ~2 attack cycles instead of ~1 |

### Training Command

```powershell
tools\inferno_rl\venv\Scripts\activate.ps1; python -m tools.inferno_rl.train_gpu --load models/bc_warmstart.pt --phase climb --start-wave 55 --max-wave 66 --promote-after 5 --min-waves-to-advance 1 --save-dir models/V27_climb --log-dir logs/V27_climb --n-envs 16 --n-steps 1024 --batch-size 256 --n-epochs 5 --lr 1e-4 --entropy-start 0.03 --entropy-end 0.015 --gae-lambda 0.95 --vf-coef 0.75 --max-grad-norm 0.5 --normalize-obs --normalize-reward --checkpoint-every 100 --timesteps 140000000 --device cuda --log-reward-terms --aux-damage-coef 0.03 --aux-damage-lookahead 8
```

---
