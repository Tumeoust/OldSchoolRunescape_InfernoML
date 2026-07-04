# V24 TensorBoard Tracking

Nibbler-aware observation space (220 → 262 floats). Fresh BC warmstart from transformed V3 data. Adds 6 dedicated
nibbler slots (7 floats each) between entity slots and wave context — gives the policy explicit visibility into
individual nibbler positions, HP, target pillar (NE vs non-NE), and urgency (distance to target pillar).

**Run 1** (0–3.1M): `logs/V24_climb/`
**Phase**: Climb (W49→65, promote-after 5) → mastery (W55-65, W63-focused)
**Load**: `models/bc_warmstart_v4.pt` (fresh, 262-dim obs, LSTM 128, seq-len 32)
**Changes**: Observation space 220 → 262 floats (nibbler slots). Reverted n-envs 24→16 and batch 384→256 (V21 proven
config) after V23 post-mortem showed sustained KL >0.03 and entropy collapse with 24 envs.
**Stopped**: 3.1M steps. KL sustained 0.04-0.05, clip fraction rising to 0.28, entropy -1.65 dropping fast.

**Run 2** (3.1M–6.6M): `logs/V24_climb/` (continues)
**Load**: `models/V24_climb/inferno_gpu_w49-65_20260226_120210_200.pt` (checkpoint from Run 1)
**Changes**: n-epochs 5→3. Clip fraction 0.28 and sustained KL >0.04 indicated 5 epochs over the same batch was producing
too-large policy updates for the 262-dim obs space. Reducing epochs shrinks per-update policy shift without changing
other dynamics.

**Run 3** (6.6M–): `logs/V24_climb/` (continues)
**Load**: `models/V24_climb/inferno_gpu_w49-65_20260226_124918_200.pt` (checkpoint from Run 2)
**Changes**: (1) Removed INFERNO_COMPLETE_REWARD (+2000). Wave 66 is trivially easy (3 nibblers + 2 magers) — the
completion bonus was the largest single reward in the system for the easiest wave, distorting learning during mastery
training. Wave completion rewards already handle progression. (2) Added EARLY_ICE_BARRAGE_NIBBLER_REWARD (+8). Rewards
using ice barrage on nibblers within the first 3 ticks of a wave (ticks 0-2). Guides the policy toward the baseline
strategy of barraging the center nibbler before they spread toward pillars.

### Observation Changes

| Section        | V23 (220)          | V24 (262)          | Detail                                              |
|----------------|--------------------|--------------------|-----------------------------------------------------|
| Player state   | [0:10]             | [0:10]             | Unchanged                                           |
| Pillar state   | [10:22]            | [10:22]            | Unchanged                                           |
| Entity slots   | [22:214] (16×12)   | [22:214] (16×12)   | Unchanged                                           |
| Nibbler slots  | —                  | [214:256] (6×7)    | NEW: exists, x, y, hp, target_is_NE, dist_pillar, dist_player |
| Wave context   | [214:220]          | [256:262]          | Shifted +42                                         |

### Nibbler Slot Encoding (7 floats per slot, 6 slots)

| Offset | Feature             | Encoding                  | Notes                                    |
|--------|---------------------|---------------------------|------------------------------------------|
| 0      | exists              | 0 or 1                    | Slot presence flag                       |
| 1      | x                   | x / GRID_WIDTH            | Position                                 |
| 2      | y                   | y / GRID_HEIGHT           | Position                                 |
| 3      | hp_ratio            | hp / 10.0                 | Signals damaged nibblers                 |
| 4      | target_is_NE        | 0 or 1                    | Critical — is this nibbler threatening NE pillar? |
| 5      | dist_to_target_pillar | dist / MAX_DISTANCE     | Urgency — how close to reaching pillar   |
| 6      | dist_to_player      | dist / MAX_DISTANCE       | Range check for barrage/attack           |

Sort order: by distance to target pillar ascending (most urgent first).

**BC warmstart accuracy**: 95.1% (10 epochs, lr=1e-3, batch 512). Nibbler slots are all zeros in BC data — the model
learns nibbler awareness entirely during PPO. BC teaches positioning, weapon switching, and attack timing.

### Start Command — Run 1 (PowerShell, n-epochs=5, stopped at 3.1M)

```powershell
python -m tools.inferno_rl.train_gpu --load models/bc_warmstart_v4.pt --phase climb --start-wave 49 --max-wave 65 --promote-after 5 --save-dir models/V24_climb --log-dir logs/V24_climb --n-envs 16 --n-steps 1024 --batch-size 256 --n-epochs 5 --lr 1e-4 --entropy-start 0.015 --entropy-end 0.002 --gae-lambda 0.97 --vf-coef 0.75 --max-grad-norm 0.5 --normalize-obs --normalize-reward --checkpoint-every 100 --timesteps 100000000 --device cuda --log-reward-terms
```

### Resume Command — Run 2 (PowerShell, n-epochs=3, stopped at 6.6M)

```powershell
python -m tools.inferno_rl.train_gpu --load models/V24_climb/inferno_gpu_w49-65_20260226_120210_200.pt --phase climb --start-wave 49 --max-wave 65 --promote-after 5 --save-dir models/V24_climb --log-dir logs/V24_climb --n-envs 16 --n-steps 1024 --batch-size 256 --n-epochs 3 --lr 1e-4 --entropy-start 0.015 --entropy-end 0.002 --gae-lambda 0.97 --vf-coef 0.75 --max-grad-norm 0.5 --normalize-obs --normalize-reward --checkpoint-every 100 --timesteps 100000000 --device cuda --log-reward-terms
```

### Resume Command — Run 3 (PowerShell, removed inferno completion bonus)

```powershell
python -m tools.inferno_rl.train_gpu --load models/V24_climb/inferno_gpu_w49-65_20260226_124918_200.pt --phase climb --start-wave 49 --max-wave 65 --promote-after 5 --save-dir models/V24_climb --log-dir logs/V24_climb --n-envs 16 --n-steps 1024 --batch-size 256 --n-epochs 3 --lr 1e-4 --entropy-start 0.015 --entropy-end 0.002 --gae-lambda 0.97 --vf-coef 0.75 --max-grad-norm 0.5 --normalize-obs --normalize-reward --checkpoint-every 100 --timesteps 100000000 --device cuda --log-reward-terms
```

---

## How to Read This Document

Each snapshot records the same metrics at the same step intervals. Comparing across snapshots reveals:

1. **Is the frontier advancing?** — The primary success signal. Should climb from 49 toward 65.
2. **Are deaths trending down?** — Indicates the policy is learning survival, not just exploration.
3. **Is EV stable >0.80?** — Value function is calibrated. If EV drops below 0.70, something is wrong.
4. **Is entropy collapsing?** — entropy_loss approaching 0 = policy becoming deterministic too early. Should decay slowly
   with the entropy coefficient schedule, not crash.
5. **Is mean_reward trending up?** — Noisy but should show upward movement over 5-10M steps.
6. **Is KL stable <0.02?** — Measures how much the policy changes per update. Healthy range is 0.005–0.02. Sustained
   >0.03 means updates are too aggressive (policy thrashing). Spikes after reward changes are expected but should settle
   within 1-2M steps.
7. **Is grad_norm stable ~1.0–1.6?** — Gradient magnitude before clipping (max-grad-norm=0.5). Stable values mean the
   loss landscape is smooth. Sudden spikes (>3.0) or drops (<0.5) indicate the optimizer is struggling.
8. **Is FPS holding?** — Rollout collection speed. Drops indicate env bottlenecks.

### Early Warning Signs (kill the run)

- Entropy loss goes to 0 or positive → entropy collapse, policy is deterministic
- EV drops below 0.60 and doesn't recover within 2M steps → value function diverging
- Deaths increase monotonically over 5M+ steps → reward signal is counterproductive
- Frontier hasn't moved from W49 after 10M steps → agent isn't learning to clear waves at all
- KL sustained above 0.03 for 5M+ steps → policy is thrashing, not converging
- Grad norm spikes above 3.0 repeatedly → loss landscape unstable, check reward scaling

---

## Manual Observations & Progress

| Steps | Phase | Frontier | f.mean | Deaths | Waves | Timeouts | MeanRwd | EV | Entropy | KL | Grad | FPS | Note |
|-------|-------|----------|--------|--------|-------|----------|---------|----|---------|----|------|-----|------|
| 0.9M | 1 | 62 | 61.0 | 136 | 62 | 0 | -0.15 | 0.56 | -0.80 | 0.050 | 2.15 | 2556 | First snapshot. Frontier 62 from BC warmstart. **KL 0.050 (watch)**, EV 0.56 (watch) — very early, likely initial settling |
| 3.1M | 1 | 65 | 63.9 | 128 | 46 | 0 | -0.44 | 0.83 | -1.65 | 0.044 | 1.64 | 2391 | Frontier hit 65 (ceiling). EV recovered 0.56→0.83. **KL 0.044 (watch)** sustained >0.03. Entropy -1.65 dropping fast — on V23 trajectory |
| | | | | | | | | | | | | | **--- Run 2: n-epochs 5→3 (resume from _200.pt) ---** |
| 6.6M | 1 | 65 | 64.4 | 111 | 62 | 0 | 0.19 | 0.89 | -1.76 | 0.030 | 1.39 | 2275 | n-epochs fix working. KL 0.044→0.030, deaths 128→111, reward now positive. Entropy decay slowed (−0.11 in 3.5M vs −0.85 in 2.2M prior) |
| | | | | | | | | | | | | | **--- Run 3: removed INFERNO_COMPLETE_REWARD +2000, added EARLY_ICE_BARRAGE_NIBBLER +8 (resume from _200.pt) ---** |

---

## All Changes: V23 → V24 Startup

Two changes from V23: (1) observation space expansion with nibbler slots, (2) n-envs and batch reverted to V21 proven
config after V23 post-mortem showed 24 envs / batch 384 caused sustained KL >0.03 and entropy collapse.

### Architecture

| Change          | V23                         | V24                                   | When changed |
|-----------------|-----------------------------|---------------------------------------|--------------|
| Observation dim | 220                         | 262                                   | V24 start    |
| Nibbler slots   | —                           | 6 slots × 7 floats = 42              | V24 start    |
| BC warmstart    | bc_warmstart_v3.pt (220-dim)| bc_warmstart_v4.pt (262-dim)          | V24 start    |
| n-envs          | 24                          | **16**                                | V24 start (revert to V21) |
| Batch size      | 384                         | **256**                               | V24 start (revert to V21) |
| Max wave        | 65                          | 65                                    | Unchanged    |
| Curriculum      | Climb → mastery             | Climb → mastery                       | Unchanged    |
| Actor network   | [256, 128]                  | [256, 128]                            | Unchanged    |
| Critic network  | [256, 256]                  | [256, 256]                            | Unchanged    |
| LSTM hidden     | 128                         | 128                                   | Unchanged    |
| LSTM seq_len    | 32                          | 32                                    | Unchanged    |

### Reward Settings (inherited from V23, inferno completion removed in Run 3)

| Setting                                   | Value     |
|-------------------------------------------|-----------|
| DAMAGE_PENALTY_PER_HP                     | -2.5      |
| Single-LOS engagement bonus               | 2.5       |
| Single-LOS requires attack within 5 ticks | Yes       |
| Mager priority bonus per NPC              | 0.3       |
| Resurrection penalty                      | -12       |
| Melee resurrections exempt                | Yes       |
| Bat kill reward                           | 8         |
| Weapon switch penalty                     | -0.5      |
| Movement penalty                          | -0.2/tick |
| Pillar damage (non-NE)                    | Removed   |
| Initial barrage heuristic                 | Disabled  |
| Death penalty                             | -120      |
| Nibbler kill reward                       | 6         |
| Inferno completion bonus                  | **Removed in Run 3** (was +2000) |
| Early ice barrage nibbler reward          | **+8 (added Run 3)** — within first 3 ticks of wave |

### What to Watch For

- **KL should stay <0.02** — V23's sustained KL >0.03 with 24 envs was a key failure signal. Reverting to 16 envs /
  batch 256 should restore V21-like KL (0.010–0.013). If KL still >0.03 at 5M steps, something else is wrong.
- **Entropy should decay slowly** — V23 hit -2.00 at 7.5M. V21 was -0.37 at 5.3M. Expect V24 to track closer to V21.
  If entropy drops below -1.5 before 10M steps, the mastery mode W63 concentration may still be too aggressive.
- **Nibbler target_is_NE should reduce wasted barrage actions** — the policy can now distinguish NE-targeting nibblers
  (barrage immediately) from non-NE nibblers (deprioritize). Watch for reduced NE pillar damage in later waves.
- **Nibbler slots are all zeros in BC data** — the model has no BC prior for nibbler behavior. It must learn nibbler
  awareness entirely during PPO. The benefit appears once the agent encounters nibbler waves and correlates target_is_NE
  with NE pillar damage rewards.
- **Compare to V21 at equivalent step counts** — V24 has the same envs/batch as V21. The variables are: 262-dim obs
  (vs 186), nibbler slots, mastery curriculum (vs level-up), W49 start (vs W55). Deaths trending down by 10M is the
  success signal.
