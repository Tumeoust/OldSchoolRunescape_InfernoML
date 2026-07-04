# V29 TensorBoard Tracking

Copy PVP project training setup: larger network, larger batches, higher LR, lower entropy,
reward magnitudes in PVP-like range (~10-15 per successful wave).

## What Changed (V28 → V29)

### Philosophy

V28's grad norm fix (÷5 rewards, 1 epoch, removed aux head) was a manual workaround. The
`osrs-pvp-reinforcement-learning` project successfully trains PvP agents with a different setup:
larger network, larger batches, higher LR, lower entropy, and reward magnitudes in a tighter range.
V29 copies their setup wholesale.

### Architecture

| Setting      | V28     | V29         |
|--------------|---------|-------------|
| actor-sizes  | 256,256 | **512,512** |
| critic-sizes | 256,256 | **512,512** |
| params       | 371K    | ~1.2M       |

~~BC warmstart zero-padded into [512,512]: first 256 units get BC weights, rest init to zero.~~ (Run 1 — failed, grad norm 512)

**Run 2:** Native [512,512] warmstart via KL distillation from V21 (checkpoint 6800, 484M steps).
Collected 2.6M steps (2000 episodes W55-66, stochastic), distilled 10 epochs (KL 0.77→0.25, acc 65%).

### Training Settings

| Setting          | V28   | V29       | Rationale                                        |
|------------------|-------|-----------|--------------------------------------------------|
| lr               | 1e-4  | **3e-4**  | PVP reference: larger network needs higher LR    |
| batch-size       | 256   | **2048**  | PVP reference: smooths gradients with more envs  |
| n-envs           | 16    | **48**    | More parallel data for larger batches            |
| entropy-start    | 0.03  | **0.01**  | PVP reference: less entropy needed               |
| entropy-end      | 0.015 | **0.001** | PVP reference                                    |
| normalize-reward | yes   | **no**    | PVP reference: raw rewards with tight magnitudes |
| n-epochs         | 1     | 1         | Same                                             |
| n-steps          | 1024  | 1024      | Same                                             |
| gamma            | 0.995 | 0.995     | Same                                             |
| gae-lambda       | 0.95  | 0.95      | Same                                             |
| vf-coef          | 0.5   | 0.5       | Same                                             |
| max-grad-norm    | 0.5   | 0.5       | Same                                             |

48 envs × 1024 steps = 49,152 samples/rollout → 24 gradient steps per epoch at batch 2048.

### Code Changes

1. **Auto-detect resize in `train_gpu.py`**: `PPO.load()` now auto-detects when CLI `--actor-sizes`/`--critic-sizes`
   differ from the checkpoint and uses `load_with_resize()` automatically. No need for `--resize-lstm` flag.

2. **Reward rescaling**: All rewards scaled to PVP landscape (per-tick 0.005-0.02, per-HP 0.01-0.06,
   per-event 0.15-2.0). Target: ~10-15 reward per successful wave.

### Reward Changes

| Reward                        | V28   | V29        | Factor |
|-------------------------------|-------|------------|--------|
| DAMAGE_PENALTY_PER_HP         | -0.3  | **-0.05**  | ÷6     |
| DAMAGE_DEALT_REWARD_PER_HP    | 0.1   | **0.01**   | ÷10    |
| BLOOD_BARRAGE_HEAL/HP         | 0.4   | **0.06**   | ÷6.7   |
| SURVIVAL_REWARD_PER_TICK      | 0.02  | **0.005**  | ÷4     |
| WAVE_END_HP_BONUS             | 8.0   | **1.5**    | ÷5.3   |
| WAVE_COMPLETE_BASE            | 10.0  | **2.0**    | ÷5     |
| WAVE_COMPLETE_INCREMENT       | 2.0   | **0.3**    | ÷6.7   |
| KILL: MAGER                   | 2.8   | **0.6**    | ÷4.7   |
| KILL: RANGER                  | 1.2   | **0.25**   | ÷4.8   |
| KILL: MELEE                   | 1.8   | **0.35**   | ÷5.1   |
| KILL: BLOB                    | 1.0   | **0.2**    | ÷5     |
| KILL: BLOB_MAGE/RANGE/MELEE   | 1.2   | **0.15**   | ÷8     |
| KILL: BAT                     | 1.2   | **0.15**   | ÷8     |
| KILL: NIBBLER                 | 1.6   | **0.2**    | ÷8     |
| KILL: JAD                     | 40.0  | **8.0**    | ÷5     |
| KILL: HEALER                  | 5.0   | **1.0**    | ÷5     |
| KILL: ZUK                     | 400.0 | **80.0**   | ÷5     |
| KILL: ZUK_HEALER              | 5.0   | **1.0**    | ÷5     |
| TILE_A_MAX_REWARD             | 0.2   | **0.04**   | ÷5     |
| STALL_BASE_PENALTY            | -0.4  | **-0.08**  | ÷5     |
| STALL_ESCALATION              | 0.1   | **0.02**   | ÷5     |
| MULTI_LOS_PENALTY             | -0.05 | **-0.01**  | ÷5     |
| SINGLE_LOS_ENGAGEMENT_BONUS   | 0.05  | **0.01**   | ÷5     |
| INVALID_ACTION_PENALTY        | -0.6  | **-0.1**   | ÷6     |
| INVALID_ATTACK_PENALTY        | -0.3  | **-0.05**  | ÷6     |
| BLOOD_BARRAGE_HIGH_HP_PENALTY | -1.0  | **-0.2**   | ÷5     |
| PILLAR_DAMAGE_PENALTY_PER_HP  | -0.02 | **-0.004** | ÷5     |
| NE_PILLAR_ZONE_BONUS          | 0.04  | **0.008**  | ÷5     |
| NE_PILLAR_ZONE_PENALTY        | -0.1  | **-0.02**  | ÷5     |
| MAGER_RESURRECTION_PENALTY    | 2.0   | **0.4**    | ÷5     |
| MELEE_RESURRECTION_PENALTY    | 1.0   | **0.2**    | ÷5     |

Unchanged (0): DEATH_PENALTY, WAVE_TIMEOUT_PENALTY, INFERNO_COMPLETE, ATTACK_ON_COOLDOWN, MAGER_PRIORITY.
Unchanged (non-zero): STALL_WINDOW (15), GRACE_TICKS (17), TILE_A_RADIUS (5), NE_PILLAR_DAMAGE_MULTIPLIER (9.0).

### Per-Wave Reward Estimate (successful W55, ~200 ticks)

- Damage dealt: 450 HP × 0.01 = +4.5
- Damage taken: 20 HP × -0.05 = -1.0
- Kills: ~6 kills = +1.8
- Wave complete: +2.0, HP bonus: +1.2
- Survival: 200 × 0.005 = +1.0
- Positioning: ~+0.5 net
- **Total: ~+10 per wave** (matches PVP episode range of 5-20)

### Checkpoint

- ~~**Run 1**: `models/bc_warmstart.pt` zero-padded [256,256] → [512,512] — **failed** (grad norm explosion)~~
- **Run 2**: `models/bc_warmstart_512.pt` — native [512,512] via KL distillation from V21 @ 484M steps
    - Source: 2.6M steps collected from V21 (2000 episodes W55-66, stochastic)
    - Distillation: 10 epochs, KL loss 0.77→0.25, accuracy 65%
    - Architecture: MLP (~1.2M params) — actor [512,512], critic [512,512], obs=186, actions=43

### Training Command (Run 1 — failed)

```powershell
tools\inferno_rl\venv\Scripts\activate.ps1; python -m tools.inferno_rl.train_gpu --load models/bc_warmstart.pt --actor-sizes 512,512 --critic-sizes 512,512 --phase climb --start-wave 55 --max-wave 66 --promote-after 5 --min-waves-to-advance 1 --save-dir models/V29_climb --log-dir logs/V29_climb --n-envs 48 --n-steps 1024 --batch-size 2048 --n-epochs 1 --lr 3e-4 --entropy-start 0.01 --entropy-end 0.001 --gamma 0.995 --gae-lambda 0.95 --vf-coef 0.5 --max-grad-norm 0.5 --normalize-obs --checkpoint-every 100 --timesteps 200000000 --device cuda --log-reward-terms
```

### Training Command (Run 2 — distilled warmstart)

```powershell
tools\inferno_rl\venv\Scripts\activate.ps1; python -m tools.inferno_rl.train_gpu --load models/bc_warmstart_512.pt --phase climb --start-wave 55 --max-wave 66 --promote-after 5 --min-waves-to-advance 1 --save-dir models/V29_climb --log-dir logs/V29_climb --n-envs 100 --n-steps 1024 --batch-size 2048 --n-epochs 1 --lr 3e-4 --entropy-start 0.01 --entropy-end 0.001 --gamma 0.995 --gae-lambda 0.95 --vf-coef 0.5 --max-grad-norm 0.5 --normalize-reward --normalize-obs --checkpoint-every 100 --timesteps 200000000 --device cuda --log-reward-terms
```

---

## Early Warning Checks

| Signal    | Threshold                    | Action                                                           |
|-----------|------------------------------|------------------------------------------------------------------|
| Grad norm | > 1.0 sustained              | Rewards still too large                                          |
| Grad norm | < 0.1 sustained              | Rewards too small, learning stalled                              |
| KL        | > 0.03 sustained             | lr too high or n_envs too many → drop lr to 1e-4 or n_envs to 32 |
| Entropy   | ≥ -0.05 (approaching 0)      | Entropy collapse — kill run                                      |
| EV        | < 0.50 for 3M+ steps         | Critic not learning — check reward scale                         |
| Deaths    | Monotonically increasing 5M+ | Reward counterproductive                                         |
| FPS       | < 500                        | 48 envs may be CPU-bottlenecked                                  |

**Key metric:** Grad norm should be 0.3-0.8 with these smaller rewards. If it's still >2.0,
the reward magnitudes aren't the issue.

## Risks

1. **48 envs is untested** — 24 caused KL>0.03 in V23 (but V23 used 5 epochs; 1 epoch should be safer)
2. ~~**BC warmstart zero-padded into [512,512]**~~ — **resolved:** Run 2 uses native [512,512] distilled warmstart.
3. **7 variables changed simultaneously** — if it fails, can't isolate cause. Accept this as an exploratory run.
4. **No reward normalization** — if critic struggles with raw magnitudes, add `--normalize-reward` back.

---

## TensorBoard Snapshots

### Run 1 (zero-padded warmstart) — KILLED

| Step | Frontier | Deaths/roll | EV   | Grad Norm | KL    | Entropy   | FPS  | Notes                                                                                                                                                           |
|------|----------|-------------|------|-----------|-------|-----------|------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 3.1M | 57       | 214         | 0.92 | **512.6** | 0.001 | **-0.04** | 4788 | **KILL: grad norm 512x above target (0.3-0.8), entropy -0.04 at collapse threshold (≥-0.05). Rewards too large or no reward norm catastrophic. MeanRwd -93.7.** |

### Run 2 (distilled warmstart, lr 3e-4) — KILLED

| Step | Frontier | Deaths/roll | EV | Grad Norm | KL | Entropy | FPS | Notes |
|------|----------|-------------|----|-----------|----|---------|-----|-------|
| 2.8M | 55 | 231 | 0.85 | **1.61** | **0.135** | -0.71 | 7047 | **KILL: KL 0.135 — 4.5x kill threshold (0.03). Policy thrashing at lr 3e-4.** |

### Run 3 (distilled warmstart, lr 1e-4)

| Step | Frontier | Deaths/roll | EV | Grad Norm | KL | Entropy | FPS | Notes |
|------|----------|-------------|----|-----------|----|---------|-----|-------|
| 2.0M | 55 | 271 | 0.83 | 0.73 | **0.132** | -0.81 | 6849 | **KL 0.132 — still 4.4x kill threshold even at lr 1e-4. Grad norm healthy (0.73). Frontier stuck.** |
| 5.9M | 56 | 282 | 0.87 | 0.64 | **0.077** | -0.86 | 7269 | KL trending down (0.132→0.077). Frontier 55→56. Waves 42→75. Grad norm healthy. Still above 0.03 threshold but improving. |
| 13.2M | 56 | 348 | 0.85 | 0.64 | 0.026 | -0.87 | 7081 | **KL under 0.03 threshold (0.026).** Reward normalizer stabilized. Waves 75→157, timeouts 15→5. MeanRwd -0.58→-0.18. Frontier still 56. |
| 200M | 66 | 22 | 0.76 | 0.42 | 0.005 | **-0.10** | 6686 | **Frontier 66, mastery 91%, deaths 22, 0 timeouts.** Full 200M run complete. Entropy -0.10 near collapse (watch). KL 0.005 excellent. |

### Checkpoint Eval (200 episodes, W49-66, deterministic)

| Checkpoint | Steps | Death% | Clear% | Mean Wave | Median | Notes |
|-----------|-------|--------|--------|-----------|--------|-------|
| 600 | 29M | 98.5% | 0.0% | 50.8 | 51 | Barely clears W49. Entropy still collapsing (-0.85→-0.10 happened 10M-30M). |
| 1000 | 49M | 100.0% | 0.0% | 51.5 | 51 | No improvement over 600. Policy locked in, entropy flat at -0.10. |
| 1600 | 78M | 99.5% | 0.0% | 52.0 | 52 | Marginal +0.5 mean wave. Still can't get past W52 consistently. |
| 4069 | 200M | 73.0% | **26.5%** | 60.5 | 62 | All performance came after 78M. 26.5% clear W49-66, median W62. |

**Key finding:** The [512,512] network (1.2M params) spent 78M+ steps barely functional, then improved
to 26.5% clear rate by 200M. For comparison, V21 [256,256] (371K params) achieved 22% clear at 138M steps.
V29 is marginally better (+4.5%) but at 1.5x the steps and 3.2x the parameters — not a decisive win for
the larger network. Entropy collapsed by 30M (flat at -0.10) but the policy continued improving despite
near-zero exploration, suggesting the agent was refining existing behaviors rather than discovering new ones.
