# V26 TensorBoard Tracking

Resume training from V21's best checkpoint (Run 5 @ 290M, 30% clear W49-66).

V25's LSTM+262-dim architecture (807K params, 271M steps) peaked at 26% clear W49-66 — worse than V21's MLP+186-dim
(371K params) at 290M (30%) with half the parameters. The architectural additions (LSTM, pillar-relative features,
nibbler entity slots) were net negative for training efficiency.

### Corrected V21 Baseline (100 seeds, W49-66, deterministic)

Previous "58% clear, zero deaths W55-66" was inflated by a death_analysis.py bug that counted wave timeouts as clears.
Full V21 eval sweep:

| Checkpoint | Run | Steps | Clear | Death | Timeout |
|-----------|-----|-------|-------|-------|---------|
| 092751_2100 | R2 | 134M | 20% | 72% | 8% |
| 092751_2200 | R2 | 136M | 21% | 72% | 7% |
| 092751_2300 | R2 | 138M | 22% | 73% | 5% |
| 092751_2400 | R2 | 139M | 14% | 77% | 9% |
| 143328_1500 | R4 | 164M | 19% | 74% | 7% |
| 195520_2000 | R5 | 223M | 25% | 72% | 3% |
| 195520_3000 | R5 | 239M | 29% | 63% | 8% |
| 195520_4000 | R5 | 256M | 29% | 69% | 2% |
| 195520_5000 | R5 | 272M | 27% | 67% | 6% |
| 195520_5500 | R5 | 280M | 30% | 61% | 9% |
| 195520_6000 | R5 | 288M | 26% | 63% | 11% |
| **195520_6103** | **R5** | **290M** | **30%** | **66%** | **4%** |
| 090159_2000 | R6 | 323M | 10% | 88% | 2% |
| 222731_4000 | R8 | 438M | 20% | 78% | 2% |

Run 5 (239M-290M) is a broad plateau at 29-30% clear. Ckpt 6103 (290M, 30% clear) is the end of Run 5 and the peak
before Run 6's wave expansion to W35-66 crashed performance to 10%.

## What Changed (V25 → V26)

### Code Reverted to V21 186-dim MLP

| Component | V25 | V26 (= V21 code) |
|-----------|-----|-------------------|
| Obs size | 262 (pillar-relative + nibbler slots) | 186 (player 8 + pillar 12 + entity 16×10 + wave 6) |
| Model | LSTM 256 hidden, 807K params | MLP only, 371K params |
| Entity sort | `_get_threat_priority()` with x,y tie-break | `entity_type.base_priority` |
| Weapon encoding | 4-way (BoFa/Blowpipe/Ice/Blood) | 4-way (BoFa/Blowpipe/Ice/Blood) |

### Reward settings in `rewards.py` (V21 original)

Note: The Run 5 checkpoint was trained under different rewards (DAMAGE_PEN -2.5, conditional Single-LOS). V26 training
will use the original V21 rewards below, so the policy will adapt to the new reward signals during fine-tuning.

| Reward | V25 (at end) | V26 (= V21 original) |
|--------|-------------|----------------------|
| DEATH_PENALTY | -120 | -120 |
| WAVE_TIMEOUT_PENALTY | -120 | -150 |
| DAMAGE_PENALTY_PER_HP | -2.5 | -1.0 |
| DAMAGE_DEALT_REWARD_PER_HP | 0.25 | 0.5 |
| BLOOD_BARRAGE_HEAL_REWARD_PER_HP | 2.5 | 2.0 |
| SURVIVAL_REWARD_PER_TICK | 0.3 | 0.1 |
| SINGLE_LOS_ENGAGEMENT_BONUS | 2.5 (conditional on attack in 5 ticks) | 1.5 (unconditional) |
| ATTACK_ON_COOLDOWN_BONUS | 2.0 | 1.0 |
| MAGER_RESURRECTION_PENALTY | -12 (non-melee only) | -20 (all types) |
| MAGER_PRIORITY_BONUS_PER_NPC | 0.6 | 0.5 |
| BAT kill reward | 8.0 | 5.0 |
| Time penalty | Escalating (stall detection) | Flat -0.1/tick |
| Pillar damage | NE only | All pillars (NE weighted 3×) |
| INFERNO_COMPLETE_REWARD | — | 2000.0 |
| Weapon switch penalty | -0.5 | removed |
| Movement penalty | -0.2/tick | removed |
| Early ice barrage reward | +15.0 | removed |
| NE pillar zone bonus/penalty | +0.3/+0.2 / -0.5 | removed |

### Checkpoint

- **File**: `models/V21_climb/inferno_gpu_w55-66_20260224_195520_6103.pt`
- **Trained steps**: 290,045,952 (~290M)
- **PolicyParams**: actor [256,128], critic [256,256], obs=186, actions=43, lstm=None
- **Architecture**: V21 MLP (371K params)
- **Eval** (100 seeds, W49-66, deterministic): 30% clear, 66% death, 4% timeout — V21 peak

### V21 Run Structure (for context)

- **Run 1** (0–100M): `w55-66_20260223_223356` — BC warmstart, W55-66 climb
- **Run 2** (100M–140M): `w55-66_20260224_092751` — mager priority bonus added at 100M
- **Run 3** (140M–148M): `w55-66_20260224_132121` — removed non-NE pillar damage, disabled initial barrage
- **Run 4** (148M–191M): `w55-66_20260224_143328` — continued
- **Run 5** (191M–290M): `w55-66_20260224_195520` — DAMAGE_PEN -1.0→-2.5, conditional Single-LOS
- **290M = Run 5 checkpoint 6103** — the peak before Run 6's W35-66 expansion crashed performance to 10%

The V21 run continued for 191M more steps (to 481M total) through 3 more reward changes and an LSTM architecture change
at 352M, but never recovered the Run 5 plateau.

---

## V26 Training Plan

### Hyperparameters

Proven V12_phase3 config (used for all successful V21 training):

| Setting | Value | Note |
|---------|-------|------|
| lr | 1e-4 | Proven stable |
| entropy | 0.015 → 0.002 | Linear decay |
| gae-lambda | 0.95 | V21 used 0.95 through Run 5 (changed to 0.97 at 352M with LSTM) |
| vf-coef | 0.75 | Stable EV 0.85-0.90 |
| max-grad-norm | 0.5 | Grad norms 1.4-1.6, clipping protective |
| n-envs | 16 | Ceiling — 24/32 fail |
| batch-size | 256 | 64 gradient steps per epoch |
| n-epochs | 5 | Conservative KL |
| n-steps | 1024 | Rollout length |
| promote-after | 5 | Curriculum advancement threshold |
| phase | climb | Forward curriculum |
| checkpoint-every | 100 | ~1.64M steps per checkpoint |

### Wave Range

W49-66 — same as V21's later runs. The Run 5 checkpoint was trained on W55-66; broadening to W49-66 addresses the
W49-54 weakness visible in the eval results while maintaining exploration pressure.

### Training Command

```powershell
tools\inferno_rl\venv\Scripts\activate.ps1; python -m tools.inferno_rl.train_gpu --load models/V21_climb/inferno_gpu_w55-66_20260224_195520_6103.pt --phase climb --start-wave 49 --max-wave 66 --promote-after 5 --save-dir models/V26_climb --log-dir logs/V26_climb --n-envs 16 --n-steps 1024 --batch-size 256 --n-epochs 5 --lr 1e-4 --entropy-start 0.015 --entropy-end 0.002 --gae-lambda 0.95 --vf-coef 0.75 --max-grad-norm 0.5 --normalize-obs --normalize-reward --checkpoint-every 100 --timesteps 10000000 --device cuda --log-reward-terms
```

---

## How to Read This Document

Each snapshot records the same metrics at the same step intervals. Comparing across snapshots reveals:

1. **Is the frontier advancing?** — The primary success signal.
2. **Are deaths trending down?** — Indicates the policy is learning survival, not just exploration.
3. **Is EV stable >0.80?** — Value function is calibrated. If EV drops below 0.70, something is wrong.
4. **Is entropy collapsing?** — entropy_loss approaching 0 = policy becoming deterministic too early.
5. **Is mean_reward trending up?** — Noisy but should show upward movement over 5-10M steps.
6. **Is KL stable <0.02?** — Sustained >0.03 means updates are too aggressive.
7. **Is grad_norm stable ~1.0–1.6?** — Sudden spikes (>3.0) indicate loss landscape instability.

### Early Warning Signs (kill the run)

- Entropy loss goes to 0 or positive → entropy collapse
- EV drops below 0.60 and doesn't recover within 2M steps → value function diverging
- Deaths increase monotonically over 5M+ steps → reward signal counterproductive
- KL sustained above 0.03 for 5M+ steps → policy thrashing
- Grad norm spikes above 3.0 repeatedly → loss landscape unstable

---

## Manual Observations & Progress

| Steps | Phase | Frontier | f.mean | Deaths | Waves | Timeouts | MeanRwd | EV | Entropy | KL | Grad | FPS | Note |
|-------|-------|----------|--------|--------|-------|----------|---------|------|---------|-------|------|------|------|
| 300.0M | 1 | 62 | 57.9 | 9 | 100 | 0 | 9.72 | 0.64 | -0.94 | 0.011 | 1.24 | 4788 | First V26 snapshot (+10M from V21 R5 290M). LOS rewards removed. Phase 1 (curriculum reset W49-66). Frontier 62 fast. EV 0.64 (watch — critic recalibrating after reward change). |
| 338.5M | 1 | 63 | 62.2 | 15 | 86 | 0 | 6.16 | 0.57 | -1.74 | 0.011 | 1.45 | 4585 | **EV 0.57 — below 0.60 kill threshold for 38M+ steps.** Frontier 62→63, f.mean 57.9→62.2 (curriculum advancing). But MeanRwd 9.72→6.16, deaths 9→15, waves 100→86 — performance regressing. Critic not adapting to reward changes after 48M steps. |

---
