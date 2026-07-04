# V53 TB Tracking

## Status

Planning. V53 is a research direction document collecting ideas from an RL literature survey, filtered through the
lens of what's actually applicable to Inferno RL. None of these are committed changes yet — each needs evaluation
against V52's final state before implementation.

## Context: What V52 Established

V52 runs on a 602-dim v4 observation, `flat_lstm_residual` with 256 LSTM + 512x512 actor/critic, sweep curriculum
across W31-66 with pillar death event penalties. Best eval (checkpoint 1900, 740.6M steps): 99.2% weighted clear
rate, 0.8% death rate across all 5 loadouts. As of 773M steps: Phase_Fail% ~14.5%, EV 0.98, Deaths ~2/rollout.

The model regressed from peak (741M: Phase_Fail 9.3%, Deaths 1) to end (773M: Phase_Fail 14.5%, Deaths 2, DmgTkn
-8.15→-9.93). This correlates with entropy approaching floor (0.012) — the policy became too deterministic and lost
robustness on edge cases. This is the natural end-of-run pattern with a decaying entropy schedule, not a fixable
training problem for the current run.

The core bottleneck (per TRAINING_FINDINGS.md) remains **credit assignment** — connecting actions to outcomes across
long temporal gaps. Model capacity and observation size are not the limiting factors.

## Research Ideas (prioritized)

### Idea 1: Prioritized Level Replay (PLR) for Curriculum

**What**: Replace failure-weighted sweep sampling with regret-based level selection. Score each wave by regret
(performance gap vs. expected), maintain a staleness counter to prevent forgetting, sample proportionally.

**Why**: Current sweep curriculum treats all failures equally. A wave where the agent dies 90% of the time but has
never improved is different from a wave where it just regressed from 95% to 85%. PLR would focus training on waves
where the agent is actively losing ground or has the most room to improve, rather than hammering waves that may
be structurally hard given current capabilities.

**Effort**: Low-medium. The `AdaptiveController` already tracks per-wave stats (`GlobalWaveStats`). PLR would
replace the sampling logic in `_select_start_wave()` with a regret-scored priority buffer.

**Risk**: Low. Worst case it performs like current sweep. The staleness counter prevents catastrophic forgetting
of mastered waves (a known failure mode from V21/V25).

**Key references**: Jiang et al. 2021 (Replay-Guided Adversarial Environment Design), PAIRED (Dennis et al. 2020).

### Idea 2: Self-Predictive Auxiliary Objective (SPR-style)

**What**: Add a small auxiliary head that predicts the next observation's latent representation from current
LSTM output + action. The prediction target is the encoder's output for the *next* timestep, detached from the
computation graph (stop-gradient).

**Why**: The LSTM needs to learn temporal dynamics purely from reward signal. An SPR-style objective gives it
direct supervision: "if I take this action, what will my threat forecast / entity positions / pillar HP look
like next tick?" This is especially relevant for multi-tick planning (e.g., moving to a safespot takes 2-3 ticks,
and the reward only arrives once you're there).

**Architecture integration** with `flat_lstm_residual`:

```
obs (602) → LayerNorm → lstm_input_encoder (602→256, ReLU) → LSTM (256→256)
                                                                    ↓
                                                                lstm_out (256)
                                                                    ↓
actor_features = [raw_obs(602), lstm_out(256)] → Actor MLP (512,512) → heads
critic_features = [raw_obs(602), lstm_out(256)] → Critic MLP (512,512) → value
SPR: [lstm_out[t], action[t]] → predictor MLP → predicted_latent
     target = stop_gradient(lstm_input_encoder(obs[t+1]))
     loss = 1 - cosine_similarity(predicted, target)
```

SPR gradients flow through LSTM + input encoder only. Actor and critic MLPs receive **zero SPR gradients** —
they branch off after the `[raw_obs, lstm_out]` concatenation, and `raw_obs` is detached from the LSTM path.
Even if SPR distorts the LSTM slightly, the residual connection (raw obs → actor/critic) provides a fallback.

**Effort**: Medium. Requires a small prediction MLP (lstm_out + action → latent), stop-gradient on the target,
cosine similarity loss, and an auxiliary loss term added to the PPO loss with a coefficient.

**Risk**: Low-medium. V27's auxiliary damage prediction head failed, but every failure factor is absent here:

| Factor | V27 (failed) | V53 SPR |
|--------|-------------|---------|
| Shared layers with actor/critic | Yes (same 64×64 hidden) | No (only LSTM + input encoder) |
| Loss type | MSE on noisy scalar (future damage sum) | Cosine similarity in 256-dim latent |
| Target gradient | Flows (target shares weights) | Blocked (stop-gradient) |
| Model capacity | 371K params | ~3M+ params |
| Policy stability | Fresh BC warmstart, no RL | 773M steps of converged RL |
| Gradient magnitude | Unbounded MSE (grad norm hit 74×) | Cosine loss bounded [0, 2] per sample |

V27's failure is fully explained by shared-layer + MSE-on-scalar + tiny model + no stop-gradient. The actual
risks are:

- **LSTM gradient distortion** (low): If coefficient too high, LSTM could optimize for prediction over RL. But
  actor/critic also consume raw obs directly via residual, so they're not fully LSTM-dependent. Cosine loss
  bounds gradient magnitude.
- **Representation collapse** (low): Latent space collapses to trivial. Mitigation: batch normalization on
  projection head (BYOL-style). Easy to detect (cosine similarity saturates at 1.0).
- **Wasted time from wrong coefficient** (medium): The real cost. Too low = no effect, too high = slight
  slowdown. Neither is catastrophic, but a fruitless coefficient search wastes a training run.

**Recommended approach**: Start at coefficient 0.1. Log SPR loss and LSTM grad norms separately. If SPR loss
doesn't decrease within 10-20M steps, it's not learning — increase coefficient. If LSTM grad norms spike,
decrease. If no benefit by 50M steps, zero the coefficient and continue without it.

**Key references**: Schwarzer et al. 2021 (SPR), TACO (Zheng et al. 2023).

### Idea 3: Lightweight Entity Attention

**What**: Add a small attention layer over the 14 entity slots (14 x 25 = 350 dims) that produces a fixed-size
summary vector via mean/max pooling over attention outputs. Concatenate this summary with the existing
global/forecast/temporal features. The raw entity slots still flow through to actor/critic unchanged (preserving
the `flat_lstm_residual` residual path).

**Why**: The current architecture treats entity slots as a flat vector. The policy has to learn entity-entity
relationships (e.g., "mager behind ranger means I need to move to get LOS") from scratch. An attention layer
provides relational inductive bias — it can learn "this entity matters because of its relationship to that
entity" rather than relying on positional slot ordering.

**Effort**: Medium. The `EntityPoolEncoder` already exists and does mean/max pooling. Replacing its per-entity
MLP with a single-head self-attention layer is straightforward. The tricky part is integrating it as an
*additive* path in `flat_lstm_residual` rather than a bottleneck (which is why the old `entity_pool_lstm` failed).

**Risk**: Medium-high. Training finding #2: "smaller models learn faster on this problem." Adding an attention
layer increases params and may slow convergence without improving the ceiling. The `flat_lstm_residual`
architecture works precisely because it doesn't gate the observation — adding attention risks re-introducing
the information bottleneck that killed `entity_pool_lstm`. Only try this if PLR and SPR don't move the needle.

**Key references**: Zambaldi et al. 2019 (Relational RL), Locatello et al. 2020 (Slot Attention).

### Idea 4: BC Warmstart with v4 Observation

**What**: Collect expert demonstrations using the current v4 observation space and pretrain the policy with
behavioral cloning before RL fine-tuning.

**Why**: V12's breakthrough was partly from BC warmstart. The pretrain infrastructure exists
(`pretrain/pretrain_bc.py`, `pretrain/collect_bc_data.py`). With the v4 observation space, BC data would need
to be re-collected, but the manual play tooling (`visualizer/play_human.py`) supports it.

**Effort**: Medium-high. Collecting high-quality demonstrations for W31-66 is time-consuming (human play
sessions). The data pipeline exists but may need updates for the v4 obs + current action space.

**Risk**: Low-medium. BC warmstart is well-understood and low-risk. The main question is whether the model has
already learned beyond what BC can teach — at 773M steps with 85-90% wave clear rate, the policy may already
be better than human demonstrations for most waves. BC would be most useful if starting a fresh architecture.

### Idea 5: Observation Gap Fixes (from V39 audit)

**What**: Address the open observation issues from the V39 environment audit still listed in TRAINING_FINDINGS.md:

- **OBS-1** [High]: Expose frozen timer as `min(frozen / 32.0, 1.0)` per entity slot (+10 dims if added to all
  14 slots, or +1 per existing slot replacing the binary). Policy can't distinguish "frozen for 30 ticks" from
  "unfreezing next tick".
- **OBS-5** [Medium]: Add `blob_scan_triggers` count per direction to neighborhood forecast (+8 dims). Moving
  to a tile that gives an unscanned blob LOS triggers a scan — currently invisible to the policy.
- **OBS-6** [Medium]: Entity slot overflow counter when >14 entities alive (+2 dims). After blob splits, entity
  counts can exceed 14 and the policy has no signal about the overflow.

**Effort**: Low per fix. These are observation builder changes in `observation_v4.py` with corresponding sim
state exposure.

**Risk**: Low individually, but each changes the observation space dimensionality and requires a checkpoint
break (can't warm-start from a 602-dim checkpoint into a 612-dim observation). Best bundled with other breaking
changes.

### Idea 6: Entropy Floor / Schedule Rethink

**What**: Replace the monotonically decaying entropy schedule (0.05→0.002) with one that maintains exploration
capacity at convergence. Options:

- **Higher entropy floor** (0.005-0.01 instead of 0.002). Maintains stochasticity at convergence.
- **Cyclical entropy**: decay to floor, bump back up, repeat. Forces periodic re-exploration.
- **Entropy bonus proportional to value uncertainty**: explore more in states the critic is unsure about.

**Why**: V52's regression pattern (peak at ~660-741M, degradation at 767-773M) correlates directly with entropy
approaching floor. Every version has used a decaying schedule. By the time the model is good, it has no
exploration budget left and brittly overfits to common configurations while losing robustness on edge cases.

**Effort**: Zero for a higher floor (just a hyperparameter). Low for cyclical. Medium for uncertainty-based.

**Risk**: Very low. A higher floor is a conservative change — worst case the model converges slightly slower.
The current 0.002 floor was inherited from V49 and never validated as optimal. Given that V52's best eval
(99.2% clear rate) occurred at entropy ~0.015, there's strong evidence the model performs *better* with more
stochasticity than the floor allows.

## What Doesn't Apply

These RL techniques were evaluated and ruled out for Inferno RL:

| Technique | Why Not |
|-----------|---------|
| World models (Dreamer, MuZero, IRIS) | We have a fast deterministic simulator. Learning a world model would be slower and less accurate than using the sim directly. World models shine when environment interaction is expensive. |
| LLM-based exploration (Voyager, ELLM, SayCan) | Inferno has fixed, knowable mechanics. No "common sense" gap — game rules are fully modeled in the simulator. LLM guidance would add latency with no informational advantage. |
| Pixel-based representation learning (CURL, SLATE, Slot Attention) | We work from structured simulator state, not pixels. The 14 exact-target slots already are the object-centric representation, hand-built. |
| Meta-learning (MAML, RL²) | One task with curriculum variations, not many diverse tasks. The LSTM already handles within-episode adaptation. |
| Contrastive state abstraction (DBC, bisimulation) | Useful when observations contain irrelevant perceptual detail. Our 602-dim observation is already hand-designed to be decision-relevant — there's nothing to throw away. |
| Foundation model pretraining (Gato, Decision Transformer) | Requires large diverse multi-task datasets. We have one environment. The architecture overhead isn't justified. |

## Implementation Order (recommended)

V52 has plateaued. Starting a new model with bundled improvements:

1. **Obs gap fixes** — bundle OBS-1/6 into obs v5, since we're breaking checkpoints anyway
2. **PLR curriculum** — replace sweep sampling with regret-based level selection
3. **Entropy floor** — raise to 0.005-0.008, prevent late-training collapse
4. **SPR auxiliary** — add from start at coefficient 0.1, monitor and adjust or zero by 50M steps
5. **BC warmstart** — only if starting a fresh architecture
6. **Entity attention** — skip unless 1-4 all plateau

## Start Command

Not yet determined. V53 will be defined when V52 concludes and a direction is chosen.

## Current Settings

TBD — will inherit from V52's final settings with changes per the selected idea(s).

## What to Watch

TBD.

## Metrics Log

| Steps | Eps | Deaths | Timeout% | Phase_Fail% | EV | KL | VL | Clip | Ent | Return | RVar | Grad | Notes |
|-------|-----|--------|----------|-------------|----|----|----|----|-----|--------|------|------|-------|

## Reward Terms Log

TBD — reward term columns will depend on which ideas are implemented.
