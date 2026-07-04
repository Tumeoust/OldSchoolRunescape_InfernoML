# V38 TB Tracking

## Goal

Test whether a small residual LSTM on top of **observation V3.1** improves temporal continuity and inter-wave learning without
reintroducing the heavy `v2 entity_pool_lstm` failure mode.

## Base Checkpoint

- Base checkpoint: **none (fresh start)**
- Start point: **W1**
- Note: V38 is an intentional fresh architecture run, not a warmstart from V37.

## What Changed (V37 -> V38)

### 1. Policy Architecture: `flat_lstm_residual`

V38 keeps the full normalized V3.1 observation visible to the actor/critic and adds a small recurrent memory path:

- `LayerNorm(295)`
- `Linear(295 -> 128) + ReLU`
- `1-layer LSTM(128 -> 128)`
- concatenate `[normalized_obs_t, lstm_out_t]`
- feed fused features into the existing actor/critic heads

This is explicitly different from the older `entity_pool_lstm` family, which forced the representation through a heavier recurrent
front-end.

### 2. Recurrent Settings

- `lstm-hidden-size = 128`
- `lstm-seq-len = 16`
- `lstm-burn-in = 8`

The intent is to cover short tactical windows and the 9-tick inter-wave grace period without adding the parameter cost of the older 256-wide
recurrent runs.

### 3. Observation and Rewards Held Constant

V38 keeps:

- `observation-version = v3.1`
- current V37 reward setup
- current inter-wave freedom (no between-wave override)

So the main experiment is architecture, not state or reward design.

### 4. Stateful Recurrent Inference Support

Eval, visualizer, replay, and analysis wrappers now preserve recurrent hidden state across ticks and reset it only at episode boundaries.

This is required for valid evaluation of recurrent checkpoints.

### 5. CLI / Docs Updated

`train_gpu.py` and the observation docs now allow `v3.1` with `policy_arch="flat_lstm_residual"`.

## V38 Hypothesis

V37 fixed many snapshot blind spots by surfacing the right explicit state in V3.1.

The remaining gaps are mostly temporal:

- action commitment across several ticks
- wave-transition setup
- tactical continuity under changing LOS
- recent hazard trends

V38 should improve those by adding a small recurrent memory path while still keeping direct access to the strong V3.1 snapshot.

## Training Settings

| Setting             | Value                | Notes                                                         |
|---------------------|----------------------|---------------------------------------------------------------|
| warmstart           | `none (fresh)`       | Chosen explicitly for clean attribution                       |
| curriculum-mode     | `static`             | Same as V37 bootstrap                                         |
| phase               | `climb`              | Same as V37                                                   |
| climb-sampling      | `weighted`           | Same as V37                                                   |
| promote-after       | `5`                  | Same as V37                                                   |
| start-wave          | `1`                  | Fresh from W1                                                 |
| max-wave            | `66`                 | Full Inferno                                                  |
| observation-version | `v3.1`               | Same as V37                                                   |
| policy-arch         | `flat_lstm_residual` | New for V38                                                   |
| lstm-hidden-size    | `128`                | Small recurrent memory                                        |
| lstm-seq-len        | `16`                 | Covers short tactical windows                                 |
| lstm-burn-in        | `8`                  | Warm hidden state for training windows                        |
| actor/critic sizes  | `512,512 / 512,512`  | Same head sizes as V37                                        |
| n-envs              | `16`                 | Same as current V37 command                                   |
| n-steps             | `1024`               | Same as V37                                                   |
| batch-size          | `2048`               | Same as V37                                                   |
| n-epochs            | `3`                  | Lower than V37 fresh MLP to reduce recurrent over-update risk |
| lr                  | `2e-4`               | Lower than V37 fresh MLP for stability                        |
| target-kl           | `0.015`              | Tighter than V37                                              |
| entropy-start/end   | `0.05 / 0.002`       | Same exploration envelope as V37                              |
| gamma               | `0.995`              | Same                                                          |
| gae-lambda          | `0.95`               | Same                                                          |
| vf-coef             | `0.5`                | Same                                                          |
| max-grad-norm       | `0.5`                | Same                                                          |
| normalize-reward    | yes                  | Same                                                          |
| normalize-obs       | yes                  | Same                                                          |
| checkpoint-every    | `100`                | Same                                                          |
| total budget        | `200M`               | Same first-pass budget as V37                                 |

## Run Command

```powershell
python -m tools.inferno_rl.train_gpu --curriculum-mode static --phase climb --climb-sampling weighted --promote-after 5 --start-wave 35 --max-wave 66 --observation-version v3.1 --policy-arch flat_lstm_residual --lstm-hidden-size 128 --lstm-seq-len 16 --lstm-burn-in 8 --episode-mode full --n-envs 16 --n-steps 1024 --batch-size 2048 --n-epochs 3 --lr 3e-4 --target-kl 0.015 --entropy-start 0.05 --entropy-end 0.002 --gamma 0.995 --gae-lambda 0.95 --vf-coef 0.5 --max-grad-norm 0.5 --normalize-obs --normalize-reward --actor-sizes 512,512 --critic-sizes 512,512 --save-dir models/V38_obs31_lstm128 --log-dir logs/V38_obs31_lstm128 --checkpoint-every 100 --timesteps 200000000 --device cuda --log-reward-terms --load models/V38_obs31_lstm128/inferno_gpu_w35-66_20260311_080525_300.pt
```

## Files Changed

| File                                           | Changes                                                                         |
|------------------------------------------------|---------------------------------------------------------------------------------|
| `tools/inferno_rl/ppo/policy.py`               | Added `flat_lstm_residual` recurrent front-end with observation + memory fusion |
| `tools/inferno_rl/train_gpu.py`                | Allowed `flat_lstm_residual` for `v3`/`v3.1`, updated CLI validation            |
| `tools/inferno_rl/inference_state.py`          | Shared recurrent inference-state helper for offline tools                       |
| `tools/inferno_rl/eval.py`                     | Stateful recurrent inference across ticks                                       |
| `tools/inferno_rl/visualizer/run_visual.py`    | Stateful recurrent inference for visual playback                                |
| `tools/inferno_rl/model_log_run.py`            | Stateful recurrent inference and V3/V3.1 temporal handling                      |
| `tools/inferno_rl/death_analysis.py`           | Stateful recurrent inference reset per episode                                  |
| `tools/inferno_rl/death_analysis_v21.py`       | Stateful recurrent inference reset per episode                                  |
| `tools/inferno_rl/cli/replay_episode.py`       | Stateful recurrent replay for actions/values                                    |
| `tools/inferno_rl/visualizer/review_deaths.py` | Stateful recurrent inference and temporal handling                              |
| `tools/inferno_rl/visualizer/play_human.py`    | Stateful recurrent comparison replay                                            |
| `tools/inferno_rl/docs/OBSERVATION_SPACE.md`   | Updated V3.1 recurrent support docs                                             |
| `tools/inferno_rl/docs/RL_CHANGES.md`          | Added V38 architecture entry                                                    |

## Metrics Log

| Ckpt  | Steps | Frontier | EV    | Entropy | KL    | Grad | Ep Len | FPS  | Notes                                                                                                                                                                                                                                                                |
|-------|-------|----------|-------|---------|-------|------|--------|------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ~18   | 0.3M  | 36       | -0.32 | 0.050   | 0.008 | 0.29 | 691    | 2680 | First log. Frontier 36 from W35 start. EV negative, value fn not trained yet. Stall penalty dominates (−117 ep sum). 8 deaths/1 timeout in 9 eps. Max wave from 35=38, from 36=39.                                                                                   |
| ~193  | 3.2M  | 45       | 0.74  | 0.049   | 0.010 | 0.36 | 208    | 2556 | Frontier 36→45 (+9 waves in 2.9M steps). EV jumped to 0.74 — value fn learning well. Stall penalty collapsed (−117→−0.25). Kill rewards broadening (ImKot, Ak, AkRek variants appearing). Max wave from 45=48. Promo rate 24% (13/54). 0 timeouts.                   |
| ~469  | 7.7M  | 54       | 0.63  | 0.048   | 0.008 | 0.39 | 278    | 2584 | Frontier 45→54 (+9 waves in 4.5M steps). Kill_Xil appearing. Max wave from 54=56. Promo rate improved to 37% (16/43). EV dipped 0.74→0.63 (expected at harder frontier). Blood barrage heal tripled (1.1→3.0). Return mean positive (0.088). 0 timeouts.             |
| ~734  | 12.0M | 60       | 0.76  | 0.047   | 0.012 | 0.36 | 227    | 2571 | Frontier 54→60 (+6 waves in 4.3M steps). EV recovered to 0.76. Max wave from 59=62, from 60=63 — pushing into Jad territory. Melee proximity penalty increased (−0.05→−0.54). Promo rate dipped to 22% (10/46). KL rising (0.008→0.012). 0 timeouts.                 |
| ~999  | 16.4M | 61       | 0.77  | 0.046   | 0.012 | 0.35 | 210    | 2540 | Frontier 60→61 (+1 wave in 4.4M steps) — climb slowing significantly. Max wave from 61=63. Promo rate dropped to 15% (8/53). Stall penalty rising again (−0.30→−0.56). Damage taken up (−5.1→−6.4). Max wave cleared mean flat at 58.2. 0 timeouts.                  |
| ~1254 | 20.5M | 62       | 0.81  | 0.045   | 0.011 | 0.36 | 236    | 2518 | Frontier 61→62 (+1 wave in 4.1M steps). EV up to 0.81 (best yet). Max wave from 62=64. Return mean jumped to 0.203 (best). Mager_Priority reward doubled (0.76→1.19). Promo rate 20% (8/40). Running reward mean crossed positive (0.57). 0 timeouts.                |
| ~1494 | 24.5M | 63       | 0.77  | 0.044   | 0.009 | 0.39 | 187    | 2399 | Frontier 62→63 (+1 wave in 4.0M steps). Max wave from 63=66 — agent reaches the end! Early stop firing (KL 0.015). EV dipped 0.81→0.77. Max wave cleared mean improved to 60.4. Promo rate 20% (9/44). Ep len shorter (236→187). Kill_Zek steady (0.29). 0 timeouts. |
| ~2298 | 37.7M | 61       | 0.83  | 0.048   | 0.010 | 0.34 | 252    | 2597 | Frontier dropped 63→61 (curriculum reset on restart). EV best yet at 0.83. No early stop. Return mean 0.268 (best). FPS 2597. Waves completed 43. Max wave from 61=62. Kill_ImKot/MejRah doubled. Promo rate 19% (8/43). 0 timeouts. |
| ~2572 | 42.1M | 63       | 0.84  | 0.047   | 0.011 | 0.35 | 207    | 2343 | Frontier re-climbed 61→63 (+2 in 4.4M steps). EV new best 0.84. Max wave from 63=66, from 62=64, from 60=63. Return mean 0.308 (new best). Max wave cleared mean 61.5 (up from 58.7). Promo rate 23% (7/31). Clip fraction rising (0.15). 0 timeouts. |
| ~2812 | 46.1M | 63       | 0.75  | 0.046   | 0.010 | 0.35 | 270    | 2440 | Frontier stalled at 63 for 4.0M steps. EV dropped 0.84→0.75. Promo rate collapsed to 9% (4/43). Stall penalty spiked (−0.67→−1.02). Max wave from 63=66 still holds. Max wave cleared mean dipped 61.5→60.6. Return mean 0.281 (down from 0.308). 0 timeouts. |
| ~3067 | 50.2M | 63       | 0.70  | 0.045   | 0.014 | 0.40 | 349    | 2439 | Frontier still 63 (~8M steps stalled). EV declining further 0.75→0.70. Early stop firing (KL 0.017, only 1 epoch). Clip fraction high at 0.19. Promo rate recovered to 24% (8/26), max streak 3. Ep len up to 349 (longer episodes = deeper runs). Max wave from 63=66 holds. 0 timeouts. |
| ~3322 | 54.4M | 63       | 0.86  | 0.044   | 0.009 | 0.35 | 277    | 2337 | Frontier 63 (~12M steps stalled). EV rebounded strongly 0.70→0.86 (new best). No early stop, full 3 epochs. KL settled to 0.009. Promo rate 29% (7/24), best yet at this frontier. Max wave cleared mean 62.0 (new best). Return mean 0.402 (new best). Waves completed 48. 0 timeouts. |

## Eval Results

### Broad Eval (`W49-66`, 100 seeds)

| Ckpt | Steps | Frontier | Clear | Death | Timeout | Mean Max Wave | Notes |
|------|-------|----------|-------|-------|---------|---------------|-------|
|      |       |          |       |       |         |               |       |

### Narrow Eval (`W55-66`, 100 seeds)

| Ckpt | Steps | Clear | Death | Timeout | Top Death Waves | Notes |
|------|-------|-------|-------|---------|-----------------|-------|
|      |       |       |       |         |                 |       |

## Success Criteria

1. Frontier reaches at least W20 by 15M steps.
2. Frontier reaches at least W49 by 40M steps.
3. Manual review shows the model consistently learning inter-wave repositioning and setup without heuristic control.
4. Once V38 reaches W49+, broad `W49-66` eval is at least competitive with the best available V37 checkpoint and ideally beats it.

## Failure / Stop Criteria

1. Frontier stalls below W20 after 15M steps.
2. Frontier stalls below W32 after 30M steps.
3. Inter-wave behavior remains clearly unlearned after 15M steps.
4. Training destabilizes with sustained KL above `0.03`, collapsing EV, or clear entropy collapse.

## Key Risks

1. Fresh-start cost from W1 delays direct comparison with late-wave baselines.
2. Even a small recurrent core may reduce sample efficiency early in climb.
3. If offline wrappers remain stateless, eval results for V38 would be invalid.
4. Recurrent PPO may become less stable than the flat V37 run if LR/epochs are too aggressive.

## Notes

- V38 is intentionally a clean architecture experiment, not a continuation of the live V37 run.
- The residual design keeps direct access to `v3.1` features to avoid forcing all information through recurrent state.
- Policy export remains unsupported for V38 recurrent checkpoints.
- Compare early frontier and inter-wave behavior against V37 before spending full budget.
- **Env scaling A/B test (from same checkpoint):** Tested 48 envs with both 1 epoch and 3 epochs against the baseline 16 envs / 3 epochs. Both 48-env configs showed slower frontier advancement (curriculum_frontier_max) per step and lower KL (~0.007 vs ~0.011 for 16 envs). 48 envs / 3 epochs followed the same trajectory as 48 envs / 1 epoch. Conclusion: 16 envs / 3 epochs has better sample efficiency for this architecture; sticking with it.
