# V33 TensorBoard Tracking

V33 isolates the curriculum variable. Keep the V32 training stack (LSTM burn-in,
target KL early-stop, pre-LSTM input encoder, `lstm_seq_len=16`) but revert climb
sampling to the V31 behavior (`frontier-3` fixed non-refresh starts).

## What Changed (V32 -> V33)

### Training Stack

Unchanged from V32:

- `lstm-seq-len = 16`
- `lstm-burn-in = 8`
- `target-kl = 0.02`
- pre-LSTM input encoder enabled
- actor / critic `512,512`
- LSTM hidden `256`

### Curriculum

Only one change:

- `climb_sampling = legacy`

This restores the V31 climb curriculum for non-refresh episodes:

- every normal climb episode starts at `max(start_wave, frontier - 3)`
- every 5th climb episode is still a refresh below frontier (same climb phase behavior)

This removes the V32 weighted frontier bias that overfit the training task and failed to
transfer in eval.

## Hypothesis

If V32 failed mainly because the climb task diverged from the eval task, V33 should recover
W55 transfer while retaining the optimization improvements from the current recurrent stack.

If V33 still underperforms V31 in early eval, the culprit is not just curriculum, and the
stack changes need to be ablated one by one.

## Training Settings

| Setting              | V32      | V33        | Rationale                                    |
|----------------------|----------|------------|----------------------------------------------|
| warmstart            | none     | **none**   | Fresh init; isolate this config cleanly      |
| phase                | climb    | climb      | Same                                         |
| climb-sampling       | weighted | **legacy** | Revert to V31 fixed `frontier-3` non-refresh |
| start-wave           | 49       | 49         | Same                                         |
| max-wave             | 66       | 66         | Same                                         |
| promote-after        | 5        | 5          | Same                                         |
| min-waves-to-advance | 1        | 1          | Same                                         |
| lstm-seq-len         | 16       | 16         | Keep shorter recurrent horizon               |
| lstm-burn-in         | 8        | 8          | Keep warm hidden-state reconstruction        |
| target-kl            | 0.02     | 0.02       | Keep PPO update guardrail                    |
| n-epochs             | 2        | 2          | Same                                         |
| n-envs               | 48       | 48         | Same                                         |
| batch-size           | 2048     | 2048       | Same                                         |
| lr                   | 1e-4     | 1e-4       | Same                                         |
| entropy-start        | 0.02     | 0.02       | Same                                         |
| entropy-end          | 0.002    | 0.002      | Same                                         |
| normalize-reward     | yes      | yes        | Same                                         |
| normalize-obs        | yes      | yes        | Same                                         |
| n-steps              | 1024     | 1024       | Same                                         |
| gamma                | 0.995    | 0.995      | Same                                         |
| gae-lambda           | 0.95     | 0.95       | Same                                         |
| vf-coef              | 0.5      | 0.5        | Same                                         |
| max-grad-norm        | 0.5      | 0.5        | Same                                         |

## Run Command

```powershell
python -m tools.inferno_rl.train_gpu `
  --phase climb --climb-sampling legacy --start-wave 49 --max-wave 66 `
  --lstm-hidden-size 256 --lstm-seq-len 16 --lstm-burn-in 8 `
  --n-envs 48 --n-steps 1024 --batch-size 2048 --n-epochs 2 `
  --lr 1e-4 --target-kl 0.02 `
  --entropy-start 0.02 --entropy-end 0.002 `
  --gamma 0.995 --gae-lambda 0.95 `
  --vf-coef 0.5 --max-grad-norm 0.5 `
  --normalize-obs --normalize-reward `
  --actor-sizes 512,512 --critic-sizes 512,512 `
  --save-dir models/V33_climb --log-dir logs/V33_climb `
  --checkpoint-every 100 --timesteps 200000000 --device cuda --log-reward-terms
```

## Eval Cadence

Do not wait until 200M to validate transfer.

- Run W55 eval every `5-10M` steps from the latest checkpoint.
- Keep the branch only if W55 clear is above V32's `0%` baseline by `20-30M`.
- Compare only against V31 `R1-1000` / `R1-2000` pace, not against V32.

## Kill Criteria

Kill the run early if any of these persist for more than one eval window:

1. W55 clear remains `0%`.
2. W55 deaths cluster at W55-W56 with rising timeout share.
3. Frontier advances in rollout but W55 sequential eval does not improve.
4. EV drops below `0.60` while KL and deaths trend upward for `10M+` steps.

## Metrics Log

| Step | Frontier | Deaths | Waves Comp | Mean Reward | EV   | KL    | Entropy | Grad Norm | FPS  | Notes                                                                                                                                                           |
|------|----------|--------|------------|-------------|------|-------|---------|-----------|------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1.4M | 50       | 402    | 114        | -0.14       | 0.80 | 0.008 | -3.33   | 0.32      | 5623 | First entry. Phase 1, frontier 49→50. EV 0.80 solid. KL 0.008 well under target 0.02. V31 @ 2.6M: frontier 51, deaths 236, EV 0.64 — V33 EV notably better. |
| 6.8M | 53       | 175    | 411        | 1.19        | 0.74 | 0.009 | -2.72   | 0.50      | 6048 | Frontier 50→53. Deaths 402→175 (-56%), waves 114→411 (+261%). Reward flipped positive. EV 0.80→0.74. V31 @ 7.8M: frontier 53 (same), deaths 192 (similar), EV 0.62 — V33 critic healthier (+0.12). |
| 12.1M | 54       | 196    | 389        | 1.41        | 0.73 | 0.008 | -2.75   | 0.50      | 5552 | Frontier 53→54 (+1 in 5.3M steps — slow). Deaths 175→196, waves 411→389. EV/KL/entropy/grad all stable. Still phase 1. V31 @ 17.7M: frontier 55 — V33 slightly behind but EV 0.73 vs V31's 0.68. |
| 20.4M | 55       | 163    | 369        | 1.68        | 0.75 | 0.007 | -2.65   | 0.48      | 5890 | Frontier 54→55. Deaths 196→163, reward 1.41→1.68. EV 0.73→0.75 (healthy). Still phase 1. V31 @ 17.7M: frontier 55, deaths 132, EV 0.68 — V33 matched frontier, more deaths but better EV (+0.07). |
| 38.1M | 61       | 89     | 314        | -0.65       | 0.34 | 0.019 | -2.40   | 0.54      | 5905 | ~~Frontier 55→61 (+6). Deaths 163→89. EV 0.34 spike (see next row).~~ |
| 38.6M | 61       | 86     | 325        | 1.84        | 0.65 | 0.011 | -2.43   | 0.65      | 5865 | EV spike recovered: 0.34→0.65. KL 0.019→0.011. Reward -0.65→1.84. Frontier held 61, f.mean 56.9. Deaths 89→86. Still phase 1. V31 @ 38M: EV ~0.68 — V33 close but slightly below. |
| 48.6M | 61       | 105    | 308        | 1.29        | 0.73 | 0.016 | -2.29   | 0.68      | 4915 | Frontier stuck at 61 for 10M steps. Deaths 86→105 (+22%), waves 325→308. EV recovered 0.65→0.73. f.mean 57.6. Still phase 1. Entropy -2.43→-2.29 (slightly less deterministic). |

## Eval Results (100 seeds per start wave)

| Checkpoint | Steps | W49 Clear | W49 Death | W55 Clear | W55 Death | W63 Clear | W63 Death |
|------------|-------|-----------|-----------|-----------|-----------|-----------|-----------|
| 400        | 19.7M | —         | —         | 0%        | 96%       | —         | —         |
| 500        | 24.6M | —         | —         | 0%        | 97%       | —         | —         |
| 700        | 34.4M | —         | —         | 0%        | 100%      | —         | —         |
| 1000       | 49.2M | —         | —         | 0%        | 99%       | —         | —         |

### W55 Death Distribution

| Ckpt | W55 | W56 | W57 | W58 | W59 | W60 | W61 | W62 | Timeout |
|------|-----|-----|-----|-----|-----|-----|-----|-----|---------|
| 400  | 41  | 48  | 6   | 3   | 2   | —   | —   | —   | 4%      |
| 500  | 26  | 40  | 21  | 9   | 3   | 1   | —   | —   | 3%      |
| 700  | 15  | 44  | 18  | 10  | 6   | 6   | —   | 1   | 0%      |
| 1000 | 18  | 46  | 12  | 11  | 5   | 4   | 2   | 1   | 1%      |

Wave columns are counts across the death cases only (they sum to 100 here). `Timeout` is the
share of all eval episodes, not part of the death-case breakdown.

**Analysis:** 0% clear across all checkpoints. Deaths spreading deeper over time: W55 deaths
dropping (41→26→15), W60+ appearing at ckpt 700 (6 deaths + 1 at W62). Zero timeouts at
ckpt 700 — fully aggressive play. V31 at ckpt 700: 66% death, 34% timeout, 92/100 stuck
on W55. V33 dramatically ahead in combat depth.

**Ckpt 1000 (49.2M):** Still 0% clear but death distribution stable — W56 remains the
primary wall (46 deaths). W61-W62 deaths appearing (3 total). V31 R1-1000 fresh re-eval
(same eval code): 0% clear, 54% death, 46% timeout — 88/100 stuck on W55. V33 pushes
far deeper (deaths spread W55-W62) but can't close waves. Combat aggression is high
(1% timeout vs V31's 46%) but survival is worse.
