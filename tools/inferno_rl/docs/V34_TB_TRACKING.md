# V34 TB Tracking

## Goal

Single fast-path branch that keeps the effective V31 training regime and legacy climb behavior, then applies the full representation stack
in one pass:

- `target_kl=0.02`
- Observation V2
- `entity_pool_lstm`
- legacy climb sampling
- no burn-in

This branch is intentionally not an ablation. If it fails, the result is not diagnostic.

## Training Settings

| Setting              | Value             | Notes                                  |
|----------------------|-------------------|----------------------------------------|
| warmstart            | none              | Fresh init                             |
| phase                | climb             | Same as V31                            |
| start-wave           | 49                | Same as V31                            |
| max-wave             | 66                | Same as V31                            |
| promote-after        | 5                 | Same as V31                            |
| min-waves-to-advance | 1                 | Same as V31                            |
| climb-sampling       | legacy            | V31-style fixed `frontier-3` starts    |
| observation-version  | v2                | Structured global + entity-slot schema |
| policy-arch          | entity_pool_lstm  | Shared entity encoder + mean/max pool  |
| lstm-hidden-size     | 256               | Same recurrent width as V31            |
| lstm-seq-len         | 32                | Effective V31 setting                  |
| lstm-burn-in         | 0                 | Explicitly disabled                    |
| actor/critic sizes   | 512,512 / 512,512 | Same as V31                            |
| target-kl            | 0.02              | Early-stop PPO updates                 |
| n-envs               | 48                | Same as V31                            |
| n-steps              | 1024              | Same as V31                            |
| batch-size           | 2048              | Same as V31                            |
| n-epochs             | 2                 | Same as V31                            |
| lr                   | 1e-4              | Same as V31                            |
| entropy-start/end    | 0.02 / 0.002      | Same as V31                            |
| gamma                | 0.995             | Same as V31                            |
| gae-lambda           | 0.95              | Same as V31                            |
| vf-coef              | 0.5               | Same as V31                            |
| max-grad-norm        | 0.5               | Same as V31                            |
| normalize-reward     | yes               | Same as V31                            |
| normalize-obs        | yes               | Same as V31                            |

## Run Command

```powershell
python -m tools.inferno_rl.train_gpu --lstm-hidden-size 256 --lstm-seq-len 32 --lstm-burn-in 0 --phase climb --start-wave 49 --max-wave 66 --promote-after 5 --min-waves-to-advance 1 --climb-sampling legacy --observation-version v2 --policy-arch entity_pool_lstm --save-dir models/V34 --log-dir logs/V34 --n-envs 48 --n-steps 1024 --batch-size 2048 --n-epochs 2 --lr 1e-4 --entropy-start 0.02 --entropy-end 0.002 --gamma 0.995 --gae-lambda 0.95 --vf-coef 0.5 --max-grad-norm 0.5 --target-kl 0.02 --normalize-reward --normalize-obs --checkpoint-every 100 --timesteps 200000000 --actor-sizes 512,512 --critic-sizes 512,512 --device cuda --log-reward-terms
```

## Metrics Log

| Step   | Frontier | Deaths | Waves Comp | Mean Reward | EV   | KL    | Entropy | Grad Norm | FPS  | Notes                                                                                     |
|--------|----------|--------|------------|-------------|------|-------|---------|-----------|------|-------------------------------------------------------------------------------------------|
| 4.8M   | 52       | 190    | 375        | 0.86        | 0.68 | 0.009 | -2.78   | 0.40      | 4829 | Fresh init, frontier climbing, EV 0.68 (watch), KL healthy                                |
| 27.0M  | 59       | 46     | 292        | 2.90        | 0.58 | 0.009 | -2.43   | 0.53      | 4864 | Phase 1, frontier 59, **EV 0.58 (watch — below 0.60)**, deaths dropping                   |
| 51.5M  | 57       | 12     | 358        | 6.96        | 0.67 | 0.019 | -2.11   | 0.64      | 4932 | Phase 2 (level-up), frontier reset 59→57, deaths 12, KL 0.019 near target                 |
| 75.6M  | 50       | 5      | 381        | 7.59        | 0.74 | 0.013 | -2.10   | 0.52      | 4895 | Phase 3 (level-up), frontier reset 57→50, deaths 5 (excellent), EV recovered 0.74         |
| 100.7M | 57       | 12     | 366        | 8.09        | 0.69 | 0.012 | -2.01   | 0.52      | 5093 | Phase 3, frontier climbing 50→57, EV dipped 0.69 (watch)                                  |
| 125.0M | 65       | 12     | 356        | 8.74        | 0.80 | 0.014 | -1.62   | 0.68      | 4773 | Phase 3, frontier 65 (near ceiling), EV 0.80 (strong), entropy -1.62 accelerating decay   |
| 150.1M | 54       | 4      | 398        | 11.71       | 0.83 | 0.012 | -1.32   | 0.63      | 5009 | Phase 4 (level-up), frontier reset 65→54, deaths 4 (best yet), reward 11.7, entropy -1.32 |
| 174.7M | 56       | 10     | 371        | 5.79        | 0.78 | 0.015 | -0.91   | 0.91      | 4922 | Phase 4, frontier 56, entropy -0.91 **(collapse risk)**, grad norm 0.91 (above clip)      |

## Eval Results (100 seeds)

### W55-66 Full Eval (100 episodes, seed 0-99)

| Checkpoint | Steps    | Clear   | Death   | Timeout | Top Death Waves           | Notes                                                   | 
|------------|----------|---------|---------|---------|---------------------------|---------------------------------------------------------|
| 2600       | 128M     | 29%     | 70%     | 1%      | W65=12, W60=10, W55=7     | W65 worst.                                              |
| 2700       | 133M     | 32%     | 68%     | 0%      | W63=13, W65=12, W55=10    | 0 timeouts.                                             |
| **2800**   | **138M** | **36%** | **64%** | **0%**  | **W56=12, W62=10, W63=9** | **Best clear rate.** 0 timeouts. Lowest death rate.     |
| 2900       | 143M     | 30%     | 69%     | 1%      | W60=9, W63=9, W65=9       | Slight regression from peak. Deaths even spread W60-65. |
| 3000       | 147M     | 30%     | 70%     | 0%      | W65=12, W56=8, W62=8      | Flat. W65 re-emerged as problem wave.                   |
| 3300       | 162M     | 23%     | 71%     | 6%      | W63=13, W56=12, W60=9     | Major regression. 6 timeouts appeared.                  |
| 3500       | 172M     | 30%     | 69%     | 1%      | W61=12, W56=9, W65=8      | Partial recovery from 3300 dip.                         |

#### Per-Wave Death Distribution (deaths only, not timeouts)

| Wave | Ckpt 2600 | Ckpt 2700 | Ckpt 2800 | Ckpt 2900 | Ckpt 3000 | Ckpt 3300 | Ckpt 3500 |
|------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| 55   | 7         | 10        | 5         | 6         | 7         | 8         | 5         |
| 56   | 6         | 6         | 12        | 6         | 8         | 12        | 9         |
| 57   | 3         | 5         | 5         | 3         | 4         | 5         | 3         |
| 58   | 4         | 2         | 1         | 3         | 2         | 7         | 2         |
| 59   | 7         | 3         | 1         | 8         | 5         | 2         | 4         |
| 60   | 10        | 3         | 6         | 9         | 4         | 9         | 7         |
| 61   | 6         | 5         | 3         | 4         | 6         | 6         | 12        |
| 62   | 2         | 5         | 10        | 6         | 8         | 4         | 8         |
| 63   | 6         | 13        | 9         | 9         | 8         | 13        | 7         |
| 64   | 5         | 4         | 2         | 6         | 4         | 3         | 2         |
| 65   | 12        | 12        | 9         | 9         | 12        | 2         | 8         |
| 66   | 2         | 0         | 1         | 0         | 2         | 0         | 2         |

### Death Distribution Notes

When recording per-wave death counts, note explicitly whether the wave columns are:

- counts over death cases only, or
- percentages over all episodes

and record timeout separately if it uses a different denominator.

## Kill Criteria

Stop the branch immediately if any of these hold:

1. At ckpt `400`, `W55` is still `0%` clear and the failure mode is not better than the V31 baseline on timeout rate or death depth.
2. At ckpt `800`, `W55` is still `0%` clear.
3. At ckpt `800`, `W61` is still `0%` clear and deaths are concentrated at the entry wave.
4. Training degrades for `10M+` steps with all of:
    - EV baseline `< 0.60`
    - rising KL toward the cap
    - rising deaths
    - stalled frontier
