# V47 Reward & Curriculum Reference

Sanity-check document for the V47 fresh RL run. V47 does not load a checkpoint. `trained_rollouts` starts at `0`,
observation and reward normalizers start fresh, and the rollout-dependent V44 shaping schedules start at their initial
values by design.

## Run Summary

V47 command:

```powershell
python -m tools.inferno_rl.train_gpu --curriculum-mode static --phase sweep --start-wave 49 --max-wave 66 --observation-version v4 --policy-arch flat_lstm_residual --lstm-hidden-size 128 --lstm-seq-len 16 --lstm-burn-in 8 --episode-mode full --n-envs 64 --n-steps 256 --batch-size 4096 --n-epochs 2 --lr 1.5e-4 --target-kl 0.02 --entropy-start 0.05 --entropy-end 0.002 --gamma 0.998 --gae-lambda 0.95 --vf-coef 0.25 --max-grad-norm 1.0 --normalize-obs --normalize-reward --actor-sizes 512,512 --critic-sizes 512,512 --save-dir models/V47 --log-dir logs/V47 --checkpoint-every 100 --timesteps 200000000 --device cuda --log-reward-terms
```

| Setting             | Value                | Notes |
|---------------------|----------------------|-------|
| init                | fresh from zero      | no `--load`, no reused normalizer stats |
| curriculum-mode     | `static`             | no adaptive controller |
| phase               | `sweep`              | failure-weighted start-wave sampling |
| episode-mode        | `full`               | per-wave attribution active |
| wave range          | `49-66`              | hardest regular RL waves in the current simulator |
| observation-version | `v4`                 | 504 public / 504 total |
| policy-arch         | `flat_lstm_residual` | actor/critic MLP + LSTM residual |
| lstm-hidden-size    | `128`                | recurrent hidden size |
| normalize-obs       | on                   | stats start fresh |
| normalize-reward    | on                   | stats start fresh |

Rollout size is `64 * 256 = 16,384` environment steps.

## Reward Space

### Active Dense Terms

| Category            | Value / behavior | Notes |
|---------------------|------------------|-------|
| Damage taken        | `-0.05 / HP`     | flat penalty |
| Damage dealt        | `+0.01 / HP`     | flat reward |
| Blood barrage heal  | `+0.06 / HP`     | only when healing |
| High-HP barrage     | `-0.2`           | guardrail penalty |
| Wave complete       | `+2.0`           | plus HP bonus up to `+1.5` |
| Inferno complete    | `0.0`            | disabled |
| Stall               | `-0.08` base     | after 15 disengaged ticks past 17-tick grace, then `-0.04` more per extra tick |
| Avoidable LOS       | `-0.01 / NPC`    | constant, no fade |
| Avoidable imminent  | `-0.01 / NPC`    | constant |
| Mager priority hit  | scheduled        | per non-mager alive when damaging a mager |
| Early mager kill    | `+0.6 + 0.15 * non_magers` | static bonus |
| Mager delay         | `-0.02`          | when a safely focusable mager is ignored |
| Resurrection        | `-0.6` / `-0.3`  | full for mager/ranger/blob, half for melee |
| NE pillar zone      | scheduled        | combat-position shaping |
| C tile              | scheduled        | ticks `0..4` only |
| Tile A              | scheduled        | between waves only |
| Pillar damage       | `-0.01 / HP`     | NE pillar gets `3x` weight |
| NPC proximity       | `-0.01`          | standing on or adjacent to dangerous NPC footprint |
| Invalid action      | `-0.1`           | non-attack invalid |
| Invalid attack      | `-0.05`          | attack invalid |

Kill rewards currently present in `KILL_REWARDS`:

| NPC            | Reward |
|----------------|--------|
| `MAGER`        | `0.6`  |
| `RANGER`       | `0.25` |
| `MELEE`        | `0.35` |
| `BLOB`         | `0.2`  |
| `BLOB_MAGE`    | `0.3`  |
| `BLOB_RANGE`   | `0.3`  |
| `BLOB_MELEE`   | `0.3`  |
| `BAT`          | `0.3`  |
| `NIBBLER`      | `0.25` |

Explicitly absent:

- no positive single-LOS reward
- no Multi-LOS penalty
- no terminal death penalty beyond episode termination
- no terminal timeout penalty beyond episode termination
- no kill rewards for `JAD`, `HEALER`, `ZUK`, or `ZUK_HEALER`
- `SURVIVAL_REWARD_PER_TICK` is defined but not currently applied in `_calculate_internal(...)`

### Rollout-Dependent Scheduled Terms

These are the terms that change as `trained_rollouts` increases.

| Term                     | Initial | Final  | Schedule |
|--------------------------|---------|--------|----------|
| Tile A max reward        | `0.04`  | `0.0`  | linear over 500 rollouts |
| C tile on                | `0.5`   | `0.0`  | linear over 500 rollouts |
| C tile adjacent          | `0.25`  | `0.0`  | linear over 500 rollouts |
| NE pillar zone bonus     | `0.008` | `0.002`| linear over 2000 rollouts |
| NE pillar zone penalty   | `-0.02` | `-0.005` | linear over 2000 rollouts |
| Mager priority bonus/NPC | `0.25`  | `0.125` | piecewise step at rollout 4000 |
| Avoidable LOS penalty    | `-0.01` | `-0.01` | constant |
| Avoidable imminent       | `-0.01` | `-0.01` | constant |
| Novelty scale            | `0.0003`| `0.0`  | linear over 300 rollouts |

Approximate step counts with 16,384 steps per rollout:

| Rollouts | Steps |
|----------|-------|
| 300      | `4,915,200` |
| 500      | `8,192,000` |
| 2000     | `32,768,000` |
| 4000     | `65,536,000` |

Practical implication for a fresh run:

- Tile A and C tile shaping are fully active at launch and fade out by about 8.2M steps.
- NE pillar zone shaping starts strong and decays slowly through about 32.8M steps.
- Mager-priority shaping stays at full strength until about 65.5M steps.
- Avoidable LOS penalties do not decay away.
- Novelty starts active and decays out by about 4.9M steps.

## Curriculum / Attribution

V47 uses static `sweep` with `episode-mode full`.

### Sweep behavior

Static sweep works like this:

1. Each env tracks its own per-wave success/fail table.
2. For the first 100 episodes per env, start waves are sampled uniformly from `49..66`.
3. After warmup, each env samples from `49..66` with weight `max(fail_rate, 0.02)`.
4. There is no frontier, promotion, prestige, or drill logic in this phase.
5. Static sweep does not globally merge wave stats across all workers. The per-wave weighting is local to each env/worker.

### Per-wave attribution

In `episode-mode full`:

- every cleared wave in the episode emits a success update
- death or timeout emits a fail update only for the terminal wave
- opener-only attribution is preserved for `episode-mode opener`, but V47 is not using opener mode

### W49-66 Composition Reference

All waves `49..66` in the current simulator still include 3 nibblers. The non-nibbler combat composition is:

| Wave | Combat composition |
|------|--------------------|
| 49   | `2 MELEE + MAGER` |
| 50   | `RANGER + MAGER` |
| 51   | `BAT + RANGER + MAGER` |
| 52   | `2 BAT + RANGER + MAGER` |
| 53   | `BLOB + RANGER + MAGER` |
| 54   | `BAT + BLOB + RANGER + MAGER` |
| 55   | `2 BAT + BLOB + RANGER + MAGER` |
| 56   | `2 BLOB + RANGER + MAGER` |
| 57   | `MELEE + RANGER + MAGER` |
| 58   | `BAT + MELEE + RANGER + MAGER` |
| 59   | `2 BAT + MELEE + RANGER + MAGER` |
| 60   | `BLOB + MELEE + RANGER + MAGER` |
| 61   | `BAT + BLOB + MELEE + RANGER + MAGER` |
| 62   | `2 BAT + BLOB + MELEE + RANGER + MAGER` |
| 63   | `2 BLOB + MELEE + RANGER + MAGER` |
| 64   | `2 MELEE + RANGER + MAGER` |
| 65   | `2 RANGER + MAGER` |
| 66   | `2 MAGER` |

This simulator does not use Jad / Triple Jad / Zuk boss-wave compositions for RL training waves `64..66`.

## Fresh-Start Sanity Checks

- `trained_rollouts` intentionally starts at `0`, so all scheduled shaping terms begin at their initial values.
- Observation normalization starts from empty running stats.
- Reward normalization starts from empty running stats.
- There is no BC behavior to preserve or destabilize because V47 is not a warmstart.
- If you want a less scaffolded run, you need to change the reward schedule inputs or hardcode later-stage reward values before launch.
