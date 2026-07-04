# V40 TB Tracking

## Goal

Test whether enriching the observation with pre-computed pillar-relative features, per-slot blob scan state, and a global mager-alive flag
reduces learning burden and improves tactical positioning compared to V39's observation baseline.

## Base Checkpoint

- Base checkpoint: **none (fresh start)**
- Start point: **W35**
- Note: V40 changes the observation shape from `267 -> 317`, so a direct warmstart from V39 is not a clean comparison.

## What Changed (V39 -> V40)

### 1. Observation Enrichment: 267-dim -> 317-dim (+50)

Four new feature groups added to the existing V3.2 observation schema:

#### Player-pillar dx/dy (replaced `in_ne_zone`)

The single binary `in_ne_zone` feature (index 22) is replaced with two continuous signed features:

- `(player_x - 18) / GRID_WIDTH` — player's horizontal offset from NE pillar center
- `(player_y - 23) / GRID_HEIGHT` — player's vertical offset from NE pillar center

This gives the policy a gradient signal for pillar-relative positioning instead of a coarse 3-tile binary zone.

#### Mager alive flag (global, index 49)

Binary flag at the end of the global extension block. `1.0` if any mager is alive, `0.0` otherwise.
Lets the policy cheaply gate mager-specific behavior (prayer priority, pillar-side selection) without scanning all entity slots.

#### Per-slot pillar-relative dx/dy (offsets +12, +13)

Each static NPC slot now includes `(closest_x - 18) / GRID_WIDTH` and `(closest_y - 23) / GRID_HEIGHT`.
Previously the policy had to reconstruct NPC-pillar geometry from player position + player-relative NPC offset — a 3-step
inference chain. Now it's a direct lookup.

#### Per-slot blob scan state (offsets +14, +15)

Two binary features per slot: `scanned_magic` and `scanned_ranged`. Only populated for blob-type entities; zero for all others
(consistent with `dig_pressure` being melee-only). Previously blob scan state was only available as aggregate counts in the
global block — now the policy can distinguish which specific blob is scanned and what prayer it targets.

### 2. Observation Layout Shift

```
Old (267):  Global(48) + Safety(30) + Threat(9) + Direction(24) + Nibbler(5) + Temporal(7) + Slots(6x2x12=144)
New (317):  Global(50) + Safety(30) + Threat(9) + Direction(24) + Nibbler(5) + Temporal(7) + Slots(6x2x16=192)
```

Safety map, threat horizon, directional forecast, nibbler summary, and temporal blocks are unchanged in content — only
their starting indices shift by +2 due to the expanded global block.

### 3. Architecture, Rewards, Curriculum Held Constant

Same `flat_lstm_residual` policy, same reward setup, same climb-style curriculum. The only variable is the observation enrichment.

## V40 Hypothesis

V39 showed the policy can learn multi-wave tactics with V3.2's forecast features. But the MLP still has to reconstruct
pillar-relative NPC geometry through multi-step inference (absolute NPC pos from player + offset, then compare to pillar center).
This is learnable but sample-inefficient and fragile.

V40 should:

- Learn pillar-side movement decisions faster (direct pillar-relative features vs. implicit reconstruction)
- Make better blob prayer-switch decisions per-entity rather than relying on aggregate scan counts
- Gate mager-specific tactics more reliably with the explicit alive flag

If the hypothesis is correct, V40 should reach equivalent frontier waves in fewer steps than V39, particularly through
the W50-66 range where pillar positioning and blob handling become critical.

## Training Settings

| Setting             | Value                | Notes                                 |
|---------------------|----------------------|---------------------------------------|
| warmstart           | `none (fresh)`       | Observation shape changed             |
| curriculum-mode     | `static`             | Same as V39                           |
| phase               | `climb`              | Same as V39                           |
| climb-sampling      | `weighted`           | Same as V39                           |
| promote-after       | `5`                  | Same as V39                           |
| start-wave          | `35`                 | Same as V39                           |
| max-wave            | `66`                 | Full Inferno                          |
| observation-version | `v3.2`               | Same tag, enriched content (317 dims) |
| policy-arch         | `flat_lstm_residual` | Same as V39                           |
| lstm-hidden-size    | `128`                | Same as V39                           |
| lstm-seq-len        | `16`                 | Same as V39                           |
| lstm-burn-in        | `8`                  | Same as V39                           |
| actor/critic sizes  | `512,512 / 512,512`  | Same head sizes as V39                |
| n-envs              | `16`                 | Same as V39                           |
| n-steps             | `1024`               | Same as V39                           |
| batch-size          | `2048`               | Same as V39                           |
| n-epochs            | `3`                  | Same as V39                           |
| lr                  | `2e-4`               | Same as V39                           |
| target-kl           | `0.015`              | Same as V39                           |
| entropy-start/end   | `0.05 / 0.002`       | Same as V39                           |
| gamma               | `0.995`              | Same                                  |
| gae-lambda          | `0.95`               | Same                                  |
| vf-coef             | `0.5`                | Same                                  |
| max-grad-norm       | `0.5`                | Same                                  |
| normalize-reward    | yes                  | Same                                  |
| normalize-obs       | yes                  | Same                                  |
| checkpoint-every    | `100`                | Same                                  |
| total budget        | `200M`               | Same first-pass budget as V39         |

## Run Command

```powershell
python -m tools.inferno_rl.train_gpu --curriculum-mode static --phase climb --climb-sampling weighted --promote-after 5 --start-wave 35 --max-wave 66 --observation-version v3.2 --policy-arch flat_lstm_residual --lstm-hidden-size 128 --lstm-seq-len 16 --lstm-burn-in 8 --episode-mode full --n-envs 16 --n-steps 1024 --batch-size 2048 --n-epochs 3 --lr 2e-4 --target-kl 0.015 --entropy-start 0.05 --entropy-end 0.002 --gamma 0.995 --gae-lambda 0.95 --vf-coef 0.5 --max-grad-norm 0.5 --normalize-obs --normalize-reward --actor-sizes 512,512 --critic-sizes 512,512 --save-dir models/V40_obs32_enriched --log-dir logs/V40_obs32_enriched --checkpoint-every 100 --timesteps 200000000 --device cuda
```

### Phase 2: 1-epoch + 48-env (from ~10M steps)

Changes from phase 1: `n-envs` 16→48, `n-epochs` 3→1, `batch-size` 2048→4096, `target-kl` 0.015→0.02. Warmstarting from ckpt 609.

```powershell
python -m tools.inferno_rl.train_gpu --load models/V40_obs32_enriched/inferno_gpu_w35-66_20260312_142707_600.pt --curriculum-mode static --phase climb --climb-sampling weighted --promote-after 5 --start-wave 35 --max-wave 66 --observation-version v3.2 --policy-arch flat_lstm_residual --lstm-hidden-size 128 --lstm-seq-len 16 --lstm-burn-in 8 --episode-mode full --n-envs 48 --n-steps 1024 --batch-size 4096 --n-epochs 1 --lr 2e-4 --target-kl 0.02 --entropy-start 0.05 --entropy-end 0.002 --gamma 0.995 --gae-lambda 0.95 --vf-coef 0.5 --max-grad-norm 0.5 --normalize-obs --normalize-reward --actor-sizes 512,512 --critic-sizes 512,512 --save-dir models/V40_obs32_enriched --log-dir logs/V40_obs32_enriched --checkpoint-every 100 --timesteps 200000000 --device cuda
```

## Files Changed

| File                                              | Changes                                                                                          |
|---------------------------------------------------|--------------------------------------------------------------------------------------------------|
| `tools/inferno_rl/training/observation_common.py` | `GLOBAL_V4_SIZE` 48->50, `STATIC_SLOT_SIZE` 12->16 (total 267->317)                              |
| `tools/inferno_rl/training/observation_v3.py`     | Player-pillar dx/dy replaces `in_ne_zone`, mager_alive flag, per-slot pillar dx/dy and blob scan |
| `tools/inferno_rl/tests/test_observation_v32.py`  | Updated layout assertions, added tests for all 4 new feature groups                              |
| `tools/inferno_rl/docs/OBSERVATION_SPACE.md`      | Full index table update reflecting 317-dim layout                                                |

## Metrics Log

| Ckpt | Steps | Frontier | Promo% | EV    | Entropy | KL    | Grad | Ep Len | FPS  | Notes                                                                                                                                         |
|------|-------|----------|--------|-------|---------|-------|------|--------|------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| ~42  | 0.7M  | 39       | 36%    | -0.32 | 0.050   | 0.007 | 0.33 | 546    | 1949 | First log. Fresh W35 start, frontier W39 at 0.7M. EV negative (expected early). max_wave_from_39=40.                                          |
| ~238 | 3.9M  | 46       | 35%    | 0.80  | 0.049   | 0.011 | 0.43 | 267    | 1641 | Frontier W39→W46 (+7). EV jumped to 0.80. Early stop firing (KL=0.018). max_wave_from_46=48. Return positive (0.06). FPS -16% (harder waves). |
| ~609 | 10.0M | 60       | 21%    | 0.77  | 0.048   | 0.009 | 0.40 | 231    | 1462 | Frontier W46→W60 (+14!). max_wave_from_60=63. Promo rate dropped to 21% (harder waves). Early stop off, full 3 epochs. Return 0.11. FPS -11%. |
| ~648 | 12.2M | 42       | 87%    | 0.75  | 0.049   | 0.005 | 0.30 | 543    | 2276 | Phase 2 (48-env, 1-epoch, batch 4096, target-kl 0.02). Frontier reset to W42 (curriculum re-climb from warmstart). Promo 87% — blazing through early waves. max_wave_from_42=49. Return 0.32. FPS 2276 (48 envs + easy waves). |
| ~646 | 12.1M | 47       | 42%    | 0.73  | 0.050   | 0.006 | 0.32 | 336    | 2168 | Run 3. Frontier W42→W47 (+5). Promo 42% (30/72). Early stop firing. LR bumped to 3e-4. max_wave_from_47=53. Return 0.25. Deaths 85. |

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

1. Frontier reaches at least W49 by 15M steps (V39 reached W48 at 6.1M, W59 at 8.9M — V40 should be competitive).
2. Frontier reaches W59+ by 25M steps.
3. Manual review shows cleaner pillar-side movement decisions than V39 at equivalent step counts.
4. Broad `W49-66` eval at least matches V39's best checkpoint.

## Failure / Stop Criteria

1. Frontier stalls below W45 after 15M steps.
2. Frontier stalls below W55 after 30M steps.
3. Training destabilizes with sustained KL above `0.03`, collapsing EV, or entropy collapse.
4. No measurable improvement over V39 at equivalent step counts after 50M steps.

## Key Risks

1. Fresh-start cost may hide the real value of the enrichment for a long time — the 50 extra dims add noise early on.
2. Pillar-relative features may be partially redundant with what the LSTM already reconstructs, yielding marginal gain.
3. The larger observation (317 vs 267) increases the input projection cost slightly, though `flat_lstm_residual` should handle it.
4. Per-slot blob scan features are sparse (only 2 of 12 slot types use them) — the policy may not learn to use them efficiently.

## Notes

- V40 is a pure observation enrichment experiment on top of V39's architecture.
- The cleanest comparison is step-for-step frontier progression: V39 vs V40 at the same step count.
- Manual replay review should focus on:
    - pillar-side movement when magers are east vs west of pillar
    - blob scan response per-entity (does the policy react differently to magic vs ranged blobs?)
    - mager-alive gated behavior (does prayer priority change when the last mager dies?)
    - player positioning gradient near the NE pillar (smooth repositioning vs binary zone-based jumps)
- Policy export remains unsupported for recurrent checkpoints.
