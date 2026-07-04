# Observation Space Reference

Observation version `v4` is a 602-dimensional flat observation. Box low = `-1.0`, box high = `1.0`.

- Actor input: 602 dims
- Critic input: 602 dims
- Privileged critic-only block: removed

Ground truth for all sizes: `training/observation_common.py` (`OBSERVATION_TOTAL_SIZE`).

Source files:

| File                             | Role                                        |
|----------------------------------|---------------------------------------------|
| `training/observation.py`        | Public facade (`build_observation`)         |
| `training/observation_common.py` | Layout constants and shared helpers         |
| `training/observation_v4.py`     | Builder implementation                      |
| `simulator/exact_targeting.py`   | Shared exact-target ordering and slot count |
| `simulator/forecast.py`          | Neighborhood forecast and threat horizon    |

## Layout

| Range       | Block                 | Size | Notes |
|-------------|-----------------------|------|-------|
| `0..50`     | Global                | 51   | Player, pillar, wave, dead-pool, prayer, and loadout-agnostic context |
| `51..158`   | Neighborhood forecast | 108  | 9 tiles x 12 features |
| `159..167`  | Threat horizon        | 9    | 3 ticks x 3 styles |
| `168..174`  | Temporal              | 7    | Rolling damage and previous-action context |
| `175..594`  | Exact target slots    | 420  | 14 slots x 30 features |
| `595..601`  | Loadout               | 7    | Informative loadout-only block |

## Global Block

The 51-dim global block contains:

- Player position, HP ratio, and attack cooldown
- Current weapon one-hot (`BOFA`, `BLOWPIPE`, `MAGE_NO_BLOOD`, `MAGE_BLOOD`)
- Active prayer one-hot
- Pillar alive / HP state for all 3 pillars
- Current wave and ticks in wave
- Nibbler count
- Split-blob count
- Player offset from the NE pillar
- Current imminent threat count and player attack range
- Between-wave flag
- Wave kill counts for `BAT`, `BLOB`, `MELEE`, `RANGER`, `MAGER`
- Scanned-blob counts for magic, ranged, and imminent blobs
- Wave spawn timer
- Queued prayer one-hot
- Nibbler counts targeting NW / NE / S pillars
- Dead-pool counts for `BAT`, `BLOB`, `MELEE`, `RANGER`, `MAGER`
- Resurrection hazard and `mager_alive`
- Attack-target-alive flag (queued attack target exists and is alive)

## Neighborhood Forecast

Produced by `forecast_neighborhood_safety(...)`.

Tile order:

1. Stay
2. N
3. S
4. E
5. W
6. NE
7. NW
8. SE
9. SW

Per tile, the feature order is:

1. `settled_step_distance`
2. `los_count`
3. `los_delta`
4. `min_attack_delay`
5. `imminent_magic`
6. `imminent_ranged`
7. `imminent_melee`
8. `unprotected_after_auto_prayer`
9. `blob_scan_triggers`
10. `priority_target_attackable`
11. `best_los_in_2_steps`
12. `steps_to_single_los`

Notes:

- Forecasting is one tick ahead.
- NPC timers are decremented before movement prediction.
- `unprotected_after_auto_prayer` uses the same one-tick prayer predictor as the simulator.
- `priority_target_attackable` is evaluated against the queued attack target if alive, else the first exact-target slot.
- Blocked directions that settle back onto the current tile are zero-filled.

## Threat Horizon

Produced by `forecast_threat_styles(...)`.

- 3 forecast ticks
- 3 style counts per tick: magic, ranged, melee
- Normalized counts only

## Temporal

The 7-dim temporal block contains:

1. `damage_taken_5tick_sum`
2. `damage_dealt_5tick_sum`
3. `ticks_since_last_attack`
4. `ticks_since_engagement`
5. `prev_action_movement`
6. `prev_action_attack`
7. `prev_action_weapon_switch`

## Exact Target Slots

The entity section is a flat exact-target block shared with the simulator targeting order.

- Slot count: 14
- Slot size: 30 dims
- Slot shape: 21-dim core + 9-dim entity-type one-hot
- No typed support buckets
- No overflow block

Ordering:

- All alive non-nibbler combat entities sorted by `combat_entity_sort_key(...)`
- Then alive nibblers sorted by pillar urgency and distance
- The first 14 entities are retained

Entity-type one-hot order:

1. `MAGER`
2. `RANGER`
3. `MELEE`
4. `BLOB`
5. `BLOB_MAGE`
6. `BLOB_RANGE`
7. `BLOB_MELEE`
8. `BAT`
9. `NIBBLER`

Slot core feature order:

1. `exists`
2. `closest_dx`
3. `closest_dy`
4. `distance`
5. `hp_ratio`
6. `signed_attack_delay`
7. `stunned_ratio`
8. `frozen_ratio`
9. `npc_to_player_los`
10. `player_to_npc_los`
11. `dig_pressure`
12. `adjacent`
13. `pillar_dx` (offset from NE pillar center)
14. `pillar_dy` (offset from NE pillar center)
15. `blob_scanned_magic`
16. `blob_scanned_ranged`
17. `can_attack_now` (cooldown 0, in range, player has LOS)
18. `is_current_attack_target`
19. `ne_angular_diff` (player vs NPC angle around NE pillar; 0 = same face, 1 = opposite)
20. `npc_angle_sin`
21. `npc_angle_cos`

## Loadout Block

The loadout block is 7 dims.

Feature order:

1. `has_blowpipe`
2. `ranged_attack_speed`
3. `ranged_attack_range` (current weapon's range, updates on weapon switch)
4. `ranged_attack_bonus`
5. `ranged_strength_bonus`
6. `mage_attack_bonus`
7. `max_health`
