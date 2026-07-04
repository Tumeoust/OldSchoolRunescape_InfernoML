
## Data Pipeline

- **Capture**: Runelite Plugin that records per-tick JSONL traces
  (player/NPC positions, LOS, attack state, pillars) using real game collision flags.
- **Analysis**: `tools/inferno_rl/analyze_trace.py` — Loads a trace and replays each tick through the
  simulator's geometry, pathfinding, and LOS to find disagreements.

## Validation Run: 2026-03-31

484 ticks, waves 0/6-9. NPC types: BAT, BLOB, BLOB splits, MELEE, NIBBLER.

### Results

| Check               | Agree | Disagree | Accuracy |
|----------------------|-------|----------|----------|
| LOS (pure visibility)| 911  | 30       | 96.8%    |
| can_attack (LOS+range)| 923 | 18       | 98.1%    |
| NPC movement         | ~400 | 26       | ~94%     |
| Attack timing        | 73   | 1        | 98.6%    |

### Issue 1: Bresenham LOS Algorithm (30 LOS + 18 can_attack disagreements)

**All disagreements are the sim being too restrictive** — it blocks LOS that the real game allows.
Every case involves rays grazing NE pillar corners.

**Root cause**: The sim uses fixed-point Bresenham (ported from osrs-sdk). The real game behavior
matches standard integer Bresenham. The two trace different paths at obstacle edges:

```
Example: NPC (17,17) -> player (16,26), NE pillar at (17-19, 22-24)

Fixed-point (sim):  steps Y-major, at y=22 x is still 17 -> hits pillar (17,22) -> BLOCKED
Standard integer:   diagonal step (17,21)->(16,22), dodges pillar             -> PASSES
```

There are also ~5 reverse cases where the sim is too permissive (BLOB_MAGE at (17,21) to player
(16,24): sim's ray misses pillar, standard Bresenham hits it). The algorithms diverge in both
directions depending on the ray angle.

**Impact on training**: The sim gives the agent a wrong model of which positions are safe near
pillar edges. ~20 of the 26 movement disagreements are downstream of wrong LOS -> wrong
move/stay decisions.

**Fix**: Replace fixed-point Bresenham in `simulator/geometry.py` with standard integer Bresenham
matching the game client. This would resolve ~48 of the 75 total disagreements.

### Issue 2: Movement Near Pillar Collision Boundaries (~6 true pathfinding disagreements)

After removing LOS-derived errors, ~6 real pathfinding disagreements remain:

- BLOB at (7,8) moved diagonally through a position where the sim's 3x3 S-pillar collision blocks
  it. The game allowed it — pillar collision tiles may be slightly more permissive than the sim's
  model, or the pillar walkability shape isn't a perfect 3x3.
- Several sub-blob disagreements near NE pillar (15-17, 21-24), same pattern.

### Issue 3: Blob Attack Timing (Missing Fast-Attack Path)

Observed BLOB intervals: [5, 10, 5, 6, 25, 6, 6, 5, 4, 6, 5, ...] (configured speed=3).
BAT intervals: [3, 3, 3, 3, 5, 6, 3, 3] (configured speed=3, matches).

Real game blob behavior:
- **With continuous LOS and stored scan**: scan (delay=3) + attack (delay=3) = **6-tick** cycle.
- **Without stored scan** (e.g. just gained LOS): attacks **1 tick faster** = **5-tick** interval.
- Intervals >6 come from LOS loss between scan and attack phases.

The sim does NOT implement the 1-tick-faster path. Both scan and attack always set
`attack_delay = 3` (`npc_combat.py:150,158`), producing a rigid 6-tick cycle regardless
of scan state. The real game's 5-tick intervals when no scan is stored are not modeled.

**Fix**: When the blob has no stored scan and gains LOS, set `attack_delay = 2` instead of 3
(or equivalent logic) so the first attack after scan arrives 1 tick sooner.

### Blob Splits

4 splits observed, all produced exactly 3 sub-blobs. Spawn positions consistent with the sim's
directional pattern (north/northeast from NE pillar).

## Coverage Gaps

This trace only covers early waves (6-9) with BAT, BLOB, MELEE, NIBBLER. Future traces needed:

- Waves 35+: RANGER, MAGER attack timing and LOS behavior
- Waves 67+: JAD multi-style attacks
- Longer sessions for more pillar-edge LOS samples
- Sessions with pillar destruction to test dead-pillar LOS changes
