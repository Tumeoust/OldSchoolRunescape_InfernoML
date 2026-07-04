# Inferno Tactics Reference

Extracted from 5 expert video guides. Focused on NE pillar play with BoFa/Blowpipe/Ice Barrage/Blood Barrage.
Prayer is handled perfectly by heuristic — but two NPCs attacking on the same tick is unblockable.

---

## 1. Kill Priority Rules

### Mager First (When Possible)

- **Mager is the priority target** — it can revive any killed NPC (except nibblers) at half HP near the arena center. Killing other NPCs first risks pillar stacks (revived NPC spawns behind mager, requiring off-ticking).
- However, **don't force mager-first if it means dying.** If the mager is out of range, behind pillar, or reaching it requires standing in multi-LOS, deal with immediate threats first (melee about to dig, blob in melee range, etc.). Surviving is more important than kill order.

### Melee Second (Dig Timer Pressure)

- Melees die fast (low HP) but have a ~50-tick dig timer. If not dealt with, they teleport to the player and force repositioning.
- Kill melee before ranger/blob when mager is dead.
- When mager is alive: kill mager first, then melee. Exception: pick off melee if it's about to dig and mager is high HP.

### Ranger — Hold If Possible

- **Avoid killing rangers before magers.** Killing a ranger causes it to respawn behind the mager, creating a pillar stack. "Try not to kill
  rangers first ever." (V5)
- Rangers are good blood barrage targets for healing — another reason to keep them alive.
- Exception: W65 (double ranger + mager) — kill the ranger that won't create a stack.

### Blob — Kill After Mager

- **Never kill a blob before the mager.** If the mager revives a blob, all three bloblets also come back — massive HP to re-deal.
- Bats can be killed freely before mager — "killing bats isn't a big deal, who cares if those respawn." (V3)

### Nibblers — Kill When Safe

- Late waves (50+): lower priority than solving the wave. "Solve the wave first then kill the nibblers." (V4)
- One nibbler on a pillar is not urgent. Two or three is a problem.
- Pillar health is a resource — protect pillars early (easy waves) so you can afford to ignore nibblers on hard waves.

### Summary

```
With mager alive:  Mager > Melee > Bats > Blob > Ranger (hold ranger)
Without mager:     Melee > Ranger > Bats > Blob
Nibblers:          Kill when safe/free, ignore on hard waves
```

---

## 2. Pillar Play & Positioning

### Default Position — North of NE Pillar

- After freezing nibblers at wave start, move north of NE pillar to assess the wave.
- Monsters east of the pillar get stuck in the corner, giving time to plan.

### Corner Trapping

- **Southwest tile brain rule:** A monster's pathfinding originates from its southwest tile. If the SW tile can see you but the path is
  blocked by pillar, the monster is trapped.
- **Melee pathing priority:** East/west before north/south. Melees trying to go east into the pillar get stuck, creating a corner trap.
- All four sides of the pillar support corner traps.
- **Mager as pillar:** The mager's 3x3 size can be used as a blocking obstacle, same as the pillar. "The mage is also a pillar." (V5)

### Space Past the Mager

- When a mager is positioned on one side of the pillar, you can go past it. The mager blocks melee from following. "The melee can't come
  through because the mage is too far down." (V5)

### Pulling/Dragging NPCs

- Step briefly toward a side to pull NPCs behind the pillar (out of LOS).
- Step south of pillar to drag mager into attack range.
- Avoid moving while killing — dragging extends kill time.

---

## 3. Melee Dig Mechanics

- **Dig timer: ~50 ticks.** If melee hasn't hit you in 50 ticks, it teleports adjacent to you.
- **Timer resets on hit:** Taking a melee hit resets the dig timer. Sometimes intentionally tanking a hit is correct to prevent a disruptive
  dig.
- **Dig destination:** Melee tries to align its SW tile with you.
- **Corner trap after dig:** If you're against the pillar when it digs, it lands adjacent and can be immediately corner trapped.

---

## 4. Weapon Selection Rules

### BoFa — Primary Weapon

- Use on: magers, melees, high-HP targets. 10-tile range.
- Standing one tile closer to target reduces arrow travel time.

### Blowpipe — Close Range / Low HP

- Use on: bats, rangers, bloblets, low-HP targets.
- **HP cutoff: 34.** Below 34 HP, blowpipe is faster than BoFa. "34 is the cutoff." (V3)
- Blowpipe is BiS on rangers — they have low defense. "Pipe is best in slot on rangers." (V4)
- **Blowpipe spec is a kill button** — use to burst down dangerous NPCs fast, healing is a bonus.

### Ice Barrage — Wave Start / Freeze

- Cast on nibblers tick 1-2 every wave.
- **Blood barrage is often better on nibblers** — they clump better with blood barrage since ice freezes them apart. "Blood barrage is
  always better on nibblers because they clump up better." (V3)
- Use ice barrage when you need nibblers frozen in place (preventing pillar damage on dangerous waves).

### Blood Barrage — Healing

- **Good targets:** Rangers (squishy, low mage def), bloblets, nibblers.
- **Bad targets:** Magers (high mage level, very inaccurate), melees (slow/wasteful).
- One barrage on grouped bloblets is time-efficient (5-tick barrage vs 6-tick for 3 pipe shots). Then switch to pipe.

---

## 5. Blob Mechanics

### 6-Tick Cycle

- Blobs register your prayer 3 ticks after gaining LOS, then attack with the opposite style 3 ticks later.
- First attack: 3 ticks after first LOS. Then 6-tick cycle while LOS is maintained.
- **Default pray mage against blobs** — mage hit is ~2x the damage of range hit.
- Blob melee: if blob is adjacent, it can melee you and break the cycle.

### Bloblet Management

- **West-of-pillar pop:** Ranger bloblet blocks the other two — only need to pray range.
- **East-of-pillar pop:** Melee bloblet gets corner trapped — only deal with two.
- Walk up to blob when killing it — bloblets group up better.
- Kill melee bloblet first (dies a tick slower than others).

### When to Pop Blobs

- **Leave alive when mager is up** — bloblets get revived with the blob.
- **Pop when:** Mager is dead and you need to clear.

---

## 6. Off-Ticking & Multi-LOS

**Critical:** Even with perfect prayer, two NPCs attacking on the same tick is unblockable. Reducing simultaneous LOS and off-ticking
attackers is essential.

### 1-Tick Alternating (Ranger+Blob / Mager+Blob)

- When ranger/mager and blob attack on alternating ticks, rapidly switching prayer blocks both. This is handled by the prayer heuristic IF
  the attacks are off-tick.
- If they attack on the same tick, one hit is unblocked.

### Off-Ticking via Pillar

- Walk around a pillar corner to stagger when NPCs first gain LOS, putting their attack cycles out of sync.
- "Anytime you've got off-tick, going around a pillar to force it is the easiest way." (V4)
- Melee + ranger/mager: pray melee every 4th tick, pray range/mage the other 3 — but only works if they're off-tick.

### Reducing LOS

- The core goal of positioning: have at most 1 dangerous NPC with LOS at a time.
- Use pillar to break LOS on NPCs you're not actively fighting.
- Pillar stacks (vertical alignment of mager+ranger) force multi-LOS situations — avoid by killing mager first.

### When to Tank

- **Never tank mager hits.** Max 70, almost always accurate.
- **Ranger melee (punch) is tankable.** Max 19, very inaccurate. Standing adjacent to a ranger and taking a potential melee hit is
  acceptable. Ranger ranged hits are NOT tankable.
- **Melee hits:** Max ~49-56, fairly inaccurate. Better to tank melee than mager. But avoid when possible.
- **Blob mage > blob range:** If unsure which blob prayer to use, pray mage.

---

## 7. Mager Revive Mechanics

- Revive takes 8 ticks (2 attack cycles). Mager doesn't attack during revive.
- ~10% chance to revive per attack cycle.
- Revived monsters spawn near center of arena with half HP.
