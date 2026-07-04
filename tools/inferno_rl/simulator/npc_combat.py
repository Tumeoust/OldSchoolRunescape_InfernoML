"""
NPC attacks, damage application, death processing, and blob spawning.

NpcCombatMixin handles NPC attack dispatch, per-type attack logic (nibbler,
blob scan/attack, mager resurrection, standard NPC), dead entity processing,
and blob-split spawning.
"""

import random
from typing import Optional, Tuple, List

from .entity import PlacedEntity, EntityTypes, AttackStyle
from .geometry import SimulatorGeometry, InfernoLineOfSight
from .pathfinding import NpcCollisionResolver
from .combat import roll_npc_damage
from .step_result import PlayerDamageEvent


# Probability that a non-melee NPC uses melee when the player is cardinally adjacent.
# In OSRS, adjacent NPCs have roughly a 40% chance to melee instead of their primary style.
NPC_ADJACENT_MELEE_CHANCE = 0.40


class NpcCombatMixin:
    """Mixin providing NPC attack, death, and blob-split logic."""

    def _apply_player_damage(
        self,
        attacker: PlacedEntity,
        attack_style: str,
        rolled_damage: int,
    ) -> None:
        """Apply NPC hit to player HP and record attribution when damage is dealt."""
        if rolled_damage <= 0:
            return

        hp_before = self.state.player_health
        self.state.player_health = max(0, hp_before - rolled_damage)
        damage_applied = hp_before - self.state.player_health
        if damage_applied <= 0:
            return

        self.player_damage_events_this_tick.append(
            PlayerDamageEvent(
                tick=self.state.current_tick,
                attacker_id=attacker.id,
                attacker_type=attacker.entity_type.name,
                attacker_x=attacker.x,
                attacker_y=attacker.y,
                attack_style=attack_style,
                damage=damage_applied,
            )
        )

    def _process_npc_attacks(self):
        """Process attacks from all NPCs."""
        player_x = self.state.player_x
        player_y = self.state.player_y
        pillar_alive = self.state.pillar_alive

        for entity in self.state.entities:
            if entity.is_dead():
                continue

            # Nibblers attack pillars
            if entity.entity_type == EntityTypes.NIBBLER:
                self._process_nibbler_attack(entity)
                continue

            # Blobs have special scan/attack mechanics
            if entity.entity_type == EntityTypes.BLOB:
                self._process_blob_attack(entity, player_x, player_y, pillar_alive)
                continue

            # Magers have flicker + resurrection mechanics
            if entity.entity_type == EntityTypes.MAGER:
                self._process_mager_attack(entity, player_x, player_y, pillar_alive)
                continue

            # Standard NPC attack
            self._process_standard_npc_attack(entity, player_x, player_y, pillar_alive)

    def _process_nibbler_attack(self, nibbler: PlacedEntity):
        """Process nibbler pillar attack."""
        target_pillar = nibbler.target_pillar_index
        if target_pillar < 0 or not self.state.is_pillar_alive(target_pillar):
            # Nibbler dies if target pillar is dead
            nibbler.take_damage(nibbler.current_health)
            return

        # Check if adjacent to pillar
        pillar_center = PlacedEntity.get_pillar_center(target_pillar)
        if pillar_center is None:
            return

        dist = SimulatorGeometry.chebyshev_distance(
            nibbler.x, nibbler.y, pillar_center[0], pillar_center[1]
        )

        if dist <= 2 and nibbler.can_attack():  # Adjacent to 3x3 pillar
            nibbler.attacked_this_tick = True
            nibbler.attack_delay = nibbler.entity_type.attack_speed

            damage = random.randint(1, 5)
            pillar_died = self.state.damage_pillar(target_pillar, damage)
            if pillar_died:
                self.state.schedule_pillar_collapse(target_pillar)
                self._kill_nibblers_targeting_pillar(target_pillar)

    def _kill_nibblers_targeting_pillar(self, pillar_index: int):
        """Kill all nibblers that were targeting a destroyed pillar."""
        for entity in self.state.entities:
            if (entity.entity_type == EntityTypes.NIBBLER and
                entity.target_pillar_index == pillar_index and
                not entity.is_dead()):
                entity.take_damage(entity.current_health)
                entity_type = entity.entity_type
                self.kills_this_tick[entity_type] = self.kills_this_tick.get(entity_type, 0) + 1

    def _process_blob_attack(self, blob: PlacedEntity,
                              player_x: int, player_y: int,
                              pillar_alive: List[bool]):
        """
        Process blob scan/attack cycle.

        Blob attack cycle (from JalAk.ts):
        1. SCAN: When gaining LOS OR when attackDelay <= 0 with LOS and no scan
        2. WAIT: Set attackDelay = 3 ticks
        3. ATTACK: When attackDelay <= 0 with stored scan (can attack through LOS loss!)
        """
        has_los = InfernoLineOfSight.can_entity_attack_player(
            blob, player_x, player_y, pillar_alive
        )

        # Scan trigger conditions:
        # 1. Just gained LOS (!had_los && has_los), OR
        # 2. Has LOS AND no current scan AND attack_delay <= 0
        just_gained_los = has_los and not blob.had_los
        scan_ready = has_los and blob.scanned_prayer is None and blob.attack_delay <= 0

        if just_gained_los or scan_ready:
            # Scan current prayer - blob will attack with OPPOSITE style.
            # If player is on melee prayer or unprayed, blob randomizes 50/50.
            if self.state.active_prayer == "PROTECT_FROM_MAGIC":
                blob.scanned_prayer = "RANGED"
            elif self.state.active_prayer == "PROTECT_FROM_MISSILES":
                blob.scanned_prayer = "MAGIC"
            else:
                blob.scanned_prayer = random.choice(["MAGIC", "RANGED"])
            blob.attack_delay = 3  # Attack in 3 ticks

        blob.had_los = has_los

        # Attack phase: has stored scan AND attack_delay <= 0
        # Note: Blobs can attack THROUGH LOS loss if they have a scan!
        if blob.scanned_prayer is not None and blob.attack_delay <= 0:
            blob.attacked_this_tick = True
            blob.attack_delay = 3
            self.npcs_attacked_player_this_tick += 1

            # Adjacent blobs have a chance to melee instead of scan-based attack
            if (self._is_player_melee_adjacent_to_npc(blob, player_x, player_y)
                    and random.random() < NPC_ADJACENT_MELEE_CHANCE):
                blob.last_attack_style = "MELEE"
                blocked = self.state.active_prayer == "PROTECT_FROM_MELEE"

                if not blocked:
                    damage = roll_npc_damage(EntityTypes.BLOB, self.state.current_preset, "melee", self.combat_tables)
                    self._apply_player_damage(blob, "MELEE", damage)
            else:
                # Normal ranged/magic attack based on scan
                blob.last_attack_style = blob.scanned_prayer
                blocked = False
                if blob.scanned_prayer == "MAGIC" and self.state.active_prayer == "PROTECT_FROM_MAGIC":
                    blocked = True
                elif blob.scanned_prayer == "RANGED" and self.state.active_prayer == "PROTECT_FROM_MISSILES":
                    blocked = True

                if not blocked:
                    style = "magic" if blob.scanned_prayer == "MAGIC" else "ranged"
                    damage = roll_npc_damage(EntityTypes.BLOB, self.state.current_preset, style, self.combat_tables)
                    self._apply_player_damage(blob, style.upper(), damage)

            blob.scanned_prayer = None

    def _process_mager_attack(self, mager: PlacedEntity,
                               player_x: int, player_y: int,
                               pillar_alive: List[bool]):
        """
        Process mager attack with resurrection mechanics.

        Magers attack immediately when ready (no flicker delay).
        10% chance to resurrect a dead mob instead of attacking (waves < 69).
        Resurrected mobs spawn at 50% HP.
        """
        if not mager.can_attack():
            return

        has_los = InfernoLineOfSight.can_entity_attack_player(
            mager, player_x, player_y, pillar_alive
        )
        mager.had_los = has_los

        if not has_los:
            return

        # Check if player is under mager (can't attack)
        is_under = NpcCollisionResolver.is_player_under_npc(
            mager.x, mager.y, mager.entity_type.size_in_tiles, player_x, player_y
        )
        if is_under:
            return

        # 10% chance to resurrect instead of attacking (only on waves < 69)
        should_resurrect = (
            random.random() < 0.1 and
            self.state.current_wave < 69 and
            len(self.dead_mobs) > 0
        )

        if should_resurrect:
            resurrected = self._mager_resurrect_mob(mager)
            self.mager_resurrected_this_tick += 1
            if resurrected and resurrected.entity_type == EntityTypes.MELEE:
                self.melee_resurrected_this_tick += 1
            elif resurrected and resurrected.entity_type == EntityTypes.BAT:
                self.bat_resurrected_this_tick += 1
        else:
            self._mager_do_attack(mager)

        mager.attack_delay = mager.entity_type.attack_speed

    def _mager_do_attack(self, mager: PlacedEntity):
        """Execute mager attack - magic normally, chance to melee if player is adjacent."""
        mager.attacked_this_tick = True
        self.npcs_attacked_player_this_tick += 1

        player_x = self.state.player_x
        player_y = self.state.player_y

        # Magers can melee from diagonal tiles too (Chebyshev distance 1), not just cardinal.
        # LOS is already guaranteed by _process_mager_attack before calling this method.
        mager_dist = InfernoLineOfSight.get_distance_from_npc(
            mager.x, mager.y, mager.entity_type.size_in_tiles, player_x, player_y
        )
        if (mager_dist <= 1
                and random.random() < NPC_ADJACENT_MELEE_CHANCE):
            mager.last_attack_style = "MELEE"
            blocked = self.state.active_prayer == "PROTECT_FROM_MELEE"

            if not blocked:
                damage = roll_npc_damage(EntityTypes.MAGER, self.state.current_preset, "melee", self.combat_tables)
                self._apply_player_damage(mager, "MELEE", damage)
        else:
            # Normal magic attack
            mager.last_attack_style = "MAGIC"
            blocked = self.state.active_prayer == "PROTECT_FROM_MAGIC"

            if not blocked:
                damage = roll_npc_damage(EntityTypes.MAGER, self.state.current_preset, "magic", self.combat_tables)
                self._apply_player_damage(mager, "MAGIC", damage)

    def _mager_resurrect_mob(self, mager: PlacedEntity) -> Optional[PlacedEntity]:
        """Resurrect a dead mob. Returns the resurrected entity, or None."""
        if not self.dead_mobs:
            return None

        # Select random dead mob
        mob_to_resurrect = random.choice(self.dead_mobs)
        self.dead_mobs.remove(mob_to_resurrect)
        mob_to_resurrect.has_resurrected = True

        # Reset mob state
        mob_to_resurrect.current_health = mob_to_resurrect.entity_type.max_health // 2
        mob_to_resurrect.attack_delay = mob_to_resurrect.entity_type.attack_speed
        mob_to_resurrect.stunned = 1  # 1 tick stun after resurrection
        mob_to_resurrect.frozen = 0

        # Find respawn location
        spawn_pos = self._find_resurrection_spawn(mob_to_resurrect.entity_type.size_in_tiles)
        mob_to_resurrect.x = spawn_pos[0]
        mob_to_resurrect.y = spawn_pos[1]

        # Add back to entities
        self.state.add_entity(mob_to_resurrect)

        # Mager has longer delay after resurrection (8 ticks instead of 4)
        mager.attack_delay = 8
        return mob_to_resurrect

    def _find_resurrection_spawn(self, size: int) -> Tuple[int, int]:
        """
        Find a valid spawn location for a resurrected mob.

        From InfernoTrainer: spawns in area (26, 24) to (32, 37) in grid coords
        which is roughly (15+11, 10+14) to (21+11, 22+14) in their system.
        """
        # Search area for resurrection spawns (center-ish of arena)
        for x in range(15, 22):
            for y in range(14, 23):
                if self._is_valid_spawn_position(x, y, size):
                    return (x, y)

        # Fallback position
        return (17, 18)

    def _is_valid_spawn_position(self, x: int, y: int, size: int) -> bool:
        """Check if position is valid for spawning (no collisions)."""
        # Check terrain validity
        if not SimulatorGeometry.is_valid_tile_for_size(x, y, size, self.state.pillar_alive):
            return False

        # Check collision with other NPCs
        for entity in self.state.entities:
            if entity.is_dead():
                continue
            if SimulatorGeometry.do_footprints_overlap(
                x, y, size,
                entity.x, entity.y, entity.entity_type.size_in_tiles
            ):
                return False

        # Check collision with player
        if SimulatorGeometry.do_footprints_overlap(x, y, size,
                                                    self.state.player_x, self.state.player_y, 1):
            return False

        return True

    def _process_standard_npc_attack(self, entity: PlacedEntity,
                                      player_x: int, player_y: int,
                                      pillar_alive: List[bool]):
        """Process standard NPC attack."""
        if not entity.can_attack():
            return

        has_los = InfernoLineOfSight.can_entity_attack_player(
            entity, player_x, player_y, pillar_alive
        )
        entity.had_los = has_los

        if not has_los:
            return

        entity.attacked_this_tick = True
        entity.attack_delay = entity.entity_type.attack_speed
        self.npcs_attacked_player_this_tick += 1

        # Non-melee NPCs have a chance to melee when player is adjacent
        use_melee = (
            entity.entity_type.attack_style != AttackStyle.MELEE
            and self._is_player_melee_adjacent_to_npc(entity, player_x, player_y)
            and random.random() < NPC_ADJACENT_MELEE_CHANCE
        )
        entity.last_attack_style = "MELEE" if use_melee else (
            "MAGIC" if entity.entity_type.attack_style in (
                AttackStyle.MAGIC,
                AttackStyle.MAGIC_RANGED,
                AttackStyle.MAGIC_RANGED_MELEE,
            ) else (
                "RANGED" if entity.entity_type.attack_style == AttackStyle.RANGED else "MELEE"
            )
        )

        if use_melee:
            # Ranger melees when player is adjacent
            blocked = self.state.active_prayer == "PROTECT_FROM_MELEE"

            if not blocked:
                damage = roll_npc_damage(entity.entity_type, self.state.current_preset, "melee", self.combat_tables)
                self._apply_player_damage(entity, "MELEE", damage)
        else:
            # Normal attack based on entity's attack style
            attack_style = entity.entity_type.attack_style
            blocked = False

            if attack_style in (AttackStyle.MAGIC, AttackStyle.MAGIC_RANGED,
                               AttackStyle.MAGIC_RANGED_MELEE):
                blocked = self.state.active_prayer == "PROTECT_FROM_MAGIC"
                style = "magic"
            elif attack_style == AttackStyle.RANGED:
                blocked = self.state.active_prayer == "PROTECT_FROM_MISSILES"
                style = "ranged"
            elif attack_style == AttackStyle.MELEE:
                blocked = self.state.active_prayer == "PROTECT_FROM_MELEE"
                style = "melee"
            else:
                style = "magic"

            if not blocked:
                damage = roll_npc_damage(entity.entity_type, self.state.current_preset, style, self.combat_tables)
                self._apply_player_damage(entity, style.upper(), damage)

    def _process_dead_entities(self):
        """Process dead entities and track kills."""
        blobs_to_split = []

        for entity in list(self.state.entities):
            if entity.is_dead() and id(entity) in self.entities_alive_at_step_start:
                # Track kill
                entity_type = entity.entity_type
                self.kills_this_tick[entity_type] = self.kills_this_tick.get(entity_type, 0) + 1
                self.state.wave_kills[entity_type] = self.state.wave_kills.get(entity_type, 0) + 1

                # Store in death store for mager resurrection
                # Magers cannot resurrect: nibblers, mini blobs (splits from main blob), or already-resurrected mobs
                non_resurrectable = (
                    EntityTypes.NIBBLER,
                    EntityTypes.BLOB_MAGE,
                    EntityTypes.BLOB_RANGE,
                    EntityTypes.BLOB_MELEE,
                )
                if not entity.has_resurrected and entity_type not in non_resurrectable:
                    self.dead_mobs.append(entity)

                # Queue main blobs for splitting
                if entity_type == EntityTypes.BLOB:
                    blobs_to_split.append(entity)

        # Split dead blobs into 3 smaller blobs
        for blob in blobs_to_split:
            self._spawn_blob_splits(blob)

        self.state.remove_dead_entities()

    def _get_blob_spawn_pattern(self, blob_x: int, blob_y: int) -> list[tuple[int, int]]:
        """
        Determine mini-blob spawn offsets based on blob position relative to NE pillar.

        Blob is 3x3, occupies (blob_x, blob_y) to (blob_x+2, blob_y+2).
        NE pillar occupies (17, 22) to (19, 24).

        Spawn patterns are directional - mini-blobs spawn away from the pillar
        or toward open space, NOT hardcoded to southeast.

        Verified from OSRS gameplay:
        - Blob at (16,19) south of pillar: spawns NORTH [(0,1), (0,2), (1,2)]
        - Blob at (14,21) west of pillar: spawns EAST [(1,0), (2,0), (2,1)]
        - Blob at (20,24) east of pillar: spawns NE diagonal [(0,0), (1,1), (2,2)]

        Returns:
            List of (dx, dy) offsets for [MELEE, RANGED, MAGIC] in that order
        """
        pillar_x, pillar_y = 17, 22
        pillar_size = 3

        # Calculate blob center for comparison
        blob_center_x = blob_x + 1
        blob_center_y = blob_y + 1
        pillar_center_x = pillar_x + 1
        pillar_center_y = pillar_y + 1

        # Determine primary direction from pillar to blob
        dx_from_pillar = blob_center_x - pillar_center_x
        dy_from_pillar = blob_center_y - pillar_center_y

        # Choose spawn pattern based on blob-pillar relationship
        if abs(dx_from_pillar) > abs(dy_from_pillar):
            # East-west dominates
            if dx_from_pillar > 0:
                # Blob is EAST of pillar: spawn northeast diagonal
                return [(0, 0), (1, 1), (2, 2)]  # Ket, Xil, Mej
            else:
                # Blob is WEST of pillar: spawn east
                return [(1, 0), (2, 0), (2, 1)]
        else:
            # North-south dominates
            if dy_from_pillar > 0:
                # Blob is NORTH of pillar: spawn north
                return [(0, 1), (0, 2), (1, 2)]
            else:
                # Blob is SOUTH of pillar: spawn north (toward pillar)
                return [(0, 1), (0, 2), (1, 2)]

        # Fallback (shouldn't reach): use northeast diagonal
        return [(0, 0), (1, 1), (2, 2)]

    def _spawn_blob_splits(self, dead_blob: PlacedEntity):
        """
        Spawn 3 small blobs when a main blob dies.

        Spawn positions are dynamic based on blob position relative to NE pillar.
        Mini-blob order: Ket (melee), Xil (ranged), Mej (magic)
        All have 4 tick attack cooldown after spawning.
        """
        blob_x = dead_blob.x
        blob_y = dead_blob.y
        current_tick = self.state.current_tick

        # Get directional spawn pattern based on blob-pillar relationship
        spawn_offsets = self._get_blob_spawn_pattern(blob_x, blob_y)
        # Returns: [(ket_dx, ket_dy), (xil_dx, xil_dy), (mej_dx, mej_dy)]

        # Assign offsets to specific blob types (order: Ket, Xil, Mej)
        spawn_configs = [
            (EntityTypes.BLOB_MELEE, blob_x + spawn_offsets[0][0], blob_y + spawn_offsets[0][1]),  # Ket
            (EntityTypes.BLOB_RANGE, blob_x + spawn_offsets[1][0], blob_y + spawn_offsets[1][1]),  # Xil
            (EntityTypes.BLOB_MAGE,  blob_x + spawn_offsets[2][0], blob_y + spawn_offsets[2][1]),  # Mej
        ]

        for entity_type, x, y in spawn_configs:
            # Clamp position to valid arena bounds
            clamped_x = max(0, min(x, 28 - entity_type.size_in_tiles))
            clamped_y = max(0, min(y, 29 - entity_type.size_in_tiles))

            # Find valid spawn position (not inside pillar)
            spawn_x, spawn_y = self._find_valid_blob_split_position(
                clamped_x, clamped_y, entity_type.size_in_tiles, blob_x, blob_y
            )

            small_blob = PlacedEntity(
                entity_type=entity_type,
                x=spawn_x,
                y=spawn_y,
                placed_tick=current_tick
            )
            # 4 tick cooldown before first attack (matching InfernoTrainer)
            small_blob.attack_delay = 4
            small_blob.stunned = 1  # Standard spawn stun

            self.state.add_entity(small_blob)

    def _find_valid_blob_split_position(self, x: int, y: int, size: int,
                                         origin_x: int, origin_y: int) -> Tuple[int, int]:
        """
        Find a valid spawn position for a mini blob, avoiding pillars.

        If the intended position overlaps a pillar, search nearby tiles
        in expanding rings until a valid position is found.

        Args:
            x, y: Intended spawn position
            size: Entity size in tiles
            origin_x, origin_y: Original blob death position (fallback center)

        Returns:
            (x, y) valid spawn position
        """
        # First check if intended position is valid
        if SimulatorGeometry.is_valid_tile_for_size(x, y, size, self.state.pillar_alive):
            return (x, y)

        # Search in expanding rings around the intended position
        # Priority: closest to intended position, then closest to origin
        for radius in range(1, 10):
            best_pos = None
            best_dist_to_origin = float('inf')

            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    # Only check tiles at this radius (on the ring edge)
                    if abs(dx) != radius and abs(dy) != radius:
                        continue

                    test_x = x + dx
                    test_y = y + dy

                    # Clamp to arena bounds
                    test_x = max(0, min(test_x, 28 - size))
                    test_y = max(0, min(test_y, 29 - size))

                    if SimulatorGeometry.is_valid_tile_for_size(test_x, test_y, size, self.state.pillar_alive):
                        # Prefer position closest to original blob death location
                        dist = abs(test_x - origin_x) + abs(test_y - origin_y)
                        if dist < best_dist_to_origin:
                            best_dist_to_origin = dist
                            best_pos = (test_x, test_y)

            if best_pos is not None:
                return best_pos

        # Final fallback: spawn at original blob position (should always be valid
        # since the blob was there)
        fallback_x = max(0, min(origin_x, 28 - size))
        fallback_y = max(0, min(origin_y, 29 - size))
        return (fallback_x, fallback_y)
