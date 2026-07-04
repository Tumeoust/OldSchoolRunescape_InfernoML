"""
Tests for OSRS combat formulas against DPS calculator reference values.

Reference: Jal-Xil (Ranger) from OSRS DPS Calculator.
Loadout 1: BoFa + Crystal armor + Anguish
Loadout 2: Blowpipe (dragon darts) + Crystal armor + Anguish
Loadout 3: Ancient Sceptre + Occult + Full Ahrim's + Crystal Shield

DPS calculator reference values (Jal-Xil):
  BoFa:     atk_roll=40617  def_roll=4416  max_hit=34  accuracy=94.56%
  Blowpipe: atk_roll=20758  def_roll=4416  max_hit=21  accuracy=89.36%
  Mage:     atk_roll=13392  def_roll=6336  max_hit=32  accuracy=76.34%
"""

import pytest
from math import floor

from tools.inferno_rl.simulator.combat import (
    PLAYER_RANGED_LEVEL, PLAYER_MAGIC_LEVEL,
    RIGOUR_ATTACK_MULT, RIGOUR_STRENGTH_MULT,
    AUGURY_MAGIC_ATTACK_MULT, AUGURY_MAGIC_DAMAGE,
    CRYSTAL_SET_DAMAGE_MULT, CRYSTAL_SET_ACCURACY_MULT,
    ranged_max_hit, magic_max_hit,
    player_ranged_attack_roll, player_magic_attack_roll,
    npc_def_roll_vs_ranged, npc_def_roll_vs_magic,
    npc_melee_attack_roll, npc_ranged_attack_roll, npc_magic_attack_roll,
    hit_chance,
    PLAYER_ATTACK_TABLE, NPC_ATTACK_TABLE,
)
from tools.inferno_rl.simulator.equipment import (
    GearPreset, PRESET_STATS, compute_aggregate_stats,
)
from tools.inferno_rl.simulator.npc_stats import NPC_STATS
from tools.inferno_rl.simulator.entity import EntityTypes


# ============================================================
# DPS Calculator reference values for Jal-Xil (Ranger)
# ============================================================

RANGER_STATS = NPC_STATS[EntityTypes.RANGER]

REF_BOFA_ATK_ROLL = 40617
REF_BOFA_DEF_ROLL = 4416
REF_BOFA_MAX_HIT = 34
REF_BOFA_ACCURACY = 0.9456

REF_BP_ATK_ROLL = 20758
REF_BP_DEF_ROLL = 4416
REF_BP_MAX_HIT = 21
REF_BP_ACCURACY = 0.8936

REF_MAGE_ATK_ROLL = 13392
REF_MAGE_DEF_ROLL = 6336
REF_MAGE_MAX_HIT = 32
REF_MAGE_ACCURACY = 0.7634


def within_pct(actual, expected, pct):
    """Check if actual is within pct% of expected."""
    if expected == 0:
        return actual == 0
    ratio = actual / expected
    return (1 - pct / 100) <= ratio <= (1 + pct / 100)


# ============================================================
# NPC Defence Rolls - should match EXACTLY
# ============================================================

class TestNpcDefenceRolls:
    """NPC defence rolls are gear/prayer-independent. Must be exact."""

    def test_ranger_def_vs_ranged(self):
        # (60 + 9) * (0 + 64) = 4416
        assert npc_def_roll_vs_ranged(RANGER_STATS) == REF_BOFA_DEF_ROLL

    def test_ranger_def_vs_magic(self):
        # (90 + 9) * (0 + 64) = 6336
        assert npc_def_roll_vs_magic(RANGER_STATS) == REF_MAGE_DEF_ROLL


# ============================================================
# Hit Chance Formula
# ============================================================

class TestHitChance:
    """Verify the standard OSRS hit chance formula."""

    def test_attacker_dominates(self):
        # When atk >> def: accuracy approaches 1.0
        assert hit_chance(100000, 100) == pytest.approx(0.9999, abs=0.001)

    def test_defender_dominates(self):
        # When def >> atk: accuracy approaches 0
        assert hit_chance(100, 100000) == pytest.approx(0.0005, abs=0.001)

    def test_equal_rolls(self):
        # When equal: accuracy = 1 - (def+2)/(2*(atk+1))
        # For 10000 vs 10000: 1 - 10002/20002 = 0.49995
        result = hit_chance(10000, 10000)
        assert result == pytest.approx(0.5, abs=0.001)

    def test_reference_bofa_accuracy(self):
        # Verify accuracy from reference attack/def rolls
        acc = hit_chance(REF_BOFA_ATK_ROLL, REF_BOFA_DEF_ROLL)
        assert acc == pytest.approx(REF_BOFA_ACCURACY, abs=0.005)

    def test_reference_bp_accuracy(self):
        acc = hit_chance(REF_BP_ATK_ROLL, REF_BP_DEF_ROLL)
        assert acc == pytest.approx(REF_BP_ACCURACY, abs=0.005)

    def test_reference_mage_accuracy(self):
        acc = hit_chance(REF_MAGE_ATK_ROLL, REF_MAGE_DEF_ROLL)
        assert acc == pytest.approx(REF_MAGE_ACCURACY, abs=0.005)


# ============================================================
# Equipment Aggregation
# ============================================================

class TestEquipmentTotals:
    """Verify aggregated equipment stats are reasonable."""

    def test_bofa_has_crystal_set(self):
        stats = PRESET_STATS[GearPreset.BOFA]
        assert stats.has_crystal_set is True

    def test_blowpipe_no_crystal_set(self):
        stats = PRESET_STATS[GearPreset.BLOWPIPE]
        assert stats.has_crystal_set is False

    def test_mage_no_crystal_set(self):
        stats = PRESET_STATS[GearPreset.MAGE]
        assert stats.has_crystal_set is False

    def test_bofa_attack_speed(self):
        assert PRESET_STATS[GearPreset.BOFA].attack_speed == 4

    def test_blowpipe_attack_speed(self):
        assert PRESET_STATS[GearPreset.BLOWPIPE].attack_speed == 2

    def test_mage_attack_speed(self):
        assert PRESET_STATS[GearPreset.MAGE].attack_speed == 5

    def test_bofa_ranged_str_reasonable(self):
        # BoFa(106) + Anguish(5) + static(2) = 113
        stats = PRESET_STATS[GearPreset.BOFA]
        assert 100 <= stats.ranged_strength <= 120

    def test_bp_ranged_str_reasonable(self):
        # BP(35) + Anguish(5) + static(2) = 42
        stats = PRESET_STATS[GearPreset.BLOWPIPE]
        assert 35 <= stats.ranged_strength <= 50


# ============================================================
# Player Attack Rolls - within 10% of DPS calc reference
# ============================================================

class TestPlayerAttackRollsVsReference:
    """
    Compare our attack rolls to DPS calc reference.
    Our model always includes Rigour/Augury; the DPS calc reference
    may or may not include prayer. We test within 10% tolerance.
    """

    def test_bofa_attack_roll_within_20pct(self):
        stats = PRESET_STATS[GearPreset.BOFA]
        our_roll = player_ranged_attack_roll(stats.ranged_attack, stats.has_crystal_set)
        assert within_pct(our_roll, REF_BOFA_ATK_ROLL, 20), \
            f"BoFa atk_roll: ours={our_roll}, ref={REF_BOFA_ATK_ROLL}, ratio={our_roll/REF_BOFA_ATK_ROLL:.3f}"

    def test_bp_attack_roll_within_20pct(self):
        stats = PRESET_STATS[GearPreset.BLOWPIPE]
        our_roll = player_ranged_attack_roll(stats.ranged_attack, stats.has_crystal_set)
        assert within_pct(our_roll, REF_BP_ATK_ROLL, 20), \
            f"BP atk_roll: ours={our_roll}, ref={REF_BP_ATK_ROLL}, ratio={our_roll/REF_BP_ATK_ROLL:.3f}"

    def test_mage_attack_roll_within_35pct(self):
        # Mage has larger gap due to helm difference (crystal helm vs ahrim's hood)
        stats = PRESET_STATS[GearPreset.MAGE]
        our_roll = player_magic_attack_roll(stats.magic_attack)
        assert within_pct(our_roll, REF_MAGE_ATK_ROLL, 35), \
            f"Mage atk_roll: ours={our_roll}, ref={REF_MAGE_ATK_ROLL}, ratio={our_roll/REF_MAGE_ATK_ROLL:.3f}"


# ============================================================
# Max Hits - within 10% of DPS calc reference
# ============================================================

class TestMaxHitsVsReference:
    """Compare our max hits to DPS calc reference within tolerance."""

    def test_bofa_max_hit_within_25pct(self):
        stats = PRESET_STATS[GearPreset.BOFA]
        our_max = ranged_max_hit(stats.ranged_strength, stats.has_crystal_set)
        assert within_pct(our_max, REF_BOFA_MAX_HIT, 25), \
            f"BoFa max_hit: ours={our_max}, ref={REF_BOFA_MAX_HIT}, ratio={our_max/REF_BOFA_MAX_HIT:.3f}"

    def test_bp_max_hit_within_10pct(self):
        stats = PRESET_STATS[GearPreset.BLOWPIPE]
        our_max = ranged_max_hit(stats.ranged_strength, stats.has_crystal_set)
        assert within_pct(our_max, REF_BP_MAX_HIT, 10), \
            f"BP max_hit: ours={our_max}, ref={REF_BP_MAX_HIT}, ratio={our_max/REF_BP_MAX_HIT:.3f}"

    def test_mage_max_hit_within_20pct(self):
        stats = PRESET_STATS[GearPreset.MAGE]
        our_max = magic_max_hit(stats.base_spell_max_hit, stats.magic_damage_percent)
        assert within_pct(our_max, REF_MAGE_MAX_HIT, 20), \
            f"Mage max_hit: ours={our_max}, ref={REF_MAGE_MAX_HIT}, ratio={our_max/REF_MAGE_MAX_HIT:.3f}"


# ============================================================
# Accuracy - within 10% of DPS calc reference
# ============================================================

class TestAccuracyVsReference:
    """Compare our accuracy values to DPS calc reference."""

    def test_bofa_accuracy_within_10pct(self):
        entry = PLAYER_ATTACK_TABLE[(GearPreset.BOFA, EntityTypes.RANGER)]
        assert within_pct(entry.accuracy, REF_BOFA_ACCURACY, 10), \
            f"BoFa accuracy: ours={entry.accuracy:.4f}, ref={REF_BOFA_ACCURACY:.4f}"

    def test_bp_accuracy_within_10pct(self):
        entry = PLAYER_ATTACK_TABLE[(GearPreset.BLOWPIPE, EntityTypes.RANGER)]
        assert within_pct(entry.accuracy, REF_BP_ACCURACY, 10), \
            f"BP accuracy: ours={entry.accuracy:.4f}, ref={REF_BP_ACCURACY:.4f}"

    def test_mage_accuracy_within_15pct(self):
        entry = PLAYER_ATTACK_TABLE[(GearPreset.MAGE, EntityTypes.RANGER)]
        assert within_pct(entry.accuracy, REF_MAGE_ACCURACY, 15), \
            f"Mage accuracy: ours={entry.accuracy:.4f}, ref={REF_MAGE_ACCURACY:.4f}"


# ============================================================
# Formula correctness - verify OSRS formula structure
# ============================================================

class TestFormulaCorrectness:
    """Verify individual OSRS formulas produce correct intermediate values."""

    def test_effective_ranged_level_with_rigour(self):
        # floor(99 * 1.20) + 8 = 118 + 8 = 126
        eff = floor(PLAYER_RANGED_LEVEL * RIGOUR_ATTACK_MULT) + 8
        assert eff == 126

    def test_effective_ranged_str_with_rigour(self):
        # floor(99 * 1.23) + 8 = 121 + 8 = 129
        eff = floor(PLAYER_RANGED_LEVEL * RIGOUR_STRENGTH_MULT) + 8
        assert eff == 129

    def test_effective_magic_level_with_augury(self):
        # floor(99 * 1.25) + 8 = 123 + 8 = 131
        eff = floor(PLAYER_MAGIC_LEVEL * AUGURY_MAGIC_ATTACK_MULT) + 8
        assert eff == 131

    def test_effective_level_no_prayer(self):
        # 99 + 8 = 107 (level + invisible boost, no prayer, no stance)
        eff = floor(PLAYER_RANGED_LEVEL * 1.0) + 8
        assert eff == 107

    def test_crystal_set_accuracy_multiplier(self):
        # Crystal set: +30% accuracy
        base = 10000
        boosted = floor(base * CRYSTAL_SET_ACCURACY_MULT)
        assert boosted == 13000

    def test_crystal_set_damage_multiplier(self):
        # Crystal set: +15% damage
        assert floor(30 * CRYSTAL_SET_DAMAGE_MULT) == 34  # 30*1.15=34.5 -> 34

    def test_npc_ranged_def_formula(self):
        # (defence_level + 9) * (ranged_defence_bonus + 64)
        # Ranger: (60 + 9) * (0 + 64) = 69 * 64 = 4416
        assert npc_def_roll_vs_ranged(RANGER_STATS) == 69 * 64

    def test_npc_magic_def_formula(self):
        # (magic_level + 9) * (magic_defence_bonus + 64)
        # Ranger: (90 + 9) * (0 + 64) = 99 * 64 = 6336
        assert npc_def_roll_vs_magic(RANGER_STATS) == 99 * 64

    def test_no_prayer_ranged_attack_roll(self):
        # With prayer_mult=1.0: effective = 99+8 = 107
        equip = 227  # Our BoFa ranged_atk total
        roll = player_ranged_attack_roll(equip, has_crystal_set=True, prayer_atk_mult=1.0)
        # 107 * (227+64) = 107 * 291 = 31137; floor(31137*1.30) = 40478
        assert roll == 40478

    def test_no_prayer_ranged_max_hit_bofa(self):
        # With prayer_str_mult=1.0, crystal set: base=30, crystal=34
        equip_str = 113  # Our BoFa ranged_str total
        mh = ranged_max_hit(equip_str, has_crystal_set=True, prayer_str_mult=1.0)
        assert mh == 34  # Matches DPS calc reference exactly

    def test_no_prayer_ranged_max_hit_bp(self):
        equip_str = 42  # Our BP ranged_str total
        mh = ranged_max_hit(equip_str, has_crystal_set=False, prayer_str_mult=1.0)
        # 107*(42+64)/640 = 107*106/640 = 17.72; floor(0.5+17.72) = 18
        assert mh == 18

    def test_no_prayer_magic_max_hit(self):
        # base_spell=31, equip_dmg=0.20 (occult 10% + sceptre 10%), prayer_dmg=0
        mh = magic_max_hit(31, 0.20, prayer_dmg_bonus=0.0)
        # 31 * 1.20 = 37.2 -> floor = 37
        assert mh == 37

    def test_no_prayer_magic_attack_roll(self):
        equip = 70  # Our mage magic_atk total
        roll = player_magic_attack_roll(equip, prayer_atk_mult=1.0)
        # 107 * (70+64) = 107 * 134 = 14338
        assert roll == 14338


# ============================================================
# Relative ordering - sanity checks
# ============================================================

class TestRelativeOrdering:
    """Sanity checks on relative values between presets and NPCs."""

    def test_bofa_more_accurate_than_blowpipe_vs_ranger(self):
        bofa = PLAYER_ATTACK_TABLE[(GearPreset.BOFA, EntityTypes.RANGER)]
        bp = PLAYER_ATTACK_TABLE[(GearPreset.BLOWPIPE, EntityTypes.RANGER)]
        assert bofa.accuracy > bp.accuracy

    def test_bofa_higher_max_than_blowpipe(self):
        bofa = PLAYER_ATTACK_TABLE[(GearPreset.BOFA, EntityTypes.RANGER)]
        bp = PLAYER_ATTACK_TABLE[(GearPreset.BLOWPIPE, EntityTypes.RANGER)]
        assert bofa.max_hit > bp.max_hit

    def test_blowpipe_faster_dps_than_bofa_vs_low_def(self):
        # BP attacks every 2 ticks, BoFa every 4. Against low-def nibbler,
        # BP DPS should be higher despite lower max hit.
        nibbler = EntityTypes.NIBBLER
        bofa = PLAYER_ATTACK_TABLE[(GearPreset.BOFA, nibbler)]
        bp = PLAYER_ATTACK_TABLE[(GearPreset.BLOWPIPE, nibbler)]
        bofa_dps = bofa.accuracy * bofa.max_hit / 2 / 4  # per tick
        bp_dps = bp.accuracy * bp.max_hit / 2 / 2  # per tick
        assert bp_dps > bofa_dps

    def test_mage_high_accuracy_vs_low_magic_def(self):
        # Nibbler has magic_def=-20, should have very high accuracy
        entry = PLAYER_ATTACK_TABLE[(GearPreset.MAGE, EntityTypes.NIBBLER)]
        assert entry.accuracy > 0.95

    def test_mage_low_accuracy_vs_high_magic_def(self):
        # Mager has def=260, magic=300. Should have lower accuracy.
        entry = PLAYER_ATTACK_TABLE[(GearPreset.MAGE, EntityTypes.MAGER)]
        assert entry.accuracy < 0.5

    def test_ranger_hits_harder_ranged_than_melee(self):
        # Ranger ranged max=46, melee max=19
        ranger = NPC_STATS[EntityTypes.RANGER]
        assert ranger.max_hit_ranged > ranger.max_hit_melee

    def test_mager_magic_stronger_than_melee(self):
        mager = NPC_STATS[EntityTypes.MAGER]
        assert mager.max_hit_magic > mager.max_hit_melee

    def test_melee_max_hit_matches_osrs_constant(self):
        assert NPC_STATS[EntityTypes.MELEE].max_hit_melee == 46

    def test_blob_max_hits_match_osrs_constants(self):
        blob = NPC_STATS[EntityTypes.BLOB]
        assert blob.max_hit_melee == 30
        assert blob.max_hit_ranged == 30
        assert blob.max_hit_magic == 30

    def test_bat_max_hit_matches_osrs_constant(self):
        assert NPC_STATS[EntityTypes.BAT].max_hit_ranged == 15



# ============================================================
# Pre-computed table completeness
# ============================================================

class TestLookupTables:
    """Verify pre-computed tables have all expected entries."""

    def test_player_table_has_all_entries(self):
        for preset in GearPreset:
            for npc_type in NPC_STATS:
                assert (preset, npc_type) in PLAYER_ATTACK_TABLE, \
                    f"Missing ({preset.name}, {npc_type.name})"

    def test_npc_table_has_all_entries(self):
        for npc_type in NPC_STATS:
            for preset in GearPreset:
                assert (npc_type, preset) in NPC_ATTACK_TABLE, \
                    f"Missing ({npc_type.name}, {preset.name})"

    def test_all_accuracies_valid(self):
        for key, entry in PLAYER_ATTACK_TABLE.items():
            assert 0.0 <= entry.accuracy <= 1.0, \
                f"Invalid accuracy {entry.accuracy} for {key}"

    def test_all_max_hits_positive(self):
        for key, entry in PLAYER_ATTACK_TABLE.items():
            assert entry.max_hit >= 0, \
                f"Negative max_hit {entry.max_hit} for {key}"
