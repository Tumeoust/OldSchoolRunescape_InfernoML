"""Tests that prayer multipliers are correctly applied per loadout tier."""

from tools.inferno_rl.simulator.combat import (
    build_combat_tables,
    compute_expected_barrage_damage,
    ranged_max_hit,
    player_ranged_attack_roll,
    magic_max_hit,
    player_magic_attack_roll,
    ALL_COMBAT_TABLES,
)
from tools.inferno_rl.simulator.equipment import (
    GearPreset,
    LoadoutId,
    LOADOUTS,
    RIGOUR_AUGURY,
    EAGLE_EYE_MYSTIC_MIGHT,
)
from tools.inferno_rl.simulator.entity import EntityTypes


def test_budget_rcb_uses_eagle_eye():
    """BUDGET_RCB should use Eagle Eye (1.15) not Rigour (1.20/1.23)."""
    loadout = LOADOUTS[LoadoutId.BUDGET_RCB]
    assert loadout.prayers == EAGLE_EYE_MYSTIC_MIGHT
    assert loadout.prayers.ranged_atk == 1.15
    assert loadout.prayers.ranged_str == 1.15


def test_crystal_bp_uses_rigour():
    """CRYSTAL_BP should use Rigour+Augury (defaults)."""
    loadout = LOADOUTS[LoadoutId.CRYSTAL_BP]
    assert loadout.prayers == RIGOUR_AUGURY
    assert loadout.prayers.ranged_atk == 1.20
    assert loadout.prayers.ranged_str == 1.23


def test_eagle_eye_produces_lower_max_hit_than_rigour():
    """Same equipment, same level — Eagle Eye gives lower max hit than Rigour."""
    equip_str = 112
    has_crystal = False
    level = 90

    mh_eagle = ranged_max_hit(equip_str, has_crystal, ranged_level=level, prayer_str_mult=1.15)
    mh_rigour = ranged_max_hit(equip_str, has_crystal, ranged_level=level, prayer_str_mult=1.23)
    assert mh_eagle < mh_rigour


def test_eagle_eye_produces_lower_attack_roll_than_rigour():
    """Same equipment, same level — Eagle Eye gives lower attack roll than Rigour."""
    equip_atk = 173
    has_crystal = False
    level = 80

    roll_eagle = player_ranged_attack_roll(equip_atk, has_crystal, ranged_level=level, prayer_atk_mult=1.15)
    roll_rigour = player_ranged_attack_roll(equip_atk, has_crystal, ranged_level=level, prayer_atk_mult=1.20)
    assert roll_eagle < roll_rigour


def test_mystic_might_produces_lower_barrage_damage_than_augury():
    """Mystic Might (0% dmg bonus) vs Augury (4% dmg bonus)."""
    base_hit = 30
    equip_dmg_pct = 0.0

    mh_mystic = magic_max_hit(base_hit, equip_dmg_pct, prayer_dmg_bonus=0.0)
    mh_augury = magic_max_hit(base_hit, equip_dmg_pct, prayer_dmg_bonus=0.04)
    assert mh_mystic < mh_augury


def test_budget_combat_tables_lower_damage_than_crystal():
    """BUDGET_RCB tables should produce lower player damage than CRYSTAL_BP tables
    for the same NPC type, since Eagle Eye < Rigour and lower levels."""
    budget_tables = ALL_COMBAT_TABLES[LoadoutId.BUDGET_RCB]
    crystal_tables = ALL_COMBAT_TABLES[LoadoutId.CRYSTAL_BP]

    # BoFa vs Mager
    budget_mager = budget_tables.player_attack[(GearPreset.BOFA, EntityTypes.MAGER)]
    crystal_mager = crystal_tables.player_attack[(GearPreset.BOFA, EntityTypes.MAGER)]
    assert budget_mager.max_hit < crystal_mager.max_hit

    # Mage vs Ranger (barrage)
    budget_ranger = budget_tables.player_attack[(GearPreset.MAGE, EntityTypes.RANGER)]
    crystal_ranger = crystal_tables.player_attack[(GearPreset.MAGE, EntityTypes.RANGER)]
    assert budget_ranger.max_hit <= crystal_ranger.max_hit


def test_budget_barrage_expected_damage_lower_than_crystal():
    """Barrage expected damage should be lower for BUDGET_RCB than CRYSTAL_BP."""
    budget_barrage = compute_expected_barrage_damage(
        tables=ALL_COMBAT_TABLES[LoadoutId.BUDGET_RCB]
    )
    crystal_barrage = compute_expected_barrage_damage(
        tables=ALL_COMBAT_TABLES[LoadoutId.CRYSTAL_BP]
    )

    # Check a common target
    budget_mager_dmg = budget_barrage[EntityTypes.MAGER]["expected_damage"]
    crystal_mager_dmg = crystal_barrage[EntityTypes.MAGER]["expected_damage"]
    assert budget_mager_dmg < crystal_mager_dmg


def test_all_loadouts_have_prayers():
    """Every loadout should have a prayers field."""
    for lid, loadout in LOADOUTS.items():
        assert loadout.prayers is not None, f"{lid} missing prayers"
        assert loadout.prayers.ranged_atk > 0
        assert loadout.prayers.ranged_str > 0
        assert loadout.prayers.magic_atk > 0
