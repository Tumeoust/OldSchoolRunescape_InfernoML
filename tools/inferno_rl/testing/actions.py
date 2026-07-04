"""
InfernoAction constants for the heuristic testing/visualizer tools.

Mirrors the exact-target legacy action layout without depending on the RL
training stack.
"""

from ..simulator.exact_targeting import get_exact_target_slots

NUM_ACTIONS = 52
MOVEMENT_START = 0
ATTACK_START = 33
NO_ACTION = 47
WEAPON_SWITCH_START = 48
NUM_ATTACK_ACTIONS = 14


class InfernoAction:
    STAY = 0
    ATTACK_TARGET_1 = 33
    ATTACK_TARGET_2 = 34
    ATTACK_TARGET_3 = 35
    ATTACK_TARGET_4 = 36
    ATTACK_TARGET_5 = 37
    ATTACK_TARGET_6 = 38
    ATTACK_TARGET_7 = 39
    ATTACK_TARGET_8 = 40
    ATTACK_TARGET_9 = 41
    ATTACK_TARGET_10 = 42
    ATTACK_TARGET_11 = 43
    ATTACK_TARGET_12 = 44
    ATTACK_TARGET_13 = 45
    ATTACK_TARGET_14 = 46
    NO_ACTION_IDX = 47
    SWITCH_BOFA = 48
    SWITCH_BLOWPIPE = 49
    SWITCH_ICE_BARRAGE = 50
    SWITCH_BLOOD_BARRAGE = 51

    @staticmethod
    def is_movement(action: int) -> bool:
        return 1 <= action <= 32

    @staticmethod
    def is_attack(action: int) -> bool:
        return ATTACK_START <= action < ATTACK_START + NUM_ATTACK_ACTIONS

    @staticmethod
    def is_weapon_switch(action: int) -> bool:
        return WEAPON_SWITCH_START <= action <= 51

    @staticmethod
    def is_no_op(action: int) -> bool:
        return action in (InfernoAction.STAY, InfernoAction.NO_ACTION_IDX)

    @staticmethod
    def get_target_index(action: int) -> int:
        if InfernoAction.is_attack(action):
            return action - ATTACK_START
        return -1

    @staticmethod
    def action_for_target_index(target_index: int) -> int:
        if not 0 <= target_index < NUM_ATTACK_ACTIONS:
            raise ValueError(f"target_index must be in [0, {NUM_ATTACK_ACTIONS - 1}], got {target_index}")
        return ATTACK_START + target_index


def get_action_mask(state) -> list:
    mask = [True] * NUM_ACTIONS
    for slot_index in range(NUM_ATTACK_ACTIONS):
        mask[ATTACK_START + slot_index] = False
    for slot_index, _ in enumerate(get_exact_target_slots(state)):
        mask[ATTACK_START + slot_index] = True
    if not getattr(state, "has_blowpipe", True):
        mask[InfernoAction.SWITCH_BLOWPIPE] = False
    return mask


def get_movement_params(action: int):
    if action == 0:
        return ("STAY", 0)
    movements = {
        1: ("N", 1), 2: ("N", 2), 3: ("N", 3), 4: ("N", 4),
        5: ("S", 1), 6: ("S", 2), 7: ("S", 3), 8: ("S", 4),
        9: ("E", 1), 10: ("E", 2), 11: ("E", 3), 12: ("E", 4),
        13: ("W", 1), 14: ("W", 2), 15: ("W", 3), 16: ("W", 4),
        17: ("NE", 1), 18: ("NE", 2), 19: ("NE", 3), 20: ("NE", 4),
        21: ("NW", 1), 22: ("NW", 2), 23: ("NW", 3), 24: ("NW", 4),
        25: ("SE", 1), 26: ("SE", 2), 27: ("SE", 3), 28: ("SE", 4),
        29: ("SW", 1), 30: ("SW", 2), 31: ("SW", 3), 32: ("SW", 4),
    }
    return movements.get(action, ("UNK", 0))
