"""
Action space definitions for the Inferno RL agent.

The simulator executes a legacy 52-action interface. PPO samples a factored
MultiDiscrete action that is decoded into that legacy space:

- `mode`: stay, move, attack, switch, noop
- `move`: 32 legacy movement actions (1-32)
- `attack`: 14 exact target slots (33-46)
- `switch`: 4 legacy weapon switches (48-51)
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Iterable, Sequence

import numpy as np

from ..simulator.exact_targeting import MAX_TARGET_SLOTS, get_exact_target_slots
from ..simulator.forecast import (
    MovementResolutionTable,
    TickThreatCache,
    build_movement_resolution_table,
)
from ..simulator.movement_actions import (
    get_movement_params as get_shared_movement_params,
    iter_legacy_movement_actions as iter_shared_legacy_movement_actions,
)
from ..simulator.state import SimulatorState


NUM_ACTIONS = 52
NUM_MOVEMENT_ACTIONS = 33
NUM_ATTACK_ACTIONS = MAX_TARGET_SLOTS
NUM_WEAPON_SWITCH_ACTIONS = 4

MOVEMENT_START = 0
ATTACK_START = 33
NO_ACTION = 47
WEAPON_SWITCH_START = 48


MODE_STAY = 0
MODE_MOVE = 1
MODE_ATTACK = 2
MODE_SWITCH = 3
MODE_NOOP = 4
NUM_MODE_ACTIONS = 5

MOVE_HEAD_SIZE = 32
ATTACK_HEAD_SIZE = NUM_ATTACK_ACTIONS
SWITCH_HEAD_SIZE = NUM_WEAPON_SWITCH_ACTIONS

ACTION_HEAD_SIZES = [NUM_MODE_ACTIONS, MOVE_HEAD_SIZE, ATTACK_HEAD_SIZE, SWITCH_HEAD_SIZE]
POLICY_ACTION_MASK_SIZE = sum(ACTION_HEAD_SIZES)
POLICY_ACTION_DEPENDENCIES = {
    1: {
        i: {"require_all": [(0, MODE_MOVE)]}
        for i in range(MOVE_HEAD_SIZE)
    },
    2: {
        i: {"require_all": [(0, MODE_ATTACK)]}
        for i in range(ATTACK_HEAD_SIZE)
    },
    3: {
        i: {"require_all": [(0, MODE_SWITCH)]}
        for i in range(SWITCH_HEAD_SIZE)
    },
}


class ActionType(Enum):
    MOVEMENT = auto()
    ATTACK = auto()
    WEAPON_SWITCH = auto()
    NONE = auto()


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
        return ATTACK_START <= action < ATTACK_START + ATTACK_HEAD_SIZE

    @staticmethod
    def is_weapon_switch(action: int) -> bool:
        return WEAPON_SWITCH_START <= action < WEAPON_SWITCH_START + SWITCH_HEAD_SIZE

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
        if not 0 <= target_index < ATTACK_HEAD_SIZE:
            raise ValueError(f"target_index must be in [0, {ATTACK_HEAD_SIZE - 1}], got {target_index}")
        return ATTACK_START + target_index

    @staticmethod
    def get_action_type(action: int) -> ActionType:
        if 0 <= action <= 32:
            return ActionType.MOVEMENT
        if InfernoAction.is_attack(action):
            return ActionType.ATTACK
        if InfernoAction.is_weapon_switch(action):
            return ActionType.WEAPON_SWITCH
        return ActionType.NONE


def get_legacy_action_mask(state: SimulatorState) -> np.ndarray:
    return policy_action_mask_to_legacy_mask(get_policy_action_mask(state))


def get_action_mask(state: SimulatorState) -> np.ndarray:
    return get_legacy_action_mask(state)


def uses_factored_policy_actions(action_head_sizes: Sequence[int]) -> bool:
    return tuple(int(v) for v in action_head_sizes) == tuple(ACTION_HEAD_SIZES)


def get_expected_action_mask_size(action_head_sizes: Sequence[int]) -> int:
    if uses_factored_policy_actions(action_head_sizes):
        return POLICY_ACTION_MASK_SIZE
    return int(sum(int(v) for v in action_head_sizes))


def legacy_action_mask_to_policy_mask(action_mask: Sequence[bool] | np.ndarray) -> np.ndarray:
    mask = np.asarray(action_mask, dtype=bool)
    if mask.shape[-1] != NUM_ACTIONS:
        raise ValueError(f"Expected legacy action mask with {NUM_ACTIONS} entries, got {mask.shape}")

    move_mask = mask[..., 1:33]
    attack_mask = mask[..., ATTACK_START:ATTACK_START + ATTACK_HEAD_SIZE]
    switch_mask = mask[..., WEAPON_SWITCH_START:WEAPON_SWITCH_START + SWITCH_HEAD_SIZE]
    mode_mask = np.stack(
        [
            mask[..., InfernoAction.STAY],
            np.any(move_mask, axis=-1),
            np.any(attack_mask, axis=-1),
            np.any(switch_mask, axis=-1),
            mask[..., InfernoAction.NO_ACTION_IDX],
        ],
        axis=-1,
    )
    return np.concatenate([mode_mask, move_mask, attack_mask, switch_mask], axis=-1)


def policy_action_mask_to_legacy_mask(action_mask: Sequence[bool] | np.ndarray) -> np.ndarray:
    mask = np.asarray(action_mask, dtype=bool)
    if mask.shape[-1] != POLICY_ACTION_MASK_SIZE:
        raise ValueError(
            f"Expected policy action mask with {POLICY_ACTION_MASK_SIZE} entries, got {mask.shape}"
        )

    mode_mask = mask[..., :NUM_MODE_ACTIONS]
    offset = NUM_MODE_ACTIONS
    move_mask = mask[..., offset:offset + MOVE_HEAD_SIZE]
    offset += MOVE_HEAD_SIZE
    attack_mask = mask[..., offset:offset + ATTACK_HEAD_SIZE]
    offset += ATTACK_HEAD_SIZE
    switch_mask = mask[..., offset:offset + SWITCH_HEAD_SIZE]

    legacy_mask = np.zeros(mask.shape[:-1] + (NUM_ACTIONS,), dtype=bool)
    legacy_mask[..., InfernoAction.STAY] = mode_mask[..., MODE_STAY]
    legacy_mask[..., 1:33] = move_mask & mode_mask[..., MODE_MOVE][..., None]
    legacy_mask[..., ATTACK_START:ATTACK_START + ATTACK_HEAD_SIZE] = (
        attack_mask & mode_mask[..., MODE_ATTACK][..., None]
    )
    legacy_mask[..., InfernoAction.NO_ACTION_IDX] = mode_mask[..., MODE_NOOP]
    legacy_mask[..., WEAPON_SWITCH_START:WEAPON_SWITCH_START + SWITCH_HEAD_SIZE] = (
        switch_mask & mode_mask[..., MODE_SWITCH][..., None]
    )
    return legacy_mask


def ensure_action_mask_for_policy(
    action_mask: Sequence[bool] | np.ndarray,
    action_head_sizes: Sequence[int],
) -> np.ndarray:
    mask = np.asarray(action_mask, dtype=bool)
    expected_size = get_expected_action_mask_size(action_head_sizes)
    if mask.shape[-1] == expected_size:
        return mask
    if uses_factored_policy_actions(action_head_sizes) and mask.shape[-1] == NUM_ACTIONS:
        return legacy_action_mask_to_policy_mask(mask)
    if (not uses_factored_policy_actions(action_head_sizes)) and mask.shape[-1] == POLICY_ACTION_MASK_SIZE:
        return policy_action_mask_to_legacy_mask(mask)
    raise ValueError(
        f"Cannot coerce action mask with shape {mask.shape} to action heads {list(action_head_sizes)}"
    )


def get_mask_for_action_space(
    state: SimulatorState,
    action_head_sizes: Sequence[int],
) -> np.ndarray:
    if uses_factored_policy_actions(action_head_sizes):
        return get_policy_action_mask(state)
    return get_legacy_action_mask(state)


def get_policy_action_mask(
    state: SimulatorState,
    movement_table: MovementResolutionTable | None = None,
    tick_threat_cache: TickThreatCache | None = None,
) -> np.ndarray:
    if tick_threat_cache is not None:
        movement_table = tick_threat_cache.movement_table
    if movement_table is None:
        movement_table = build_movement_resolution_table(
            state.player_x,
            state.player_y,
            state.pillar_alive,
        )

    move_mask = np.zeros(MOVE_HEAD_SIZE, dtype=bool)
    for move_index in range(MOVE_HEAD_SIZE):
        legacy_action = move_index + 1
        settled_x = movement_table.destination_x_for_action(legacy_action)
        settled_y = movement_table.destination_y_for_action(legacy_action)
        move_mask[move_index] = (
            settled_x != state.player_x or settled_y != state.player_y
        )

    exact_targets = get_exact_target_slots(state)
    attack_mask = np.zeros(ATTACK_HEAD_SIZE, dtype=bool)
    attack_mask[:len(exact_targets)] = True

    switch_mask = np.ones(SWITCH_HEAD_SIZE, dtype=bool)
    if not state.has_blowpipe:
        switch_mask[1] = False

    mode_mask = np.array(
        [
            True,
            bool(move_mask.any()),
            bool(attack_mask.any()),
            bool(switch_mask.any()),
            True,
        ],
        dtype=bool,
    )
    return np.concatenate([mode_mask, move_mask, attack_mask, switch_mask])


def decode_policy_action(action: int | np.integer | Sequence[int] | np.ndarray) -> int:
    if isinstance(action, np.ndarray):
        if action.shape == ():
            return int(action.item())
        if action.ndim != 1:
            raise ValueError(f"Expected 1D policy action, got shape {action.shape}")
        values = action.astype(np.int32).tolist()
    elif isinstance(action, (int, np.integer)):
        return int(action)
    else:
        values = [int(v) for v in action]

    if len(values) == 1:
        return int(values[0])
    if len(values) != len(ACTION_HEAD_SIZES):
        raise ValueError(
            f"Expected {len(ACTION_HEAD_SIZES)} policy action heads, got {len(values)}"
        )

    mode, move_index, attack_index, switch_index = values
    if mode == MODE_STAY:
        return InfernoAction.STAY
    if mode == MODE_MOVE:
        return int(move_index) + 1
    if mode == MODE_ATTACK:
        return ATTACK_START + int(attack_index)
    if mode == MODE_SWITCH:
        return WEAPON_SWITCH_START + int(switch_index)
    return InfernoAction.NO_ACTION_IDX


def encode_policy_action(legacy_action: int) -> np.ndarray:
    if legacy_action == InfernoAction.STAY:
        return np.array([MODE_STAY, 0, 0, 0], dtype=np.int32)
    if 1 <= legacy_action <= 32:
        return np.array([MODE_MOVE, legacy_action - 1, 0, 0], dtype=np.int32)
    if ATTACK_START <= legacy_action < ATTACK_START + ATTACK_HEAD_SIZE:
        return np.array([MODE_ATTACK, 0, legacy_action - ATTACK_START, 0], dtype=np.int32)
    if WEAPON_SWITCH_START <= legacy_action < WEAPON_SWITCH_START + SWITCH_HEAD_SIZE:
        return np.array([MODE_SWITCH, 0, 0, legacy_action - WEAPON_SWITCH_START], dtype=np.int32)
    return np.array([MODE_NOOP, 0, 0, 0], dtype=np.int32)


def policy_action_probabilities_to_legacy(
    probabilities: Sequence[float] | np.ndarray,
) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float32)
    if probs.shape[-1] != POLICY_ACTION_MASK_SIZE:
        raise ValueError(
            f"Expected policy probability vector with {POLICY_ACTION_MASK_SIZE} entries, got {probs.shape}"
        )

    mode_probs = probs[..., :NUM_MODE_ACTIONS]
    offset = NUM_MODE_ACTIONS
    move_probs = probs[..., offset:offset + MOVE_HEAD_SIZE]
    offset += MOVE_HEAD_SIZE
    attack_probs = probs[..., offset:offset + ATTACK_HEAD_SIZE]
    offset += ATTACK_HEAD_SIZE
    switch_probs = probs[..., offset:offset + SWITCH_HEAD_SIZE]

    legacy_probs = np.zeros(probs.shape[:-1] + (NUM_ACTIONS,), dtype=np.float32)
    legacy_probs[..., InfernoAction.STAY] = mode_probs[..., MODE_STAY]
    legacy_probs[..., 1:33] = mode_probs[..., MODE_MOVE][..., None] * move_probs
    legacy_probs[..., ATTACK_START:ATTACK_START + ATTACK_HEAD_SIZE] = (
        mode_probs[..., MODE_ATTACK][..., None] * attack_probs
    )
    legacy_probs[..., InfernoAction.NO_ACTION_IDX] = mode_probs[..., MODE_NOOP]
    legacy_probs[..., WEAPON_SWITCH_START:WEAPON_SWITCH_START + SWITCH_HEAD_SIZE] = (
        mode_probs[..., MODE_SWITCH][..., None] * switch_probs
    )
    return legacy_probs


def get_movement_params(action: int) -> tuple[int, int, int]:
    return get_shared_movement_params(action)


def iter_legacy_movement_actions() -> Iterable[int]:
    yield from iter_shared_legacy_movement_actions()
