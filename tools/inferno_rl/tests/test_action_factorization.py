import numpy as np

from tools.inferno_rl.training.actions import (
    ACTION_HEAD_SIZES,
    InfernoAction,
    MODE_ATTACK,
    MODE_MOVE,
    MODE_NOOP,
    MODE_STAY,
    MODE_SWITCH,
    POLICY_ACTION_DEPENDENCIES,
    POLICY_ACTION_MASK_SIZE,
    decode_policy_action,
    ensure_action_mask_for_policy,
    encode_policy_action,
    legacy_action_mask_to_policy_mask,
    policy_action_mask_to_legacy_mask,
    policy_action_probabilities_to_legacy,
)


def test_policy_action_shape_constants_are_consistent() -> None:
    assert ACTION_HEAD_SIZES == [5, 32, 14, 4]
    assert POLICY_ACTION_MASK_SIZE == sum(ACTION_HEAD_SIZES)
    assert set(POLICY_ACTION_DEPENDENCIES.keys()) == {1, 2, 3}


def test_encode_decode_round_trip_for_representative_actions() -> None:
    actions = [
        InfernoAction.STAY,
        1,
        18,
        InfernoAction.ATTACK_TARGET_1,
        InfernoAction.ATTACK_TARGET_14,
        InfernoAction.SWITCH_BOFA,
        InfernoAction.SWITCH_BLOOD_BARRAGE,
        InfernoAction.NO_ACTION_IDX,
    ]
    for legacy_action in actions:
        encoded = encode_policy_action(legacy_action)
        assert encoded.shape == (4,)
        assert decode_policy_action(encoded) == legacy_action


def test_decode_policy_action_from_explicit_heads() -> None:
    assert decode_policy_action(np.array([MODE_STAY, 0, 0, 0], dtype=np.int32)) == 0
    assert decode_policy_action(np.array([MODE_MOVE, 5, 0, 0], dtype=np.int32)) == 6
    assert decode_policy_action(np.array([MODE_ATTACK, 0, 13, 0], dtype=np.int32)) == 46
    assert decode_policy_action(np.array([MODE_SWITCH, 0, 0, 3], dtype=np.int32)) == 51
    assert decode_policy_action(np.array([MODE_NOOP, 0, 0, 0], dtype=np.int32)) == 47


def test_legacy_mask_round_trips_through_policy_conversion() -> None:
    legacy_mask = np.zeros(52, dtype=bool)
    legacy_mask[InfernoAction.STAY] = True
    legacy_mask[1] = True
    legacy_mask[4] = True
    legacy_mask[InfernoAction.ATTACK_TARGET_1] = True
    legacy_mask[InfernoAction.ATTACK_TARGET_14] = True
    legacy_mask[InfernoAction.NO_ACTION_IDX] = True
    legacy_mask[InfernoAction.SWITCH_BOFA] = True

    policy_mask = legacy_action_mask_to_policy_mask(legacy_mask)
    assert policy_mask.shape == (POLICY_ACTION_MASK_SIZE,)

    round_tripped = policy_action_mask_to_legacy_mask(policy_mask)
    np.testing.assert_array_equal(round_tripped, legacy_mask)
    np.testing.assert_array_equal(
        ensure_action_mask_for_policy(legacy_mask, ACTION_HEAD_SIZES),
        policy_mask,
    )


def test_policy_probabilities_decode_to_legacy_probabilities() -> None:
    policy_probs = np.zeros(POLICY_ACTION_MASK_SIZE, dtype=np.float32)
    policy_probs[:5] = np.array([0.2, 0.3, 0.25, 0.15, 0.1], dtype=np.float32)
    policy_probs[5:37] = 1.0 / 32.0
    policy_probs[37:51] = 0.0
    policy_probs[37] = 0.7
    policy_probs[50] = 0.3
    policy_probs[51:55] = np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float32)

    legacy_probs = policy_action_probabilities_to_legacy(policy_probs)

    assert legacy_probs.shape == (52,)
    assert np.isclose(legacy_probs[InfernoAction.STAY], 0.2)
    assert np.isclose(legacy_probs[1], 0.3 / 32.0)
    assert np.isclose(legacy_probs[InfernoAction.ATTACK_TARGET_1], 0.25 * 0.7)
    assert np.isclose(legacy_probs[InfernoAction.ATTACK_TARGET_14], 0.25 * 0.3)
    assert np.isclose(legacy_probs[InfernoAction.NO_ACTION_IDX], 0.1)
    assert np.isclose(legacy_probs[InfernoAction.SWITCH_BOFA], 0.15 * 0.4)
