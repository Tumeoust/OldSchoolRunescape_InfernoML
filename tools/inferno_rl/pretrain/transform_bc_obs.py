"""
Transform BC observation data between layout versions.

Supported transforms:
  186 → 220: Adds pillar-relative position features (player + entities)
  220 → 262: Adds 6 nibbler slots (42 floats) between entity slots and wave context

Auto-detects input dimension and applies the correct chain of transforms.

Usage:
    python -m tools.inferno_rl.pretrain.transform_bc_obs \
        --input models/bc_data/bc_w35-66_v3.npz \
        --output models/bc_data/bc_w35-66_v4.npz
"""

import argparse

import numpy as np

# ── 186 → 220 constants ─────────────────────────────────────────────────────

OLD_186_DIM = 186
OLD_186_PLAYER_POS = slice(0, 2)
OLD_186_PLAYER_REST = slice(2, 8)
OLD_186_PILLARS = slice(8, 20)
OLD_186_ENTITY_START = 20
OLD_186_ENTITY_SIZE = 10
OLD_186_WAVE_CTX = slice(180, 186)

NE_PILLAR_X_NORM = 18.0 / 29.0
NE_PILLAR_Y_NORM = 23.0 / 30.0

# ── 220 → 262 constants ─────────────────────────────────────────────────────

OLD_220_DIM = 220
NEW_262_DIM = 262
ENTITY_END_220 = 214          # [0:214] = player + pillars + entities
WAVE_CTX_START_220 = 214      # [214:220] = wave context
NIBBLER_TOTAL_SIZE = 42       # 6 slots × 7 floats
WAVE_CTX_START_262 = 256      # [256:262] = wave context in new layout

NUM_SLOTS = 16
NEW_220_ENTITY_SIZE = 12
NEW_220_ENTITY_START = 22


def transform_186_to_220(old: np.ndarray) -> np.ndarray:
    """Transform (N, 186) observations to (N, 220)."""
    assert old.shape[1] == OLD_186_DIM, f"Expected {OLD_186_DIM} cols, got {old.shape[1]}"
    N = old.shape[0]
    new = np.zeros((N, OLD_220_DIM), dtype=np.float32)

    new[:, 0:2] = old[:, OLD_186_PLAYER_POS]
    new[:, 2] = old[:, 0] - NE_PILLAR_X_NORM
    new[:, 3] = old[:, 1] - NE_PILLAR_Y_NORM
    new[:, 4:10] = old[:, OLD_186_PLAYER_REST]
    new[:, 10:22] = old[:, OLD_186_PILLARS]

    for slot in range(NUM_SLOTS):
        old_start = OLD_186_ENTITY_START + slot * OLD_186_ENTITY_SIZE
        new_start = NEW_220_ENTITY_START + slot * NEW_220_ENTITY_SIZE
        new[:, new_start:new_start + OLD_186_ENTITY_SIZE] = old[:, old_start:old_start + OLD_186_ENTITY_SIZE]
        entity_x = old[:, old_start + 2]
        entity_y = old[:, old_start + 3]
        exists = old[:, old_start]
        new[:, new_start + 10] = (entity_x - NE_PILLAR_X_NORM) * exists
        new[:, new_start + 11] = (entity_y - NE_PILLAR_Y_NORM) * exists

    new[:, WAVE_CTX_START_220:WAVE_CTX_START_220 + 6] = old[:, OLD_186_WAVE_CTX]
    return new


def transform_220_to_262(old: np.ndarray) -> np.ndarray:
    """Transform (N, 220) observations to (N, 262).

    Inserts 42 zeros for nibbler slots between entity slots and wave context.
    BC data has no nibbler observations — slots are all zeros.
    """
    assert old.shape[1] == OLD_220_DIM, f"Expected {OLD_220_DIM} cols, got {old.shape[1]}"
    N = old.shape[0]
    new = np.zeros((N, NEW_262_DIM), dtype=np.float32)

    # [0:214] player + pillars + entity slots — unchanged
    new[:, 0:ENTITY_END_220] = old[:, 0:ENTITY_END_220]

    # [214:256] nibbler slots — all zeros (already initialized)

    # [256:262] wave context — shifted from [214:220]
    new[:, WAVE_CTX_START_262:WAVE_CTX_START_262 + 6] = old[:, WAVE_CTX_START_220:WAVE_CTX_START_220 + 6]

    return new


def validate_220_to_262(old: np.ndarray, new: np.ndarray) -> None:
    """Spot-check the 220→262 transformation."""
    N = old.shape[0]
    assert new.shape == (N, NEW_262_DIM), f"Shape mismatch: {new.shape}"

    # Player + pillars + entities unchanged
    assert np.allclose(new[:, 0:ENTITY_END_220], old[:, 0:ENTITY_END_220]), \
        "Player/pillar/entity section mismatch"

    # Nibbler slots all zeros
    nibbler_sum = new[:, ENTITY_END_220:WAVE_CTX_START_262].sum()
    assert nibbler_sum == 0.0, f"Nibbler slots should be all zeros, sum={nibbler_sum}"

    # Wave context shifted correctly
    assert np.allclose(
        new[:, WAVE_CTX_START_262:WAVE_CTX_START_262 + 6],
        old[:, WAVE_CTX_START_220:WAVE_CTX_START_220 + 6],
    ), "Wave context mismatch"

    print("  Player+pillars+entities [0:214]: OK")
    print(f"  Nibbler slots [214:256] sum: {nibbler_sum}")
    print("  Wave context [256:262]: OK")
    print("  All validations passed.")


def validate_186_to_220(old: np.ndarray, new: np.ndarray) -> None:
    """Spot-check the 186→220 transformation."""
    N = old.shape[0]
    assert new.shape == (N, OLD_220_DIM), f"Shape mismatch: {new.shape}"

    assert np.allclose(new[:, 0:2], old[:, 0:2]), "Player pos mismatch"
    assert np.allclose(new[:, 4:10], old[:, 2:8]), "Player rest mismatch"
    assert np.allclose(new[:, 10:22], old[:, 8:20]), "Pillar mismatch"
    assert np.allclose(new[:, 214:220], old[:, 180:186]), "Wave context mismatch"

    expected_rel_x = old[:, 0] - NE_PILLAR_X_NORM
    expected_rel_y = old[:, 1] - NE_PILLAR_Y_NORM
    assert np.allclose(new[:, 2], expected_rel_x), "Player rel_x mismatch"
    assert np.allclose(new[:, 3], expected_rel_y), "Player rel_y mismatch"

    for slot in range(NUM_SLOTS):
        old_exists = old[:, OLD_186_ENTITY_START + slot * OLD_186_ENTITY_SIZE]
        new_exists = new[:, NEW_220_ENTITY_START + slot * NEW_220_ENTITY_SIZE]
        assert np.allclose(old_exists, new_exists), f"Slot {slot} exists mismatch"

    print("  All validations passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Transform BC obs between layout versions")
    parser.add_argument("--input", type=str, required=True, help="Input .npz file")
    parser.add_argument("--output", type=str, required=True, help="Output .npz file")
    args = parser.parse_args()

    print(f"Loading {args.input}")
    data = np.load(args.input)
    old_obs = data["observations"]
    actions = data["actions"]
    action_masks = data["action_masks"]
    input_dim = old_obs.shape[1]
    print(f"  observations: {old_obs.shape}, actions: {actions.shape}, action_masks: {action_masks.shape}")

    if input_dim == OLD_186_DIM:
        print("Transforming 186 -> 220 -> 262...")
        mid_obs = transform_186_to_220(old_obs)
        print("Validating 186 -> 220...")
        validate_186_to_220(old_obs, mid_obs)
        new_obs = transform_220_to_262(mid_obs)
        print("Validating 220 -> 262...")
        validate_220_to_262(mid_obs, new_obs)
    elif input_dim == OLD_220_DIM:
        print("Transforming 220 -> 262...")
        new_obs = transform_220_to_262(old_obs)
        print("Validating...")
        validate_220_to_262(old_obs, new_obs)
    elif input_dim == NEW_262_DIM:
        print("Input is already 262-dim, nothing to do.")
        return
    else:
        raise ValueError(f"Unsupported input dimension: {input_dim}. Expected 186, 220, or 262.")

    print(f"Saving to {args.output}")
    np.savez_compressed(args.output, observations=new_obs, actions=actions, action_masks=action_masks)
    print(f"  Output shape: {new_obs.shape}")
    print("Done.")


if __name__ == "__main__":
    main()
