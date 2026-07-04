"""
Test blob mini-spawn position patterns.

Verifies that blob splits spawn in the correct direction based on
blob position relative to the NE pillar, rather than using hardcoded
southeast offsets.
"""
import pytest
from inferno_rl.simulator.simulator import InfernoSimulator


class TestBlobSpawnPatterns:
    """Test directional blob spawn patterns based on pillar proximity."""

    def test_blob_spawn_south_of_pillar(self):
        """
        Blob at (16,19) south of NE pillar should spawn NORTH pattern.

        Expected positions:
        - Ket (melee):  (16, 20) - offset [+0, +1]
        - Xil (ranged): (16, 21) - offset [+0, +2]
        - Mej (magic):  (17, 21) - offset [+1, +2]

        NE pillar occupies (17, 22) to (19, 24).
        Blob center at (17, 20) is south of pillar center (18, 23).
        Should spawn northward toward pillar.
        """
        simulator = InfernoSimulator()
        offsets = simulator._get_blob_spawn_pattern(16, 19)

        expected_offsets = [(0, 1), (0, 2), (1, 2)]  # Ket, Xil, Mej
        assert offsets == expected_offsets, (
            f"South-of-pillar pattern failed. "
            f"Expected {expected_offsets}, got {offsets}"
        )

        # Verify absolute positions
        blob_x, blob_y = 16, 19
        expected_positions = [
            (16, 20),  # Ket
            (16, 21),  # Xil
            (17, 21),  # Mej
        ]
        actual_positions = [
            (blob_x + dx, blob_y + dy) for dx, dy in offsets
        ]
        assert actual_positions == expected_positions, (
            f"Expected positions {expected_positions}, "
            f"got {actual_positions}"
        )

    def test_blob_spawn_west_of_pillar(self):
        """
        Blob at (14,21) west of NE pillar should spawn EAST pattern.

        Expected positions:
        - Ket (melee):  (15, 21) - offset [+1, +0]
        - Xil (ranged): (16, 21) - offset [+2, +0]
        - Mej (magic):  (16, 22) - offset [+2, +1]

        NE pillar occupies (17, 22) to (19, 24).
        Blob center at (15, 22) is west of pillar center (18, 23).
        Should spawn eastward toward open space.
        """
        simulator = InfernoSimulator()
        offsets = simulator._get_blob_spawn_pattern(14, 21)

        expected_offsets = [(1, 0), (2, 0), (2, 1)]  # Ket, Xil, Mej
        assert offsets == expected_offsets, (
            f"West-of-pillar pattern failed. "
            f"Expected {expected_offsets}, got {offsets}"
        )

        # Verify absolute positions
        blob_x, blob_y = 14, 21
        expected_positions = [
            (15, 21),  # Ket
            (16, 21),  # Xil
            (16, 22),  # Mej
        ]
        actual_positions = [
            (blob_x + dx, blob_y + dy) for dx, dy in offsets
        ]
        assert actual_positions == expected_positions, (
            f"Expected positions {expected_positions}, "
            f"got {actual_positions}"
        )

    def test_blob_spawn_east_of_pillar(self):
        """
        Blob at (20,24) east of NE pillar should spawn NORTHEAST diagonal pattern.

        Expected positions:
        - Ket (melee):  (20, 24) - offset [+0, +0]
        - Xil (ranged): (21, 25) - offset [+1, +1]
        - Mej (magic):  (22, 26) - offset [+2, +2]

        NE pillar occupies (17, 22) to (19, 24).
        Blob center at (21, 25) is east of pillar center (18, 23).
        Should spawn northeast diagonal toward open space.
        """
        simulator = InfernoSimulator()
        offsets = simulator._get_blob_spawn_pattern(20, 24)

        expected_offsets = [(0, 0), (1, 1), (2, 2)]  # Ket, Xil, Mej
        assert offsets == expected_offsets, (
            f"East-of-pillar pattern failed. "
            f"Expected {expected_offsets}, got {offsets}"
        )

        # Verify absolute positions
        blob_x, blob_y = 20, 24
        expected_positions = [
            (20, 24),  # Ket
            (21, 25),  # Xil
            (22, 26),  # Mej
        ]
        actual_positions = [
            (blob_x + dx, blob_y + dy) for dx, dy in offsets
        ]
        assert actual_positions == expected_positions, (
            f"Expected positions {expected_positions}, "
            f"got {actual_positions}"
        )

    def test_blob_spawn_north_of_pillar(self):
        """
        Blob north of NE pillar should spawn NORTH pattern.

        Blob at (17, 26) - north of pillar.
        NE pillar occupies (17, 22) to (19, 24).
        Blob center at (18, 27) is north of pillar center (18, 23).
        Should spawn northward away from pillar.
        """
        simulator = InfernoSimulator()
        offsets = simulator._get_blob_spawn_pattern(17, 26)

        expected_offsets = [(0, 1), (0, 2), (1, 2)]  # Ket, Xil, Mej
        assert offsets == expected_offsets, (
            f"North-of-pillar pattern failed. "
            f"Expected {expected_offsets}, got {offsets}"
        )

    def test_blob_spawn_patterns_deterministic(self):
        """
        Verify pattern selection is deterministic for the same blob position.
        """
        simulator = InfernoSimulator()

        # Test same position multiple times
        position = (16, 19)
        first_result = simulator._get_blob_spawn_pattern(*position)
        for _ in range(5):
            result = simulator._get_blob_spawn_pattern(*position)
            assert result == first_result, (
                "Spawn pattern should be deterministic for same position"
            )

    @pytest.mark.parametrize("blob_x,blob_y,expected_pattern", [
        # South of pillar
        (16, 19, [(0, 1), (0, 2), (1, 2)]),
        (17, 19, [(0, 1), (0, 2), (1, 2)]),
        (18, 19, [(0, 1), (0, 2), (1, 2)]),

        # West of pillar
        (14, 21, [(1, 0), (2, 0), (2, 1)]),
        (14, 22, [(1, 0), (2, 0), (2, 1)]),
        (14, 23, [(1, 0), (2, 0), (2, 1)]),

        # East of pillar
        (20, 24, [(0, 0), (1, 1), (2, 2)]),
        (21, 24, [(0, 0), (1, 1), (2, 2)]),
        (22, 24, [(0, 0), (1, 1), (2, 2)]),

        # North of pillar
        (17, 26, [(0, 1), (0, 2), (1, 2)]),
        (18, 27, [(0, 1), (0, 2), (1, 2)]),
    ])
    def test_blob_spawn_pattern_coverage(self, blob_x, blob_y, expected_pattern):
        """
        Test spawn patterns for various blob positions around NE pillar.
        """
        simulator = InfernoSimulator()
        offsets = simulator._get_blob_spawn_pattern(blob_x, blob_y)
        assert offsets == expected_pattern, (
            f"Pattern for ({blob_x}, {blob_y}) failed. "
            f"Expected {expected_pattern}, got {offsets}"
        )
