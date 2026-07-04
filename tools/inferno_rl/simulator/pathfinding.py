"""
OSRS-accurate pathfinding mechanics.

Implements NPC pathfinding and NPC collision resolution.

OSRS uses "dumb" pathfinding where entities move directly toward their target
without smart obstacle avoidance. When blocked, they try cardinal directions
in a specific order.

Collision Resolution:
When a player "steps under" an NPC (occupies one of the NPC's footprint tiles),
the NPC enters a collision-resolution state instead of using normal follow:
1. NPC selects a RANDOM cardinal direction (N/S/E/W - no diagonals)
2. If that direction is passable, the NPC moves one tile
3. If blocked, the NPC does NOT try other directions - it waits until next tick
"""

from typing import Callable, List, Optional, Tuple
from collections import deque
import random


# Direction priority for OSRS BFS (W, E, S, N, SW, SE, NW, NE)
OSRS_BFS_DIRECTIONS = [
    (-1, 0),   # West
    (1, 0),    # East
    (0, -1),   # South
    (0, 1),    # North
    (-1, -1),  # South-west
    (1, -1),   # South-east
    (-1, 1),   # North-west
    (1, 1)     # North-east
]


# Type alias for walkability checker
WalkabilityChecker = Callable[[int, int, int], bool]


class OSRSPathfinding:
    """OSRS pathfinding utilities."""

    @staticmethod
    def calculate_next_move(current_x: int, current_y: int,
                             target_x: int, target_y: int,
                             size: int,
                             checker: WalkabilityChecker) -> Tuple[int, int]:
        """
        Calculate the next move for an NPC using OSRS "dumb" pathfinding
        with diagonal corner safespotting support.

        OSRS NPCs use a simple direct-vector algorithm:
        1. Calculate intended diagonal move
        2. Apply corner safespotting: if destination overlaps player, cancel Y
        3. Try diagonal move (if still diagonal after overlap check)
        4. Try X cardinal (always prioritized over Y)
        5. Try Y cardinal (only if Chebyshev distance > 1)
        6. If all blocked: NO movement

        Corner safespotting: When an NPC's intended diagonal destination would
        overlap the player, the Y-axis movement is canceled while X-axis
        movement continues. This creates directional corner trapping.

        Args:
            current_x, current_y: Current position
            target_x, target_y: Target position (usually player position)
            size: Entity size in tiles
            checker: Walkability checker function

        Returns:
            (dx, dy) move delta, or (0, 0) if no movement
        """
        from .geometry import SimulatorGeometry

        # OSRS NPC pathfinding: direction is calculated from SW corner to target,
        # NOT from bounding box. This ensures large NPCs slide along obstacles
        # to align their SW tile with the player.
        dx = -1 if target_x < current_x else (1 if target_x > current_x else 0)
        dy = -1 if target_y < current_y else (1 if target_y > current_y else 0)

        # Already at target
        if dx == 0 and dy == 0:
            return (0, 0)

        # Calculate intended destination
        intended_x = current_x + dx
        intended_y = current_y + dy

        # Corner safespotting: if intended destination overlaps player, cancel Y
        # This allows corner trapping when player is in certain diagonal positions
        if dx != 0 and dy != 0:  # Only applies to diagonal movement intent
            if SimulatorGeometry.would_npc_overlap_player_at(
                intended_x, intended_y, size, target_x, target_y
            ):
                dy = 0  # Cancel Y movement, keep X

        # 1. Try diagonal first (if still diagonal after overlap check)
        if dx != 0 and dy != 0:
            if OSRSPathfinding.can_move_diagonally(current_x, current_y, dx, dy, size, checker):
                return (dx, dy)

        # 2. Always try X first (OSRS priority: diagonal > X > Y)
        if dx != 0 and checker(current_x + dx, current_y, size):
            return (dx, 0)
        # 3. Try Y only if Chebyshev distance > 1 (RuneLite WorldArea guard)
        if dy != 0 and max(abs(target_x - current_x), abs(target_y - current_y)) > 1:
            if checker(current_x, current_y + dy, size):
                return (0, dy)

        # No valid movement found
        return (0, 0)

    @staticmethod
    def can_move_diagonally(x: int, y: int, dx: int, dy: int,
                            size: int, checker: WalkabilityChecker) -> bool:
        """
        Check if a diagonal move is valid according to OSRS clipping rules.
        
        A diagonal move requires:
        1. Destination must be walkable
        2. BOTH intermediate cardinal tiles must be passable
        """
        # Destination must be walkable
        if not checker(x + dx, y + dy, size):
            return False

        # Both intermediates must be passable (no corner-cutting)
        cardinal_x_passable = checker(x + dx, y, size)
        cardinal_y_passable = checker(x, y + dy, size)

        return cardinal_x_passable and cardinal_y_passable

    @staticmethod
    def simulate_npc_movement(current_x: int, current_y: int,
                               target_x: int, target_y: int,
                               size: int, max_tiles: int,
                               checker: WalkabilityChecker) -> Tuple[int, int]:
        """
        Simulate NPC movement over multiple ticks using dumb pathfinding.
        
        Args:
            current_x, current_y: Starting position
            target_x, target_y: Target position
            size: Entity size
            max_tiles: Maximum tiles to move (1 for walking, 2 for running)
            checker: Walkability checker
            
        Returns:
            (new_x, new_y) position after movement
        """
        x, y = current_x, current_y
        tiles_moved = 0

        while tiles_moved < max_tiles:
            dx, dy = OSRSPathfinding.calculate_next_move(x, y, target_x, target_y, size, checker)

            if dx == 0 and dy == 0:
                break  # Can't move further

            x += dx
            y += dy
            tiles_moved += 1

            # Check if reached target
            if target_x >= x and target_x < x + size and target_y >= y and target_y < y + size:
                break

        return (x, y)

    @staticmethod
    def find_player_path(start_x: int, start_y: int,
                          target_x: int, target_y: int,
                          checker: WalkabilityChecker,
                          max_dist: int = 50) -> List[Tuple[int, int]]:
        """
        Find a path from start to target using BFS.
        
        BFS guarantees the shortest path in terms of number of steps.
        OSRS uses a fixed direction order: W, E, S, N, SW, SE, NW, NE
        
        Returns:
            List of (x, y) coordinates forming the path, or empty if no path
        """
        return OSRSPathfinding.find_player_path_to_endpoints(
            start_x, start_y, [(target_x, target_y)], checker, max_dist
        )

    @staticmethod
    def find_player_path_to_endpoints(start_x: int, start_y: int,
                                       endpoints: List[Tuple[int, int]],
                                       checker: WalkabilityChecker,
                                       max_dist: int = 50) -> List[Tuple[int, int]]:
        """
        Find shortest path from start to ANY of the endpoint tiles using BFS.
        
        This matches osrs-sdk constructPaths behavior:
        - BFS explores from start position
        - Returns path when ANY endpoint is reached
        - Endpoints should be ordered with SW tile first for tie-breaking
        - OSRS direction order: W, E, S, N, SW, SE, NW, NE
        
        Args:
            start_x, start_y: Starting position
            endpoints: List of valid destination tiles, SW tile first
            checker: Walkability checker function
            max_dist: Maximum BFS search distance
            
        Returns:
            List of (x, y) coordinates forming the path, or empty if no path
        """
        if not endpoints:
            return []
        
        # Convert endpoints to set for O(1) lookup
        endpoint_set = set(endpoints)
        
        # Check if already at an endpoint
        if (start_x, start_y) in endpoint_set:
            return []

        queue = deque()
        visited = {}

        start_node = (start_x, start_y, None, 0)  # (x, y, parent, steps)
        queue.append(start_node)
        visited[(start_x, start_y)] = start_node

        while queue:
            current = queue.popleft()
            x, y, parent, steps = current

            # Check if reached ANY endpoint
            if (x, y) in endpoint_set:
                return OSRSPathfinding._reconstruct_path(current, visited)

            # Don't search too far
            if steps >= max_dist:
                continue

            # Try all 8 directions in OSRS order
            for dx, dy in OSRS_BFS_DIRECTIONS:
                nx, ny = x + dx, y + dy
                key = (nx, ny)

                if key in visited:
                    continue

                # For diagonal movement, use proper clipping check
                if dx != 0 and dy != 0:
                    if not OSRSPathfinding.can_move_diagonally(x, y, dx, dy, 1, checker):
                        continue
                else:
                    # Cardinal movement - just check destination
                    if not checker(nx, ny, 1):
                        continue

                new_node = (nx, ny, (x, y), steps + 1)
                visited[key] = new_node
                queue.append(new_node)

        # No path found
        return []

    @staticmethod
    def _reconstruct_path(end_node: tuple, visited: dict) -> List[Tuple[int, int]]:
        """Reconstruct path from BFS result."""
        path = []
        x, y, parent, _ = end_node

        while parent is not None:
            path.append((x, y))
            node = visited.get(parent)
            if node is None:
                break
            x, y, parent, _ = node

        path.reverse()
        return path

    @staticmethod
    def simulate_player_movement(current_x: int, current_y: int,
                                  target_x: int, target_y: int,
                                  max_tiles: int,
                                  checker: WalkabilityChecker) -> Tuple[int, int]:
        """
        Simulate player movement using OSRS-style BFS pathfinding.
        
        Args:
            current_x, current_y: Starting position
            target_x, target_y: Target position
            max_tiles: Maximum tiles to move (1 for walking, 2 for running)
            checker: Walkability checker
            
        Returns:
            (new_x, new_y) position after movement
        """
        if current_x == target_x and current_y == target_y:
            return (current_x, current_y)

        # Calculate the full BFS path
        path = OSRSPathfinding.find_player_path(
            current_x, current_y, target_x, target_y, checker
        )

        if not path:
            # No path found - stay in place
            return (current_x, current_y)

        # Follow the path for up to max_tiles steps
        x, y = current_x, current_y
        tiles_moved = 0

        for step_x, step_y in path:
            if tiles_moved >= max_tiles:
                break
            x, y = step_x, step_y
            tiles_moved += 1

        return (x, y)


class NpcCollisionResolver:
    """
    Handles NPC collision resolution when an NPC and player occupy the same tile(s).
    
    In OSRS, when a player "steps under" an NPC (occupies one of the NPC's footprint tiles),
    the NPC enters a collision-resolution state instead of using the normal follow algorithm:
    
    1. NPC selects a RANDOM cardinal direction (N/S/E/W - no diagonals)
    2. If that direction is passable, the NPC moves one tile
    3. If blocked, the NPC does NOT try other directions - it waits until next tick
    
    This creates stochastic "jittering" behavior where NPCs may remain stacked for multiple
    ticks depending on how many directions are blocked.
    
    Probability of movement per tick = (free cardinal directions) / 4
    - Open space (4 free): 100% chance to move
    - Corner (2 free): 50% chance to move  
    - U-trap (1 free): 25% chance to move
    - Full trap (0 free): 0% - NPC is stuck
    """
    
    # Cardinal directions only (no diagonals for collision resolution)
    CARDINAL_DIRECTIONS = [
        (0, 1),   # North
        (0, -1),  # South
        (1, 0),   # East
        (-1, 0)   # West
    ]
    
    @staticmethod
    def is_player_under_npc(npc_x: int, npc_y: int, npc_size: int,
                            player_x: int, player_y: int) -> bool:
        """
        Check if a player is occupying any tile within an NPC's footprint.
        
        Args:
            npc_x, npc_y: NPC's SW anchor coordinate
            npc_size: NPC size in tiles (e.g., 3 for a 3x3 NPC)
            player_x, player_y: Player's coordinate
            
        Returns:
            True if player is under (overlapping with) the NPC
        """
        return (npc_x <= player_x < npc_x + npc_size and
                npc_y <= player_y < npc_y + npc_size)
    
    @staticmethod
    def resolve_collision(npc_x: int, npc_y: int, npc_size: int,
                          checker: WalkabilityChecker) -> Tuple[int, int]:
        """
        Resolve collision by attempting to move the NPC in a random cardinal direction.
        
        This implements OSRS's collision resolution behavior:
        - Only cardinal directions are considered (N/S/E/W)
        - A single random direction is chosen
        - If that direction is blocked, NPC doesn't move (waits for next tick)
        
        Args:
            npc_x, npc_y: NPC's SW anchor coordinate
            npc_size: NPC size in tiles
            checker: Walkability checker for the new position
            
        Returns:
            (dx, dy) move delta, or (0, 0) if blocked
        """
        # Pick a single random cardinal direction
        dx, dy = random.choice(NpcCollisionResolver.CARDINAL_DIRECTIONS)
        
        # Check if NPC can move in that direction
        if checker(npc_x + dx, npc_y + dy, npc_size):
            return (dx, dy)
        
        # Direction blocked - NPC stays put this tick
        return (0, 0)
    
    @staticmethod
    def resolve_collision_try_all(npc_x: int, npc_y: int, npc_size: int,
                                  checker: WalkabilityChecker) -> Tuple[int, int]:
        """
        Alternative resolution that tries all cardinal directions in random order.
        
        This is a more lenient variant that increases the chance of movement
        but is less authentic to OSRS behavior. Use this if you want NPCs
        to escape more reliably.
        
        Returns:
            (dx, dy) for the first valid direction, or (0, 0) if all blocked
        """
        shuffled = list(NpcCollisionResolver.CARDINAL_DIRECTIONS)
        random.shuffle(shuffled)
        
        for dx, dy in shuffled:
            if checker(npc_x + dx, npc_y + dy, npc_size):
                return (dx, dy)
        
        # All directions blocked - NPC is fully trapped
        return (0, 0)
