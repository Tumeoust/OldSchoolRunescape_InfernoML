from __future__ import annotations


PLAYER_MOVE_DIRECTIONS = (
    (0, 1),
    (0, -1),
    (1, 0),
    (-1, 0),
    (1, 1),
    (-1, 1),
    (1, -1),
    (-1, -1),
)

DIRECTIONAL_MOVE_ACTIONS = (
    2,
    6,
    10,
    14,
    18,
    22,
    26,
    30,
)

_MOVEMENT_PARAMS = (
    (0, 0, 0),
    (0, 1, 1), (0, 1, 2), (0, 1, 3), (0, 1, 4),
    (0, -1, 1), (0, -1, 2), (0, -1, 3), (0, -1, 4),
    (1, 0, 1), (1, 0, 2), (1, 0, 3), (1, 0, 4),
    (-1, 0, 1), (-1, 0, 2), (-1, 0, 3), (-1, 0, 4),
    (1, 1, 1), (1, 1, 2), (1, 1, 3), (1, 1, 4),
    (-1, 1, 1), (-1, 1, 2), (-1, 1, 3), (-1, 1, 4),
    (1, -1, 1), (1, -1, 2), (1, -1, 3), (1, -1, 4),
    (-1, -1, 1), (-1, -1, 2), (-1, -1, 3), (-1, -1, 4),
)


def get_movement_params(action: int) -> tuple[int, int, int]:
    if 0 <= action < len(_MOVEMENT_PARAMS):
        return _MOVEMENT_PARAMS[action]
    return (0, 0, 0)


def iter_legacy_movement_actions():
    for action in range(1, len(_MOVEMENT_PARAMS)):
        yield action
