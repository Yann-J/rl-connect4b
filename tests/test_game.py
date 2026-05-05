from __future__ import annotations

import numpy as np

from connect4.game import Position, apply_move, canonicalize, is_terminal, mirror_board, mirror_policy, new_game, winner


def test_legal_and_apply_move() -> None:
    pos = new_game()
    pos = apply_move(pos, 0)
    assert pos.board[5, 0] == 1


def test_winner_all_directions() -> None:
    b = np.zeros((6, 7), dtype=np.int8)
    b[5, :4] = 1
    assert winner(b) == 1
    b = np.zeros((6, 7), dtype=np.int8)
    b[2:6, 0] = -1
    assert winner(b) == -1
    b = np.zeros((6, 7), dtype=np.int8)
    for i in range(4):
        b[2 + i, i] = 1
    assert winner(b) == 1
    b = np.zeros((6, 7), dtype=np.int8)
    for i in range(4):
        b[2 + i, 6 - i] = -1
    assert winner(b) == -1


def test_canonicalize_roundtrip_players() -> None:
    b = np.zeros((6, 7), dtype=np.int8)
    b[5, 0] = 1
    b[5, 1] = -1
    p1 = canonicalize(Position(b, 1))
    p2 = canonicalize(Position(b, -1))
    assert p1[0, 5, 0] == 1 and p1[1, 5, 1] == 1
    assert p2[0, 5, 1] == 1 and p2[1, 5, 0] == 1


def test_mirror_involution_and_policy() -> None:
    b = np.arange(42, dtype=np.int8).reshape(6, 7)
    assert np.array_equal(mirror_board(mirror_board(b)), b)
    p = np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.float32)
    assert np.array_equal(mirror_policy(mirror_policy(p)), p)


def test_terminal_draw() -> None:
    b = np.array(
        [
            [1, 1, -1, -1, 1, 1, -1],
            [-1, -1, 1, 1, -1, -1, 1],
            [1, 1, -1, -1, 1, 1, -1],
            [-1, -1, 1, 1, -1, -1, 1],
            [1, 1, -1, -1, 1, 1, -1],
            [-1, -1, 1, 1, -1, -1, 1],
        ],
        dtype=np.int8,
    )
    t, w = is_terminal(Position(b, 1))
    assert t and w == 0

