from __future__ import annotations

from dataclasses import dataclass
import numpy as np

ROWS = 6
COLS = 7


@dataclass(frozen=True)
class Position:
    board: np.ndarray  # shape (6, 7), values in {-1, 0, 1}
    to_play: int  # 1 or -1


def new_game() -> Position:
    return Position(board=np.zeros((ROWS, COLS), dtype=np.int8), to_play=1)


def legal_moves(pos: Position) -> list[int]:
    return [c for c in range(COLS) if pos.board[0, c] == 0]


def apply_move(pos: Position, col: int) -> Position:
    board = pos.board.copy()
    for row in range(ROWS - 1, -1, -1):
        if board[row, col] == 0:
            board[row, col] = pos.to_play
            return Position(board=board, to_play=-pos.to_play)
    raise ValueError(f"column {col} is full")


def winner(board: np.ndarray) -> int:
    for r in range(ROWS):
        for c in range(COLS):
            p = board[r, c]
            if p == 0:
                continue
            if c <= COLS - 4 and all(board[r, c + i] == p for i in range(4)):
                return int(p)
            if r <= ROWS - 4 and all(board[r + i, c] == p for i in range(4)):
                return int(p)
            if r <= ROWS - 4 and c <= COLS - 4 and all(board[r + i, c + i] == p for i in range(4)):
                return int(p)
            if r <= ROWS - 4 and c >= 3 and all(board[r + i, c - i] == p for i in range(4)):
                return int(p)
    return 0


def is_terminal(pos: Position) -> tuple[bool, int]:
    w = winner(pos.board)
    if w != 0:
        return True, w
    if not legal_moves(pos):
        return True, 0
    return False, 0


def canonicalize(pos: Position) -> np.ndarray:
    own = (pos.board == pos.to_play).astype(np.float32)
    opp = (pos.board == -pos.to_play).astype(np.float32)
    return np.stack([own, opp], axis=0)


def mirror_board(board: np.ndarray) -> np.ndarray:
    return np.flip(board, axis=1).copy()


def mirror_policy(policy: np.ndarray) -> np.ndarray:
    return policy[::-1].copy()

