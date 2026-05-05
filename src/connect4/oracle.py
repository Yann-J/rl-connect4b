from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

from .game import Position, apply_move, is_terminal, legal_moves


@dataclass(frozen=True)
class OracleResult:
    value: float
    best_move: int


class Oracle(Protocol):
    def evaluate(self, pos: Position) -> OracleResult: ...


class MinimaxOracle:
    def __init__(self, depth: int = 6, use_tt: bool = True) -> None:
        self.depth = depth
        self.use_tt = use_tt
        self._tt: dict[tuple[bytes, int, int], tuple[float, int]] = {}

    def evaluate(self, pos: Position) -> OracleResult:
        value, move = self._search(pos, self.depth, -math.inf, math.inf)
        if move == -1:
            legal = legal_moves(pos)
            move = legal[0] if legal else -1
        return OracleResult(value=float(value), best_move=int(move))

    def _search(
        self,
        pos: Position,
        depth: int,
        alpha: float,
        beta: float,
    ) -> tuple[float, int]:
        terminal, winner = is_terminal(pos)
        if terminal:
            if winner == 0:
                return 0.0, -1
            return (1.0 if winner == pos.to_play else -1.0), -1
        if depth <= 0:
            return 0.0, -1

        key = (pos.board.tobytes(), pos.to_play, depth)
        if self.use_tt and key in self._tt:
            return self._tt[key]

        best_value = -math.inf
        best_move = -1
        for move in legal_moves(pos):
            child = apply_move(pos, move)
            child_value, _ = self._search(child, depth - 1, -beta, -alpha)
            value = -child_value
            if value > best_value:
                best_value = value
                best_move = move
            alpha = max(alpha, value)
            if alpha >= beta:
                break

        result = (best_value, best_move)
        if self.use_tt:
            self._tt[key] = result
        return result


class PonsOracle:
    _ROWS = 6
    _COLS = 7
    _BITS_PER_COL = _ROWS + 1
    _CENTER_FIRST = (3, 2, 4, 1, 5, 0, 6)

    def __init__(self, use_tt: bool = True) -> None:
        self.use_tt = use_tt
        self._tt: dict[tuple[int, int], int] = {}
        self._best_move_tt: dict[tuple[int, int], int] = {}
        self._bottom_masks = tuple(
            1 << (col * self._BITS_PER_COL) for col in range(self._COLS)
        )
        self._top_masks = tuple(
            1 << (col * self._BITS_PER_COL + self._ROWS - 1)
            for col in range(self._COLS)
        )

    def evaluate(self, pos: Position) -> OracleResult:
        terminal, winner = is_terminal(pos)
        if terminal:
            if winner == 0:
                return OracleResult(value=0.0, best_move=-1)
            return OracleResult(
                value=1.0 if winner == pos.to_play else -1.0,
                best_move=-1,
            )
        current, mask = self._encode_position(pos)
        best_move = -1
        best_score = -2
        alpha = -1
        beta = 1
        for move in self._ordered_moves(mask):
            if self._is_winning_move(current, mask, move):
                return OracleResult(value=1.0, best_move=move)
            next_current, next_mask = self._play(current, mask, move)
            score = -self._solve(next_current, next_mask, -beta, -alpha)
            if score > best_score:
                best_score = score
                best_move = move
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break
        if best_move == -1:
            legal = legal_moves(pos)
            best_move = legal[0] if legal else -1
            best_score = 0 if best_score == -2 else best_score
        return OracleResult(value=float(best_score), best_move=int(best_move))

    def _solve(self, current: int, mask: int, alpha: int, beta: int) -> int:
        if self._has_alignment(mask ^ current):
            return -1
        if mask == self._full_mask():
            return 0
        key = self._tt_key(current, mask)
        if self.use_tt:
            cached = self._tt.get(key)
            if cached is not None:
                return cached
        for move in self._ordered_moves(mask):
            if self._is_winning_move(current, mask, move):
                if self.use_tt:
                    self._tt[key] = 1
                    self._best_move_tt[key] = move
                return 1

        best = -1
        best_move = -1
        for move in self._ordered_moves(mask):
            next_current, next_mask = self._play(current, mask, move)
            score = -self._solve(next_current, next_mask, -beta, -alpha)
            if score > best:
                best = score
                best_move = move
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break
        if self.use_tt:
            self._tt[key] = best
            if best_move != -1:
                self._best_move_tt[key] = best_move
        return best

    def _ordered_moves(self, mask: int) -> list[int]:
        return [col for col in self._CENTER_FIRST if self._can_play(mask, col)]

    def _play(self, current: int, mask: int, col: int) -> tuple[int, int]:
        return current ^ mask, mask | (mask + self._bottom_masks[col])

    def _can_play(self, mask: int, col: int) -> bool:
        return (mask & self._top_masks[col]) == 0

    def _is_winning_move(self, current: int, mask: int, col: int) -> bool:
        pos = current | (mask + self._bottom_masks[col])
        return self._has_alignment(pos)

    def _has_alignment(self, bitboard: int) -> bool:
        m = bitboard & (bitboard >> self._BITS_PER_COL)
        if m & (m >> (2 * self._BITS_PER_COL)):
            return True
        m = bitboard & (bitboard >> (self._BITS_PER_COL - 1))
        if m & (m >> (2 * (self._BITS_PER_COL - 1))):
            return True
        m = bitboard & (bitboard >> (self._BITS_PER_COL + 1))
        if m & (m >> (2 * (self._BITS_PER_COL + 1))):
            return True
        m = bitboard & (bitboard >> 1)
        return bool(m & (m >> 2))

    def _full_mask(self) -> int:
        full = 0
        for col in range(self._COLS):
            full |= ((1 << self._ROWS) - 1) << (col * self._BITS_PER_COL)
        return full

    def _encode_position(self, pos: Position) -> tuple[int, int]:
        current = 0
        mask = 0
        for col in range(self._COLS):
            for row in range(self._ROWS):
                piece = int(pos.board[self._ROWS - 1 - row, col])
                if piece == 0:
                    continue
                bit = 1 << (col * self._BITS_PER_COL + row)
                mask |= bit
                if piece == pos.to_play:
                    current |= bit
        return current, mask

    def _mirror(self, bitboard: int) -> int:
        mirrored = 0
        for col in range(self._COLS):
            col_bits = bitboard >> (col * self._BITS_PER_COL)
            col_bits &= (1 << self._BITS_PER_COL) - 1
            dst_col = self._COLS - 1 - col
            mirrored |= col_bits << (dst_col * self._BITS_PER_COL)
        return mirrored

    def _tt_key(self, current: int, mask: int) -> tuple[int, int]:
        key = (current, mask)
        mirror_key = (self._mirror(current), self._mirror(mask))
        return key if key <= mirror_key else mirror_key


def build_oracle(
    backend: str = "minimax",
    *,
    depth: int = 6,
    use_tt: bool = True,
) -> Oracle:
    backend_name = backend.strip().lower()
    if backend_name == "minimax":
        return MinimaxOracle(depth=depth, use_tt=use_tt)
    if backend_name == "pons":
        return PonsOracle(use_tt=use_tt)
    raise ValueError(f"unknown oracle backend: {backend}")
