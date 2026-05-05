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
    def evaluate(self, pos: Position) -> OracleResult:
        ...


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
        raise NotImplementedError(
            "oracle backend 'pons' is not implemented yet",
        )
    raise ValueError(f"unknown oracle backend: {backend}")
