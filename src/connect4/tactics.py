from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .game import Position, apply_move, canonicalize, is_terminal, legal_moves
from .mcts import select_move
from .nn import TinyNet


@dataclass(frozen=True)
class Tactic:
    name: str
    pos: Position
    move: int
    kind: str


def _from_bottom(rows: tuple[str, ...], to_play: int) -> Position:
    board = np.zeros((6, 7), dtype=np.int8)
    if len(rows) > 6:
        raise ValueError("at most 6 rows")
    for r, line in enumerate(rows):
        line = line.strip()
        if len(line) != 7:
            raise ValueError(f"row must have 7 cells, got {line!r}")
        board_row = 5 - r
        for col, ch in enumerate(line):
            if ch == "X":
                board[board_row, col] = 1
            elif ch == "O":
                board[board_row, col] = -1
            elif ch != ".":
                raise ValueError(f"unknown cell {ch!r}")
    return Position(board=board, to_play=to_play)


def tactics_suite() -> tuple[Tactic, ...]:
    return (
        Tactic(
            "forced_win_horizontal",
            _from_bottom(("XXX....",), 1),
            move=3,
            kind="forced_win",
        ),
        Tactic(
            "forced_win_vertical",
            _from_bottom(("...X...", "...X...", "...X..."), 1),
            move=3,
            kind="forced_win",
        ),
        Tactic(
            "forced_win_diagonal",
            _from_bottom(
                (
                    "X..O...",
                    ".X.O...",
                    "..XO...",
                ),
                1,
            ),
            move=3,
            kind="forced_win",
        ),
        Tactic(
            "forced_block_horizontal",
            _from_bottom(("OOO....",), 1),
            move=3,
            kind="forced_block",
        ),
        Tactic(
            "forced_block_vertical",
            _from_bottom(("...O...", "...O...", "...O..."), 1),
            move=3,
            kind="forced_block",
        ),
        Tactic(
            "forced_block_diagonal",
            _from_bottom(
                (
                    "O..O...",
                    ".O.O...",
                    "..OX...",
                    ".....XX",
                    ".....XX",
                ),
                1,
            ),
            move=3,
            kind="forced_block",
        ),
        Tactic(
            "double_threat_horizontal",
            _from_bottom(
                (
                    "XX..XX.",
                    "OO..OO.",
                ),
                1,
            ),
            move=3,
            kind="double_threat",
        ),
        Tactic(
            "double_threat_horizontal_mirror",
            _from_bottom(
                (
                    ".XX..XX",
                    ".OO..OO",
                ),
                1,
            ),
            move=3,
            kind="double_threat",
        ),
    )


def policy_move(net: TinyNet, pos: Position) -> int:
    legal = legal_moves(pos)
    mask = np.zeros((1, 7), dtype=np.float32)
    mask[0, legal] = 1.0
    probs, _values = net.forward(canonicalize(pos)[None, ...], mask)
    return int(np.argmax(probs[0]))


def _accuracy(hits: list[bool]) -> float:
    if not hits:
        return 0.0
    return float(sum(hits) / len(hits))


def evaluate_tactics(
    net: TinyNet,
    *,
    search_sims: int = 200,
    suite: tuple[Tactic, ...] | None = None,
) -> dict[str, float]:
    suite = tactics_suite() if suite is None else suite
    win_hits: list[bool] = []
    block_hits: list[bool] = []
    threat_hits: list[bool] = []
    for tactic in suite:
        if tactic.kind in ("forced_win", "forced_block"):
            hit = policy_move(net, tactic.pos) == tactic.move
            if tactic.kind == "forced_win":
                win_hits.append(hit)
            else:
                block_hits.append(hit)
        elif tactic.kind == "double_threat":
            move, _, _ = select_move(
                net, tactic.pos, sims=search_sims, selfplay=False,
            )
            threat_hits.append(move == tactic.move)
    return {
        "tactics_forced_win_policy": _accuracy(win_hits),
        "tactics_forced_block_policy": _accuracy(block_hits),
        "tactics_double_threat_search": _accuracy(threat_hits),
    }
