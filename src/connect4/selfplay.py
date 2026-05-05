from __future__ import annotations

import numpy as np

from .game import Position, new_game, apply_move, canonicalize, is_terminal, legal_moves
from .mcts import select_move
from .nn import TinyNet


def play_one_game(net: TinyNet, max_moves: int = 42) -> list[tuple[np.ndarray, np.ndarray, float, np.ndarray]]:
    pos: Position = new_game()
    traj: list[tuple[np.ndarray, np.ndarray, int, np.ndarray]] = []
    while True:
        terminal, w = is_terminal(pos)
        if terminal:
            out = []
            for x, pi, player, mask in traj:
                z = 0.0 if w == 0 else (1.0 if w == player else -1.0)
                out.append((x, pi, z, mask))
            return out
        x = canonicalize(pos)
        mask = np.zeros((7,), dtype=np.float32)
        mask[legal_moves(pos)] = 1.0
        move, pi = select_move(net, pos, selfplay=True, move_index=len(traj))
        traj.append((x, pi, pos.to_play, mask))
        pos = apply_move(pos, move)
        if len(traj) >= max_moves:
            return [(x_, pi_, 0.0, mask_) for x_, pi_, _p, mask_ in traj]

