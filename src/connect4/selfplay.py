from __future__ import annotations

import numpy as np

from .game import (
    Position,
    new_game,
    apply_move,
    canonicalize,
    is_terminal,
    legal_moves,
)
from .mcts import select_move
from .nn import TinyNet


def play_one_game(
    current_net: TinyNet,
    opponent_net: TinyNet | None = None,
    current_player: int = 1,
    max_moves: int = 42,
    sims: int = 100,
    rng: np.random.Generator | None = None,
) -> list[tuple[np.ndarray, np.ndarray, float, np.ndarray]]:
    if rng is None:
        rng = np.random.default_rng()
    pos: Position = new_game()
    if opponent_net is None:
        opponent_net = current_net
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
        actor_net = (
            current_net if pos.to_play == current_player else opponent_net
        )
        move, pi = select_move(
            actor_net,
            pos,
            sims=sims,
            selfplay=True,
            move_index=len(traj),
            rng=rng,
        )
        traj.append((x, pi, pos.to_play, mask))
        pos = apply_move(pos, move)
        if len(traj) >= max_moves:
            return [(x_, pi_, 0.0, mask_) for x_, pi_, _p, mask_ in traj]
