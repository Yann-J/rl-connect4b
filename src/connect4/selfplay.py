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
from .mcts import Node, select_move
from .nn import TinyNet


def play_one_game(
    current_net: TinyNet,
    opponent_net: TinyNet | None = None,
    current_player: int = 1,
    randomize_start_player: bool = True,
    max_moves: int = 42,
    sims: int = 100,
    c_puct: float = 1.5,
    dirichlet_alpha: float = 1.0,
    dirichlet_eps: float = 0.25,
    rng: np.random.Generator | None = None,
) -> list[tuple[np.ndarray, np.ndarray, float, np.ndarray]]:
    if rng is None:
        rng = np.random.default_rng()
    start_player = 1
    if randomize_start_player:
        start_player = 1 if rng.random() < 0.5 else -1
    pos: Position = new_game(start_player=start_player)
    if opponent_net is None:
        opponent_net = current_net
    same_net = opponent_net is current_net
    traj: list[tuple[np.ndarray, np.ndarray, int, np.ndarray]] = []
    # Tree-reuse state. When both players use the same net, a single shared tree
    # is sufficient; otherwise each net needs its own tree because Q/prior estimates differ.
    current_root: Node | None = None
    opponent_root: Node | None = None
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
        is_current_turn = pos.to_play == current_player
        actor_net = current_net if is_current_turn else opponent_net
        if same_net:
            actor_root = current_root
        else:
            actor_root = current_root if is_current_turn else opponent_root
        move, pi, next_root = select_move(
            actor_net,
            pos,
            sims=sims,
            selfplay=True,
            move_index=len(traj),
            c_puct=c_puct,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_eps=dirichlet_eps,
            rng=rng,
            root=actor_root,
        )
        traj.append((x, pi, pos.to_play, mask))
        if same_net or is_current_turn:
            current_root = next_root
        if not same_net and not is_current_turn:
            opponent_root = next_root
        pos = apply_move(pos, move)
        if len(traj) >= max_moves:
            return [(x_, pi_, 0.0, mask_) for x_, pi_, _p, mask_ in traj]
