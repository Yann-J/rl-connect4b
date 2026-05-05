from __future__ import annotations

import numpy as np

from .game import Position, canonicalize, legal_moves
from .nn import TinyNet


def select_move(
    net: TinyNet,
    pos: Position,
    sims: int = 100,
    selfplay: bool = False,
    move_index: int = 0,
    dirichlet_alpha: float = 1.0,
    dirichlet_eps: float = 0.25,
) -> tuple[int, np.ndarray]:
    del sims  # tracer-bullet skeleton keeps this as an interface knob.
    legal = legal_moves(pos)
    mask = np.zeros((1, 7), dtype=np.float32)
    mask[0, legal] = 1.0
    probs, _ = net.forward(canonicalize(pos)[None, ...], mask)
    pi = probs[0]
    if selfplay and move_index == 0 and legal:
        noise = np.random.default_rng().dirichlet(np.full((len(legal),), dirichlet_alpha, dtype=np.float32))
        pi = pi.copy()
        pi[legal] = (1.0 - dirichlet_eps) * pi[legal] + dirichlet_eps * noise
        pi = pi / np.clip(pi.sum(), 1e-8, None)
    tau = 1.0 if (selfplay and move_index < 10) else 0.0
    if tau == 0.0:
        move = int(np.argmax(pi))
    else:
        move = int(np.random.default_rng().choice(np.arange(7), p=pi))
    if move not in legal:
        move = legal[0]
    pi[[i for i in range(7) if i not in legal]] = 0.0
    pi = pi / np.clip(pi.sum(), 1e-8, None)
    return move, pi

