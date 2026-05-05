from __future__ import annotations

import numpy as np

from connect4.game import Position, legal_moves, mirror_policy, new_game
from connect4.mcts import select_move
from connect4.nn import TinyNet
from connect4.replay_buffer import ReplayBuffer


def test_mcts_masks_illegal_moves() -> None:
    b = np.zeros((6, 7), dtype=np.int8)
    b[:, 0] = 1
    pos = Position(b, 1)
    _m, pi = select_move(TinyNet(), pos, selfplay=False)
    assert pi[0] == 0.0
    assert np.isclose(pi.sum(), 1.0)


def test_dirichlet_applies_only_selfplay() -> None:
    net = TinyNet(seed=1)
    pos = new_game()
    _m1, p1 = select_move(net, pos, selfplay=False)
    _m2, p2 = select_move(net, pos, selfplay=False)
    assert np.allclose(p1, p2)

    _m3, p3 = select_move(net, pos, selfplay=True)
    _m4, p4 = select_move(net, pos, selfplay=True)
    assert not np.allclose(p3, p4)


def test_tau_zero_is_deterministic() -> None:
    net = TinyNet(seed=2)
    pos = new_game()
    m1, _ = select_move(net, pos, selfplay=True, move_index=20)
    m2, _ = select_move(net, pos, selfplay=True, move_index=20)
    assert m1 == m2


def test_replay_buffer_fifo_and_mirror() -> None:
    rb = ReplayBuffer(capacity=4)
    x = np.zeros((2, 6, 7), dtype=np.float32)
    x[0, 5, 0] = 1.0
    pi = np.arange(7, dtype=np.float32)
    pi = pi / pi.sum()
    m = np.ones((7,), dtype=np.float32)
    rb.add(x, pi, 1.0, m)
    rb.add(x, pi, -1.0, m)
    assert len(rb) == 4

    xs, _ms, pis, _zs = rb.sample(4, np.random.default_rng(0))
    mirrored = any(np.array_equal(p, mirror_policy(pi)) for p in pis)
    assert mirrored
    assert all(np.isclose(p.sum(), 1.0) for p in pis)
    assert all((x_.shape == (2, 6, 7)) for x_ in xs)

