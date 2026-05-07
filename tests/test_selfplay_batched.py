from __future__ import annotations

import numpy as np

from connect4.nn import TinyNet
from connect4.selfplay_batched import play_games_batched


def _tiny() -> TinyNet:
    return TinyNet(seed=0, channels=8, blocks=1)


def test_play_games_batched_returns_one_traj_per_spec() -> None:
    net = _tiny()
    specs = [(net, net, 1), (net, net, -1), (net, net, 1)]
    trajs = play_games_batched(
        specs, parallel_games=2, sims=4, rng=np.random.default_rng(0),
    )
    assert len(trajs) == len(specs)
    for traj in trajs:
        assert len(traj) > 0
        for x, pi, z, mask in traj:
            assert x.shape == (2, 6, 7)
            assert pi.shape == (7,)
            assert mask.shape == (7,)
            assert z in (-1.0, 0.0, 1.0)
            assert np.isclose(pi.sum(), 1.0)


def test_play_games_batched_handles_mixed_nets() -> None:
    # Two different nets should be batched independently per actor.
    net_a = TinyNet(seed=0, channels=8, blocks=1)
    net_b = TinyNet(seed=1, channels=8, blocks=1)
    specs = [(net_a, net_a, 1), (net_a, net_b, 1), (net_a, net_b, -1)]
    trajs = play_games_batched(
        specs, parallel_games=3, sims=4, rng=np.random.default_rng(1),
    )
    assert len(trajs) == 3
    assert all(len(t) > 0 for t in trajs)


def test_play_games_batched_clamps_parallel_to_specs() -> None:
    # parallel_games > len(specs) should not crash; the active pool just shrinks.
    net = _tiny()
    specs = [(net, net, 1)]
    trajs = play_games_batched(
        specs, parallel_games=8, sims=4, rng=np.random.default_rng(2),
    )
    assert len(trajs) == 1
    assert len(trajs[0]) > 0


def test_play_games_batched_empty_input() -> None:
    net = _tiny()
    trajs = play_games_batched(
        [], parallel_games=4, sims=4, rng=np.random.default_rng(3),
    )
    assert trajs == []


def test_play_games_batched_completes_more_games_than_pool() -> None:
    # With parallel_games < num_games, slots must be refilled as games finish.
    net = _tiny()
    specs = [(net, net, 1)] * 5
    trajs = play_games_batched(
        specs, parallel_games=2, sims=4, rng=np.random.default_rng(4),
    )
    assert len(trajs) == 5
    assert all(len(t) > 0 for t in trajs)
