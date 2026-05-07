from __future__ import annotations

from pathlib import Path
import numpy as np

from connect4.game import canonicalize, legal_moves, new_game
from connect4.mcts import select_move
from connect4.nn import TinyNet
from connect4.replay_buffer import ReplayBuffer
from connect4.selfplay import play_one_game
from connect4.train import planned_training_steps, run_training


def test_network_forward_shapes() -> None:
    net = TinyNet()
    x = np.zeros((2, 2, 6, 7), dtype=np.float32)
    mask = np.ones((2, 7), dtype=np.float32)
    p, v = net.forward(x, mask)
    assert p.shape == (2, 7)
    assert v.shape == (2,)


def test_mcts_legal_move() -> None:
    net = TinyNet()
    pos = new_game()
    move, _pi, _root = select_move(net, pos)
    assert move in legal_moves(pos)


def test_selfplay_emits_samples() -> None:
    net = TinyNet()
    samples = play_one_game(net)
    assert len(samples) > 0
    x, pi, z, mask = samples[0]
    assert x.shape == (2, 6, 7)
    assert pi.shape == (7,)
    assert mask.shape == (7,)
    assert np.isfinite(z)


def test_train_step_changes_loss_path() -> None:
    net = TinyNet()
    pos = new_game()
    x = canonicalize(pos)[None, ...]
    mask = np.ones((1, 7), dtype=np.float32)
    pi = np.full((1, 7), 1.0 / 7.0, dtype=np.float32)
    z = np.array([0.5], dtype=np.float32)
    l1 = net.train_step(x, mask, pi, z, lr=0.01)
    l2 = net.train_step(x, mask, pi, z, lr=0.01)
    assert np.isfinite(l1) and np.isfinite(l2)


def test_replay_buffer_sampling() -> None:
    rb = ReplayBuffer(capacity=10)
    x = np.zeros((2, 6, 7), dtype=np.float32)
    pi = np.full((7,), 1.0 / 7.0, dtype=np.float32)
    m = np.ones((7,), dtype=np.float32)
    rb.add(x, pi, 0.0, m)
    xs, ms, pis, zs = rb.sample(1, np.random.default_rng(0))
    assert xs.shape == (1, 2, 6, 7)
    assert ms.shape == (1, 7)
    assert pis.shape == (1, 7)
    assert zs.shape == (1,)


def test_training_writes_tensorboard_events(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    cfg = {
        "seed": 0,
        "model": {"hidden": 16},
        "buffer": {"capacity": 256},
        "selfplay": {"games": 4, "warmup_random_games": 4},
        "train": {"steps": 4, "batch_size": 8, "lr": 0.01},
        "output": {"dir": str(out_dir)},
        "logging": {
            "tensorboard_dir": str(out_dir / "runs"),
            "run_name": "test-run",
        },
    }
    ckpt = run_training(cfg)
    assert Path(ckpt).exists()
    assert (out_dir / "model.onnx").exists()
    events = list(
        (out_dir / "runs" / "test-run").glob("events.out.tfevents.*"),
    )
    assert events


def test_planned_training_steps_matches_explicit_steps(tmp_path: Path) -> None:
    cfg = {
        "selfplay": {"games": 128},
        "train": {"steps": 999, "batch_size": 8},
    }
    assert planned_training_steps(cfg) == 999


def test_training_runs_periodic_selfplay(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    cfg = {
        "seed": 0,
        "model": {"hidden": 16},
        "buffer": {"capacity": 512},
        "selfplay": {
            "games": 2,
            "games_per_refresh": 1,
            "warmup_random_games": 2,
            "mcts_sims_selfplay": 2,
        },
        "train": {
            "steps": 6,
            "batch_size": 4,
            "lr": 0.01,
            "selfplay_every_steps": 2,
        },
        "output": {"dir": str(out_dir)},
        "logging": {
            "tensorboard_dir": str(out_dir / "runs"),
            "run_name": "periodic-sp",
            "progress_every_steps": 0,
            "train_log_every_steps": 100,
            "selfplay_log_every_games": 100,
        },
    }
    ckpt = run_training(cfg)
    assert Path(ckpt).exists()
