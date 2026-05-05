from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import sys
import types

from connect4.game import apply_move, legal_moves, new_game, Position
from connect4.eval import EvalConfig, load_heldout_dataset, run_eval_panel
from connect4.league import LeaguePool
from connect4.nn import TinyNet
from connect4.oracle import MinimaxOracle, build_oracle


def test_oracle_returns_legal_move_and_is_deterministic() -> None:
    pos = new_game()
    pos = apply_move(pos, 3)
    pos = apply_move(pos, 2)
    oracle = MinimaxOracle(depth=4, use_tt=True)
    r1 = oracle.evaluate(pos)
    r2 = oracle.evaluate(pos)
    assert r1.best_move in legal_moves(pos)
    assert r1.best_move == r2.best_move
    assert np.isclose(r1.value, r2.value)


def test_oracle_tt_and_no_tt_agree() -> None:
    pos = new_game()
    for move in [3, 3, 2, 2, 1]:
        pos = apply_move(pos, move)
    with_tt = MinimaxOracle(depth=4, use_tt=True).evaluate(pos)
    no_tt = MinimaxOracle(depth=4, use_tt=False).evaluate(pos)
    assert with_tt.best_move == no_tt.best_move
    assert np.isclose(with_tt.value, no_tt.value)


def test_oracle_factory_returns_minimax_backend() -> None:
    pos = new_game()
    oracle = build_oracle("minimax", depth=4)
    result = oracle.evaluate(pos)
    assert result.best_move in legal_moves(pos)
    assert np.isfinite(result.value)


def test_oracle_factory_returns_pons_backend() -> None:
    pos = new_game()
    for move in [3, 2, 3, 2, 3, 1]:
        pos = apply_move(pos, move)
    oracle = build_oracle("pons")
    result = oracle.evaluate(pos)
    assert result.best_move == 3
    assert result.value == 1.0


def test_pons_oracle_returns_legal_move_and_is_deterministic() -> None:
    pos = new_game()
    for move in [3, 2, 4, 2, 5]:
        pos = apply_move(pos, move)
    oracle = build_oracle("pons")
    r1 = oracle.evaluate(pos)
    r2 = oracle.evaluate(pos)
    assert r1.best_move in legal_moves(pos)
    assert r1.best_move == r2.best_move
    assert np.isclose(r1.value, r2.value)


def test_league_fifo_sampling_and_persistence(tmp_path: Path) -> None:
    league = LeaguePool(size=3, current_vs_current_prob=0.5, seed=0)
    league.snapshot("a.ckpt")
    league.snapshot("b.ckpt")
    league.snapshot("c.ckpt")
    league.snapshot("d.ckpt")
    assert [m.checkpoint for m in league.members] == [
        "b.ckpt",
        "c.ckpt",
        "d.ckpt",
    ]

    draws = [league.sample_opponent("current.ckpt") for _ in range(200)]
    current_ratio = sum(x == "current.ckpt" for x in draws) / len(draws)
    assert 0.35 <= current_ratio <= 0.65

    path = tmp_path / "league.json"
    league.save(str(path))
    restored = LeaguePool.load(str(path), seed=0)
    assert [m.checkpoint for m in restored.members] == [
        m.checkpoint for m in league.members
    ]


def test_onnx_export_parity(tmp_path: Path) -> None:
    pytest.importorskip("onnxruntime")
    import onnxruntime as ort

    # Export from the same checkpoint format used by training/CLI paths.
    net = TinyNet(seed=0)
    ckpt_path = tmp_path / "model.ckpt"
    net.save(str(ckpt_path))
    net = TinyNet.load(str(ckpt_path))

    x = np.zeros((3, 2, 6, 7), dtype=np.float32)
    x[1, 0, 5, 3] = 1.0
    x[1, 1, 5, 2] = 1.0
    x[2, 0, 5, 0] = 1.0
    mask = np.ones((3, 7), dtype=np.float32)
    mask[2, 0] = 0.0

    torch_policy, torch_value = net.forward(x, mask)
    onnx_path = tmp_path / "model.onnx"
    net.export_onnx(str(onnx_path))

    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    x_input = next(inp for inp in session.get_inputs() if inp.name == "x")
    assert x_input.shape[1:] == [2, 6, 7]
    onnx_policy, onnx_value = session.run(
        ["policy", "value"],
        {"x": x, "legal_mask": mask.astype(np.bool_)},
    )
    np.testing.assert_allclose(torch_policy, onnx_policy, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(torch_value, onnx_value, rtol=1e-5, atol=1e-5)
    assert onnx_policy[2, 0] == 0.0


def test_eval_panel_includes_league_elo(tmp_path: Path) -> None:
    pytest.importorskip("kaggle_environments")
    net = TinyNet(seed=0)
    ckpt_path = tmp_path / "member.ckpt"
    net.save(str(ckpt_path))
    league = LeaguePool(size=10, current_vs_current_prob=0.5, seed=0)
    league.snapshot(str(ckpt_path))
    panel = run_eval_panel(
        net,
        EvalConfig(
            games=2,
            mcts_sims_eval=8,
            heldout_size=8,
            heldout_seed=0,
            league_games_per_pair=2,
        ),
        league=league,
    )
    assert "league_elo_current" in panel
    assert "league_elo_mean_pool" in panel


def test_eval_panel_handles_kaggle_dict_and_namespace_obs(monkeypatch) -> None:
    class FakeEnv:
        def __init__(self) -> None:
            self.configuration = {"columns": 7}
            self.agents = {
                "random": self.random_agent,
                "negamax": self.negamax_agent,
            }

        @staticmethod
        def random_agent(obs, cfg):
            legal = [c for c in range(cfg["columns"]) if obs["board"][c] == 0]
            return legal[0]

        @staticmethod
        def negamax_agent(obs, cfg):
            legal = [c for c in range(cfg.columns) if obs.board[c] == 0]
            return legal[0]

    fake_module = types.SimpleNamespace(
        make=lambda *_args, **_kwargs: FakeEnv(),
    )
    monkeypatch.setitem(sys.modules, "kaggle_environments", fake_module)
    net = TinyNet(seed=0)
    panel = run_eval_panel(
        net,
        EvalConfig(
            games=1,
            mcts_sims_eval=2,
            heldout_size=0,
            heldout_seed=0,
            league_games_per_pair=1,
        ),
        league=None,
    )
    assert "winrate_vs_random" in panel
    assert "winrate_vs_negamax" in panel


def test_load_heldout_dataset_from_repo_file() -> None:
    heldout = load_heldout_dataset(
        path="data/heldout_positions_v1.json",
        size=32,
        seed=0,
    )
    assert len(heldout) == 32
    assert all(isinstance(p, Position) for p in heldout)
