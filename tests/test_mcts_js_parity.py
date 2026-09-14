from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from connect4.game import Position, new_game
from connect4.mcts import select_move


class ConstantValueNet:
    def __init__(self, value: float) -> None:
        self.value = value

    def forward(
        self,
        x: np.ndarray,
        legal_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        pi = legal_mask.astype(np.float32).copy()
        totals = pi.sum(axis=1, keepdims=True)
        pi = np.divide(pi, np.clip(totals, 1e-8, None))
        values = np.full((x.shape[0],), self.value, dtype=np.float32)
        return pi, values


def _python_board_to_js(pos: Position) -> list[list[int]]:
    mapping = {1: 1, -1: 2, 0: 0}
    return [[mapping[int(cell)] for cell in row] for row in pos.board]


def _run_js_search(pos: Position, *, value: float, sims: int) -> tuple[int, np.ndarray]:
    runner = Path(__file__).resolve().parent / "mcts_parity_runner.mjs"
    spec = {
        "board": _python_board_to_js(pos),
        "toPlay": 1 if pos.to_play == 1 else 2,
        "sims": sims,
        "value": value,
    }
    proc = subprocess.run(
        ["node", str(runner)],
        input=json.dumps(spec),
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    return int(payload["move"]), np.array(payload["visitPolicy"], dtype=np.float32)


def test_js_search_matches_python_on_constant_value_net() -> None:
    pos = new_game()
    value = 1.0
    sims = 40
    net = ConstantValueNet(value)
    py_move, py_pi, _ = select_move(
        net, pos, sims=sims, selfplay=False, rng=np.random.default_rng(0),
    )
    js_move, js_pi = _run_js_search(pos, value=value, sims=sims)
    assert py_move == js_move
    assert np.allclose(py_pi, js_pi, atol=1e-6)
