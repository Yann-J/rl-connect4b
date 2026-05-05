from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from connect4.nn import TinyNet  # noqa: E402


POLICY_TEMPLATE = """import os
import numpy as np
import onnxruntime as ort

_SESSION = None


def _session():
    global _SESSION
    if _SESSION is None:
        model_path = os.path.join(os.path.dirname(__file__), "model.onnx")
        _SESSION = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
    return _SESSION


def _legal_mask(obs):
    board = np.asarray(obs["board"], dtype=np.int8).reshape(6, 7)
    return (board[0] == 0).astype(np.float32)


def _canonical(obs):
    board = np.asarray(obs["board"], dtype=np.int8).reshape(6, 7)
    mark = int(obs["mark"])
    own = (board == mark).astype(np.float32)
    opp = (board != 0).astype(np.float32) - own
    return np.stack([own, opp], axis=0)


def agent(obs, config):
    x = _canonical(obs)[None, ...]
    mask = _legal_mask(obs)[None, ...]
    policy, _value = _session().run(
        ["policy", "value"],
        {"x": x, "legal_mask": mask.astype(np.bool_)},
    )
    pi = policy[0] * mask[0]
    total = float(pi.sum())
    if total <= 0:
        legal = np.flatnonzero(mask[0] > 0.0)
        return int(legal[0])
    pi = pi / total
    return int(np.argmax(pi))
"""


MCTS_TEMPLATE = """import os
import numpy as np
import onnxruntime as ort

_SESSION = None


def _session():
    global _SESSION
    if _SESSION is None:
        model_path = os.path.join(os.path.dirname(__file__), "model.onnx")
        _SESSION = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
    return _SESSION


def _legal_moves(board):
    return [c for c in range(7) if board[0, c] == 0]


def _apply_move(board, to_play, col):
    out = board.copy()
    for row in range(5, -1, -1):
        if out[row, col] == 0:
            out[row, col] = to_play
            return out, -to_play
    raise ValueError("column is full")


def _winner(board):
    for r in range(6):
        for c in range(7):
            p = board[r, c]
            if p == 0:
                continue
            if c <= 3 and all(board[r, c + i] == p for i in range(4)):
                return int(p)
            if r <= 2 and all(board[r + i, c] == p for i in range(4)):
                return int(p)
            if (
                r <= 2
                and c <= 3
                and all(board[r + i, c + i] == p for i in range(4))
            ):
                return int(p)
            if (
                r <= 2
                and c >= 3
                and all(board[r + i, c - i] == p for i in range(4))
            ):
                return int(p)
    return 0


def _terminal(board):
    w = _winner(board)
    if w != 0:
        return True, w
    if len(_legal_moves(board)) == 0:
        return True, 0
    return False, 0


def _canonical(board, to_play):
    own = (board == to_play).astype(np.float32)
    opp = (board == -to_play).astype(np.float32)
    return np.stack([own, opp], axis=0)


def _infer(board, to_play):
    legal = _legal_moves(board)
    mask = np.zeros((1, 7), dtype=np.float32)
    mask[0, legal] = 1.0
    policy, value = _session().run(
        ["policy", "value"],
        {
            "x": _canonical(board, to_play)[None, ...],
            "legal_mask": mask.astype(np.bool_),
        },
    )
    pi = policy[0] * mask[0]
    total = float(pi.sum())
    if total <= 0.0:
        pi[legal] = 1.0 / len(legal)
    else:
        pi = pi / total
    return pi, float(value[0])


def _select_move(board, to_play, sims):
    legal = _legal_moves(board)
    if len(legal) == 1:
        return legal[0]
    root_visits = np.zeros((7,), dtype=np.float32)
    root_q = np.zeros((7,), dtype=np.float32)
    root_prior, _ = _infer(board, to_play)
    c_puct = 1.5
    for _ in range(max(1, sims)):
        moves = []
        boards = [board]
        players = [to_play]
        visits = [root_visits]
        q_vals = [root_q]
        priors = [root_prior]
        while True:
            current_board = boards[-1]
            current_player = players[-1]
            terminal, winner = _terminal(current_board)
            if terminal:
                if winner == 0:
                    leaf_value = 0.0
                else:
                    leaf_value = 1.0 if winner == current_player else -1.0
                break
            legal_now = _legal_moves(current_board)
            if len(moves) > 0 and len(legal_now) == len(priors[-1]):
                pass
            if len(moves) == 0:
                n = visits[-1]
                q = q_vals[-1]
                p = priors[-1]
            else:
                p, leaf_value = _infer(current_board, current_player)
                break
            total_n = float(np.sqrt(max(1.0, n.sum())))
            best_score = -1e18
            best_move = legal_now[0]
            for m in legal_now:
                score = -q[m] + c_puct * p[m] * (total_n / (1.0 + n[m]))
                if score > best_score:
                    best_score = score
                    best_move = m
            next_board, next_player = _apply_move(
                current_board,
                current_player,
                best_move,
            )
            moves.append(best_move)
            boards.append(next_board)
            players.append(next_player)
            visits.append(np.zeros((7,), dtype=np.float32))
            q_vals.append(np.zeros((7,), dtype=np.float32))
            priors.append(np.zeros((7,), dtype=np.float32))
        value = leaf_value
        for depth in range(len(moves) - 1, -1, -1):
            m = moves[depth]
            n = visits[depth]
            q = q_vals[depth]
            n[m] += 1.0
            q[m] += (value - q[m]) / n[m]
            value = -value
    return int(np.argmax(root_visits))


def agent(obs, config):
    board = np.asarray(obs["board"], dtype=np.int8).reshape(6, 7)
    mark = int(obs["mark"])
    to_play = 1 if mark == 1 else -1
    return _select_move(board, to_play, sims=48)
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("out_dir")
    parser.add_argument(
        "--mode",
        choices=("policy", "mcts"),
        default="policy",
        help="submission runtime policy type",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.onnx"
    TinyNet.load(args.checkpoint).export_onnx(str(model_path), opset=17)
    submission_path = out_dir / "submission.py"
    submission_source = (
        MCTS_TEMPLATE
        if args.mode == "mcts"
        else POLICY_TEMPLATE
    )
    submission_path.write_text(submission_source, encoding="utf-8")
    shutil.copyfile(model_path, out_dir / "submission_model.onnx")
    print(str(submission_path))


if __name__ == "__main__":
    main()
