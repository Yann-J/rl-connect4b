from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from connect4.nn import TinyNet  # noqa: E402


SUBMISSION_TEMPLATE = """import os
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("out_dir")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.onnx"
    TinyNet.load(args.checkpoint).export_onnx(str(model_path), opset=17)
    submission_path = out_dir / "submission.py"
    submission_path.write_text(SUBMISSION_TEMPLATE, encoding="utf-8")
    shutil.copyfile(model_path, out_dir / "submission_model.onnx")
    print(str(submission_path))


if __name__ == "__main__":
    main()
