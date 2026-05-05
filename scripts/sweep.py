from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from connect4.eval import EvalConfig, run_eval_panel  # noqa: E402
from connect4.nn import TinyNet  # noqa: E402
from connect4.train import run_training  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_config")
    parser.add_argument("--out", default="checkpoints/sweep/report.md")
    args = parser.parse_args()

    base = yaml.safe_load(Path(args.base_config).read_text(encoding="utf-8"))
    variants = [
        {"name": "small", "hidden": 64, "sims": 100},
        {"name": "medium-lite", "hidden": 128, "sims": 200},
    ]
    lines = ["# Sweep report", ""]
    for variant in variants:
        cfg = deepcopy(base)
        cfg["model"]["hidden"] = variant["hidden"]
        cfg["selfplay"]["mcts_sims_selfplay"] = variant["sims"]
        cfg["eval"]["mcts_sims_eval"] = max(400, variant["sims"])
        cfg["output"]["dir"] = f"checkpoints/sweep/{variant['name']}"
        ckpt = run_training(cfg)
        metrics = run_eval_panel(
            TinyNet.load(ckpt),
            EvalConfig(games=20, mcts_sims_eval=cfg["eval"]["mcts_sims_eval"]),
        )
        lines.append(f"## {variant['name']}")
        lines.append(f"- checkpoint: `{ckpt}`")
        lines.extend([f"- {k}: {v:.3f}" for k, v in sorted(metrics.items())])
        lines.append("")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(str(out))


if __name__ == "__main__":
    main()
