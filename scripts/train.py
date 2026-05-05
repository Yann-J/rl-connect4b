from __future__ import annotations

import argparse
from pathlib import Path
import sys
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from connect4.train import run_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to YAML config")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    path = run_training(cfg)
    print(path)
    if cfg.get("output", {}).get("export_onnx", True):
        onnx_name = str(cfg.get("output", {}).get("onnx_file", "model.onnx"))
        onnx_path = Path(cfg["output"]["dir"]) / onnx_name
        if onnx_path.exists():
            print(f"onnx={onnx_path}")
    tb_hint = Path(cfg["output"]["dir"]) / "latest_tb_run.txt"
    if tb_hint.exists():
        print(f"tensorboard_run={tb_hint.read_text().strip()}")


if __name__ == "__main__":
    main()

