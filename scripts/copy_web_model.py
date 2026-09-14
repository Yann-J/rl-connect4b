#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy a trained ONNX export into web/model.onnx for Human matches.",
    )
    parser.add_argument(
        "--source",
        default="checkpoints/small/model.onnx",
        help="ONNX file produced by training (default: checkpoints/small/model.onnx)",
    )
    parser.add_argument(
        "--dest",
        default="web/model.onnx",
        help="Browser model path (default: web/model.onnx)",
    )
    args = parser.parse_args()
    src = Path(args.source)
    dst = Path(args.dest)
    if not src.is_file():
        raise SystemExit(f"source ONNX not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    print(f"copied {src} -> {dst}")


if __name__ == "__main__":
    main()
