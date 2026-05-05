from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from connect4.nn import TinyNet  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    args = parser.parse_args()
    net = TinyNet.load(args.checkpoint)
    out = net.export_onnx(args.output, opset=17)
    print(out)


if __name__ == "__main__":
    main()
