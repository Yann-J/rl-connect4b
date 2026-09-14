#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from connect4.eval import EvalConfig, run_eval_panel
from connect4.league import LeaguePool
from connect4.nn import TinyNet


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Gate eval: 200 Search-play games vs Kaggle negamax.",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/small/model.ckpt",
        help="PyTorch checkpoint to evaluate",
    )
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--sims", type=int, default=128)
    parser.add_argument(
        "--league-dir",
        default="checkpoints/small/league.json",
        help="Optional league JSON for Elo side metrics",
    )
    args = parser.parse_args()
    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        raise SystemExit(f"checkpoint not found: {ckpt}")
    net = TinyNet.load(str(ckpt))
    league_path = Path(args.league_dir)
    league = (
        LeaguePool.load(str(league_path))
        if league_path.is_file()
        else None
    )
    cfg = EvalConfig(
        games=args.games,
        mcts_sims_eval=args.sims,
        kaggle_matches=True,
        minimax_depths=(2,),
        heldout_size=0,
        league_games_per_pair=0,
        tactics_enabled=True,
        tactics_search_sims=200,
    )
    panel = run_eval_panel(net, cfg, league=league)
    wr = panel.get("winrate_vs_negamax")
    print(yaml.safe_dump(panel, sort_keys=True))
    if wr is not None:
        print(f"gate winrate_vs_negamax={wr:.3f} (target > 0.5 over {args.games} games)")


if __name__ == "__main__":
    main()
