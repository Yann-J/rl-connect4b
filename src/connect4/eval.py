from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
import numpy as np
from types import SimpleNamespace
from typing import Any

from .game import Position, apply_move, canonicalize, is_terminal, legal_moves, new_game
from .league import LeaguePool
from .mcts import select_move
from .nn import TinyNet
from .oracle import OracleResult, build_oracle
from .tactics import evaluate_tactics


@dataclass(frozen=True)
class EvalConfig:
    games: int = 200
    mcts_sims_eval: int = 400
    kaggle_matches: bool = True
    heldout_size: int = 10000
    heldout_seed: int = 0
    league_games_per_pair: int = 2
    minimax_depths: tuple[int, ...] = (2, 4, 6, 8)
    oracle_depth: int = 10
    oracle_backend: str = "minimax"
    heldout_dataset_path: str = "data/heldout_positions_v1.json"
    tactics_enabled: bool = True
    tactics_search_sims: int = 200


_HELDOUT_LABEL_CACHE: dict[
    tuple[int, int, int, str, str],
    tuple[list[Position], list[OracleResult], float],
] = {}
_HELDOUT_DATASET_CACHE: dict[str, list[Position]] = {}


def _play_vs_minimax(net: TinyNet, depth: int, games: int, sims: int) -> float:
    oracle = build_oracle("minimax", depth=depth)
    wins = 0
    for game_idx in range(games):
        pos = new_game()
        agent_player = 1 if game_idx % 2 == 0 else -1
        while True:
            terminal, winner = is_terminal(pos)
            if terminal:
                if winner == agent_player:
                    wins += 1
                break
            if pos.to_play == agent_player:
                move, _, _ = select_move(net, pos, sims=sims, selfplay=False)
            else:
                move = oracle.evaluate(pos).best_move
            pos = apply_move(pos, move)
    return wins / max(1, games)


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return {}


def _as_namespace(value: Any) -> SimpleNamespace:
    if isinstance(value, SimpleNamespace):
        return value
    if isinstance(value, dict):
        return SimpleNamespace(**value)
    if hasattr(value, "__dict__"):
        return SimpleNamespace(**vars(value))
    return SimpleNamespace()


def _call_kaggle_agent(
    agent_fn: Any,
    obs_dict: dict[str, Any],
    cfg_raw: Any,
) -> int:
    cfg_dict = _as_dict(cfg_raw)
    cfg_ns = _as_namespace(cfg_raw)
    attempts = [
        (obs_dict, cfg_raw),
        (_as_namespace(obs_dict), cfg_raw),
        (obs_dict, cfg_dict),
        (_as_namespace(obs_dict), cfg_ns),
    ]
    last_err: Exception | None = None
    for obs_arg, cfg_arg in attempts:
        try:
            return int(agent_fn(obs_arg, cfg_arg))
        except (AttributeError, TypeError, KeyError) as err:
            last_err = err
    if last_err is not None:
        raise last_err
    raise RuntimeError("failed to call kaggle agent")


def _play_vs_kaggle_agent(net: TinyNet, agent_name: str, games: int, sims: int) -> tuple[float, float, float]:
    try:
        from kaggle_environments import make
    except Exception:
        if agent_name == "random":
            return _play_vs_random(net, games=games, sims=sims), 0.0, 0.0
        raise RuntimeError(
            "kaggle_environments is required for non-random kaggle eval agents",
        ) from None

    wins = 0
    draws = 0
    total_moves = 0
    rng = np.random.default_rng(0)
    env = make("connectx", debug=False)
    cfg = env.configuration
    for game_idx in range(games):
        pos = new_game()
        agent_player = 1 if game_idx % 2 == 0 else -1
        while True:
            terminal, winner = is_terminal(pos)
            if terminal:
                if winner == agent_player:
                    wins += 1
                if winner == 0:
                    draws += 1
                break
            if pos.to_play == agent_player:
                move, _, _ = select_move(net, pos, sims=sims, selfplay=False)
            else:
                legal = legal_moves(pos)
                board = np.where(pos.board == 1, 1, np.where(pos.board == -1, 2, 0)).astype(np.int8)
                board_flat = [int(x) for x in board.reshape(-1)]
                mark = 1 if pos.to_play == 1 else 2
                obs = {"board": board_flat, "mark": mark}
                move = _call_kaggle_agent(env.agents[agent_name], obs, cfg)
                if move not in legal:
                    move = int(rng.choice(legal))
            total_moves += 1
            pos = apply_move(pos, move)
    return wins / max(1, games), draws / max(1, games), total_moves / max(1, games)


def _policy_entropy(pi: np.ndarray) -> float:
    clipped = np.clip(pi, 1e-8, 1.0)
    return float(-(clipped * np.log(clipped)).sum())


def _heldout_metrics_with_labels(
    net: TinyNet,
    heldout: list[Position],
    oracle_labels: list[OracleResult],
) -> tuple[float, float, int, float]:
    if not heldout:
        return 0.0, 0.0, 0, 0.0
    if len(heldout) != len(oracle_labels):
        raise ValueError("heldout positions and oracle labels length mismatch")
    batch_size = 256
    agree = 0
    sqerr = 0.0
    nan_inf_count = 0
    t0 = time.perf_counter()
    for start in range(0, len(heldout), batch_size):
        chunk = heldout[start:start + batch_size]
        x = np.stack([canonicalize(pos) for pos in chunk], axis=0).astype(np.float32, copy=False)
        mask = np.zeros((len(chunk), 7), dtype=np.float32)
        for i, pos in enumerate(chunk):
            mask[i, legal_moves(pos)] = 1.0
        pi_batch, v_batch = net.forward(x, mask)
        for i, pos in enumerate(chunk):
            pi0 = pi_batch[i]
            v0 = float(v_batch[i])
            nan_inf_count += int(np.isnan(pi0).any() or np.isinf(pi0).any() or not np.isfinite(v0))
            pred_move = int(np.argmax(pi0))
            target = oracle_labels[start + i]
            agree += int(pred_move == target.best_move)
            sqerr += (v0 - target.value) ** 2
    forward_seconds = time.perf_counter() - t0
    return agree / len(heldout), sqerr / len(heldout), nan_inf_count, forward_seconds


def _get_heldout_with_oracle_labels(
    heldout_size: int,
    heldout_seed: int,
    oracle_depth: int,
    oracle_backend: str,
    heldout_dataset_path: str,
) -> tuple[list[Position], list[OracleResult], float, bool]:
    key = (
        heldout_size,
        heldout_seed,
        oracle_depth,
        oracle_backend,
        heldout_dataset_path,
    )
    cached = _HELDOUT_LABEL_CACHE.get(key)
    if cached is not None:
        heldout, labels, label_time_s = cached
        return heldout, labels, label_time_s, True
    heldout = load_heldout_dataset(
        path=heldout_dataset_path,
        size=heldout_size,
        seed=heldout_seed,
    )
    oracle = build_oracle(oracle_backend, depth=oracle_depth, use_tt=True)
    t0 = time.perf_counter()
    labels = [oracle.evaluate(pos) for pos in heldout]
    label_time_s = time.perf_counter() - t0
    _HELDOUT_LABEL_CACHE[key] = (heldout, labels, label_time_s)
    return heldout, labels, label_time_s, False


def load_heldout_dataset(path: str, size: int, seed: int) -> list[Position]:
    dataset = _HELDOUT_DATASET_CACHE.get(path)
    if dataset is None:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        positions_raw = payload.get("positions", [])
        dataset = [
            Position(
                board=np.asarray(item["board"], dtype=np.int8),
                to_play=int(item["to_play"]),
            )
            for item in positions_raw
        ]
        _HELDOUT_DATASET_CACHE[path] = dataset
    if size <= 0:
        return []
    if size > len(dataset):
        raise ValueError(
            f"requested heldout_size={size} exceeds dataset size={len(dataset)} at {path}",
        )
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=size, replace=False)
    return [dataset[int(i)] for i in indices]


def _play_head_to_head(net_a: TinyNet, net_b: TinyNet, games: int, sims: int) -> tuple[int, int, int]:
    wins_a = 0
    wins_b = 0
    draws = 0
    for game_idx in range(games):
        pos = new_game()
        a_player = 1 if game_idx % 2 == 0 else -1
        while True:
            terminal, winner = is_terminal(pos)
            if terminal:
                if winner == 0:
                    draws += 1
                elif winner == a_player:
                    wins_a += 1
                else:
                    wins_b += 1
                break
            actor = net_a if pos.to_play == a_player else net_b
            move, _, _ = select_move(actor, pos, sims=sims, selfplay=False)
            pos = apply_move(pos, move)
    return wins_a, wins_b, draws


def _league_elo(
    current_net: TinyNet,
    league: LeaguePool,
    games_per_pair: int,
    sims: int,
) -> tuple[float, float]:
    if len(league) == 0:
        return 0.0, 0.0

    participants: list[tuple[str, TinyNet]] = [("current", current_net)]
    for idx, member in enumerate(league.members):
        participants.append((f"member_{idx}", TinyNet.load(member.checkpoint)))

    ratings: dict[str, float] = {name: 1000.0 for name, _ in participants}
    k = 24.0
    for i in range(len(participants)):
        name_a, net_a = participants[i]
        for j in range(i + 1, len(participants)):
            name_b, net_b = participants[j]
            wins_a, wins_b, draws = _play_head_to_head(
                net_a,
                net_b,
                games=games_per_pair,
                sims=sims,
            )
            total = max(1, wins_a + wins_b + draws)
            score_a = (wins_a + 0.5 * draws) / total
            score_b = 1.0 - score_a
            exp_a = 1.0 / (1.0 + 10.0 ** ((ratings[name_b] - ratings[name_a]) / 400.0))
            exp_b = 1.0 - exp_a
            ratings[name_a] += k * (score_a - exp_a)
            ratings[name_b] += k * (score_b - exp_b)
    league_mean = float(np.mean([ratings[name] for name in ratings if name != "current"]))
    return float(ratings["current"]), league_mean


def run_eval_panel(net: TinyNet, cfg: EvalConfig, league: LeaguePool | None = None) -> dict[str, float]:
    eval_start = time.perf_counter()
    metrics: dict[str, float] = {}
    total_eval_games = 0
    print(
        "[eval] panel start "
        f"games={cfg.games} sims={cfg.mcts_sims_eval} "
        f"heldout={cfg.heldout_size} oracle_depth={cfg.oracle_depth} "
        f"tactics={int(cfg.tactics_enabled)}",
    )
    if cfg.tactics_enabled:
        metrics.update(
            evaluate_tactics(net, search_sims=cfg.tactics_search_sims),
        )
    random_time_s = 0.0
    negamax_time_s = 0.0
    mean_len_for_sims: float | None = None
    if cfg.kaggle_matches:
        t0 = time.perf_counter()
        wr_random, _, _ = _play_vs_kaggle_agent(net, "random", games=cfg.games, sims=cfg.mcts_sims_eval)
        random_time_s = time.perf_counter() - t0
        total_eval_games += cfg.games
        t0 = time.perf_counter()
        wr_negamax, draw_rate, mean_len = _play_vs_kaggle_agent(
            net,
            "negamax",
            games=cfg.games,
            sims=cfg.mcts_sims_eval,
        )
        negamax_time_s = time.perf_counter() - t0
        total_eval_games += cfg.games
        mean_len_for_sims = mean_len
        metrics["winrate_vs_random"] = wr_random
        metrics["winrate_vs_negamax"] = wr_negamax
        metrics["diag_draw_rate"] = draw_rate
        metrics["diag_mean_game_length"] = mean_len
    t0 = time.perf_counter()
    for depth in cfg.minimax_depths:
        metrics[f"winrate_vs_minimax_d{depth}"] = _play_vs_minimax(
            net,
            depth=depth,
            games=cfg.games,
            sims=cfg.mcts_sims_eval,
        )
        total_eval_games += cfg.games
    minimax_sweep_time_s = time.perf_counter() - t0
    print(
        "[eval] game suites done "
        f"random={random_time_s:.2f}s negamax={negamax_time_s:.2f}s minimax={minimax_sweep_time_s:.2f}s",
    )
    t0 = time.perf_counter()
    heldout, oracle_labels, label_time_s, label_cache_hit = _get_heldout_with_oracle_labels(
        heldout_size=cfg.heldout_size,
        heldout_seed=cfg.heldout_seed,
        oracle_depth=cfg.oracle_depth,
        oracle_backend=cfg.oracle_backend,
        heldout_dataset_path=cfg.heldout_dataset_path,
    )
    heldout_prep_time_s = time.perf_counter() - t0
    print(
        "[eval] heldout labels "
        f"cache_hit={int(label_cache_hit)} prep={heldout_prep_time_s:.2f}s label={label_time_s:.2f}s",
    )
    heldout_forward_time_s = 0.0
    if heldout:
        acc, mse, nan_inf, heldout_forward_time_s = _heldout_metrics_with_labels(
            net,
            heldout,
            oracle_labels,
        )
        metrics["near_optimal_move_accuracy"] = acc
        metrics["value_mse_vs_oracle"] = mse
        metrics["diag_nan_inf_count"] = float(nan_inf)
    start = new_game()
    _, pi0, _ = select_move(net, start, sims=cfg.mcts_sims_eval, selfplay=False)
    metrics["diag_root_policy_entropy"] = _policy_entropy(pi0)
    league_time_s = 0.0
    if league is not None and len(league) > 0:
        t0 = time.perf_counter()
        current_elo, league_mean_elo = _league_elo(
            net,
            league=league,
            games_per_pair=cfg.league_games_per_pair,
            sims=cfg.mcts_sims_eval,
        )
        league_time_s = time.perf_counter() - t0
        total_eval_games += cfg.league_games_per_pair * len(league)
        metrics["league_elo_current"] = current_elo
        metrics["league_elo_mean_pool"] = league_mean_elo
    total_time_s = time.perf_counter() - eval_start
    approx_eval_sims = 0.0
    if mean_len_for_sims is not None:
        approx_eval_sims = total_eval_games * mean_len_for_sims * cfg.mcts_sims_eval
    if cfg.kaggle_matches:
        metrics["diag_timing_random_games_s"] = random_time_s
        metrics["diag_timing_negamax_games_s"] = negamax_time_s
    metrics["diag_timing_minimax_sweep_s"] = minimax_sweep_time_s
    metrics["diag_timing_heldout_prep_s"] = heldout_prep_time_s
    metrics["diag_timing_heldout_oracle_label_s"] = label_time_s
    if heldout:
        metrics["diag_timing_heldout_forward_s"] = heldout_forward_time_s
    metrics["diag_timing_league_elo_s"] = league_time_s
    metrics["diag_timing_eval_total_s"] = total_time_s
    metrics["diag_eval_games_per_s"] = (
        total_eval_games / total_time_s if total_time_s > 0.0 else 0.0
    )
    if mean_len_for_sims is not None:
        metrics["diag_eval_sims_per_s"] = (
            approx_eval_sims / total_time_s if total_time_s > 0.0 else 0.0
        )
    metrics["diag_kaggle_matches_enabled"] = 1.0 if cfg.kaggle_matches else 0.0
    metrics["diag_heldout_oracle_cache_hit"] = 1.0 if label_cache_hit else 0.0
    return metrics


def _play_vs_random(net: TinyNet, games: int, sims: int) -> float:
    rng = np.random.default_rng(0)
    wins = 0
    for game_idx in range(games):
        pos = new_game()
        agent_player = 1 if game_idx % 2 == 0 else -1
        while True:
            terminal, winner = is_terminal(pos)
            if terminal:
                if winner == agent_player:
                    wins += 1
                break
            if pos.to_play == agent_player:
                move, _, _ = select_move(net, pos, sims=sims, selfplay=False)
            else:
                move = int(rng.choice(legal_moves(pos)))
            pos = apply_move(pos, move)
    return wins / max(1, games)

