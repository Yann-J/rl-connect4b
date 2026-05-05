from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from types import SimpleNamespace

from .game import Position, apply_move, canonicalize, is_terminal, legal_moves, new_game
from .league import LeaguePool
from .mcts import select_move
from .nn import TinyNet
from .oracle import MinimaxOracle


@dataclass(frozen=True)
class EvalConfig:
    games: int = 200
    mcts_sims_eval: int = 400
    heldout_size: int = 10000
    heldout_seed: int = 0
    league_games_per_pair: int = 2


def _play_vs_minimax(net: TinyNet, depth: int, games: int, sims: int) -> float:
    oracle = MinimaxOracle(depth=depth)
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
                move, _ = select_move(net, pos, sims=sims, selfplay=False)
            else:
                move = oracle.evaluate(pos).best_move
            pos = apply_move(pos, move)
    return wins / max(1, games)


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
                move, _ = select_move(net, pos, sims=sims, selfplay=False)
            else:
                legal = legal_moves(pos)
                board = np.where(pos.board == 1, 1, np.where(pos.board == -1, 2, 0)).astype(np.int8)
                board_flat = [int(x) for x in board.reshape(-1)]
                mark = 1 if pos.to_play == 1 else 2
                obs = SimpleNamespace(board=board_flat, mark=mark)
                move = int(env.agents[agent_name](obs, cfg))
                if move not in legal:
                    move = int(rng.choice(legal))
            total_moves += 1
            pos = apply_move(pos, move)
    return wins / max(1, games), draws / max(1, games), total_moves / max(1, games)


def _policy_entropy(pi: np.ndarray) -> float:
    clipped = np.clip(pi, 1e-8, 1.0)
    return float(-(clipped * np.log(clipped)).sum())


def _heldout_metrics(net: TinyNet, heldout: list[Position], oracle: MinimaxOracle) -> tuple[float, float, int]:
    if not heldout:
        return 0.0, 0.0, 0
    batch_size = 256
    agree = 0
    sqerr = 0.0
    nan_inf_count = 0
    for start in range(0, len(heldout), batch_size):
        chunk = heldout[start : start + batch_size]
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
            target = oracle.evaluate(pos)
            agree += int(pred_move == target.best_move)
            sqerr += (v0 - target.value) ** 2
    return agree / len(heldout), sqerr / len(heldout), nan_inf_count


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
            move, _ = select_move(actor, pos, sims=sims, selfplay=False)
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
    metrics: dict[str, float] = {}
    wr_random, _, _ = _play_vs_kaggle_agent(net, "random", games=cfg.games, sims=cfg.mcts_sims_eval)
    metrics["winrate_vs_random"] = wr_random
    wr_negamax, draw_rate, mean_len = _play_vs_kaggle_agent(
        net,
        "negamax",
        games=cfg.games,
        sims=cfg.mcts_sims_eval,
    )
    metrics["winrate_vs_negamax"] = wr_negamax
    metrics["diag_draw_rate"] = draw_rate
    metrics["diag_mean_game_length"] = mean_len
    for depth in (2, 4, 6, 8):
        metrics[f"winrate_vs_minimax_d{depth}"] = _play_vs_minimax(
            net,
            depth=depth,
            games=cfg.games,
            sims=cfg.mcts_sims_eval,
        )
    heldout = make_heldout_dataset(size=cfg.heldout_size, seed=cfg.heldout_seed)
    oracle = MinimaxOracle(depth=10, use_tt=True)
    acc, mse, nan_inf = _heldout_metrics(net, heldout, oracle)
    metrics["near_optimal_move_accuracy"] = acc
    metrics["value_mse_vs_oracle"] = mse
    metrics["diag_nan_inf_count"] = float(nan_inf)
    start = new_game()
    _, pi0 = select_move(net, start, sims=cfg.mcts_sims_eval, selfplay=False)
    metrics["diag_root_policy_entropy"] = _policy_entropy(pi0)
    if league is not None and len(league) > 0:
        current_elo, league_mean_elo = _league_elo(
            net,
            league=league,
            games_per_pair=cfg.league_games_per_pair,
            sims=cfg.mcts_sims_eval,
        )
        metrics["league_elo_current"] = current_elo
        metrics["league_elo_mean_pool"] = league_mean_elo
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
                move, _ = select_move(net, pos, sims=sims, selfplay=False)
            else:
                move = int(rng.choice(legal_moves(pos)))
            pos = apply_move(pos, move)
    return wins / max(1, games)


def make_heldout_dataset(size: int = 1000, seed: int = 0) -> list[Position]:
    rng = np.random.default_rng(seed)
    out: list[Position] = []
    while len(out) < size:
        pos = new_game()
        steps = int(rng.integers(0, 20))
        for _ in range(steps):
            legal = legal_moves(pos)
            if not legal:
                break
            pos = apply_move(pos, int(rng.choice(legal)))
            terminal, _ = is_terminal(pos)
            if terminal:
                break
        terminal, _ = is_terminal(pos)
        if not terminal:
            out.append(pos)
    return out
