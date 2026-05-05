from __future__ import annotations

from concurrent.futures.thread import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
import random
import time

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from .eval import EvalConfig, run_eval_panel
from .game import apply_move, canonicalize, is_terminal, legal_moves, new_game
from .league import LeaguePool
from .nn import TinyNet
from .replay_buffer import ReplayBuffer
from .selfplay import play_one_game


def _build_eval_cfg(eval_cfg: dict, seed: int, profile: str) -> EvalConfig:
    profile_cfg = eval_cfg.get(profile, {}) if isinstance(eval_cfg.get(profile, {}), dict) else {}

    def get(name: str, default: int) -> int:
        if name in profile_cfg:
            return int(profile_cfg[name])
        return int(eval_cfg.get(name, default))

    minimax_depths_raw = profile_cfg.get("minimax_depths", eval_cfg.get("minimax_depths", [2, 4, 6, 8]))
    minimax_depths = tuple(int(d) for d in minimax_depths_raw)
    return EvalConfig(
        games=get("games", 200),
        mcts_sims_eval=get("mcts_sims_eval", 400),
        heldout_size=get("heldout_size", 10000),
        heldout_seed=int(eval_cfg.get("heldout_seed", seed)),
        league_games_per_pair=get("league_games_per_pair", 2),
        minimax_depths=minimax_depths,
        oracle_depth=get("oracle_depth", 10),
    )


def run_training(config: dict) -> str:
    seed = int(config.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    model_cfg = config.get("model", {})
    model = TinyNet(
        channels=int(model_cfg.get("channels", model_cfg.get("hidden", 64))),
        blocks=int(model_cfg.get("blocks", 5)),
        seed=seed,
    )
    buffer = ReplayBuffer(capacity=int(config["buffer"]["capacity"]))
    warmup_games = int(config["selfplay"].get("warmup_random_games", 1000))
    output_dir = Path(config["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    league_cfg = config.get("league", {})
    league = LeaguePool(
        size=int(league_cfg.get("size", 10)),
        current_vs_current_prob=float(
            league_cfg.get("current_vs_current_prob", 0.5),
        ),
        seed=seed,
    )
    league_dir = output_dir / "league"
    league_dir.mkdir(parents=True, exist_ok=True)
    league_state_path = league_dir / "league.json"
    net_cache: dict[str, TinyNet] = {}

    def snapshot_league(epoch_idx: int) -> str:
        ckpt_path = league_dir / f"epoch_{epoch_idx:04d}.ckpt"
        model.save(str(ckpt_path))
        league.snapshot(str(ckpt_path))
        league.save(str(league_state_path))
        return str(ckpt_path)

    run_name = config.get("logging", {}).get("run_name") or datetime.now().strftime(
        "%Y%m%d-%H%M%S",
    )
    tb_root = Path(
        config.get("logging", {}).get("tensorboard_dir", output_dir / "runs"),
    )
    run_dir = tb_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir))
    wandb_run = None
    if config.get("logging", {}).get("wandb", {}).get("enabled", False):
        import wandb  # pylint: disable=import-error,import-outside-toplevel

        wandb_run = wandb.init(
            project=config.get("logging", {}).get("wandb", {}).get(
                "project",
                "rl-connect4b",
            ),
            config=config,
            name=run_name,
        )
    train_step_idx = 0
    progress_every = int(config.get("logging", {}).get("progress_every_steps", 50))
    last_async_step_time = time.perf_counter()
    last_train_step_time = time.perf_counter()
    selfplay_sims = int(config["selfplay"].get("mcts_sims_selfplay", 100))

    def ensure_finite_loss(
        loss: float,
        x: np.ndarray,
        mask: np.ndarray,
        pi: np.ndarray,
        z: np.ndarray,
        stage: str,
        step_idx: int,
    ) -> None:
        if np.isfinite(loss):
            return
        mask_row_sum = (
            mask.sum(axis=1)
            if mask.ndim == 2
            else np.asarray([], dtype=np.float32)
        )
        pi_row_sum = pi.sum(axis=1) if pi.ndim == 2 else np.asarray([], dtype=np.float32)
        raise RuntimeError(
            (
                f"non-finite loss detected at stage={stage} step={step_idx} loss={loss}. "
                f"x_nan={int(np.isnan(x).sum())} x_inf={int(np.isinf(x).sum())} "
                f"pi_nan={int(np.isnan(pi).sum())} pi_inf={int(np.isinf(pi).sum())} "
                f"z_nan={int(np.isnan(z).sum())} z_inf={int(np.isinf(z).sum())} "
                "mask_row_sum_min="
                f"{float(mask_row_sum.min()) if mask_row_sum.size else 0.0} "
                "mask_row_sum_max="
                f"{float(mask_row_sum.max()) if mask_row_sum.size else 0.0} "
                "pi_row_sum_min="
                f"{float(pi_row_sum.min()) if pi_row_sum.size else 0.0} "
                "pi_row_sum_max="
                f"{float(pi_row_sum.max()) if pi_row_sum.size else 0.0} "
                f"z_min={float(np.min(z)) if z.size else 0.0} "
                f"z_max={float(np.max(z)) if z.size else 0.0}"
            )
        )

    for _ in range(warmup_games):
        pos = new_game()
        trajectory: list[tuple[np.ndarray, np.ndarray, int, np.ndarray]] = []
        while True:
            term, w = is_terminal(pos)
            if term:
                for x, pi, player, mask in trajectory:
                    z = 0.0 if w == 0 else (1.0 if w == player else -1.0)
                    buffer.add(x, pi, z, mask)
                break
            legal = legal_moves(pos)
            move = rng.choice(legal).item()
            mask = np.zeros((7,), dtype=np.float32)
            mask[legal] = 1.0
            pi = np.zeros((7,), dtype=np.float32)
            pi[legal] = 1.0 / len(legal)
            trajectory.append((canonicalize(pos), pi, pos.to_play, mask))
            pos = apply_move(pos, move)
    writer.add_scalar("warmup/games", warmup_games, 0)
    writer.add_scalar("buffer/size_after_warmup", len(buffer), 0)
    print(
        f"[train] warmup done: games={warmup_games} buffer_size={len(buffer)}",
    )

    current_snapshot = snapshot_league(epoch_idx=0)
    writer.add_scalar("league/size", len(league), 0)

    def produce_games() -> None:
        selfplay_stage_start = time.perf_counter()
        for game_idx in range(int(config["selfplay"]["games"])):
            game_t0 = time.perf_counter()
            opponent_ckpt = league.sample_opponent(current_snapshot)
            if opponent_ckpt == current_snapshot:
                opponent_net = model
            else:
                if opponent_ckpt not in net_cache:
                    net_cache[opponent_ckpt] = TinyNet.load(opponent_ckpt)
                opponent_net = net_cache[opponent_ckpt]
            current_player = 1 if rng.random() < 0.5 else -1
            samples = play_one_game(
                model,
                opponent_net=opponent_net,
                current_player=current_player,
                sims=int(config["selfplay"].get("mcts_sims_selfplay", 100)),
            )
            for sample in samples:
                buffer.add(*sample)
            game_dt = time.perf_counter() - game_t0
            game_sims = max(1, len(samples)) * selfplay_sims
            writer.add_scalar("selfplay/game_length", len(samples), game_idx)
            writer.add_scalar(
                "selfplay/games_per_s",
                (1.0 / game_dt) if game_dt > 0.0 else 0.0,
                game_idx,
            )
            writer.add_scalar(
                "selfplay/sims_per_s",
                (game_sims / game_dt) if game_dt > 0.0 else 0.0,
                game_idx,
            )
            writer.add_scalar(
                "selfplay/draw",
                float(all(np.isclose(s[2], 0.0) for s in samples)),
                game_idx,
            )
            writer.add_scalar("buffer/size", len(buffer), game_idx)
        total_dt = time.perf_counter() - selfplay_stage_start
        total_games = int(config["selfplay"]["games"])
        total_sims = max(1, total_games) * selfplay_sims
        print(
            "[selfplay] "
            f"games={total_games} wall={total_dt:.2f}s "
            f"games/s={(total_games / total_dt) if total_dt > 0.0 else 0.0:.2f} "
            f"sims/s={(total_sims / total_dt) if total_dt > 0.0 else 0.0:.2f}",
        )

    with ThreadPoolExecutor(max_workers=2) as ex:
        fut = ex.submit(produce_games)
        while not fut.done() or len(buffer) < int(config["train"]["batch_size"]):
            if len(buffer) >= int(config["train"]["batch_size"]):
                x, mask, pi, z = buffer.sample(
                    batch_size=int(config["train"]["batch_size"]),
                    rng=rng,
                )
                loss = model.train_step(x, mask, pi, z, lr=float(config["train"]["lr"]))
                ensure_finite_loss(loss, x, mask, pi, z, "async", train_step_idx)
                now = time.perf_counter()
                dt = now - last_async_step_time
                fps_async = 1.0 / dt if dt > 0 else 0.0
                last_async_step_time = now
                writer.add_scalar("train/loss_async", loss, train_step_idx)
                writer.add_scalar(
                    "train/grad_norm_async",
                    model.last_grad_norm,
                    train_step_idx,
                )
                writer.add_scalar("train/fps_async", fps_async, train_step_idx)
                train_step_idx += 1
            else:
                _ = fut.running()
    print(f"[train] selfplay collection done: buffer_size={len(buffer)}")

    if len(buffer) == 0:
        raise RuntimeError("self-play produced no samples")

    eval_cfg = config.get("eval", {})
    eval_enabled = bool(eval_cfg.get("enabled", False))
    eval_interval = int(
        eval_cfg.get(
            "eval_interval_steps",
            max(1, int(config["train"]["steps"])),
        ),
    )
    eval_once_at_start = bool(eval_cfg.get("eval_once_at_start", True))
    full_panel_every = max(1, int(eval_cfg.get("full_panel_every", 1)))
    epochs = int(config["train"].get("epochs", 1))
    total_steps = int(config["train"]["steps"])
    steps_per_epoch = max(1, total_steps // max(1, epochs))
    remainder_steps = total_steps % max(1, epochs)

    if eval_enabled and eval_once_at_start:
        print("[eval] running initial eval panel...")
        initial_eval_profile = "quick" if "quick" in eval_cfg else "full"
        eval_panel_cfg = _build_eval_cfg(eval_cfg, seed=seed, profile=initial_eval_profile)
        eval_t0 = time.perf_counter()
        panel = run_eval_panel(
            model,
            eval_panel_cfg,
            league=league,
        )
        eval_wall = time.perf_counter() - eval_t0
        for key, value in panel.items():
            writer.add_scalar(f"eval/{key}", value, train_step_idx)
        writer.flush()
        print(
            "[eval] initial panel logged "
            f"profile={initial_eval_profile} wall={eval_wall:.2f}s "
            f"heldout={panel.get('diag_timing_heldout_oracle_label_s', 0.0):.2f}s",
        )

    completed_steps = 0
    for epoch_idx in range(epochs):
        this_epoch_steps = steps_per_epoch + (
            1 if epoch_idx < remainder_steps else 0
        )
        for _ in range(this_epoch_steps):
            x, mask, pi, z = buffer.sample(
                batch_size=int(config["train"]["batch_size"]),
                rng=rng,
            )
            loss = model.train_step(x, mask, pi, z, lr=float(config["train"]["lr"]))
            ensure_finite_loss(loss, x, mask, pi, z, "train", train_step_idx)
            now = time.perf_counter()
            dt = now - last_train_step_time
            fps = 1.0 / dt if dt > 0 else 0.0
            last_train_step_time = now
            writer.add_scalar("train/loss", loss, train_step_idx)
            writer.add_scalar("train/grad_norm", model.last_grad_norm, train_step_idx)
            writer.add_scalar("train/fps", fps, train_step_idx)
            nan_inf_count = int(
                np.isnan(x).sum()
                + np.isinf(x).sum()
                + np.isnan(pi).sum()
                + np.isinf(pi).sum(),
            )
            writer.add_scalar(
                "diag/nan_inf_count_train_batch",
                nan_inf_count,
                train_step_idx,
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/loss": loss,
                        "train/grad_norm": model.last_grad_norm,
                    },
                    step=train_step_idx,
                )
            completed_steps += 1
            if progress_every > 0 and completed_steps % progress_every == 0:
                print(
                    f"[train] step={completed_steps}/{total_steps} "
                    f"loss={loss:.4f} grad_norm={model.last_grad_norm:.4f} buffer={len(buffer)}",
                )
                writer.flush()
            train_step_idx += 1
            if eval_enabled and completed_steps % eval_interval == 0:
                print(f"[eval] running eval at step={completed_steps}")
                run_idx = completed_steps // eval_interval
                full_panel = (run_idx % full_panel_every) == 0
                eval_profile = "full" if full_panel else "quick"
                eval_panel_cfg = _build_eval_cfg(eval_cfg, seed=seed, profile=eval_profile)
                eval_t0 = time.perf_counter()
                panel = run_eval_panel(
                    model,
                    eval_panel_cfg,
                    league=league,
                )
                eval_wall = time.perf_counter() - eval_t0
                for key, value in panel.items():
                    writer.add_scalar(f"eval/{key}", value, train_step_idx)
                writer.flush()
                print(
                    "[eval] logged panel "
                    f"profile={eval_profile} step={completed_steps} wall={eval_wall:.2f}s "
                    f"games/s={panel.get('diag_eval_games_per_s', 0.0):.2f} "
                    f"sims/s={panel.get('diag_eval_sims_per_s', 0.0):.2f} "
                    f"heldout={panel.get('diag_timing_heldout_oracle_label_s', 0.0):.2f}s",
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {f"eval/{k}": v for k, v in panel.items()},
                        step=train_step_idx,
                    )
        current_snapshot = snapshot_league(epoch_idx=epoch_idx + 1)
        writer.add_scalar("league/size", len(league), train_step_idx)

    out_path = str(output_dir / "model.ckpt")
    model.save(out_path)
    export_onnx = bool(config.get("output", {}).get("export_onnx", True))
    if export_onnx:
        onnx_name = str(config.get("output", {}).get("onnx_file", "model.onnx"))
        onnx_path = str(output_dir / onnx_name)
        model.export_onnx(onnx_path, opset=17)
        print(f"[train] exported onnx: {onnx_path}")
    (output_dir / "latest_tb_run.txt").write_text(f"{run_dir}\n", encoding="utf-8")
    writer.flush()
    writer.close()
    if wandb_run is not None:
        wandb_run.finish()
    return out_path
