from __future__ import annotations

from pathlib import Path
import random
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from .nn import TinyNet
from .replay_buffer import ReplayBuffer
from .selfplay import play_one_game
from .game import new_game, legal_moves, apply_move, is_terminal, canonicalize


def run_training(config: dict) -> str:
    seed = int(config.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    model = TinyNet(hidden=int(config["model"]["hidden"]), seed=seed)
    buffer = ReplayBuffer(capacity=int(config["buffer"]["capacity"]))
    warmup_games = int(config["selfplay"].get("warmup_random_games", 1000))
    output_dir = Path(config["output"]["dir"])
    run_name = config.get("logging", {}).get("run_name") or datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_root = Path(config.get("logging", {}).get("tensorboard_dir", output_dir / "runs"))
    run_dir = tb_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir))
    train_step_idx = 0

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

    def produce_games() -> None:
        for game_idx in range(int(config["selfplay"]["games"])):
            samples = play_one_game(model)
            for sample in samples:
                buffer.add(*sample)
            writer.add_scalar("selfplay/game_length", len(samples), game_idx)
            writer.add_scalar("selfplay/draw", float(all(np.isclose(s[2], 0.0) for s in samples)), game_idx)
            writer.add_scalar("buffer/size", len(buffer), game_idx)

    with ThreadPoolExecutor(max_workers=2) as ex:
        fut = ex.submit(produce_games)
        while not fut.done() or len(buffer) < int(config["train"]["batch_size"]):
            if len(buffer) >= int(config["train"]["batch_size"]):
                x, mask, pi, z = buffer.sample(batch_size=int(config["train"]["batch_size"]), rng=rng)
                loss = model.train_step(x, mask, pi, z, lr=float(config["train"]["lr"]))
                writer.add_scalar("train/loss_async", loss, train_step_idx)
                train_step_idx += 1
            else:
                _ = fut.running()

    if len(buffer) == 0:
        raise RuntimeError("self-play produced no samples")

    for _ in range(int(config["train"]["steps"])):
        x, mask, pi, z = buffer.sample(batch_size=int(config["train"]["batch_size"]), rng=rng)
        loss = model.train_step(x, mask, pi, z, lr=float(config["train"]["lr"]))
        writer.add_scalar("train/loss", loss, train_step_idx)
        train_step_idx += 1

    out_path = str(output_dir / "model.ckpt")
    model.save(out_path)
    (output_dir / "latest_tb_run.txt").write_text(f"{run_dir}\n", encoding="utf-8")
    writer.flush()
    writer.close()
    return out_path

