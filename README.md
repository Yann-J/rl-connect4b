# rl-connect4b

Connect-4 training pipeline with league self-play, minimax/oracle evaluation, ONNX export, and a pure web demo, auto-published via GitHub pages at [RL-Connect4](https://yann-j.github.io/rl-connect4b/).

## General architecture

- AlphaZero-style network of residual CNN blocks (2D convolutional layers + batch norm + linear/ReLu) of configurable size - convolution seems appropriate since Connect-4 has a strong focus on local spatial patterns
- Game state model is 2 one-hot vectors of `(current_player_stones, opponent_stones)` (swapped every turn)
- Policy (both training and runtime) in not pure neural net but also uses Monte-Carlo Tree Search (MCTS) simulations to pick the best moves among the model's inference results, with state values estimated with PUCT algorithm - this is a big trade-off because it means much of the training time is spent in python (cpu) and won't run on GPU
- Training uses only self-play, but to avoid collapse / overfitting to a single play style, self-play is using a 50/50 mix of current model + recent checkpoints
- Eval using mainly win rate vs minimax players, and play variance from reference perfect solver (Pons algorithm)
- Regular leagues with Elo scoring against pool of recent checkpoints to make sure we keep learning (but we don't use it to selectively pick model for further training, since this can be a weak signal)
- Simple, sparse terminal reward (-1/0/+1)
- Initial bootstrapping with plays against random policy
- Various game-specific optimizations (use horizontal symmetry to double the game samples, randomized start player)

## What is in this repo

- `src/connect4`: core modules (game logic, MCTS, network, self-play, training loop, eval, oracle, league).
- `scripts/train.py`: entrypoint to launch training from a YAML config.
- `configs/*.yaml`: ready-made experiment presets (`small`, `medium`, `large`, and variants).
- `tests`: smoke tests and eval/oracle-specific tests.
- `web`: static browser app loading an exported ONNX model.
- `data/heldout_positions_v1.json`: heldout dataset for eval metrics.

## Requirements

- Python 3.11+
- pip (or uv/pipx equivalent)

Install dependencies:

```bash
pip install -e .
```

Install dev dependencies (tests):

```bash
pip install -e ".[dev]"
```

## Quick start

Run a medium preset training:

```bash
python scripts/train.py configs/medium.yaml
```

Run a larger preset:

```bash
python scripts/train.py configs/large.yaml
```

Each run writes checkpoints and artifacts under the configured `output.dir` (for example `checkpoints/medium`), including:

- model checkpoint
- league checkpoints in `output.dir/league`:
  - `epoch_XXXX.ckpt` (always, once per epoch + initial `epoch_0000`)
  - `step_XXXXXXXX.ckpt` (optional, when `train.checkpoint_every_steps` is set)
- `model.onnx` (if ONNX export is enabled)
- TensorBoard run info (`latest_tb_run.txt` when available)

## Evaluation notes

Evaluation is integrated in training through the `eval` section of each YAML config.

Typical metrics include:

- win rate vs Kaggle `negamax`
- win rate vs minimax at configured depths
- heldout near-optimal move accuracy
- value MSE vs oracle labels
- league Elo diagnostics (when league members exist)

Tune eval cost by changing fields such as:

- `eval.games`
- `eval.mcts_sims_eval`
- `eval.heldout_size`
- `eval.minimax_depths`
- `eval.full_panel_every`

## TensorBoard

If tensorboard logging is enabled in config:

```bash
tensorboard --logdir checkpoints/medium/runs
```

Use the run directory matching your current experiment.

## Testing

Run all tests:

```bash
pytest
```

Run specific suites:

```bash
pytest tests/test_smoke.py
pytest tests/test_oracle_league_onnx.py
```

## Configuration guide

The main top-level config sections are:

- `seed`
- `model`
- `buffer`
- `selfplay`
- `league`
- `eval`
- `train`
- `output`
- `logging`

Start from `configs/small.yaml` for quick iterations, then scale to `medium`/`large`.

Checkpoint cadence can be tuned with:

- `train.epochs`: controls default epoch checkpoint frequency.
- `train.checkpoint_every_steps`: optional step-based checkpoint interval (disabled when missing or <= 0).

## Web app

The `web` folder contains a static app (`index.html`, `app.js`, `styles.css`) and a model file (`model.onnx`).

After training, copy the export into the browser app:

```bash
python scripts/copy_web_model.py
```

The web UI defaults to `web/model.onnx`. Loop eval during training logs vs minimax depth 2 and the Forced win/block fixture; Gate eval (200 vs Kaggle negamax) is:

```bash
uv run python scripts/run_gate_eval.py --checkpoint checkpoints/small/model.ckpt
```
