# Outstanding Tasks

Legend: `[ ]` todo, `[~]` in progress, `[x]` done.

## P0 - Blocking / correctness

- [x] Add fail-fast guard for non-finite training loss with useful batch diagnostics.
- [x] Stabilize training numerics after ResNet + real MCTS (investigate prior `loss=nan` events).
- [x] Keep eval compatible with `kaggle_environments` API differences and lock this with tests.
- [x] Normalize `src/connect4/train.py` formatting/style so future changes remain readable.

## P2 - PRD alignment (post-Phase 1)

- [x] Implement PRD training cadence (`+100 train steps per +1k new positions`) instead of fixed total-step loop.
- [x] Add cosine LR schedule.
- [x] Introduce Oracle interface with pluggable backends (minimax now, future Pons backend).
- [x] Ship a persistent heldout dataset in-repo (instead of generating on the fly).
- [x] Add slim MCTS mode to Kaggle submission packager (currently policy-greedy).

## P3 - Developer ergonomics

- [ ] Add run shortcuts (`train-fast`, `train-medium`, `eval-full`, `profile-eval`).
- [ ] Write run metadata artifact per checkpoint (config hash, git sha, wall time, key metrics).
- [ ] Document expected bottlenecks (CPU-bound self-play/oracle can leave GPU underutilized).
