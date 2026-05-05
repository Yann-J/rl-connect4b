# Outstanding Tasks

Legend: `[ ]` todo, `[~]` in progress, `[x]` done.

## P0 - Blocking / correctness

- [x] Add fail-fast guard for non-finite training loss with useful batch diagnostics.
- [x] Stabilize training numerics after ResNet + real MCTS (investigate prior `loss=nan` events).
- [x] Keep eval compatible with `kaggle_environments` API differences and lock this with tests.
- [x] Normalize `src/connect4/train.py` formatting/style so future changes remain readable.

## P1 - Throughput / iteration speed

- [x] Create `configs/medium_fast.yaml` (quick eval loop) and keep `configs/medium.yaml` as the fuller profile.
- [x] Add timing logs for self-play, training, and eval sub-stages (games/s, sims/s, heldout time).
- [x] Cache oracle labels for heldout positions to avoid recomputing deep minimax every eval.
- [x] Add eval knobs for oracle depth and "quick vs full" panel selection.

## P1 - Phase 1 hardening

- [ ] Add tests that assert PUCT sims budget behavior impacts visit distribution.
- [ ] Add tests for root-only Dirichlet noise during self-play.
- [ ] Add tests that illegal moves always get zero visit probability.
- [ ] Add regression test confirming CLI visit policy is visit-count derived.
- [ ] Add checkpoint backward-compatibility test for old `hidden` checkpoints.

## P2 - PRD alignment (post-Phase 1)

- [ ] Implement PRD training cadence (`+100 train steps per +1k new positions`) instead of fixed total-step loop.
- [ ] Add cosine LR schedule.
- [ ] Introduce Oracle interface with pluggable backends (minimax now, future Pons backend).
- [ ] Ship a persistent heldout dataset in-repo (instead of generating on the fly).
- [ ] Add slim MCTS mode to Kaggle submission packager (currently policy-greedy).

## P3 - Developer ergonomics

- [ ] Add run shortcuts (`train-fast`, `train-medium`, `eval-full`, `profile-eval`).
- [ ] Write run metadata artifact per checkpoint (config hash, git sha, wall time, key metrics).
- [ ] Document expected bottlenecks (CPU-bound self-play/oracle can leave GPU underutilized).
