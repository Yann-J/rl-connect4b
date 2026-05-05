# PRD — Connect-4 RL Agent (AlphaZero-style with League Self-Play)

Status: needs-triage
Owner: Yann
Source plan: `.cursor/plans/connect4_alphazero_league_*.plan.md`

## Problem Statement

I want to build a strong Connect-4 reinforcement-learning agent end-to-end, learn AlphaZero-style techniques in the process, and deploy the trained agent in a static JavaScript web UI I already maintain. Pre-trained snippets and Kaggle notebooks don't give me the engineering experience I want, and most published AlphaZero reference implementations either target Go-scale compute or skip the parts I care about most: principled evaluation against a perfect-play oracle, league self-play to fight strategy collapse, and clean delivery as both a Kaggle submission and an in-browser ONNX model.

I need a focused, single-GPU-friendly, well-engineered AlphaZero pipeline with first-class evaluation, modular code I can extend, and clear delivery artifacts.

## Solution

A self-contained Connect-4 AlphaZero agent. The current network plays self-play games using PUCT-MCTS, sometimes against itself and sometimes against a rolling pool of recent past-self checkpoints (a small "league"). Training samples — `(canonicalized_state, MCTS_visit_policy, game_outcome)` — flow into a sliding replay buffer, are augmented with horizontal mirrors (Connect-4 is left-right symmetric), and feed a configurable ResNet with policy and value heads.

Every training epoch the current network is evaluated using the `kaggle_environments` Python package as harness. The primary metric is winrate vs its built-in `negamax` reference agent (200 games, alternating starts). Secondary metrics include winrate vs minimax baselines at depths 2/4/6/8, near-optimal-move accuracy and value-head MSE against an in-repo Python deep-minimax oracle (alpha-beta + transposition table, depth ~10), and internal Elo via round-robin among the league pool. Diagnostics (policy entropy, draw rate, gradient norm) are logged to TensorBoard.

The trained agent ships as: a terminal CLI for human play, a Kaggle ConnectX submission packager (numpy + onnxruntime, offline-eligible), and an ONNX export that loads in the existing static JS web UI via `onnxruntime-web` — with a reference JS MCTS so the in-browser agent has the same playing style as the trained Python agent, not just policy-greedy.

## User Stories

1. As an ML engineer, I want to launch a full training run from a single YAML config, so that experiments are reproducible and easy to compare.
2. As an ML engineer, I want network depth and width to be config knobs (default small, medium variant available), so that I can scale up later without code changes.
3. As an ML engineer, I want MCTS simulations-per-move to be configurable for self-play and evaluation independently, so that I can trade self-play throughput for data quality.
4. As an ML engineer, I want the league opponent-sampling mix (current-vs-current vs current-vs-past) to be configurable, so that I can ablate the anti-collapse pressure.
5. As an ML engineer, I want a 1k-game random-vs-random warm-up to seed the replay buffer cheaply before MCTS self-play kicks in, so that the first epoch isn't dominated by garbage MCTS targets over a near-empty buffer.
6. As an ML engineer, I want self-play and training to run concurrently on a single GPU, so that the GPU is never idle.
7. As an ML engineer, I want every training epoch to snapshot the current network into the league pool with FIFO eviction at size 10, so that the past-self curriculum keeps moving forward.
8. As an ML engineer, I want every replay-buffer sample to also produce its horizontal-mirror counterpart, so that I get a free 2× data multiplier and the value head learns the symmetry.
9. As an ML engineer, I want the policy head to mask illegal columns to negative infinity before the softmax, in training, MCTS, evaluation, and ONNX export, so that the agent never wastes probability mass on illegal moves and the four code paths agree.
10. As an ML engineer, I want all states fed to the network to be canonicalized to "current player's stones, opponent's stones," so that the network never sees raw player identity and a single weight set handles both colors.
11. As an evaluator, I want a single-number primary strength metric — winrate vs `kaggle_environments` `negamax` over 200 games with alternating starts — so that I can track playing strength against a fixed, well-known reference without depending on a perfect solver.
12. As an evaluator, I want a difficulty-stratified secondary learning curve via winrate-vs-minimax at depths 2/4/6/8 (200 games each, alternating starts), so that early training shows movement before the primary metric becomes informative.
13. As an evaluator, I want a held-out 10k-position dataset labeled by the in-repo depth-10 minimax oracle, with metrics for "near-optimal-move accuracy" (chosen move agrees with the oracle) and value-head MSE against the oracle's values, so that I can monitor agreement with a near-optimal reference and value-head calibration without external dependencies.
14. As an evaluator, I want internal league Elo via round-robin among the pool's 10 checkpoints, so that I can see relative progress between two epochs even when the absolute metrics are saturated or noisy.
15. As an evaluator, I want diagnostic plots (root policy entropy, mean game length, draw rate, gradient norm, NaN/inf counters) on every epoch, so that strategy collapse and training instability are visible early.
16. As an evaluator, I want the entire eval panel to depend only on pure-Python packages (`kaggle_environments`, numpy, torch) with no C++ toolchain, so that evaluation runs anywhere the trainer runs (laptop, Colab, CI).
17. As a Kaggle competitor, I want a packager that bundles the trained network and a slim MCTS into a single offline-eligible `submission.py` (numpy + onnxruntime only), so that I can submit to ConnectX without network access at evaluation time.
18. As an end-user playing in the terminal, I want a CLI that renders the board in ASCII, accepts column-number moves, and shows the agent's chosen column with its visit-count distribution, so that I can play and inspect the agent's reasoning.
19. As an end-user playing in the browser, I want the trained agent exported to ONNX with a stable input signature `(N, 2, 6, 7)`, so that it loads in `onnxruntime-web` and runs entirely client-side.
20. As an end-user playing in the browser, I want a reference JS MCTS that calls the ONNX model for evaluations, so that the in-browser agent plays at MCTS strength rather than policy-greedy strength, and matches the Python agent's style.
21. As an end-user playing in the browser, I want a "fast" mode that uses policy-only (no MCTS) for instant moves, so that there's a snappier playthrough option on slow devices.
22. As a future contributor, I want the game engine, MCTS, oracle, replay buffer, and league pool to each be deep modules with small stable interfaces, so that I can swap implementations (e.g. add a transposition table to MCTS) without ripple effects.
23. As a future contributor, I want the perfect-play oracle to live behind a single `evaluate(board) → (value, best_move)` interface with at least two backends (Pons binary + minimax fallback), so that the rest of the codebase is oracle-agnostic.
24. As a future contributor, I want exhaustive unit tests on the five deep modules, so that I can refactor confidently.
25. As a future contributor, I want a cross-runtime parity test that verifies the ONNX model returns identical outputs to the PyTorch model on a fixed input batch, so that the browser agent isn't silently weaker than the trained Python agent.
26. As a future contributor, I want canonicalization, mirror, and illegal-move masking covered by their own tests with property-based inputs, so that the most error-prone invariants are continuously validated.
27. As an ML engineer, I want clear phased milestones (engine + tests → vanilla AZ → league + symmetry + solver eval → delivery → tuning), each ending in a runnable artifact, so that I always have a working agent and can stop early if needed.
28. As an ML engineer, I want all deliverables (CLI, Kaggle, ONNX) to consume the same on-disk checkpoint format, so that there's one source of truth for "the trained model."

## Implementation Decisions

### Algorithm and training scheme
- AlphaZero (PUCT-MCTS + policy/value network + self-play). No MuZero, no model-free DQN/PPO.
- League self-play: current network plus a FIFO pool of the 10 most-recent checkpoints. Snapshot cadence: end of each training epoch. No champion-gating — continuous training; the league's diversity is the regularizer.
- Random-vs-random warm-up of ~1k games seeds the replay buffer before MCTS self-play takes over.
- Reward is sparse terminal `{+1 win, 0 draw, −1 loss}` from the current player's view, γ=1, no shaping. The value-head target is the final game outcome propagated back through the trajectory.

### State representation
- Two binary planes of shape `(2, 6, 7)`, always canonicalized to `(current_player_stones, opponent_stones)`. No to-move plane, no history planes, no valid-moves plane.
- A single horizontal-mirror operation maps both states and policy targets via column `c ↔ 6−c`; row index is unchanged. Mirror augmentation is applied at replay-buffer insertion time.

### Network
- Small ResNet (default `B=5` blocks × `C=64` channels, ~150k params). Initial 3×3 conv → BN → ReLU; `B` residual blocks of (conv 3×3 → BN → ReLU → conv 3×3 → BN → +skip → ReLU); two heads.
- Policy head: 1×1 conv → 2 channels → flatten → linear → 7 logits → mask illegal columns to `−inf` → softmax.
- Value head: 1×1 conv → 1 channel → flatten → linear(`C`) → ReLU → linear(1) → tanh.
- Loss: `(z − v)^2 − π·log p + 1e-4 ‖θ‖²`.
- Optimizer: Adam, lr=1e-3, weight decay 1e-4, gradient clip 1.0, cosine LR schedule across the planned step budget.
- Width and depth are exposed as YAML knobs; a `medium.yaml` (`B=10`, `C=128`) ships alongside `small.yaml` for ablation.

### MCTS
- PUCT formula with `c_puct = 1.5`.
- Dirichlet noise at the root only, only during self-play: `α = 1.0`, `ε = 0.25`.
- Temperature schedule: `τ=1` for moves 0–10 of self-play (sample from visit-count distribution); `τ→0` (argmax) thereafter and at all evaluation time.
- Default sims/move: 100 in self-play, 400 in evaluation. Both config-tunable per role.
- No transposition table, no virtual loss, no parallelized MCTS in v1. Documented as a future ~2× speedup.

### League
- Pool size 10, FIFO eviction.
- Per-game opponent sampling (default 50/50, configurable): half of self-play games are current-vs-current, half are current-vs-uniform-random-past. Both sides use MCTS with Dirichlet noise.

### Replay buffer and training cadence
- Sliding window over the last ~50k games (~1M positions). Uniform sampling.
- Mirror augmentation applied at insertion: every sample produces itself plus its mirror.
- For every 1k positions added, run 100 gradient steps at `batch_size=512`. Self-play and trainer run concurrently.

### Oracle and evaluation
- Eval harness: `kaggle_environments` Python package (pip-installable, pure Python). Provides the `connectx` env and built-in `random` + `negamax` reference agents. Replaces the Pons C++ toolchain.
- Single `Oracle.evaluate(board) → (value, best_move)` interface with one backend in v1: a Python deep-minimax (alpha-beta + transposition table, depth ~10). Near-optimal but not perfect. The interface is preserved so a perfect solver can be added later as a second backend without disturbing callers.
- Held-out 10k-position dataset shipped in the repo, labeled offline by the depth-10 oracle, used for near-optimal-move accuracy and value-head MSE.
- Metric panel run every epoch: winrate vs `kaggle_environments` negamax (PRIMARY, 200 games alternating starts), winrate vs minimax depths 2/4/6/8 (200 games each, alternating starts), near-optimal-move accuracy on the held-out set, value-head MSE on the held-out set, internal league Elo via round-robin, plus diagnostics.
- Logging: TensorBoard always; Weights & Biases optional behind a flag.

### Modules (deep)
- **`game`** — bitboard board state, win detection on horizontal/vertical/both diagonals, legal-move enumeration, terminal detection, canonicalization, horizontal mirror, plane export.
- **`mcts`** — PUCT search, root Dirichlet noise, temperature-controlled action selection, illegal-move masking, batched leaf evaluation against the network.
- **`oracle`** — abstract evaluator with one backend in v1 (Python deep-minimax, alpha-beta + transposition table, depth ~10); identical contract preserved for future backends; deterministic on equal positions.
- **`replay_buffer`** — sliding window store with mirror augmentation at insertion, uniform mini-batch sampling.
- **`league`** — FIFO checkpoint pool, configurable opponent sampling, persists to disk.

### Modules (orchestration)
- **`nn`** — network module, checkpoint save/load, ONNX export with stable `(N, 2, 6, 7)` signature and opset 17.
- **`selfplay`** — orchestrates one game using `mcts` + `game` + `league`, emits samples to `replay_buffer`.
- **`train`** — gradient step loop using `nn` + `replay_buffer`, optimizer/scheduler, periodic checkpointing, league snapshotting.
- **`eval`** — drives the metric panel using `oracle` + `mcts` + `game`; writes structured metrics for the logger.
- **`play`** — terminal human-vs-agent CLI.
- **`kaggle`** — packages a slim numpy+onnxruntime submission for ConnectX.
- **`web`** — ONNX exporter and a reference JS MCTS adapter for the existing static UI.

### Cross-cutting decisions
- Single canonicalization convention end-to-end (game → MCTS → trainer → ONNX).
- Single illegal-move-masking convention end-to-end (training, MCTS expansion, evaluation, ONNX inference).
- One on-disk checkpoint format consumed by every deliverable.
- Configs are YAML; each module reads from its own sub-section; no constants buried in code.
- Python + PyTorch for training; the Kaggle artifact and the web artifact only need numpy + onnxruntime (no PyTorch dependency at inference).
- `kaggle_environments` is a training-time dependency (used by the eval module). It is a pure-Python pip install with no C/C++ toolchain.

## Testing Decisions

### What makes a good test for this project
- Tests assert **external behavior** of a module's small public interface, not internal data structures. We must be able to swap a bitboard representation, a PUCT variant, or a buffer implementation without touching tests.
- Determinism is enforced via fixed seeds wherever stochasticity exists (Dirichlet noise, sampling, augmentation).
- Property-based tests are preferred for invariants (mirror symmetry, canonicalization round-trip, legal-move masking) over enumerated cases.

### Modules with dedicated test suites
- **`game`** — exhaustive rule coverage (all four win directions including both diagonals); terminal detection on full and partial boards; legal-move enumeration matches the column-fill semantics; canonicalization round-trips for both players; mirror is an involution and preserves outcomes; plane export shape and values for sample positions.
- **`mcts`** — PUCT visit counts concentrate on higher-prior or higher-Q children; illegal columns receive zero visits and zero policy mass; Dirichlet noise is applied only at the root and only in self-play mode; temperature `τ→0` is deterministic given a fixed tree; sims-budget is honored exactly; one-step-to-mate positions are found within a small budget.
- **`oracle`** — depth-10 minimax backend honors the contract on a curated set of small positions (terminal positions return correct game-theoretic values; `best_move` is always legal); deterministic on equal inputs and across runs given the same seed; transposition table preserves correctness (results match a pure depth-10 minimax without TT on a fixed test set).
- **`replay_buffer`** — sliding-window eviction is FIFO at the configured capacity; for every inserted sample its mirror is present and correctly transformed (state planes flipped, policy permuted, value unchanged); uniform sampling is uniform within tolerance over a large draw.
- **`league`** — FIFO eviction at configured size; opponent sampling distribution matches the configured mix within tolerance over a large draw; persistence round-trips cleanly.

### Smoke tests for orchestration modules
- **`nn`** — forward pass returns the correct shapes and value range; illegal-move mask propagates to the policy output; ONNX export reloads and matches PyTorch outputs to a tight numerical tolerance on a fixed input batch (cross-runtime parity).
- **`selfplay`** — one game runs end-to-end and produces a non-empty buffer of well-shaped, finite-valued samples.
- **`train`** — one gradient step decreases the loss on a tiny synthetic batch.
- **`eval`** — the metric panel runs end-to-end on a saved checkpoint against the minimax fallback oracle and produces a complete metrics dict.

### Prior art
- The publicly available "AlphaZero-General" repo and `muzero-general` use comparable test patterns (game-rule enumeration, MCTS sanity, training smoke tests). We follow the same shape.
- Pytest is the test runner; fixtures hold tiny sample boards, a deterministic mock network, and a small minimax oracle for cross-checks.

## Out of Scope

- Distributed or multi-GPU training; multi-machine self-play actors.
- MuZero / learned-dynamics agents.
- Hand-crafted Connect-4 features fed into the network.
- Transposition table / DAG-MCTS, virtual loss, parallelized MCTS (called out as a future ~2× speedup).
- AlphaGo-Zero-style champion gating between checkpoints.
- AlphaStar-style "exploiter" agents in the league.
- Generalization to other m,n,k-games or arbitrary board sizes.
- Persistent league across separate training runs (within a run only, in v1).
- Live online competitive play / matchmaking server.

## Further Notes

### Phasing
The implementation proceeds in five phases, each ending in a runnable artifact:
- **Phase 0** — game engine + mirror + canonicalize + exhaustive unit tests.
- **Phase 1** — vanilla AZ end-to-end (ResNet + MCTS + single-process self-play + replay buffer + trainer + terminal CLI). Working, small.
- **Phase 2** — league pool + horizontal-mirror augmentation + Python deep-minimax oracle + `kaggle_environments` integration + full eval panel + TensorBoard.
- **Phase 3** — ONNX export + Kaggle ConnectX packager + integration with the existing static JS web UI.
- **Phase 4** — hyperparameter sweep over `(B, C, sims/move)`; target ≥ 99% optimal-move accuracy.

### Known risks and mitigations
- **AZ instability on Connect-4** (cycles, value-head collapse): mitigated by league play, mirror augmentation, and continuous (non-gated) training.
- **Illegal-move masking divergence** between training, MCTS, evaluation, and ONNX: covered by a dedicated invariant test that asserts identical masked outputs from all four code paths on the same input.
- **Canonicalization divergence** across modules: enforced by a single helper plus round-trip tests.
- **ONNX/JS parity**: cross-runtime parity test on a fixed input batch; the JS reference MCTS uses the same PUCT formula, the same masking, and the same temperature as the Python implementation.
- **Imperfect oracle**: the depth-10 minimax oracle is strong but not perfect. Near-optimal-move accuracy will saturate below 100%, and the metric cannot distinguish two near-perfect agents. Mitigation: rely on winrate vs `kaggle_environments` `negamax` as the primary metric; keep the `Oracle` interface so a perfect solver (e.g. Pons) can be added later as a drop-in second backend without code churn elsewhere.

### References
- Silver et al., *Mastering the game of Go without human knowledge* (AlphaGo Zero, 2017).
- Silver et al., *A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play* (AlphaZero, 2018).
- Kaggle ConnectX competition and the `kaggle_environments` Python package (eval harness, `negamax` reference agent, submission format).
- Pascal Pons, *Solving Connect 4: how to build a perfect AI* (`connect4.gamesolver.org`) — kept for reference; a future second oracle backend.

