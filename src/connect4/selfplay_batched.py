"""Multi-game leaf-batched self-play.

Instead of running games sequentially with batch-size-1 forward passes (as
``play_one_game`` does), this module keeps up to ``parallel_games`` games in
flight and batches every MCTS leaf evaluation across all of them. This is the
biggest single throughput win on GPU because the network forward pass becomes
launch-overhead-bound at batch size 1.

Each "round of moves" works as follows:

1.  For every active game, descend its actor's MCTS tree from root to a leaf
    using stale Q values (no virtual loss is needed because the trees are
    independent — different games means different roots).
2.  Group the resulting non-terminal leaves by the actor net that owns them
    (with a league, several games may use different opponents). Run one batched
    forward pass per group.
3.  Distribute (priors, value) back: expand each leaf with priors, backprop the
    value along the path. Terminal leaves are handled inline without a forward
    pass.
4.  After ``sims`` such rounds, pick a move per game from visit counts (with
    the same temperature schedule as the scalar self-play), record the training
    sample, advance each tree to the chosen move, and apply the move.

When a game finishes (terminal or move cap), we replace it in the active pool
with the next pending game spec so the batch stays full.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence
import numpy as np

from .game import (
    Position,
    apply_move,
    canonicalize,
    is_terminal,
    legal_moves,
    new_game,
)
from .mcts import (
    Node,
    _backprop,
    _inject_root_dirichlet,
    _select_child,
    _visit_policy,
    advance_root,
)
from .nn import TinyNet


Sample = tuple[np.ndarray, np.ndarray, float, np.ndarray]
GameSpec = tuple[TinyNet, TinyNet, int]  # (current_net, opponent_net, current_player)


@dataclass
class _GameState:
    spec_idx: int
    current_net: TinyNet
    opponent_net: TinyNet
    current_player: int
    pos: Position
    traj: list[tuple[np.ndarray, np.ndarray, int, np.ndarray]] = field(default_factory=list)
    current_root: Node | None = None
    opponent_root: Node | None = None


_LeafItem = tuple[_GameState, Node, list[Node], TinyNet]


def _is_current_turn(g: _GameState) -> bool:
    return g.pos.to_play == g.current_player


def _actor_net(g: _GameState) -> TinyNet:
    return g.current_net if _is_current_turn(g) else g.opponent_net


def _get_actor_root(g: _GameState) -> Node | None:
    if g.current_net is g.opponent_net:
        return g.current_root
    return g.current_root if _is_current_turn(g) else g.opponent_root


def _set_actor_root(g: _GameState, root: Node | None) -> None:
    if g.current_net is g.opponent_net:
        g.current_root = root
    elif _is_current_turn(g):
        g.current_root = root
    else:
        g.opponent_root = root


def _terminal_value(pos: Position) -> float:
    """Value at a terminal position from the to_play player's perspective.

    AlphaZero canonicalizes everything to the player to move, so the network
    target is +1 for "to_play wins", -1 for "to_play loses", 0 for draw.
    """
    _, w = is_terminal(pos)
    if w == 0:
        return 0.0
    return 1.0 if w == pos.to_play else -1.0


def _batched_eval(groups: dict[int, list[_LeafItem]]) -> None:
    """For each group of leaves sharing an actor net, run one forward pass and
    distribute the priors/value back to expand and backprop."""
    for items in groups.values():
        if not items:
            continue
        net = items[0][3]
        n = len(items)
        batch_x = np.empty((n, 2, 6, 7), dtype=np.float32)
        batch_mask = np.zeros((n, 7), dtype=np.float32)
        for i, (_g, leaf, _path, _net) in enumerate(items):
            batch_x[i] = canonicalize(leaf.pos)
            batch_mask[i, legal_moves(leaf.pos)] = 1.0
        probs, values = net.forward(batch_x, batch_mask)
        for (_g, leaf, path, _net), prob_row, v in zip(items, probs, values):
            for move in legal_moves(leaf.pos):
                child_pos = apply_move(leaf.pos, move)
                leaf.children[move] = Node(pos=child_pos, prior=float(prob_row[move]))
            _backprop(path, float(v))


def _run_one_move(
    games: list[_GameState],
    *,
    sims: int,
    c_puct: float,
    dirichlet_alpha: float,
    dirichlet_eps: float,
    rng: np.random.Generator,
) -> list[tuple[_GameState, int, np.ndarray]]:
    """Run ``sims`` lock-step batched MCTS rounds for every game, then return
    ``(game, chosen_move, visit_pi)`` per game."""

    # 1. Make sure every game has an actor root at its current position.
    for g in games:
        actor_root = advance_root(_get_actor_root(g), g.pos)
        if actor_root is None:
            actor_root = Node(pos=g.pos, prior=1.0)
        _set_actor_root(g, actor_root)

    # 2. Batch the initial root expansion for any roots that aren't expanded yet.
    initial_groups: dict[int, list[_LeafItem]] = {}
    for g in games:
        actor_root = _get_actor_root(g)
        assert actor_root is not None
        if actor_root.children:
            continue
        initial_groups.setdefault(id(_actor_net(g)), []).append(
            (g, actor_root, [actor_root], _actor_net(g)),
        )
    _batched_eval(initial_groups)

    # 3. Inject Dirichlet noise at every root (self-play training only).
    for g in games:
        actor_root = _get_actor_root(g)
        assert actor_root is not None
        _inject_root_dirichlet(
            actor_root,
            alpha=dirichlet_alpha,
            eps=dirichlet_eps,
            rng=rng,
        )

    # 4. Run sims rounds of leaf-batched MCTS.
    for _ in range(max(1, sims)):
        leaf_groups: dict[int, list[_LeafItem]] = {}
        for g in games:
            actor_root = _get_actor_root(g)
            assert actor_root is not None
            node = actor_root
            path: list[Node] = [node]
            while node.children:
                _, node = _select_child(node, c_puct=c_puct)
                path.append(node)
            term, _w = is_terminal(node.pos)
            if term:
                _backprop(path, _terminal_value(node.pos))
            else:
                leaf_groups.setdefault(id(_actor_net(g)), []).append(
                    (g, node, path, _actor_net(g)),
                )
        _batched_eval(leaf_groups)

    # 5. Pick a move per game from visit counts.
    chosen: list[tuple[_GameState, int, np.ndarray]] = []
    for g in games:
        actor_root = _get_actor_root(g)
        assert actor_root is not None
        visit_pi = _visit_policy(actor_root)
        move_index = len(g.traj)
        # Same schedule as scalar selfplay: stochastic for the opening, greedy after.
        tau = 1.0 if move_index < 10 else 0.0
        if tau == 0.0:
            move = int(np.argmax(visit_pi))
        else:
            probs = visit_pi.copy()
            probs = probs / np.clip(probs.sum(), 1e-8, None)
            move = int(rng.choice(np.arange(7), p=probs))
        chosen.append((g, move, visit_pi))
    return chosen


def play_games_batched(
    game_specs: Sequence[GameSpec],
    *,
    parallel_games: int,
    sims: int,
    randomize_start_player: bool = True,
    c_puct: float = 1.5,
    dirichlet_alpha: float = 1.0,
    dirichlet_eps: float = 0.25,
    max_moves: int = 42,
    rng: np.random.Generator | None = None,
) -> list[list[Sample]]:
    """Play ``len(game_specs)`` self-play games keeping up to ``parallel_games``
    in flight. Returns one trajectory list per spec, in the input order."""
    if rng is None:
        rng = np.random.default_rng()
    if not game_specs:
        return []
    parallel_games = max(1, min(parallel_games, len(game_specs)))

    pending = list(enumerate(game_specs))[::-1]  # popped from the end
    results: list[list[Sample] | None] = [None] * len(game_specs)
    active: list[_GameState] = []

    def _spawn() -> _GameState | None:
        if not pending:
            return None
        idx, (cnet, onet, cp) = pending.pop()
        start_player = 1
        if randomize_start_player:
            start_player = 1 if rng.random() < 0.5 else -1
        return _GameState(
            spec_idx=idx,
            current_net=cnet,
            opponent_net=onet,
            current_player=cp,
            pos=new_game(start_player=start_player),
        )

    while len(active) < parallel_games:
        g = _spawn()
        if g is None:
            break
        active.append(g)

    while active:
        # Finalize any games that ended after the previous round of moves and
        # spawn replacements so the batch stays full.
        next_active: list[_GameState] = []
        for g in active:
            term, w = is_terminal(g.pos)
            if term or len(g.traj) >= max_moves:
                if term:
                    samples = [
                        (
                            x,
                            pi,
                            (0.0 if w == 0 else (1.0 if w == player else -1.0)),
                            mask,
                        )
                        for x, pi, player, mask in g.traj
                    ]
                else:
                    samples = [(x, pi, 0.0, mask) for x, pi, _p, mask in g.traj]
                results[g.spec_idx] = samples
            else:
                next_active.append(g)
        while len(next_active) < parallel_games:
            spawned = _spawn()
            if spawned is None:
                break
            next_active.append(spawned)
        active = next_active
        if not active:
            break

        chosen = _run_one_move(
            active,
            sims=sims,
            c_puct=c_puct,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_eps=dirichlet_eps,
            rng=rng,
        )
        for g, move, visit_pi in chosen:
            x = canonicalize(g.pos)
            mask = np.zeros((7,), dtype=np.float32)
            mask[legal_moves(g.pos)] = 1.0
            g.traj.append((x, visit_pi, g.pos.to_play, mask))
            actor_root = _get_actor_root(g)
            assert actor_root is not None
            next_root = actor_root.children.get(move)
            _set_actor_root(g, next_root)
            g.pos = apply_move(g.pos, move)

    return [r if r is not None else [] for r in results]
