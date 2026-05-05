from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .game import Position, apply_move, canonicalize, is_terminal, legal_moves
from .nn import TinyNet


@dataclass
class Node:
    pos: Position
    prior: float = 0.0
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[int, "Node"] = field(default_factory=dict)

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


def _masked_policy_and_value(
    net: TinyNet,
    pos: Position,
) -> tuple[np.ndarray, float]:
    legal = legal_moves(pos)
    mask = np.zeros((1, 7), dtype=np.float32)
    mask[0, legal] = 1.0
    probs, values = net.forward(canonicalize(pos)[None, ...], mask)
    pi = probs[0]
    pi[[i for i in range(7) if i not in legal]] = 0.0
    total = float(pi.sum())
    if total <= 0.0 and legal:
        pi[legal] = 1.0 / len(legal)
    else:
        pi = pi / np.clip(total, 1e-8, None)
    return pi, float(values[0])


def _expand(net: TinyNet, node: Node) -> float:
    terminal, winner = is_terminal(node.pos)
    if terminal:
        if winner == 0:
            return 0.0
        return 1.0 if winner == node.pos.to_play else -1.0

    pi, value = _masked_policy_and_value(net, node.pos)
    for move in legal_moves(node.pos):
        child_pos = apply_move(node.pos, move)
        node.children[move] = Node(pos=child_pos, prior=float(pi[move]))
    return value


def _select_child(node: Node, c_puct: float) -> tuple[int, Node]:
    parent_sqrt = float(np.sqrt(max(1, node.visit_count)))
    best_score = -1e18
    best_move = -1
    best_child: Node | None = None
    for move, child in node.children.items():
        q = -child.q_value
        u = c_puct * child.prior * (parent_sqrt / (1 + child.visit_count))
        score = q + u
        if score > best_score:
            best_score = score
            best_move = move
            best_child = child
    if best_child is None:
        raise RuntimeError("PUCT selection called on leaf node")
    return best_move, best_child


def _inject_root_dirichlet(
    root: Node,
    alpha: float,
    eps: float,
    rng: np.random.Generator,
) -> None:
    if not root.children:
        return
    moves = list(root.children.keys())
    noise = rng.dirichlet(
        np.full((len(moves),), alpha, dtype=np.float32),
    )
    for idx, move in enumerate(moves):
        child = root.children[move]
        child.prior = float((1.0 - eps) * child.prior + eps * noise[idx])


def _backprop(path: list[Node], leaf_value: float) -> None:
    value = leaf_value
    for node in reversed(path):
        node.visit_count += 1
        node.value_sum += value
        value = -value


def _visit_policy(root: Node) -> np.ndarray:
    visits = np.zeros((7,), dtype=np.float32)
    for move, child in root.children.items():
        visits[move] = float(child.visit_count)
    total = float(visits.sum())
    if total <= 0:
        legal = legal_moves(root.pos)
        if legal:
            visits[legal] = 1.0 / len(legal)
        return visits
    return visits / total


def select_move(
    net: TinyNet,
    pos: Position,
    sims: int = 100,
    selfplay: bool = False,
    move_index: int = 0,
    dirichlet_alpha: float = 1.0,
    dirichlet_eps: float = 0.25,
    c_puct: float = 1.5,
    rng: np.random.Generator | None = None,
) -> tuple[int, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng()
    legal = legal_moves(pos)
    if not legal:
        return -1, np.zeros((7,), dtype=np.float32)

    root = Node(pos=pos, prior=1.0)
    _expand(net, root)

    if selfplay and move_index == 0:
        _inject_root_dirichlet(
            root,
            alpha=dirichlet_alpha,
            eps=dirichlet_eps,
            rng=rng,
        )

    for _ in range(max(1, sims)):
        node = root
        path = [node]
        while node.children:
            _, node = _select_child(node, c_puct=c_puct)
            path.append(node)
        leaf_value = _expand(net, node)
        _backprop(path, leaf_value)

    visit_pi = _visit_policy(root)
    tau = 1.0 if (selfplay and move_index < 10) else 0.0
    if tau == 0.0:
        move = int(np.argmax(visit_pi))
    else:
        probs = visit_pi.copy()
        probs = probs / np.clip(probs.sum(), 1e-8, None)
        move = int(rng.choice(np.arange(7), p=probs))
    return move, visit_pi
