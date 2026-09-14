from __future__ import annotations

import numpy as np

from connect4.game import apply_move, is_terminal, legal_moves
from connect4.tactics import evaluate_tactics, policy_move, tactics_suite


class ConstantValueNet:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def forward(
        self,
        x: np.ndarray,
        legal_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        pi = legal_mask.astype(np.float32).copy()
        totals = pi.sum(axis=1, keepdims=True)
        pi = np.divide(pi, np.clip(totals, 1e-8, None))
        values = np.full((x.shape[0],), self.value, dtype=np.float32)
        return pi, values


class PointingNet:
    def __init__(self, move: int) -> None:
        self.move = move

    def forward(
        self,
        x: np.ndarray,
        legal_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        pi = np.zeros_like(legal_mask, dtype=np.float32)
        pi[:, self.move] = 1.0
        pi = pi * legal_mask
        totals = pi.sum(axis=1, keepdims=True)
        empty = totals[:, 0] <= 0
        if np.any(empty):
            fallback = legal_mask.astype(np.float32)
            fallback = fallback / np.clip(
                fallback.sum(axis=1, keepdims=True), 1e-8, None,
            )
            pi[empty] = fallback[empty]
        else:
            pi = pi / np.clip(totals, 1e-8, None)
        values = np.zeros((x.shape[0],), dtype=np.float32)
        return pi, values


def test_forced_win_move_wins_immediately() -> None:
    for tactic in tactics_suite():
        if tactic.kind != "forced_win":
            continue
        nxt = apply_move(tactic.pos, tactic.move)
        terminal, winner = is_terminal(nxt)
        assert terminal and winner == tactic.pos.to_play, tactic.name


def test_forced_block_is_the_only_save() -> None:
    for tactic in tactics_suite():
        if tactic.kind != "forced_block":
            continue
        opponent = -tactic.pos.to_play
        for move in legal_moves(tactic.pos):
            nxt = apply_move(tactic.pos, move)
            if move == tactic.move:
                assert not is_terminal(nxt)[0], tactic.name
                continue
            reply = apply_move(nxt, tactic.move)
            terminal, winner = is_terminal(reply)
            assert terminal and winner == opponent, tactic.name


def test_double_threat_creates_two_forced_wins() -> None:
    for tactic in tactics_suite():
        if tactic.kind != "double_threat":
            continue
        after = apply_move(tactic.pos, tactic.move)
        assert not is_terminal(after)[0], tactic.name
        winning_replies = []
        for opp_move in legal_moves(after):
            opp_pos = apply_move(after, opp_move)
            wins = [
                mv
                for mv in legal_moves(opp_pos)
                if is_terminal(apply_move(opp_pos, mv))[1] == tactic.pos.to_play
            ]
            assert wins, tactic.name
            winning_replies.append(frozenset(wins))
        union = set.union(*[set(s) for s in winning_replies])
        assert len(union) >= 2, tactic.name


def test_pointing_net_scores_perfect_policy_tactics() -> None:
    net = PointingNet(3)
    metrics = evaluate_tactics(net, search_sims=8)
    assert metrics["tactics_forced_win_policy"] == 1.0
    assert metrics["tactics_forced_block_policy"] == 1.0


def test_uniform_net_misses_policy_tactics_but_search_finds_double_threat() -> None:
    net = ConstantValueNet(0.0)
    for tactic in tactics_suite():
        if tactic.kind in ("forced_win", "forced_block"):
            assert policy_move(net, tactic.pos) != tactic.move
    metrics = evaluate_tactics(net, search_sims=200)
    assert metrics["tactics_forced_win_policy"] == 0.0
    assert metrics["tactics_forced_block_policy"] == 0.0
    assert metrics["tactics_double_threat_search"] == 1.0
