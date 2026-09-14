from __future__ import annotations

from connect4.game import new_game
from connect4.mcts import Node, _select_child


def test_puct_uses_negated_child_q() -> None:
    pos = new_game()
    root = Node(pos=pos)
    root.visit_count = 20
    good_for_opponent = Node(pos=pos, prior=0.5)
    good_for_opponent.visit_count = 10
    good_for_opponent.value_sum = 10.0
    good_for_us = Node(pos=pos, prior=0.5)
    good_for_us.visit_count = 10
    good_for_us.value_sum = -10.0
    root.children[0] = good_for_opponent
    root.children[3] = good_for_us
    move, _child = _select_child(root, c_puct=1.5)
    assert move == 3
