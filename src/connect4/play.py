from __future__ import annotations

from .game import Position, apply_move, is_terminal, legal_moves, new_game
from .mcts import select_move
from .nn import TinyNet


def _render(pos: Position) -> str:
    symbols = {1: "X", -1: "O", 0: "."}
    lines = [" ".join(symbols[int(v)] for v in row) for row in pos.board]
    lines.append("0 1 2 3 4 5 6")
    return "\n".join(lines)


def play_cli(checkpoint: str) -> None:
    net = TinyNet.load(checkpoint)
    pos = new_game()
    while True:
        print(_render(pos))
        terminal, w = is_terminal(pos)
        if terminal:
            print("Draw" if w == 0 else f"Winner: {'Human' if w == 1 else 'Agent'}")
            return
        if pos.to_play == 1:
            move = int(input("Your move (0-6): ").strip())
            if move not in legal_moves(pos):
                print("Illegal move")
                continue
            pos = apply_move(pos, move)
        else:
            move, pi, _ = select_move(net, pos)
            print(f"Agent move: {move} | visit-policy: {pi.round(3).tolist()}")
            pos = apply_move(pos, move)

