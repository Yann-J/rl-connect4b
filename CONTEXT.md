# Connect-4 Agent

An AlphaZero-style Connect-4 player: a policy/value network plus PUCT search, trained only by self-play, then played by humans in the browser.

## Language

**Agent**:
The AlphaZero-style Connect-4 player. Strength comes from a trained network plus PUCT search.
_Avoid_: bot, engine, solver

**Human match**:
A game against a person. The weakness we observed is here, in the web UI.
_Avoid_: real game

**Eval match**:
A game against a fixed reference opponent (Kaggle random or negamax, or minimax at a set depth) used to measure strength.

**Policy play**:
Choosing a column from the network policy alone. Web Fast is this.
_Avoid_: greedy, raw net, no MCTS

**Search play**:
Choosing a column after PUCT. Web Strong, the CLI, and self-play are this.
_Avoid_: MCTS play, strong mode (UI label only)

**Shipping bar**:
The Agent is shippable when a person loses most Search-play Human matches, and Policy play does not miss a Forced win or Forced block.
_Avoid_: real performance, primary metric

**Eval proxy**:
Winrate of Search play against Kaggle negamax. Automatable stand-in for the shipping bar; the gate is 0.5 over 200 Eval matches.
_Avoid_: primary metric, Elo

**Forced win**:
A position where the side to move can complete four-in-a-row this turn.

**Forced block**:
A position where the side to move must occupy a column this turn or the opponent wins on the next turn.

**Double threat**:
A position where one move creates two Forced wins at once, so a single Forced block cannot save the opponent.
_Avoid_: two-threat, win-in-2 (use only when we mean a two-move force that is not a Double threat)

**Loop eval**:
Cheap periodic measurement while training: Search play vs minimax depth 2, plus Forced win / Forced block / Double threat fixtures.
_Avoid_: full panel, primary metric

**Gate eval**:
200 Search-play Eval matches vs Kaggle negamax. Run when Loop eval has moved; pass at winrate > 0.5.

