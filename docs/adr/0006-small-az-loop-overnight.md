# First small run is a real AZ loop, capped overnight

`small.yaml` today is one dump of 64 MCTS games and ~500 steps, which cannot teach Forced wins. The first retrain keeps C64×5, 100 sims, and `lr=1e-3`, but adds self-play refresh and a large step ceiling. We stop when Loop eval moves (vs minimax depth 2, or the Forced-win suite) or when overnight wall-clock is up. That run is not a promise of the 0.5 Gate eval.
