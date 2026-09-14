# First retrain is a small net at 1e-3, 100 sims

The first retrain stays C64×5 with 100 self-play sims so it fits a laptop. Learning rate drops from `small.yaml`’s 0.01 to 1e-3 (PRD-ish); eval game counts follow the Loop eval, not the old 10-game panel. Architecture and sim budget stay frozen; other knobs are not a hunting license yet.
