# Loop eval is cheap; gate eval is 200 vs negamax

Laptop training cannot afford 200 Kaggle negamax games on every eval. The inner loop watches Search play vs minimax depth 2 and a Forced win / Forced block / Double threat fixture. The 0.5-over-200 vs-negamax Gate eval runs only after those loop numbers have moved.
