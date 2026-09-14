# Python and JS Search play share one PUCT formula

The browser is the Human-match surface, and it currently selects with unnegated Q while Python uses `-child.Q`. Search play in Python and JS must use the same PUCT formula, Q-sign, and tree shape; small numeric drift from float/ONNX is acceptable. JS follows the Python `select_move` tree (including root visit handling), not the other way around. Kaggle’s 1-ply bandit is not that tree and is not the parity target. Self-play exploration choices such as Dirichlet at every root stay as they are for this pass.
