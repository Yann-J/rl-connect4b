from __future__ import annotations

from collections import deque
import numpy as np

from .game import mirror_policy

class ReplayBuffer:
    def __init__(self, capacity: int = 50000 * 21):
        self._data: deque[tuple[np.ndarray, np.ndarray, float, np.ndarray]] = deque(maxlen=capacity)

    def add(self, x: np.ndarray, pi: np.ndarray, z: float, legal_mask: np.ndarray) -> None:
        self._data.append((x, pi, z, legal_mask))
        # Mirror augmentation at insertion keeps training/eval invariants aligned.
        x_m = np.flip(x, axis=2).copy()
        pi_m = mirror_policy(pi)
        mask_m = mirror_policy(legal_mask)
        self._data.append((x_m, pi_m, z, mask_m))

    def __len__(self) -> int:
        return len(self._data)

    def sample(self, batch_size: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idx = rng.integers(0, len(self._data), size=batch_size)
        xs, pis, zs, masks = zip(*(self._data[i] for i in idx))
        return np.stack(xs), np.stack(masks), np.stack(pis), np.asarray(zs, dtype=np.float32)

