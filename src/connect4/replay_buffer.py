from __future__ import annotations

import numpy as np

from .game import mirror_policy


Sample = tuple[np.ndarray, np.ndarray, float, np.ndarray]


class ReplayBuffer:
    """FIFO replay buffer backed by a Python list used as a ring buffer.

    Random-access indexing is O(1) (vs O(N) on a ``collections.deque``), which
    matters because ``sample`` does ``batch_size`` independent index lookups
    every training step.
    """

    def __init__(self, capacity: int = 50000 * 21):
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self._capacity = int(capacity)
        self._data: list[Sample] = []
        self._write_idx = 0

    def _append(self, sample: Sample) -> None:
        if len(self._data) < self._capacity:
            self._data.append(sample)
        else:
            self._data[self._write_idx] = sample
            self._write_idx = (self._write_idx + 1) % self._capacity

    def add(self, x: np.ndarray, pi: np.ndarray, z: float, legal_mask: np.ndarray) -> None:
        self._append((x, pi, float(z), legal_mask))
        # Mirror augmentation at insertion keeps training/eval invariants aligned.
        x_m = np.flip(x, axis=2).copy()
        pi_m = mirror_policy(pi)
        mask_m = mirror_policy(legal_mask)
        self._append((x_m, pi_m, float(z), mask_m))

    def __len__(self) -> int:
        return len(self._data)

    def sample(
        self,
        batch_size: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idx = rng.integers(0, len(self._data), size=batch_size)
        data = self._data
        xs = np.stack([data[i][0] for i in idx])
        pis = np.stack([data[i][1] for i in idx])
        zs = np.fromiter((data[i][2] for i in idx), dtype=np.float32, count=batch_size)
        masks = np.stack([data[i][3] for i in idx])
        return xs, masks, pis, zs
