from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import json
import random


@dataclass(frozen=True)
class LeagueMember:
    checkpoint: str


class LeaguePool:
    def __init__(
        self,
        size: int = 10,
        current_vs_current_prob: float = 0.5,
        seed: int = 0,
    ) -> None:
        self.size = size
        self.current_vs_current_prob = current_vs_current_prob
        self._rng = random.Random(seed)
        self._members: deque[LeagueMember] = deque(maxlen=size)

    def __len__(self) -> int:
        return len(self._members)

    @property
    def members(self) -> list[LeagueMember]:
        return list(self._members)

    def snapshot(self, checkpoint: str) -> None:
        self._members.append(LeagueMember(checkpoint=checkpoint))

    def sample_opponent(self, current_checkpoint: str) -> str:
        if len(self._members) == 0 or self._rng.random() < self.current_vs_current_prob:
            return current_checkpoint
        return self._rng.choice(self.members).checkpoint

    def save(self, path: str) -> str:
        data = {
            "size": self.size,
            "current_vs_current_prob": self.current_vs_current_prob,
            "members": [m.checkpoint for m in self._members],
        }
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return str(out)

    @classmethod
    def load(cls, path: str, seed: int = 0) -> "LeaguePool":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        pool = cls(
            size=int(data["size"]),
            current_vs_current_prob=float(data["current_vs_current_prob"]),
            seed=seed,
        )
        for checkpoint in data["members"]:
            pool.snapshot(checkpoint)
        return pool
