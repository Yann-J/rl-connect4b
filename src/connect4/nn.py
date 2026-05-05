from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class TinyNet:
    hidden: int = 64
    seed: int = 0

    def __post_init__(self) -> None:
        torch.manual_seed(self.seed)
        in_dim = 2 * 6 * 7
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, self.hidden),
            nn.ReLU(),
        ).to(self.device)
        self.policy_head = nn.Linear(self.hidden, 7).to(self.device)
        self.value_head = nn.Linear(self.hidden, 1).to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.policy_head.parameters()) + list(self.value_head.parameters()),
            lr=1e-3,
            weight_decay=1e-4,
        )

    def forward(self, x: np.ndarray, legal_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        with torch.no_grad():
            x_t = torch.as_tensor(x, dtype=torch.float32, device=self.device)
            mask_t = torch.as_tensor(legal_mask, dtype=torch.bool, device=self.device)
            h = self.model(x_t)
            logits = self.policy_head(h)
            logits = logits.masked_fill(~mask_t, float("-inf"))
            probs = F.softmax(logits, dim=1)
            values = torch.tanh(self.value_head(h)).squeeze(1)
        return probs.cpu().numpy(), values.cpu().numpy()

    def train_step(self, x: np.ndarray, legal_mask: np.ndarray, target_pi: np.ndarray, target_z: np.ndarray, lr: float) -> float:
        self.model.train()
        for group in self.optimizer.param_groups:
            group["lr"] = lr

        x_t = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        mask_t = torch.as_tensor(legal_mask, dtype=torch.bool, device=self.device)
        pi_t = torch.as_tensor(target_pi, dtype=torch.float32, device=self.device)
        z_t = torch.as_tensor(target_z, dtype=torch.float32, device=self.device)

        h = self.model(x_t)
        logits = self.policy_head(h)
        logits = logits.masked_fill(~mask_t, float("-inf"))
        log_probs = F.log_softmax(logits, dim=1)
        values = torch.tanh(self.value_head(h)).squeeze(1)

        policy_loss = -(pi_t * log_probs).sum(dim=1).mean()
        value_loss = F.mse_loss(values, z_t)
        l2_term = torch.tensor(0.0, device=self.device)
        for p in list(self.model.parameters()) + list(self.policy_head.parameters()) + list(self.value_head.parameters()):
            l2_term = l2_term + p.pow(2).sum()
        loss = policy_loss + value_loss + 1e-4 * l2_term

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.model.parameters()) + list(self.policy_head.parameters()) + list(self.value_head.parameters()),
            1.0,
        )
        self.optimizer.step()
        return float(loss.item())

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "hidden": self.hidden,
                "seed": self.seed,
                "model": self.model.state_dict(),
                "policy_head": self.policy_head.state_dict(),
                "value_head": self.value_head.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "TinyNet":
        state = torch.load(path, map_location="cpu")
        net = cls(hidden=int(state.get("hidden", 64)), seed=int(state.get("seed", 0)))
        net.model.load_state_dict(state["model"])
        net.policy_head.load_state_dict(state["policy_head"])
        net.value_head.load_state_dict(state["value_head"])
        return net
