from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        out = self.bn2(out)
        return F.relu(x + out, inplace=True)


class AZBackbone(nn.Module):
    def __init__(self, channels: int, blocks: int) -> None:
        super().__init__()
        self.stem_conv = nn.Conv2d(
            2,
            channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.stem_bn = nn.BatchNorm2d(channels)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(blocks)],
        )

        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 6 * 7, 7)

        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(6 * 7, channels)
        self.value_fc2 = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.stem_conv(x)
        h = self.stem_bn(h)
        h = F.relu(h, inplace=True)
        h = self.res_blocks(h)

        p = self.policy_conv(h)
        p = self.policy_bn(p)
        p = F.relu(p, inplace=True)
        p = p.flatten(1)
        logits = self.policy_fc(p)

        v = self.value_conv(h)
        v = self.value_bn(v)
        v = F.relu(v, inplace=True)
        v = v.flatten(1)
        v = self.value_fc1(v)
        v = F.relu(v, inplace=True)
        v = self.value_fc2(v).squeeze(1)
        return logits, v


@dataclass
class TinyNet:
    channels: int = 64
    blocks: int = 5
    hidden: int | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        torch.manual_seed(self.seed)
        # Backward compatibility for old config/checkpoint field name.
        if self.hidden is not None:
            self.channels = int(self.hidden)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu",
        )
        self.model = AZBackbone(
            channels=self.channels,
            blocks=self.blocks,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=1e-3,
            weight_decay=1e-4,
        )
        self.last_grad_norm = 0.0

    class _ExportModel(nn.Module):
        def __init__(
            self,
            model: nn.Module,
        ) -> None:
            super().__init__()
            self.model = model

        def forward(
            self,
            x: torch.Tensor,
            legal_mask: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            logits, value_logits = self.model(x)
            logits = logits.masked_fill(~legal_mask.bool(), float("-inf"))
            probs = torch.softmax(logits, dim=1)
            values = torch.tanh(value_logits)
            return probs, values

    def forward(
        self,
        x: np.ndarray,
        legal_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        with torch.no_grad():
            x_t = torch.as_tensor(x, dtype=torch.float32, device=self.device)
            mask_t = torch.as_tensor(
                legal_mask,
                dtype=torch.bool,
                device=self.device,
            )
            logits, value_logits = self.model(x_t)
            logits = logits.masked_fill(~mask_t, float("-inf"))
            probs = F.softmax(logits, dim=1)
            values = torch.tanh(value_logits)
        return probs.cpu().numpy(), values.cpu().numpy()

    def train_step(
        self,
        x: np.ndarray,
        legal_mask: np.ndarray,
        target_pi: np.ndarray,
        target_z: np.ndarray,
        lr: float,
    ) -> float:
        self.model.train()
        for group in self.optimizer.param_groups:
            group["lr"] = lr

        x_t = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        mask_t = torch.as_tensor(
            legal_mask,
            dtype=torch.bool,
            device=self.device,
        )
        pi_t = torch.as_tensor(
            target_pi,
            dtype=torch.float32,
            device=self.device,
        )
        z_t = torch.as_tensor(
            target_z,
            dtype=torch.float32,
            device=self.device,
        )

        logits, value_logits = self.model(x_t)
        logits = logits.masked_fill(~mask_t, float("-inf"))
        log_probs = F.log_softmax(logits, dim=1)
        values = torch.tanh(value_logits)

        # Avoid NaNs from 0 * -inf on illegal moves.
        log_probs = torch.where(mask_t, log_probs, torch.zeros_like(log_probs))
        safe_pi = pi_t * mask_t.float()
        safe_pi = safe_pi / safe_pi.sum(dim=1, keepdim=True).clamp_min(1e-8)
        policy_loss = -(safe_pi * log_probs).sum(dim=1).mean()
        value_loss = F.mse_loss(values, z_t)
        # L2 regularization is already handled by optimizer
        # weight_decay.
        loss = policy_loss + value_loss

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            1.0,
        )
        grad_norm_val = grad_norm.item() if hasattr(grad_norm, "item") else grad_norm
        self.last_grad_norm = float(grad_norm_val)
        self.optimizer.step()
        return float(loss.item())

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "hidden": self.channels,
                "channels": self.channels,
                "blocks": self.blocks,
                "seed": self.seed,
                "model": self.model.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "TinyNet":
        state = torch.load(path, map_location="cpu")
        net = cls(
            channels=int(state.get("channels", state.get("hidden", 64))),
            blocks=int(state.get("blocks", 5)),
            seed=int(state.get("seed", 0)),
        )
        net.model.load_state_dict(state["model"])
        return net

    def export_onnx(self, path: str, opset: int = 17) -> str:
        export_model = self._ExportModel(self.model).to(self.device)
        export_model.eval()
        x = torch.zeros((1, 2, 6, 7), dtype=torch.float32, device=self.device)
        legal_mask = torch.ones((1, 7), dtype=torch.bool, device=self.device)
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(
            export_model,
            (x, legal_mask),
            str(out_path),
            input_names=["x", "legal_mask"],
            output_names=["policy", "value"],
            dynamic_axes={
                "x": {0: "batch"},
                "legal_mask": {0: "batch"},
                "policy": {0: "batch"},
                "value": {0: "batch"},
            },
            opset_version=opset,
        )
        return str(out_path)
