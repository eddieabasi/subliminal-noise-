"""Split-head networks: a body f_theta mapping inputs to a d-dim latent, plus two
linear readouts on that latent: a class head (n logits) and an aux head (m logits).

Matches Table 1 / Table 2 of Brockers et al. (2026), "Learning Through Noise".
"""
from __future__ import annotations

import torch
from torch import nn


class SplitHeadNet(nn.Module):
    def __init__(self, body: nn.Module, latent_dim: int, n_classes: int, m_aux: int, input_kind: str):
        super().__init__()
        self.body = body
        self.class_head = nn.Linear(latent_dim, n_classes)
        self.aux_head = nn.Linear(latent_dim, m_aux)
        self.latent_dim = latent_dim
        self.input_kind = input_kind  # "flat" (B, 784) or "image" (B, 1, 28, 28)

    def prep(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_kind == "flat":
            return x.reshape(x.shape[0], -1)
        return x.reshape(x.shape[0], 1, 28, 28)

    def latent(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(self.prep(x))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.latent(x)
        return self.class_head(z), self.aux_head(z)

    def body_layers(self) -> list[nn.Module]:
        """Parameterised body layers in order (Linear / Conv2d)."""
        return [mod for mod in self.body.modules() if isinstance(mod, (nn.Linear, nn.Conv2d))]


def mlp_body(widths: tuple[int, ...], in_dim: int = 784) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for w in widths:
        layers += [nn.Linear(prev, w), nn.ReLU()]
        prev = w
    return nn.Sequential(*layers)


def cnn_body(channels: tuple[int, int] = (32, 128), latent_dim: int = 256) -> nn.Sequential:
    """Table 2 CNN: conv(1->32,k3,p1)-ReLU-maxpool2, conv(32->128,k3,p1)-ReLU-maxpool4,
    flatten (128*3*3 = 1152) -> Linear(1152, 256)-ReLU."""
    c1, c2 = channels
    return nn.Sequential(
        nn.Conv2d(1, c1, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(c1, c2, 3, padding=1), nn.ReLU(), nn.MaxPool2d(4),
        nn.Flatten(),
        nn.Linear(c2 * 3 * 3, latent_dim), nn.ReLU(),
    )


def build(arch: str, widths: tuple[int, ...], m_aux: int, n_classes: int = 10) -> SplitHeadNet:
    if arch == "mlp":
        return SplitHeadNet(mlp_body(widths), widths[-1], n_classes, m_aux, "flat")
    if arch == "cnn":
        # widths = (c1, c2, latent)
        c1, c2, d = widths
        return SplitHeadNet(cnn_body((c1, c2), d), d, n_classes, m_aux, "image")
    raise ValueError(f"unknown arch {arch!r}")


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
