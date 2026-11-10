"""Layerwise shared / independent initialisation, head perturbation and freezing.

Component names: body layers are ``l1, l2, ...`` (the parameterised layers of the
body in order, Linear or Conv2d), plus ``class`` and ``aux`` for the two heads.
"""
from __future__ import annotations

import copy

import torch
from torch import nn

from .models import SplitHeadNet, build


def components(model: SplitHeadNet) -> dict[str, nn.Module]:
    comps: dict[str, nn.Module] = {f"l{i + 1}": layer for i, layer in enumerate(model.body_layers())}
    comps["class"] = model.class_head
    comps["aux"] = model.aux_head
    return comps


def reinit_(module: nn.Module, seed: int) -> None:
    """Fresh PyTorch-default init for one layer, from an independent seed."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        module.reset_parameters()


def make_teacher(arch: str, widths: tuple[int, ...], m_aux: int, seed: int, n_classes: int = 10) -> SplitHeadNet:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return build(arch, widths, m_aux, n_classes)


def make_student(
    teacher_init: SplitHeadNet,
    seed: int,
    arch: str | None = None,
    widths: tuple[int, ...] | None = None,
    reinit: tuple[str, ...] = (),
) -> SplitHeadNet:
    """Student starting from the teacher's *initial* weights.

    Same architecture: an exact copy of ``teacher_init`` with the components named in
    ``reinit`` re-drawn independently ("R" in the paper's notation; others are "A").
    Different architecture: the whole body is independent and only the heads are
    copied from the teacher (unless they are also listed in ``reinit``).
    """
    if arch is None:  # same architecture as the teacher
        student = copy.deepcopy(teacher_init)
    else:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 7919)
            student = build(arch, widths, teacher_init.aux_head.out_features, teacher_init.class_head.out_features)
        assert student.latent_dim == teacher_init.latent_dim, "heads need a shared latent width"
        student.class_head.load_state_dict(teacher_init.class_head.state_dict())
        student.aux_head.load_state_dict(teacher_init.aux_head.state_dict())
    comps = components(student)
    for k, name in enumerate(reinit):
        if name == "body":
            for j, layer in enumerate(student.body_layers()):
                reinit_(layer, seed * 1000 + 100 + j)
        else:
            reinit_(comps[name], seed * 1000 + k + 1)
    return student


@torch.no_grad()
def perturb_(module: nn.Module, delta: float, seed: int) -> None:
    """Add iid N(0, delta^2) noise to every weight and bias of ``module``."""
    gen = torch.Generator().manual_seed(seed)
    for p in module.parameters():
        p.add_(torch.randn(p.shape, generator=gen) * delta)


def freeze_(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad_(False)


def flat(module: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in module.parameters()])


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)).item())
