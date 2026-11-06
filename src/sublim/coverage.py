"""Coverage screen from Madl (2026), "Channel Location Constrains the Auditability
of Subliminal Learning", Sec. 2-3.

    d0       = -grad_{theta_body} L_distill(theta_0)    (student's first update, at its init)
    u_T      = Delta theta_T / ||Delta theta_T||          (teacher's training displacement)
    coverage = cos(d0, u_T)

Computed before the student is trained. Madl defines it under a shared
initialisation. When the student's body starts elsewhere we report two versions:

* ``coverage_tdisp``: u_T = theta_T,final - theta_T,init  (the teacher's own displacement)
* ``coverage_s2t``  : u_T = theta_T,final - theta_S,init  (direction from the student's start
                      to the teacher's end point)

Both coincide when the bodies share their initialisation.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .models import SplitHeadNet
from .train import NoiseFn


def body_vector(model: SplitHeadNet) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.body.parameters()])


def initial_update(student: SplitHeadNet, teacher: SplitHeadNet, noise_fn: NoiseFn, *, seed: int,
                   n_batches: int = 10, batch: int = 1000) -> torch.Tensor:
    """-grad of the distillation loss w.r.t. the student's body, averaged over noise batches."""
    student.eval()
    teacher.eval()
    gen = torch.Generator().manual_seed(seed)
    params = list(student.body.parameters())
    grads = [torch.zeros_like(p) for p in params]
    for _ in range(n_batches):
        xb = noise_fn(batch, gen)
        with torch.no_grad():
            target = teacher(xb)[1]
        loss = F.mse_loss(student(xb)[1], target)
        for g, gi in zip(grads, torch.autograd.grad(loss, params)):
            g += gi / n_batches
    return -torch.cat([g.reshape(-1) for g in grads])


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a[None], b[None]).item())


def coverage(student_init: SplitHeadNet, teacher_init: SplitHeadNet, teacher_final: SplitHeadNet,
             noise_fn: NoiseFn, *, seed: int) -> dict[str, float]:
    d0 = initial_update(student_init, teacher_final, noise_fn, seed=seed)
    t_end, t_start, s_start = body_vector(teacher_final), body_vector(teacher_init), body_vector(student_init)
    if not (t_end.shape == s_start.shape):
        return {}
    return {
        "coverage_tdisp": _cos(d0, t_end - t_start),
        "coverage_s2t": _cos(d0, t_end - s_start),
        "grad_norm": float(d0.norm()),
    }
