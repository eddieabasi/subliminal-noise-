"""Invariants the experiments rely on (milestone 2)."""
import copy
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sublim import noise  # noqa: E402
from sublim.coverage import coverage  # noqa: E402
from sublim.init import components, flat, make_student, make_teacher, perturb_  # noqa: E402
from sublim.models import build, n_params  # noqa: E402
from sublim.train import distill_student, train_teacher  # noqa: E402


def same(a, b):
    return torch.equal(flat(a), flat(b))


def test_param_counts_match_paper_tables():
    assert n_params(build("mlp", (256, 256), 10)) == 271_892          # Table 1
    assert n_params(build("mlp", (256, 256, 256), 10)) == 337_684     # Table 2 teacher
    assert n_params(build("cnn", (32, 128, 256), 10)) == 337_620      # Table 2 student


def test_shared_init_is_identical_and_deterministic():
    t = make_teacher("mlp", (256, 256), 10, seed=3)
    s = make_student(t, seed=3)
    for name, comp in components(t).items():
        assert same(comp, components(s)[name]), name
    t2 = make_teacher("mlp", (256, 256), 10, seed=3)
    assert same(t, t2)
    assert not same(t, make_teacher("mlp", (256, 256), 10, seed=4))


@pytest.mark.parametrize("reinit", [["l1"], ["l2"], ["l1", "l2"], ["aux"], ["class"]])
def test_reinit_changes_only_named_components(reinit):
    t = make_teacher("mlp", (256, 256), 10, seed=0)
    s = make_student(t, seed=0, reinit=tuple(reinit))
    for name, comp in components(t).items():
        assert same(comp, components(s)[name]) == (name not in reinit), name


def test_cross_architecture_student_shares_only_heads():
    t = make_teacher("mlp", (256, 256, 256), 10, seed=0)
    s = make_student(t, seed=0, arch="cnn", widths=(32, 128, 256))
    assert same(t.class_head, s.class_head) and same(t.aux_head, s.aux_head)
    s2 = make_student(t, seed=0, arch="mlp", widths=(256, 256, 256))
    assert not same(t.body, s2.body)


def test_teacher_training_leaves_aux_head_untouched():
    t = make_teacher("mlp", (64, 64), 5, seed=0)
    t0 = copy.deepcopy(t)
    x, y = torch.randn(256, 1, 28, 28), torch.randint(0, 10, (256,))
    train_teacher(t, x, y, seed=0, epochs=1, batch=64)
    assert same(t.aux_head, t0.aux_head)
    assert not same(t.class_head, t0.class_head)
    assert not same(t.body, t0.body)


def test_distillation_leaves_student_class_head_untouched():
    t = make_teacher("mlp", (64, 64), 5, seed=0)
    s = make_student(t, seed=0, reinit=("l1",))
    s0 = copy.deepcopy(s)
    nfn = lambda b, g: noise.sample("uniform", b, g)  # noqa: E731
    distill_student(s, t, nfn, seed=0, epochs=1, steps=3, batch=32)
    assert same(s.class_head, s0.class_head)
    assert not same(s.aux_head, s0.aux_head)
    assert not same(s.body, s0.body)


def test_perturbation_scale():
    t = make_teacher("mlp", (256, 256), 10, seed=0)
    s = make_student(t, seed=0)
    perturb_(s.aux_head, 0.05, seed=1)
    diff = flat(s.aux_head) - flat(t.aux_head)
    assert abs(float(diff.std()) - 0.05) < 0.005


@pytest.mark.parametrize("kind,kw", [("uniform", {}), ("perlin", {"res": 8}), ("gaussian", {}),
                                     ("rademacher", {}), ("spectral", {"alpha": 1.0}),
                                     ("spectral", {"alpha": -1.0}), ("lowrank", {"rank": 2}),
                                     ("uniform", {"dc": 1.0})])
def test_noise_shapes_and_determinism(kind, kw):
    a = noise.sample(kind, 8, torch.Generator().manual_seed(0), **kw)
    b = noise.sample(kind, 8, torch.Generator().manual_seed(0), **kw)
    assert a.shape == (8, 1, 28, 28) and torch.equal(a, b) and torch.isfinite(a).all()


def test_uniform_noise_range_and_perlin_smoothness():
    u = noise.sample("uniform", 64, torch.Generator().manual_seed(0))
    assert u.min() >= -1 and u.max() <= 1
    p = noise.sample("perlin", 64, torch.Generator().manual_seed(0), res=8)
    assert p.abs().max() <= 1.0 + 1e-5
    # neighbouring pixels are correlated for Perlin, not for uniform noise
    corr = lambda x: torch.corrcoef(torch.stack([x[..., :-1].reshape(-1), x[..., 1:].reshape(-1)]))[0, 1]  # noqa: E731
    assert corr(p) > 0.5 and abs(corr(u)) < 0.05


def test_coverage_positive_under_shared_init():
    """Cloud et al.'s theorem: with a shared init, the first student step has a
    non-negative inner product with the teacher's displacement."""
    t = make_teacher("mlp", (64, 64), 8, seed=0)
    t0 = copy.deepcopy(t)
    x, y = torch.randn(512, 1, 28, 28), torch.randint(0, 10, (512,))
    train_teacher(t, x, y, seed=0, epochs=1, batch=64)
    s = make_student(t0, seed=0)
    nfn = lambda b, g: noise.sample("uniform", b, g)  # noqa: E731
    cov = coverage(s, t0, t, nfn, seed=0)
    assert cov["coverage_tdisp"] > 0
    assert cov["coverage_tdisp"] == pytest.approx(cov["coverage_s2t"])
