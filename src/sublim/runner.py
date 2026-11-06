"""One job = one seeded teacher plus every student condition distilled from it.

Grouping students under their teacher means each teacher is trained once and the
students of a job share the same noise stream (common random numbers), which
makes conditions directly comparable within a seed.
"""
from __future__ import annotations

import copy
import time
from functools import partial
from typing import Any

import torch

from . import noise as noise_mod
from .coverage import body_vector, coverage
from .data import load_mnist
from .init import components, cosine, flat, freeze_, make_student, make_teacher, perturb_
from .models import n_params
from .train import accuracy, aux_loss, distill_student, train_teacher

_DATA: dict[str, torch.Tensor] | None = None

TEACHER_DEFAULTS = {"arch": "mlp", "widths": [256, 256], "m": 10, "freeze_class": False, "epochs": 5}
STUDENT_DEFAULTS = {
    "arch": None, "widths": None, "reinit": [], "perturb": None, "freeze_aux": False,
    "noise": {"kind": "uniform"}, "steps": 60, "batch": 1000, "epochs": 5, "coverage": True,
}


def data() -> dict[str, torch.Tensor]:
    global _DATA
    if _DATA is None:
        _DATA = load_mnist()
    return _DATA


def noise_fn(spec: dict[str, Any]):
    kw = {k: v for k, v in spec.items() if k != "kind"}
    return partial(lambda b, g, kind, kw: noise_mod.sample(kind, b, g, **kw), kind=spec["kind"], kw=kw)


def _rel_change(before: torch.Tensor, after: torch.Tensor) -> float:
    return float((after - before).norm() / (before.norm() + 1e-12))


def run_job(job: dict[str, Any]) -> list[dict[str, Any]]:
    torch.set_num_threads(int(job.get("threads", 1)))
    d = data()
    seed = int(job["seed"])
    tcfg = {**TEACHER_DEFAULTS, **job.get("teacher", {})}

    t0 = time.time()
    teacher = make_teacher(tcfg["arch"], tuple(tcfg["widths"]), tcfg["m"], seed)
    if tcfg["freeze_class"]:
        freeze_(teacher.class_head)
    teacher_init = copy.deepcopy(teacher)
    train_teacher(teacher, d["x_train"], d["y_train"], seed=seed + 1, epochs=tcfg["epochs"])
    teacher_acc = accuracy(teacher, d["x_test"], d["y_test"])
    t_comps_i, t_comps_f = components(teacher_init), components(teacher)
    teacher_time = time.time() - t0

    rows = []
    for scfg_raw in job["students"]:
        scfg = {**STUDENT_DEFAULTS, **scfg_raw}
        s0 = time.time()
        nfn = noise_fn(scfg["noise"])
        student = make_student(teacher_init, seed, scfg["arch"],
                               tuple(scfg["widths"]) if scfg["widths"] else None, tuple(scfg["reinit"]))
        if scfg["perturb"]:
            perturb_(components(student)[scfg["perturb"]["head"]], scfg["perturb"]["delta"], seed + 3)
        if scfg["freeze_aux"]:
            freeze_(student.aux_head)
        student_init = copy.deepcopy(student)

        row: dict[str, Any] = {
            "exp": job["exp"], "seed": seed, "student": scfg["name"],
            "teacher_cfg": tcfg, "student_cfg": scfg,
            "n_params_teacher": n_params(teacher), "n_params_student": n_params(student),
            "teacher_acc": teacher_acc,
            "student_init_acc": accuracy(student, d["x_test"], d["y_test"]),
            "aux_loss_init": aux_loss(student, teacher, nfn, seed=10_000 + seed),
        }
        if scfg["coverage"]:
            row.update(coverage(student_init, teacher_init, teacher, nfn, seed=20_000 + seed))

        distill_student(student, teacher, nfn, seed=seed + 2, epochs=scfg["epochs"],
                        steps=scfg["steps"], batch=scfg["batch"])

        row["student_acc"] = accuracy(student, d["x_test"], d["y_test"])
        row["aux_loss_final"] = aux_loss(student, teacher, nfn, seed=10_000 + seed)
        s_comps_i, s_comps_f = components(student_init), components(student)
        # Head compatibility (Fig. 5b,d): student heads vs the teacher's final heads.
        for h in ("class", "aux"):
            row[f"{h}_cos_init"] = cosine(flat(s_comps_i[h]), flat(t_comps_f[h]))
            row[f"{h}_cos_final"] = cosine(flat(s_comps_f[h]), flat(t_comps_f[h]))
        # Relative weight change per component (Fig. 6c) and update alignment (Fig. 7).
        for name in t_comps_i:
            row[f"tchange_{name}"] = _rel_change(flat(t_comps_i[name]), flat(t_comps_f[name]))
        for name in s_comps_i:
            row[f"schange_{name}"] = _rel_change(flat(s_comps_i[name]), flat(s_comps_f[name]))
            if name in t_comps_i and flat(s_comps_i[name]).shape == flat(t_comps_i[name]).shape:
                s_upd = flat(s_comps_f[name]) - flat(s_comps_i[name])
                t_upd = flat(t_comps_f[name]) - flat(t_comps_i[name])
                if s_upd.norm() > 0 and t_upd.norm() > 0:  # a frozen/untrained head has no direction
                    row[f"update_cos_{name}"] = cosine(s_upd, t_upd)
        tb_i, tb_f, sb_i, sb_f = (body_vector(m) for m in (teacher_init, teacher, student_init, student))
        if tb_i.shape == sb_i.shape:
            disp = tb_f - tb_i
            row["walk"] = float((sb_f - sb_i) @ disp / (disp @ disp))  # fraction of teacher displacement
            row["body_dist_init"] = float((sb_i - tb_f).norm() / tb_f.norm())
            row["body_dist_final"] = float((sb_f - tb_f).norm() / tb_f.norm())
        row["seconds_teacher"] = teacher_time
        row["seconds_student"] = time.time() - s0
        rows.append(row)
    return rows
