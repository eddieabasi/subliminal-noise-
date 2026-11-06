"""E4 - head perturbation (Fig. 5a-d): Gaussian noise of scale delta added to the
student's aux or class head before distillation (baseline otherwise)."""
from .common import SEEDS20, job

DELTAS = [0.0125, 0.025, 0.05, 0.1, 0.2]


def jobs():
    out = []
    for s in SEEDS20:
        students = [{"name": "delta0", "head": "none", "delta": 0.0}]
        for head in ("aux", "class"):
            for d in DELTAS:
                students.append({"name": f"{head}_{d}", "perturb": {"head": head, "delta": d},
                                 "head": head, "delta": d})
        out.append(job("E4", s, students, cost=11, m=10))
    return out
