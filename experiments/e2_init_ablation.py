"""E2 - which shared components matter (Fig. 2): reinitialise parts of the student."""
from .common import SEEDS20, job

CONDITIONS = {
    "shared": [],
    "reinit_l1": ["l1"],
    "reinit_l2": ["l2"],
    "reinit_hidden": ["l1", "l2"],
    "reinit_aux": ["aux"],
    "reinit_class": ["class"],
}
MS = [1, 10, 50, 250]


def jobs():
    out = []
    for m in MS:
        for s in SEEDS20:
            students = [{"name": f"m{m}_{c}", "reinit": r, "m": m, "condition": c} for c, r in CONDITIONS.items()]
            out.append(job("E2", s, students, cost=6, m=m))
    return out
