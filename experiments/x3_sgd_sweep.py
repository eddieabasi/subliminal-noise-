"""X3 - X2 with plain SGD students (no momentum, lr = 3, the largest stable rate in a
calibration on uniform noise: 6 and above diverge). Coverage measures the raw gradient
direction; Adam rescales each weight's step, plain SGD follows the gradient. If the
optimiser explains our gap to Madl's rho of 0.95, it should close here."""
from .common import job
from .x2_madl_sweep import FAMILIES, SEEDS, SETTINGS

LR = 3.0


def jobs():
    out = []
    for s in SEEDS:
        for setting, (m, loss) in SETTINGS.items():
            for fam, conds in FAMILIES.items():
                students = [{"name": f"sgd_{setting}_{fam}_{c}", "setting": setting, "family": fam, "condition": c,
                             "noise": spec, "loss": loss, "m": m, "opt": "sgd", "lr": LR}
                            for c, spec in conds.items()]
                out.append(job("X3", s, students, cost=len(students), m=m))
    return out
