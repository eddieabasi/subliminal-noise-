"""E7 - latent dimension (Fig. 6a / Fig. 13). Teacher and student fully shared, MLP
(256, d): d is the width of the latent layer that feeds both heads. Four head
regimes: free, student aux head fixed, teacher class head fixed, both fixed."""
from .common import SEEDS5, job

DS = [16, 64, 256, 1024, 4096, 16384]


def jobs():
    out = []
    for s in SEEDS5:
        for d in DS:
            cost = 1 + d / 2048
            out.append(job("E7", s, [
                {"name": f"d{d}_free", "d": d, "heads": "free"},
                {"name": f"d{d}_aux_fixed", "d": d, "heads": "aux_fixed", "freeze_aux": True},
            ], cost=2 * cost, m=10, widths=[256, d]))
            out.append(job("E7", s, [
                {"name": f"d{d}_class_fixed", "d": d, "heads": "class_fixed"},
                {"name": f"d{d}_both_fixed", "d": d, "heads": "both_fixed", "freeze_aux": True},
            ], cost=2 * cost, m=10, widths=[256, d], freeze_class=True))
    return out
