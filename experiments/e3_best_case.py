"""E3 - optimal condition (Tables 3-4): m = 500, N = 10^6 noise samples per epoch,
random student hidden layers, teacher class head and student aux head frozen."""
from .common import SEEDS20, job


def jobs():
    student = {"name": "best_case", "reinit": ["l1", "l2"], "freeze_aux": True, "steps": 1000}
    return [job("E3", s, [student], cost=17, m=500, freeze_class=True) for s in SEEDS20]
