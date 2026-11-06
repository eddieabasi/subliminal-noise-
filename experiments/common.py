"""Shared helpers for experiment configs."""
# Tier 1 uses the paper's 20 seeds (runs take ~7 s each); Tier 2 and X1 use 5.
SEEDS20 = range(20)
SEEDS5 = range(5)


def job(exp: str, seed: int, students: list[dict], cost: float = 1.0, **teacher) -> dict:
    return {"exp": exp, "seed": seed, "teacher": teacher, "students": students, "cost": cost}
