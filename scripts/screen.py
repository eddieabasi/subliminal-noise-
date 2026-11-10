"""Recompute the pre-training screens for runs that already exist, without retraining
the students: rebuild each job's teacher (bit-identical, see README) and every student
initialisation, compute the screens, and join them to the recorded accuracy.

    python scripts/screen.py e2_init_ablation e4_perturbation x1_noise_sweep --workers 4

Writes results/screens.jsonl (one line per student run, keyed by exp/seed/student).
"""
from __future__ import annotations

import argparse
import copy
import importlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import torch  # noqa: E402

from sublim.init import components, freeze_, make_student, make_teacher, perturb_  # noqa: E402
from sublim.runner import PROBES, STUDENT_DEFAULTS, TEACHER_DEFAULTS, data, noise_fn, screens  # noqa: E402
from sublim.train import accuracy, train_teacher  # noqa: E402


def screen_job(job: dict) -> list[dict]:
    torch.set_num_threads(1)
    d = data()
    seed = int(job["seed"])
    tcfg = {**TEACHER_DEFAULTS, **job.get("teacher", {})}
    teacher = make_teacher(tcfg["arch"], tuple(tcfg["widths"]), tcfg["m"], seed)
    if tcfg["freeze_class"]:
        freeze_(teacher.class_head)
    teacher_init = copy.deepcopy(teacher)
    train_teacher(teacher, d["x_train"], d["y_train"], seed=seed + 1, epochs=tcfg["epochs"])
    teacher_acc = accuracy(teacher, d["x_test"], d["y_test"])
    rows = []
    for raw in job["students"]:
        scfg = {**STUDENT_DEFAULTS, **raw}
        student = make_student(teacher_init, seed, scfg["arch"],
                               tuple(scfg["widths"]) if scfg["widths"] else None, tuple(scfg["reinit"]))
        if scfg["perturb"]:
            perturb_(components(student)[scfg["perturb"]["head"]], scfg["perturb"]["delta"], seed + 3)
        if scfg["freeze_aux"]:
            freeze_(student.aux_head)
        row = {"exp": job["exp"], "seed": seed, "student": scfg["name"], "teacher_acc_rebuilt": teacher_acc}
        row.update(screens(student, teacher_init, teacher, noise_fn(scfg["noise"]), seed, scfg, d["x_train"][:PROBES]))
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiments", nargs="+")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "results" / "screens.jsonl"))
    args = ap.parse_args()
    out = Path(args.out)
    done = set()
    if out.exists():
        done = {(r["exp"], r["seed"], r["student"]) for r in map(json.loads, out.read_text().splitlines())}
    jobs = []
    for name in args.experiments:
        for job in importlib.import_module(f"experiments.{name}").jobs():
            todo = [s for s in job["students"] if (job["exp"], job["seed"], s["name"]) not in done]
            if todo:
                jobs.append({**job, "students": todo})
    print(f"{len(jobs)} jobs, {sum(len(j['students']) for j in jobs)} students to screen", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(args.workers, mp_context=get_context("spawn")) as pool, out.open("a") as fh:
        for i, fut in enumerate(as_completed([pool.submit(screen_job, j) for j in jobs]), 1):
            for r in fut.result():
                fh.write(json.dumps(r) + "\n")
            fh.flush()
            if i % 10 == 0 or i == len(jobs):
                print(f"[{i}/{len(jobs)} jobs] elapsed {time.time() - t0:.0f}s", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
