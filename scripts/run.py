"""Run one or more experiment configs in parallel and append results to a JSONL file.

    python scripts/run.py e1_baseline e2_init_ablation --workers 4

Resumable: student runs already present in the output file are skipped.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from sublim.runner import run_job  # noqa: E402


def load_jobs(names: list[str]) -> list[dict]:
    jobs = []
    for name in names:
        mod = importlib.import_module(f"experiments.{name}")
        jobs += mod.jobs()
    return jobs


def done_keys(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            keys.add((r["exp"], r["seed"], r["student"]))
    return keys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiments", nargs="+")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "results" / "runs.jsonl"))
    ap.add_argument("--seeds", type=int, default=None, help="limit to the first N seeds")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = done_keys(out)
    jobs = []
    for job in load_jobs(args.experiments):
        if args.seeds is not None and job["seed"] >= args.seeds:
            continue
        todo = [s for s in job["students"] if (job["exp"], job["seed"], s["name"]) not in done]
        if todo:
            jobs.append({**job, "students": todo})
    jobs.sort(key=lambda j: -j.get("cost", 1.0))  # heaviest first for load balance
    n_students = sum(len(j["students"]) for j in jobs)
    print(f"{len(jobs)} jobs, {n_students} student runs to do", flush=True)

    t0, finished = time.time(), 0
    ctx = get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool, out.open("a") as fh:
        futures = {pool.submit(run_job, job): job for job in jobs}
        for fut in as_completed(futures):
            job = futures[fut]
            try:
                rows = fut.result()
            except Exception as exc:  # report and keep going
                print(f"FAILED {job['exp']} seed={job['seed']}: {exc!r}", flush=True)
                continue
            for r in rows:
                fh.write(json.dumps(r) + "\n")
                finished += 1
                print(f"[{finished}/{n_students}] {r['exp']} seed={r['seed']} {r['student']}: "
                      f"teacher={r['teacher_acc']:.4f} student={r['student_acc']:.4f} "
                      f"({r['seconds_student']:.0f}s, elapsed {time.time() - t0:.0f}s)", flush=True)
            fh.flush()
    print(f"DONE in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
