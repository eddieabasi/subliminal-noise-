"""Steps 1-2: does the coverage law come back in Madl's setup (X2), and which screen
predicts subliminal transfer across head mismatches and noise types?

Reads results/runs.jsonl (+ results/screens.jsonl for runs screened after the fact)
and writes results/screens_summary.md, results/screens_summary.json and figures.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from matplotlib.ticker import MaxNLocator
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import C, FIG, INK, INK2, RES, auroc, chance, plt  # noqa: E402

TRANSFER = 0.20  # accuracy above which a run counts as "transferred" for AUROC
PREDICTORS = {
    "coverage_tdisp": "Body coverage (Madl)",
    "readout_coverage": "One-step readout coverage",
    "class_head_cos": "Class-head match alone",
    "combined": "Body coverage x class-head match",
    "early_agreement": "Early canary (agreement after 5% of training)",
}


def load() -> pd.DataFrame:
    runs = pd.DataFrame([json.loads(x) for x in (RES / "runs.jsonl").read_text().splitlines() if x.strip()])
    cfg = pd.json_normalize(runs["student_cfg"])
    runs = pd.concat([runs.drop(columns=["student_cfg", "teacher_cfg"]), cfg.add_prefix("s.")], axis=1)
    path = RES / "screens.jsonl"
    if path.exists():
        scr = pd.DataFrame([json.loads(x) for x in path.read_text().splitlines() if x.strip()])
        key = ["exp", "seed", "student"]
        runs = runs.merge(scr, on=key, how="left", suffixes=("", "_scr"))
        for col in ["coverage_tdisp", "coverage_s2t", "grad_norm", "readout_coverage", "early_agreement",
                    "class_head_cos"]:
            if f"{col}_scr" in runs:
                runs[col] = runs[col].combine_first(runs[f"{col}_scr"]) if col in runs else runs[f"{col}_scr"]
        if "teacher_acc_rebuilt" in runs:
            rebuilt = runs.dropna(subset=["teacher_acc_rebuilt"])
            runs.attrs["rebuild_mismatches"] = int((rebuilt.teacher_acc_rebuilt != rebuilt.teacher_acc).sum())
            runs.attrs["rebuilt"] = len(rebuilt)
    runs["combined"] = runs["coverage_tdisp"] * runs["class_head_cos"].clip(lower=0)
    return runs


def score(d: pd.DataFrame, col: str) -> tuple[float, float]:
    d = d.dropna(subset=[col, "student_acc"])
    if len(d) < 5 or d[col].nunique() < 2:
        return float("nan"), float("nan")
    return float(spearmanr(d[col], d.student_acc).statistic), auroc(d[col], d.student_acc > TRANSFER)


def x2(df: pd.DataFrame, out: dict, exp: str = "X2", label: str = "Adam students", fig_name: str = "x2_madl_sweep.png") -> None:
    d = df[df.exp == exp]
    if d.empty:
        return
    res, md = {}, [f"## Step 1 - {exp} ({label}): coverage law in a Madl-like setup\n",
                   "Seed-pooled Spearman ρ between coverage and student accuracy over noise conditions "
                   "(Madl reports 0.95 over his 54).\n",
                   "| Setting | Madl's 5 families (33 conditions) | + Perlin (36) | Leave-one-family-out, min | "
                   "Accuracy range |", "| --- | --- | --- | --- | --- |"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, setting in zip(axes, ["madl_kl", "madl_mse", "ours"]):
        g = d[d["s.setting"] == setting].groupby(["s.family", "s.condition"])[["coverage_tdisp", "student_acc"]].mean()
        g = g.reset_index()
        madl = g[g["s.family"] != "perlin"]
        rho5 = spearmanr(madl.coverage_tdisp, madl.student_acc).statistic
        rho6 = spearmanr(g.coverage_tdisp, g.student_acc).statistic
        lofo = min(spearmanr(madl[madl["s.family"] != f].coverage_tdisp, madl[madl["s.family"] != f].student_acc).statistic
                   for f in madl["s.family"].unique())
        fam_rho = {f: float(spearmanr(x.coverage_tdisp, x.student_acc).statistic)
                   for f, x in g.groupby("s.family") if len(x) > 2}
        res[setting] = {"rho_madl_families": rho5, "rho_with_perlin": rho6, "lofo_min": lofo,
                        "per_family": fam_rho, "rows": g.to_dict("records")}
        md.append(f"| {setting} | {rho5:.2f} | {rho6:.2f} | {lofo:.2f} | "
                  f"{100 * g.student_acc.min():.0f}-{100 * g.student_acc.max():.0f}% |")
        ax.scatter(madl.coverage_tdisp, madl.student_acc, color=C[0], s=22, label="Madl's 5 families", zorder=3)
        p = g[g["s.family"] == "perlin"]
        ax.scatter(p.coverage_tdisp, p.student_acc, color=C[1], s=22, label="Perlin", zorder=3)
        for _, r in p.iterrows():
            ax.annotate(r["s.condition"], (r.coverage_tdisp, r.student_acc), textcoords="offset points",
                        xytext=(4, 2), fontsize=7, color=INK2)
        names = {"madl_kl": "3 aux logits, KL channel", "madl_mse": "3 aux logits, MSE", "ours": "10 aux logits, MSE"}
        ax.set_title(f"{names[setting]}: ρ = {rho5:.2f}", color=INK)
        ax.set_xlabel("coverage, before training")
        ax.xaxis.set_major_locator(MaxNLocator(5))
        chance(ax)
    axes[0].set_ylabel("student MNIST accuracy")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle(f"Step 1 ({label}): coverage vs transfer over 36 fixed-RMS noise conditions (3 seeds pooled)",
                 color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / fig_name)
    plt.close(fig)
    md += ["", "Per-family Spearman ρ (seed-pooled):\n",
           "| Setting | " + " | ".join(sorted(res["ours"]["per_family"])) + " |",
           "| --- |" + " --- |" * len(res["ours"]["per_family"])]
    for setting, r in res.items():
        md.append(f"| {setting} | " + " | ".join(f"{r['per_family'][f]:.2f}" for f in sorted(r["per_family"])) + " |")
    out[exp] = res
    out["md"] += md + [""]


def step2(df: pd.DataFrame, out: dict) -> None:
    pools = {
        "Head + body mismatches (E2+E4, m = 10)": df[df.exp.isin(["E2", "E4"]) & (df["s.m"].isna() | (df["s.m"] == 10))],
        "All of E2 (m = 1-250)": df[df.exp == "E2"],
        "Noise types, shared init (X1)": df[df.exp == "X1"],
        "Fixed-RMS noise sweep (X2)": df[df.exp == "X2"],
    }
    res = {}
    md = ["## Step 2 - which screen predicts transfer?\n",
          f"Spearman ρ with student accuracy / AUROC for transfer (accuracy above {TRANSFER:.0%}), per run.\n",
          "| Screen | " + " | ".join(f"{k} (n = {len(v)})" for k, v in pools.items()) + " |",
          "| --- |" + " --- |" * len(pools)]
    for col, name in PREDICTORS.items():
        cells = []
        for pname, pdf in pools.items():
            rho, auc = score(pdf, col)
            res.setdefault(col, {})[pname] = {"spearman": rho, "auroc": auc}
            cells.append(f"{rho:.2f} / {auc:.2f}")
        md.append(f"| {name} | " + " | ".join(cells) + " |")
    out["step2"] = res
    out["md"] += md + [""]

    p1 = pools["Head + body mismatches (E2+E4, m = 10)"].copy()
    cond = p1["s.condition"].fillna("")
    head = p1["s.head"].fillna("")
    p1["mismatch"] = np.select([cond.isin(["reinit_class"]) | (head == "class"),
                                cond.isin(["reinit_aux"]) | (head == "aux"),
                                cond.isin(["reinit_l1", "reinit_l2", "reinit_hidden"])],
                               ["class head", "aux head", "hidden layers"], "nothing")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, col in zip(axes, ["coverage_tdisp", "combined", "early_agreement"]):
        nothing = p1[p1.mismatch == "nothing"]
        ax.scatter(nothing[col], nothing.student_acc, color=INK2, s=10, label="nothing (shared)", zorder=2)
        for g, c in [("hidden layers", C[0]), ("aux head", C[1]), ("class head", C[2])]:
            gd = p1[p1.mismatch == g]
            ax.scatter(gd[col], gd.student_acc, color=c, s=10, label=f"{g} differ", zorder=3,
                       edgecolor="white", linewidth=0.4)
        rho, auc = score(p1, col)
        ax.set_title(f"{PREDICTORS[col].split(' (')[0]}\nρ = {rho:.2f}, AUROC = {auc:.2f}", color=INK, fontsize=9)
        ax.set_xlabel(PREDICTORS[col].split(" (")[0].lower())
        chance(ax)
    axes[0].set_ylabel("student MNIST accuracy")
    axes[0].legend(loc="upper left", fontsize=7)
    fig.suptitle("Step 2: screens on E2 + E4 runs at m = 10 (20 seeds)", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "step2_screens.png")
    plt.close(fig)


def main() -> None:
    df = load()
    out: dict = {"md": ["# Follow-up: coverage in Madl's setup, and readout-aware screens\n"]}
    if "rebuilt" in df.attrs:
        out["md"].append(f"Teachers rebuilt for after-the-fact screening: {df.attrs['rebuilt']} runs, "
                         f"{df.attrs['rebuild_mismatches']} with a teacher accuracy different from the original run.\n")
    x2(df, out)
    x2(df, out, "X3", "plain SGD students, lr 3", "x3_sgd_sweep.png")
    if "X2" in out and "X3" in out:
        out["md"] += ["## Adam vs plain SGD (Spearman ρ, seed-pooled)\n",
                      "| Setting | Adam: 5 families | SGD: 5 families | Adam: + Perlin | SGD: + Perlin | "
                      "Adam: accuracy range | SGD: accuracy range |", "| --- | --- | --- | --- | --- | --- | --- |"]
        for setting in out["X2"]:
            a, g = out["X2"][setting], out["X3"].get(setting)
            if g is None:
                continue
            rng = lambda r: f"{100 * min(x['student_acc'] for x in r['rows']):.0f}-{100 * max(x['student_acc'] for x in r['rows']):.0f}%"
            out["md"].append(f"| {setting} | {a['rho_madl_families']:.2f} | {g['rho_madl_families']:.2f} | "
                             f"{a['rho_with_perlin']:.2f} | {g['rho_with_perlin']:.2f} | {rng(a)} | {rng(g)} |")
        out["md"].append("")
    step2(df, out)
    (RES / "screens_summary.md").write_text("\n".join(out.pop("md")))
    (RES / "screens_summary.json").write_text(json.dumps(out, indent=1, default=float))
    print((RES / "screens_summary.md").read_text())


if __name__ == "__main__":
    main()
