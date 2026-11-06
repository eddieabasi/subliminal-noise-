# subliminal-noise

A from-scratch, CPU-only replication of

> **Learning Through Noise: Why Subliminal Learning Works and When It Fails**
> Brockers, Ventzke, Neuhaus, Hidalgo-Ogalde, Priesemann. arXiv [2605.23645](https://arxiv.org/abs/2605.23645), May 2026.

plus one extension: the pre-training **coverage screen** from

> **Channel Location Constrains the Auditability of Subliminal Learning**
> Madl. arXiv [2606.22019](https://arxiv.org/abs/2606.22019), June 2026.

Subliminal learning is when a student model inherits a teacher's traits from
distillation data that never mention them. The paper studies it in a controlled
MNIST setting: the teacher learns digits through a *class head*, the student only
ever imitates the teacher's *auxiliary* logits on random noise, and yet the student
ends up classifying digits through its untouched class head. The paper's claim is
that this is governed by **compatible output heads**, not by a shared
initialisation of the hidden layers.

RESULTS_PLACEHOLDER

## Setup

Everything follows the paper's Appendix C (Table 1) unless listed under *Deviations*.

| | |
| --- | --- |
| Models | MLP 784-256-256 with a class head (10) and an aux head (m, default 10); 271,892 parameters |
| Teacher | Adam 1e-3, 5 epochs, batch 1024, cross-entropy on the class head only |
| Student | Adam 1e-3, 5 epochs x 60 steps of 1,000 noise images (N = 60,000 per epoch), MSE on the aux head only |
| Noise | U(-1, 1) per pixel; Perlin noise (resolution 8) for CNN students |
| Data | MNIST, normalised with mean 0.1307, std 0.3081 |
| Uncertainty | mean over seeds, 95% bootstrap CI over seeds |
| Compute | 4 CPU cores, no GPU; about 7 s per teacher or student run on one core |

Parameter counts match the paper's Tables 1 and 2 exactly (checked in `tests/`).

## Repository layout

```
src/sublim/
  models.py     split-head MLP and CNN (Tables 1-2)
  init.py       layerwise shared / independent init, head perturbation, freezing
  noise.py      uniform, Perlin, and the X1 noise families
  train.py      teacher training, student aux distillation, evaluation
  coverage.py   Madl's coverage screen
  runner.py     one job = one teacher + all student conditions distilled from it
experiments/    one config per experiment (E1-E7, X1)
scripts/run.py      parallel, resumable sweep launcher -> results/runs.jsonl
scripts/analyze.py  tables (results/summary.md) and figures (results/figures/)
tests/          invariants: shared init identical, heads untouched where they must be, ...
```

## Reproduce

```bash
pip install torch torchvision scipy pandas matplotlib pytest
python -m pytest -q tests
python scripts/run.py e1_baseline e2_init_ablation e3_best_case e4_perturbation --workers 4
python scripts/run.py e5_architecture e6_m_and_n e7_latent_dim x1_noise_sweep --workers 4
python scripts/analyze.py
```

MNIST downloads to `data/` on first use (override with `SUBLIM_DATA`).
