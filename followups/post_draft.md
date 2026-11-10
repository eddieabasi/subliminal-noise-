# Replicating "Learning Through Noise": the heads matter, and a pre-training screen can't see one of them

*Draft for sharing (e.g. LessWrong / Alignment Forum). Not posted. Code and data:
[eddieabasi/subliminal-noise-](https://github.com/eddieabasi/subliminal-noise-).*

## TL;DR

* We replicated the MNIST experiments of *Learning Through Noise* (Brockers et al.,
  May 2026) from scratch on 4 CPU cores. Its central claim holds: a student trained
  only on noise inherits the teacher's digit classifier as long as its **output heads**
  match the teacher's, even when its hidden layers are reinitialised (82.9% at 250 aux
  logits) or it is a different architecture. A reinitialised class head keeps it at
  chance. The best case reproduces almost exactly: student 96.7% vs. teacher 96.8%
  (paper: 96.82% vs. 96.86%).
* We also tested the pre-training "coverage" screen from *Channel Location Constrains
  the Auditability of Subliminal Learning* (Madl, June 2026), which predicted transfer
  with Spearman ρ ≈ 0.95. In our hands it reaches 0.40-0.49 across noise types, even
  in a setup matched to his. It works within DC-offset and mixture noise
  (ρ 0.86-1.00) but inverts across noise smoothness.
* More importantly for auditing: coverage is computed on the network body, so it is
  **blind to the class head**, the component that, per Brockers et al., decides whether
  transfer happens at all. Multiplying coverage by a class-head match lifts its
  correlation with transfer from 0.43 to 0.73; an "early canary" that runs the first
  5% of distillation reaches 0.58-0.78 across every pool we tested.

## Setup

A split-head MLP (784-256-256) reads its latent through two linear heads: a class
head (10 digits) and an aux head (m extra logits). The teacher learns MNIST through
the class head only. The student never sees a digit: it only matches the teacher's
aux logits on uniform noise. Transfer is the student's digit accuracy through its
untouched class head. All hyperparameters follow the paper's Appendix C; 1,065
runs for the replication and about 700 more for the follow-ups.

## What replicated

| Claim | Ours | Verdict |
| --- | --- | --- |
| Noise-only students learn MNIST | 87.7% (teacher 97.1%, chance 10%) | Reproduced |
| Shared hidden layers are not needed | Both hidden layers reinitialised: 11% at m = 1 to 82.9% at m = 250 | Reproduced |
| A mismatched head blocks transfer | Class head reinitialised: 10-11% at every m; aux head: below 25% | Reproduced |
| Best case matches the teacher | 96.7% vs 96.8% | Reproduced |
| Large latent width breaks it via aux-head drift | 14.9% at d = 16,384; 71.9% with the aux head frozen | Reproduced |
| Other architectures transfer | MLPs with one more or one fewer layer: yes. CNN: CNN_RESULT | CNN_VERDICT |
| m and N are joint bottlenecks | M1_RESULT | M1_VERDICT |
| EMNIST: more classes, weaker transfer | EMNIST_RESULT | EMNIST_VERDICT |

## The screen and its blind spot

Madl's coverage is the cosine between the student's first distillation gradient and
the teacher's training displacement, both in body-parameter space, computed before
the student trains. Two findings:

1. **It does not generalise across noise types the way the paper reports.** Over 36
   noise conditions at a fixed input scale, in five families matching Madl's (DC level,
   rank, spectral slope, distribution shape, mixtures), Spearman ρ is 0.40 with his
   channel (3 aux logits, soft-label KL), 0.49 with MSE, and 0.47 in our 10-logit
   setting. The relationship is an inverted U: smooth, low-frequency noise has the
   highest coverage but transfers poorly. Within the DC and mixture families, the
   "causal dial" of his Section 3, it holds (ρ 0.86-1.00).
2. **It cannot see the class head.** A student whose class head is reinitialised or
   perturbed gets *exactly* the same coverage as one whose head matches, yet ends at
   chance. That is Brockers et al.'s second condition, and it lives outside the
   parameters coverage looks at. We had predicted the opposite failure (that coverage
   would break when hidden layers are reinitialised); it held up there.

What helps:

| Screen (Spearman ρ / AUROC for transfer above 20%) | Head and body mismatches | All init conditions | Noise types |
| --- | --- | --- | --- |
| Body coverage | 0.43 / 0.72 | 0.48 / 0.74 | 0.43 / 0.70 |
| Body coverage x class-head cosine | **0.73 / 0.81** | **0.79 / 0.89** | 0.43 / 0.70 |
| Early canary: agreement with the teacher after 5% of training | 0.68 / 0.82 | 0.78 / 0.88 | **0.58 / 0.79** |
| One-step readout coverage (first-order change of class logits) | 0.08 / 0.55 | 0.22 / 0.59 | 0.38 / 0.65 |

The lesson matches Madl's own framing of "channel location", one level further out:
a screen has to look at the readout that will express the trait, not only at the
body that carries it. In a language model that readout is the unembedding, which is
where his token-trait results already point.

## Caveats

* MNIST toy setting; 5-20 seeds per condition.
* Our noise families are reconstructions of Madl's, not his 54 conditions, and his
  exact hyperparameters live in scripts we did not have.
* The early canary is not a pre-training screen: it needs 5% of the distillation run.

## Notes for the authors

* In the released code, `scripts/figure6/d_sweep_fixed_aux_head.sh` passes
  `--teacher-trainable true,true,false,true`, which the positional parser maps to
  `[fc1, fc2, class_head, aux_head]`, so it freezes the **class** head, and
  `d_sweep_fixed_class_head.sh` freezes the aux head. Our runs freeze heads by name
  and agree with the paper's text (aux-head drift dominates), so this looks like a
  file-naming slip rather than a wrong result.
* PERLIN_NOTE
