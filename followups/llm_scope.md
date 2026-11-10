# Follow-up scope: does head compatibility govern subliminal transfer in language models?

Status: scoped, not run. Needs a GPU and Hugging Face access, which this repo's
cloud environment does not have.

## Question

*Learning Through Noise* (Brockers et al., 2026) ends with a prediction: in a
language model, the vocabulary head plays both roles. Distillation-relevant tokens
act as the aux head and trait-relevant tokens as the class head, so subliminal
transfer should need **compatible unembedding geometry**, not a shared
initialisation of the transformer body. Our MNIST results support the head claim,
and step 2 found that a body-only screen (Madl's coverage) cannot see the readout.
The open question is whether either result survives at language-model scale.

## Minimal design

| Piece | Choice | Why |
| --- | --- | --- |
| Models | Pythia-160M at several pretraining seeds (the PolyPythia release), plus Pythia-70M/410M | Same data, tokenizer and architecture, different initialisation: the LM analogue of E2's "hidden layers reinitialised" |
| Trait | Teacher fine-tuned to raise one token's probability τ on neutral prompts (Madl's token trait) | Clean, measurable, low-harm |
| Channel | Student distilled on the teacher's next-token distribution over random-token prompts, τ masked from the loss | Mirrors aux-only distillation on noise |
| Transfer metric | Student's held-out P(τ) lift over its own base model, minus the lift of frequency-matched control tokens | Madl's metric, controls for generic peakiness |
| Head compatibility | Correlation of relative token orientations in the unembedding (Lee et al.) between teacher and student, and τ's neighbourhood overlap | The LM analogue of our class- and aux-head cosine |
| Screens | Body coverage, coverage x head match, early-agreement canary | Direct test of step 2 at scale |

Conditions, from most to least compatible:

1. Same checkpoint for teacher and student (shared initialisation).
2. Same size, different pretraining seed (different body, similar unembedding geometry).
3. Different size within the family (different body and width).
4. Different family and tokenizer (no canonical head alignment), as a negative control.

The paper predicts transfer in 1-3, ordered by head compatibility rather than by
body similarity, and none in 4.

## Budget

* About 20 teacher-student pairs x 3 seeds, each a few minutes of fine-tuning
  Pythia-160M on one mid-range GPU (an L4 or A10): roughly 5-10 GPU-hours in total.
* Pythia-70M alone could run on CPU, but at about 1-2 steps per second it would
  take days, not hours.

## What the environment needs

* A machine with a GPU (not available in Claude Code cloud environments).
* Network access to `huggingface.co` and its file CDN for the model weights.

## Risks

* PolyPythia's seeds may share more unembedding geometry than expected, making
  conditions 1 and 2 hard to separate. Mitigation: report head compatibility as a
  continuous variable, not just by condition.
* Token-trait transfer at 160M may be too weak to measure. Mitigation: Madl
  reports clear lifts down to 70M with the masked-token channel.
