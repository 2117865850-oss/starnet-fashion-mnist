# AI workflow and prompt record

This project was generated and run by the Codex AI agent on 20 September 2026.
No pre-existing student code or experiment results were supplied. The user
provided AIassign1.pdf and requested these report contents:

> Your AI assignment report in Word/PDF format should contain the following details (1) how you ask AI tools to find the algorithm, (2) algorithm description, (3) how AI implements the algorithm, (4) experiment settings and results, (5) what you have learnt from this AI assignment, and (6) a webpage link of your source codes.

When the agent asked whether to use an existing project or create one, the user
selected: **Create a new AI-generated project**.

## Actual research delegation

The main agent gave a research subagent the following task:

> Independently research StarNet CVPR 2024 Rewrite the Stars as recent deep CNN candidate for CISC3024 AI assignment. Use official paper and repository only, web browse. Report exact architecture/star operation, claims with caveats, date/primary URLs, practical small-image adaptation and pitfalls. Do not write main files or run training. User brief asks recent CNN or deep autoencoder, algorithm description, experiment settings results and AI prompting. We will run original local experiment.

Research verified the CVPR 2024 publication and the original model implementation.
The main agent also searched for the official Fashion-MNIST dataset. The research
agent corrected the original guessed code path to `imagenet/starnet.py` and warned
that the full ImageNet architecture downsamples too aggressively for 28x28 inputs.

## Implementation directions condensed from the actual agent task

The main agent requested an AI-generated, reproducible StarNet-inspired experiment
on Fashion-MNIST. The implementation directions specified the exact star block,
a three-stage small-image architecture, an otherwise identical addition ablation,
12,000 balanced training images, 2,000 validation images, the official 10,000 test
images, split seed 2026, paired training seeds 42/43/44, five epochs, AdamW,
learning rate 0.001, weight decay 0.0001, batch size 128, pixel scaling by 255,
validation-only checkpoint selection, actual timing, and raw evidence outputs.
The agent was asked to verify shape, finite gradients, split disjointness,
class balance, and identical initial parameters in each star/sum pair.

The implementation agent generated `model.py`, `train.py`, `plot_results.py`,
and documentation. It benchmarked MPS, then executed all six planned runs.
Training settings were fixed before viewing test scores. No hyperparameters were
changed after the experiment. The main agent requested a separate read-only audit.

## Report and reflection

A document agent used the real result JSON/CSV files to generate the Word/PDF
report. It was directed to distinguish our compact adaptation from the original
ImageNet experiments and to label the learning reflection as a draft for the
student's review. The report and source webpage were prepared by AI.

Condensed descriptions here are summaries of the actual workflow, not an invented
verbatim student conversation. The full conversation remains in the Codex task.
