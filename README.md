# DAS3 Neuro-Voice Controller

Safety-constrained voice-command execution for an assistive robotic arm: a locally run
Distil-Whisper ASR front-end, a Phonetic Interceptor transcript-repair stage, a
purpose-built Small Language Model (SLM) intent classifier trained from scratch, a
hierarchical Gatekeeper safety module, and an OpenSim execution layer driving the
Dynamic Arm Simulator 3 (DAS3) musculoskeletal model of the right upper limb.

This repository is organised so that anybody can **follow the work in the order it
happened** — from the first prototype (V0) through two architectural revisions to the
fourteen-participant live evaluation reported in the paper — and re-derive the published
numbers from the raw material.

The dataset was constructed and used as the experimental basis for the paper below, and is
released here as a standalone research resource — to support reproducibility and to serve
as a benchmark for future work on voice-controlled assistive manipulation.

---

## The result

A 44,189,193-parameter DistilBERT-architecture classifier, trained from scratch on a
purpose-built nine-command corpus with a custom 872-token tokenizer, reaches
**99.95 % top-1 accuracy** in live use, while a deterministic safety hierarchy reduces a
**11.5 % priority-override failure rate to 0.0 %**.

---

## Repository layout

```

dataset/                     the nine seed corpora (the "database") + command schema
src/                         all code, numbered in pipeline order
versions/                    V0 -> V1 -> V2 development trail with per-version reports
experiments/live_study_N14/  the fourteen-participant evaluation (logs, metrics, protocol)
analysis/                    scripts that re-derive the published figures from raw logs
```

---

## How to trace the work

### Step 1 — the problem, and the two architectural fixes (`versions/`)

Each version folder holds its own evaluation report, its good-command log, its failure
log, and the exact source code that produced them. The story is legible from the three
reports alone:

| | V0 | V1 | V2 |
|---|---|---|---|
| Expressions tested | 218 | 202 | 204 |
| Success | 176 (80.7 %) | 164 (81.2 %) | 201 (98.5 %) |
| Core SLM misunderstandings | 6 (2.8 %) | 3 (1.5 %) | 3 (1.5 %) |
| ASR (Voice AI) mistakes | 11 (5.0 %) | 35 (17.3 %) | **0 (0.0 %)** |
| **Priority-override failures** | **25 (11.5 %)** | **0 (0.0 %)** | **0 (0.0 %)** |

* **V0 → V1** introduced the *Hierarchical Gatekeeper*. In V0, an utterance combining a
  safety word with a movement ("stop moving the shoulder up") was classified as
  `SHOULDER_UP`, because attention weight sat on the anatomical tokens. Routing STOP/REST
  keywords around the classifier entirely took that failure mode from 25/218 to 0.
* **V1 → V2** introduced the *Phonetic Interceptor*. V1 exposed a second bottleneck: the
  ASR fractured anatomical terms ("forearm" → "four arm", "abort" → "a board"), pushing
  ASR-attributable failures to 17.3 %. Repairing transcripts before the Gatekeeper took
  that to 0.

### Step 2 — how the model was built (`dataset/`, `src/`)

`dataset/seed_corpora/` contains the nine hand-authored command files (400 expressions
each). Run the pipeline in `src/` in numbered order to reproduce the model:

```
src/02_dataset_build/build_slm_database.py        3,600 raw -> 3,591 balanced -> 2,872/719 split
src/03_tokenizer_and_training/build_slm_tokenizer.py   custom 872-token WordPiece vocabulary
src/03_tokenizer_and_training/train_slm_model.py       30 epochs, 53,220 steps, from random init
src/04_evaluation/                                     blind test set + confusion matrix / F1
```

The logs of the actual tokenizer build and training run are kept alongside the scripts
(`log_tokenizer_build.txt`, `log_training_run.txt`) so the reported figures can be checked
against the run that produced them.

### Step 3 — the live evaluation (`experiments/live_study_N14/`)

Fourteen participants, a standardised 144-command protocol organised as eight functional
arm poses. `logs_anonymised/` holds one execution log per participant (`U01.txt` …
`U14.txt`); `metrics/` holds the per-participant measurement tables; `protocol/` holds the
command script and the target-pose images.

### Step 4 — re-derive the published numbers (`analysis/`)

```bash
python analysis/verify_accounting.py     # input-attempt accounting across all 14 logs
python analysis/verify_corpus.py         # replays dataset construction from the seed files
```

`verify_accounting.py` reproduces the three-way outcome decomposition used in the paper:

```
input attempts        2,705  =  2,632 [MIC RAW] + 73 [TEXT IN]
classifier decisions  1,968  =  1,882 [SLM DECISION] + 86 [FAST-PATH BYPASS]
Gatekeeper blocks       646
safety overrides         91  =  86 OVERRIDE->REST + 5 EMERGENCY OVERRIDE->STOP
                      -----
                      2,705
```

`verify_corpus.py` reproduces the dataset figures: 400 x 9 = 3,600 raw, one exact duplicate
removed, all classes clamped to 399 -> 3,591 balanced, stratified 80/20 -> 2,872 / 719.

---

## Citation

If you use this dataset, the code, or the evaluation logs, please cite the paper:

> Emanuel Muntean, Monica Leba, Andreea Ionica, "Hearing It Right, Doing It Safely:
> A Reflex-Inspired Safety Gatekeeper for Voice-Controlled Exoskeleton Arm Manipulation",
> submitted to *Biomimetics*, special issue "Advanced Service Robots: Exoskeleton Robots 2026", 2026.

```bibtex
@article{muntean2026hearing,
  author  = {Muntean, Emanuel and Leba, Monica and Ionica, Andreea},
  title   = {Hearing It Right, Doing It Safely: A Reflex-Inspired Safety
             Gatekeeper for Voice-Controlled Exoskeleton Arm Manipulation},
  journal = {Biomimetics},
  year    = {2026},
  note    = {Submitted}
}
```

## Participant data and privacy

The live study involved human participants. **Voice recordings and the participant-identity
key are deliberately not in this repository.** See [`PRIVACY.md`](PRIVACY.md) for exactly
what was excluded and why. Logs are published under anonymous identifiers only; the
mapping from identifier to person is not distributed.
