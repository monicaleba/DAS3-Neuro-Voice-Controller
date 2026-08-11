# Development trail: V0 -> V1 -> V2

Each folder contains that version's evaluation report, its good-command log, its failure
log, and the source code that produced them. Read the three reports in order.

## Summary

| Metric | V0 | V1 | V2 |
|---|---|---|---|
| Total expressions | 218 | 202 | 204 |
| Success | 176 (80.7 %) | 164 (81.2 %) | 201 (98.5 %) |
| Core SLM misunderstandings | 6 (2.8 %) | 3 (1.5 %) | 3 (1.5 %) |
| ASR (Voice AI) mistakes | 11 (5.0 %) | 35 (17.3 %) | 0 (0.0 %) |
| Priority-override failures | 25 (11.5 %) | 0 (0.0 %) | 0 (0.0 %) |

## V0 — the baseline, and the safety flaw

218 expressions, 80.7 % success. The critical finding is the **11.5 % priority-override
failure rate**: 25 utterances that combined a safety keyword with a movement verb were
executed as movements. `Failures_V0.docx` There were 25 verbatim, as
*"STOP MOVING THE SHOULDER UP."* returned `SHOULDER_UP` at confidence 1.00. Of the 25,
15 should have resolved to `STOP` and 10 to `REST`.

Derivation: 25 / 218 = 11.47 % -> 11.5 %. The whole table reconciles: 176 + 42 = 218 and
6 + 11 + 25 = 42.

## V1 — the Hierarchical Gatekeeper

STOP and REST keyword arrays are matched *before* the SLM is invoked and dispatched
unconditionally, so no amount of anatomical co-occurrence can outvote a safety word.
Priority-override failures: **25 -> 0**. The change also made the next bottleneck visible:
ASR failures rose to 17.3 % as the safety layer stopped masking them.

## V2 — the Phonetic Interceptor

A transcript-repair stage ahead of the Gatekeeper fixes the recurrent ASR hallucinations
of this vocabulary ("four arm" -> FOREARM, "a board" -> ABORT). ASR-attributable failures:
**35 -> 0**; overall success 98.5 %.
