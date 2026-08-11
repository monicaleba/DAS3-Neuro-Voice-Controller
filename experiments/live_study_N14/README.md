# Live study — fourteen participants

## Design

Fourteen participants (age 18-50; 3 female, 11 male), all native Romanian speakers issuing
commands in English, matching the English-only scope of the deployed
`distil-whisper/distil-small.en` front-end. Each executed a standardised protocol of 144
default commands organised as eight functional arm poses, with a scripted "relax" (REST)
instruction between consecutive poses.

## Contents

* `logs_anonymised/U01.txt` … `U14.txt` — one full execution log per participant.
* `metrics/` — per-participant measurement tables.
* `protocol/Command_Protocol_144.docx` — the command script read by participants.
* `protocol/target_poses/` — the eight target arm configurations.

Identifiers are anonymous; see `../../PRIVACY.md`.

## Log format

```
[15:55:39] [SYSTEM] Initializing Master AI Pipeline...
[16:01:01]   [MIC RAW]: 'THE ELBOW UPWARD.'          <- ASR transcript
[16:01:01]   [SANITIZED]: 'FOREARM UP'               <- Phonetic Interceptor repair
[16:01:01]   [SLM DECISION] ELBOW_UP (Conf: 1.00)    <- classifier output
[16:01:03]   [GATEKEEPER] BLOCKED (Not enough valid keywords)
[17:12:44]   [GATEKEEPER] OVERRIDE TRIGGERED -> REST <- safety-tier bypass
```

## Outcome accounting

Every logged attempt terminates in exactly one of three mutually exclusive outcomes:

| Outcome | Count |
|---|---|
| Classifier-mediated decisions (`SLM DECISION` + `FAST-PATH BYPASS`) | 1,968 |
| Gatekeeper blocks | 646 |
| Safety-tier overrides (86 REST + 5 STOP) | 91 |
| **Total input attempts** (2,632 `MIC RAW` + 73 `TEXT IN`) | **2,705** |

Safety overrides belong to neither of the first two categories: they bypass the classifier
and are dispatched rather than rejected. Reproduce with
`python ../../analysis/verify_accounting.py`.

One caveat worth knowing: the raw logs contain **647** `BLOCKED` lines. One of them (U11,
17:41:16) follows a listening cycle that produced no transcript, so it has no associated
input attempt and is excluded from transcript-based analysis, giving the 646 above.
