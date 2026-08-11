# Participant data — what is included and what is not

The fourteen-participant evaluation involved human subjects. This repository is built to
be publishable, so identifiable material was removed at assembly time rather than left to
a `.gitignore` rule.

## Excluded, deliberately

| Excluded | Volume | Reason |
|---|---|---|
| `*.wav` — participant voice recordings | 33 files, ~413 MB | Voice is biometric and directly identifying. Publication would also exceed normal repository size limits. |
| `Users ID.txt` — participant-ID to name mapping | 1 file | This is the re-identification key. Publishing it alongside anonymised logs would defeat the anonymisation completely. |
| `*.rar` archives | 2 files | Contain copies of the above. |

## Included, anonymised

Execution logs are published as `experiments/live_study_N14/logs_anonymised/U01.txt` …
`U14.txt`. Two checks were run at assembly time and both pass:

1. **No participant name appears in any path** in this repository.
2. **No participant name appears inside any log file.** Names were present only in the
   original filenames, never in log content; renaming the files was therefore sufficient.

The identifier assignment (U01 … U14) was verified against the recorded per-participant
metrics: the `[MIC RAW]` count in each log matches that participant's Total Voice Inputs
figure in `metrics/Metrics_per_participant.docx` for all fourteen participants.

## What the logs still contain

Verbatim ASR transcripts of what each participant said while performing the command
protocol. These are task-directed utterances about arm movement, but they also include
off-task speech captured under continuous listening. 
