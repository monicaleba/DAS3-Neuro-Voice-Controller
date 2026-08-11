# Dataset

## Seed corpora

`seed_corpora/` holds the nine hand-authored command files that constitute the master
corpus. Each file contains exactly **400** double-quoted expressions; the file of origin
*is* the intent label.

| File | Command class | Kinematic effect |
|---|---|---|
| `stop.txt` | `STOP` | `EMERGENCY(action=stop)` |
| `rest.txt` | `REST` | `STATE(target=zero_pose)` |
| `move_up.txt` | `SHOULDER_UP` | `INCREMENT(axis=GH_z, direction=positive, amount=10)` |
| `move_down.txt` | `SHOULDER_DOWN` | `INCREMENT(axis=GH_z, direction=negative, amount=10)` |
| `move_left.txt` | `SHOULDER_LEFT` | `INCREMENT(axis=GH_y, direction=positive, amount=10)` |
| `move_right.txt` | `SHOULDER_RIGHT` | `INCREMENT(axis=GH_y, direction=negative, amount=10)` |
| `pull.txt` | `ELBOW_UP` | `INCREMENT(axis=EL_x, direction=flex, amount=10)` |
| `push.txt` | `ELBOW_DOWN` | `INCREMENT(axis=EL_x, direction=extend, amount=10)` |
| `turn.txt` | `ROTATE_WRIST` | `INCREMENT(axis=PS_y, direction=turn, amount=90)` |

The file-to-class mapping is not arbitrary: elbow flexion is sourced from `pull.txt` and
extension from `push.txt`, reflecting the pulling and pushing motions that raise and lower
the forearm.

## Construction pipeline

Run `src/02_dataset_build/build_slm_database.py`. It reproduces exactly:

```
3,600 raw expressions          (400 x 9)
   -1 exact duplicate          ("lift the forearm higher", twice inside pull.txt)
3,599 after deduplication      (ELBOW_UP reduced to 399)
3,591 balanced                 (all classes clamped to the 399 minimum)
2,872 / 719                    (stratified 80/20 split, random_state=42)
```

The training split alone is then expanded by the Signal Booster to **14,189** samples
(4.94x), which is the pool the reported 53,220 optimisation steps were run over
(30 epochs at batch size 8 => ceil(14,189/8) x 30 = 53,220).

* Deduplication is a **case-sensitive** exact match. Forty case-variant repetitions
  survive within their own classes; none crosses a class boundary.
* No expression appears in more than one class file — checked case-insensitively, zero
  cross-class collisions. The nine label sets are mutually exclusive.
