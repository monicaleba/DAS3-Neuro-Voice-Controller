#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Re-derive the dataset figures from the nine seed corpora.

Replays the deduplication and class-balancing performed by
src/02_dataset_build/build_slm_database.py and checks the published numbers:

    3,600 raw expressions   (400 x 9)
       -1 exact duplicate   ("lift the forearm higher", twice inside pull.txt)
    3,599 after dedup
    3,591 balanced          (all classes clamped to the 399 minimum)
    2,872 / 719             (stratified 80/20 split)

Also verifies that the nine label sets are mutually exclusive.

Usage:  python analysis/verify_corpus.py
"""
import io
import os
import csv
import sys
import collections
import math

HERE = os.path.dirname(os.path.abspath(__file__))
CORPORA = os.path.join(HERE, os.pardir, 'dataset', 'seed_corpora')

FILES = collections.OrderedDict([
    ('STOP', 'stop.txt'), ('REST', 'rest.txt'),
    ('SHOULDER_UP', 'move_up.txt'), ('SHOULDER_DOWN', 'move_down.txt'),
    ('SHOULDER_LEFT', 'move_left.txt'), ('SHOULDER_RIGHT', 'move_right.txt'),
    ('ELBOW_UP', 'pull.txt'), ('ELBOW_DOWN', 'push.txt'),
    ('ROTATE_WRIST', 'turn.txt'),
])


def load():
    """Read expressions exactly as the build script does."""
    rows = []
    for token, fname in FILES.items():
        path = os.path.join(CORPORA, fname)
        if not os.path.exists(path):
            sys.exit('missing corpus file: %s' % path)
        with io.open(path, encoding='utf-8') as fh:
            for row in csv.reader(fh, skipinitialspace=True):
                for expr in row:
                    clean = expr.strip(' \n\r\t"\'')
                    if clean:
                        rows.append((clean, token, fname))
    return rows


def main():
    raw = load()
    per_class = collections.Counter(t for _, t, _ in raw)

    print('raw corpus')
    for token in FILES:
        print('  %-15s %4d' % (token, per_class[token]))
    print('  %-15s %4d' % ('TOTAL', len(raw)))

    # exact, case-sensitive dedup — matches pandas drop_duplicates(subset=['input'])
    counts = collections.Counter(e for e, _, _ in raw)
    dupes = {e: n for e, n in counts.items() if n > 1}
    print('\nexact duplicates: %d' % len(dupes))
    for expr, n in dupes.items():
        where = sorted({f for e, _, f in raw if e == expr})
        print('  %r x%d  in %s' % (expr, n, ', '.join(where)))

    seen, dedup = set(), []
    for expr, token, _ in raw:
        if expr in seen:
            continue
        seen.add(expr)
        dedup.append((expr, token))
    after = collections.Counter(t for _, t in dedup)
    minimum = min(after.values())
    balanced = minimum * len(FILES)
    # sklearn's train_test_split(test_size=0.2) takes ceil() of the test fraction
    val = int(math.ceil(balanced * 0.2))
    train = balanced - val

    print('\nafter dedup      %5d' % len(dedup))
    print('minimum class    %5d  (%s)'
          % (minimum, ', '.join(t for t, c in after.items() if c == minimum)))
    print('balanced         %5d  (%d x %d)' % (balanced, minimum, len(FILES)))
    print('80/20 split      %5d train / %d validation' % (train, val))

    # mutual exclusivity of the nine label sets
    by_expr = collections.defaultdict(set)
    for expr, token, _ in raw:
        by_expr[expr.lower()].add(token)
    cross = {e: t for e, t in by_expr.items() if len(t) > 1}
    print('\ncross-class collisions (case-insensitive): %d' % len(cross))
    for expr, tokens in list(cross.items())[:10]:
        print('  %r -> %s' % (expr, sorted(tokens)))

    case_variants = sum(
        n - 1 for n, t in
        ((collections.Counter(e.lower() for e, _, _ in raw)[k], v) for k, v in by_expr.items())
        if n > 1 and len(t) == 1)
    print('case-variant duplicates surviving exact dedup: %d' % case_variants)

    expected = {'raw': 3600, 'dedup': 3599, 'balanced': 3591, 'train': 2872, 'val': 719}
    got = {'raw': len(raw), 'dedup': len(dedup), 'balanced': balanced,
           'train': train, 'val': val}
    print('\nagainst published figures:')
    ok = True
    for key in ['raw', 'dedup', 'balanced', 'train', 'val']:
        good = got[key] == expected[key]
        ok &= good
        print('  %-9s got %5d  expected %5d  %s'
              % (key, got[key], expected[key], 'OK' if good else 'MISMATCH'))
    ok &= not cross
    print('  %-9s %s' % ('disjoint', 'OK' if not cross else 'MISMATCH'))
    print('\nALL FIGURES REPRODUCE' if ok else '\nDISCREPANCY FOUND')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
