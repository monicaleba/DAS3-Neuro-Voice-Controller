#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Re-derive the input-attempt accounting of the fourteen-participant live study.

Reads every log in experiments/live_study_N14/logs_anonymised/ and reproduces the
three-way outcome decomposition reported in the paper:

    input attempts        2,705  =  2,632 [MIC RAW] + 73 [TEXT IN]
    classifier decisions  1,968  =  1,882 [SLM DECISION] + 86 [FAST-PATH BYPASS]
    Gatekeeper blocks       646
    safety overrides         91  =  86 OVERRIDE->REST + 5 EMERGENCY OVERRIDE->STOP

Usage:  python analysis/verify_accounting.py
"""
import io
import os
import sys
import glob

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(HERE, os.pardir, 'experiments', 'live_study_N14', 'logs_anonymised')

EXPECTED = {'mic': 2632, 'txt': 73, 'slm': 1882, 'fast': 86,
            'blocked': 647, 'rest': 86, 'stop': 5}


def classify(line):
    """Map a log line to (kind, counter-key), or (None, None)."""
    if '[MIC RAW]' in line:
        return 'input', 'mic'
    if '[TEXT IN]' in line:
        return 'input', 'txt'
    if '[SLM DECISION]' in line:
        return 'outcome', 'slm'
    if '[FAST-PATH BYPASS]' in line:
        return 'outcome', 'fast'
    if '[GATEKEEPER] BLOCKED' in line:
        return 'outcome', 'blocked'
    if '[GATEKEEPER] EMERGENCY OVERRIDE' in line:
        return 'outcome', 'stop'
    if '[GATEKEEPER] OVERRIDE' in line:
        return 'outcome', 'rest'
    return None, None


def main():
    paths = sorted(glob.glob(os.path.join(LOGS, 'U*.txt')))
    if not paths:
        sys.exit('no logs found in %s' % LOGS)

    total = dict.fromkeys(EXPECTED, 0)
    orphans = []       # outcome lines with no preceding input attempt

    print('%-8s %6s %5s | %6s %6s %8s %6s %6s' %
          ('log', 'MIC', 'TEXT', 'SLM', 'FAST', 'BLOCKED', 'REST', 'STOP'))
    for path in paths:
        counts = dict.fromkeys(EXPECTED, 0)
        pending = None
        for lineno, raw in enumerate(io.open(path, encoding='utf-8', errors='replace'), 1):
            kind, key = classify(raw)
            if kind is None:
                continue
            counts[key] += 1
            if kind == 'input':
                pending = lineno
            elif pending is None:
                orphans.append((os.path.basename(path), lineno, raw.strip()))
            else:
                pending = None      # this input is now resolved
        for k in total:
            total[k] += counts[k]
        print('%-8s %6d %5d | %6d %6d %8d %6d %6d' %
              (os.path.basename(path)[:-4], counts['mic'], counts['txt'], counts['slm'],
               counts['fast'], counts['blocked'], counts['rest'], counts['stop']))

    inputs = total['mic'] + total['txt']
    classifier = total['slm'] + total['fast']
    overrides = total['rest'] + total['stop']
    blocks_with_transcript = inputs - classifier - overrides

    print('-' * 60)
    print('input attempts        %5d  =  %d [MIC RAW] + %d [TEXT IN]'
          % (inputs, total['mic'], total['txt']))
    print('classifier decisions  %5d  =  %d [SLM DECISION] + %d [FAST-PATH BYPASS]'
          % (classifier, total['slm'], total['fast']))
    print('Gatekeeper blocks     %5d  (of %d BLOCKED lines in the raw logs)'
          % (blocks_with_transcript, total['blocked']))
    print('safety overrides      %5d  =  %d OVERRIDE->REST + %d EMERGENCY->STOP'
          % (overrides, total['rest'], total['stop']))
    print('-' * 60)
    print('%d + %d + %d = %d   (input attempts = %d)  %s'
          % (classifier, blocks_with_transcript, overrides,
             classifier + blocks_with_transcript + overrides, inputs,
             'OK' if classifier + blocks_with_transcript + overrides == inputs else 'MISMATCH'))

    extra = total['blocked'] - blocks_with_transcript
    if extra:
        print('\nnote: %d BLOCKED line(s) follow a listening cycle that produced no '
              'transcript,\n      so they carry no input attempt and are excluded from '
              'transcript-based analysis.' % extra)

    print('\nagainst published figures:')
    ok = True
    for key, exp in sorted(EXPECTED.items()):
        good = total[key] == exp
        ok &= good
        print('  %-8s got %5d  expected %5d  %s'
              % (key, total[key], exp, 'OK' if good else 'MISMATCH'))
    print('\nALL FIGURES REPRODUCE' if ok else '\nDISCREPANCY FOUND')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
