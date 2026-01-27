#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Scan data/weights for numeric fields named 'parameter_count' or 'total_parameters'
and print their occurrences and a small summary.
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / 'data' / 'weights'
print(f"Scanning weights directory: {ROOT}")
keys = set(['parameter_count', 'total_parameters', 'total_parameters'])

results = []

for p in ROOT.rglob('*.json'):
    try:
        with open(p, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        # skip non-json or malformed
        continue

    def walk(obj, path=''):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in keys and isinstance(v, (int, float)):
                    results.append((str(p.relative_to(ROOT)), path + '/' + k, int(v)))
                walk(v, path + '/' + k)
        elif isinstance(obj, list):
            for i, item in enumerate(obj[:200]):  # limit huge lists
                walk(item, path + f'[{i}]')

    walk(data)

# Print findings grouped by file
from collections import defaultdict
by_file = defaultdict(list)
for fname, keypath, val in results:
    by_file[fname].append((keypath, val))

total = 0
if not ROOT.exists():
    print(f"\n⚠️ Weights directory does not exist: {ROOT}")
    print('\nDone.')
    raise SystemExit(1)

print('\nWeight parameter findings:')
for fname, items in by_file.items():
    print(f'\nFile: {fname}')
    for keypath, val in items:
        print(f'  {keypath}: {val:,}')
        total += val

print('\nSummary:')
print(f'  Files scanned: {len(by_file)}')
print(f'  Total reported parameters (sum of found fields): {total:,}')

# Also print a deduplicated list of top-level metadata totals
meta_totals = []
for p in ROOT.rglob('*.json'):
    try:
        with open(p, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except:
        continue
    md = data.get('metadata', {})
    if isinstance(md, dict) and 'total_parameters' in md:
        meta_totals.append((str(p.relative_to(ROOT)), int(md['total_parameters'])))

if meta_totals:
    print('\nTop-level metadata totals:')
    for fname, val in meta_totals:
        print(f'  {fname}: {val:,}')

print('\nDone.')
