"""
Process collected per-N complexity JSON files, fit log-log slope alpha with R^2,
compute bootstrap CI for alpha, and write results/complexity_sweep_full.json.
Also update docs/PROOF_NOTE_v1.md appendix with small-N equivalence/shift/convolution summaries.
"""
import json
import math
import os
import glob
import numpy as np
from statistics import median
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
DOCS = ROOT / 'docs'

# Gather per-N files named complexity_N{N}.json
files = sorted(RESULTS.glob('complexity_N*.json'))
entries = []
for f in files:
    try:
        j = json.load(open(f))
        if 'results' in j and len(j['results'])>0:
            entries.append(j['results'][0])
    except Exception as e:
        print('skip', f, e)

# Require at least 3 points
if len(entries) < 3:
    print('Not enough entries to fit slope; need >=3; found', len(entries))
    # still write an empty summary
    out = {'error': 'not enough data', 'count': len(entries)}
    json.dump(out, open(RESULTS / 'complexity_sweep_full.json','w'), indent=2)
    raise SystemExit(0)

# Build arrays
Ns = np.array([e['N'] for e in entries], dtype=float)
rft_medians = np.array([e['rft_median'] for e in entries], dtype=float)
fft_medians = np.array([e['fft_median'] for e in entries], dtype=float)

# Fit log-log: log(t) = a + b log(N)
logN = np.log(Ns)
logt = np.log(rft_medians)
A = np.vstack([logN, np.ones_like(logN)]).T
b, a = np.linalg.lstsq(A, logt, rcond=None)[0]
# b is slope (alpha), a intercept
# compute R^2
pred = a + b*logN
ss_res = np.sum((logt - pred)**2)
ss_tot = np.sum((logt - np.mean(logt))**2)
r2 = 1 - ss_res/ss_tot if ss_tot>0 else 0.0
alpha = float(b)

# bootstrap CI for alpha
rng = np.random.default_rng(12345)
boot_alphas = []
B = 1000
for _ in range(B):
    # sample with replacement per-N from original trial lists
    sampled_medians = []
    for e in entries:
        # sample times array if available
        times = e.get('rft_times')
        if times and len(times)>0:
            # bootstrap median via sampling times with replacement
            sampled = [rng.choice(times) for _ in range(len(times))]
            sampled_medians.append(median(sampled))
        else:
            sampled_medians.append(e.get('rft_median'))
    sampled_medians = np.array(sampled_medians)
    logt_s = np.log(sampled_medians)
    try:
        b_s, a_s = np.linalg.lstsq(np.vstack([logN, np.ones_like(logN)]).T, logt_s, rcond=None)[0]
        boot_alphas.append(b_s)
    except Exception:
        continue

if len(boot_alphas)>0:
    alphas = np.array(boot_alphas)
    alpha_ci = [float(np.percentile(alphas, 2.5)), float(np.percentile(alphas, 97.5))]
else:
    alpha_ci = [alpha, alpha]

# Normed efficiency t/(N log2 N)
normed = [(float(e['N']), float(e['rft_median']), float(e.get('rft_norm')) if e.get('rft_norm') is not None else None) for e in entries]

out = {
    'created_at': json.loads(json.dumps({'now':None})) ,
    'alpha': alpha,
    'r2': float(r2),
    'alpha_ci_95': alpha_ci,
    'normed': [{'N': int(n),'rft_median': t, 'rft_norm': norm} for (n,t,norm) in normed]
}

json.dump(out, open(RESULTS / 'complexity_sweep_full.json','w'), indent=2)
print('Wrote', RESULTS / 'complexity_sweep_full.json')

# Update PROOF_NOTE_v1.md appendix with small-N tables
proof_md = DOCS / 'PROOF_NOTE_v1.md'
if proof_md.exists():
    md = proof_md.read_text()
else:
    md = '# Proof Note v1\n\n'

appendix = '\n\n## Numerical appendix (small-N)\n\n'
appendix += '| N | rft_median (s) | rft_norm t/(N log2N) | fft_median (s) | fft_norm |\n'
appendix += '|---:|---:|---:|---:|---:|\n'
for e in entries:
    appendix += f"| {e['N']} | {e['rft_median']:.6e} | {e.get('rft_norm'):.6e} | {e['fft_median']:.6e} | {e.get('fft_norm'):.6e} |\n"

md = md + appendix
proof_md.write_text(md)
print('Updated', proof_md)
