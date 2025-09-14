"""
Shift diagonalization test: checks whether U* S U is diagonal for circular shift S.
Outputs JSON with diagonal energy fraction and off-diagonal Frobenius norm.

Usage:
  python tests/analysis/shift_diagonalization_test.py --n 32 --out results/shift_diag_N32.json
"""
import argparse
import json
import math
import os
import numpy as np
from time import perf_counter


def find_rft_module():
    try:
        from src.assembly.python_bindings.unitary_rft import rft
        return rft, 'src.assembly.python_bindings.unitary_rft'
    except Exception:
        try:
            from src.core.canonical_true_rft import rft
            return rft, 'src.core.canonical_true_rft'
        except Exception:
            return None, None


def build_U_matrix(rft_fn, N):
    U = np.zeros((N,N), dtype=complex)
    for k in range(N):
        e = np.zeros(N, dtype=complex)
        e[k] = 1.0
        U[:,k] = rft_fn(e)
    return U


def circular_shift_matrix(N):
    S = np.zeros((N,N), dtype=complex)
    for i in range(N):
        S[(i+1)%N, i] = 1.0
    return S


def analyze(U, S):
    # compute U* S U
    Udag = np.conjugate(U.T)
    M = Udag @ S @ U
    off_diag = M - np.diag(np.diag(M))
    frob_off = np.linalg.norm(off_diag)
    frob_total = np.linalg.norm(M)
    diag_energy = np.linalg.norm(np.diag(M))
    return {
        'frob_off': float(frob_off),
        'frob_total': float(frob_total),
        'diag_energy_fraction': float((diag_energy**2) / (frob_total**2) if frob_total>0 else 0.0),
        'frob_off_fraction': float(frob_off / frob_total if frob_total>0 else 0.0)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=32)
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()

    rft_fn, note = find_rft_module()
    if rft_fn is None:
        print('RFT implementation not found; aborting')
        return
    print('Using RFT from', note)

    U = build_U_matrix(rft_fn, args.n)
    S = circular_shift_matrix(args.n)
    stats = analyze(U, S)
    out = {'N': args.n, 'note': note, 'stats': stats}

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(out, f, indent=2)
        print('Wrote', args.out)
    else:
        print(json.dumps(out, indent=2))

if __name__ == '__main__':
    main()
