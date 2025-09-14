"""
Equivalence test: check whether the project's RFT is just a re-phased/permuted DFT.
Produces JSON output with match rates and residuals.

Usage:
  python tests/analysis/transform_equivalence_test.py --n 64 --tol 1e-6 --out results/equivalence_result_N64.json

The script will try to import an RFT implementation from known locations,
falling back to raising an informative error if none found.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
import numpy as np
from time import perf_counter


def find_rft():
    """Try to import an rft(x: ndarray) -> ndarray function from common paths in this repo."""
    candidates = [
        ("src.core.canonical_true_rft", "rft"),
        ("src.assembly.python_bindings.unitary_rft", "rft"),
        ("src.assembly.python_bindings.vertex_quantum_rft", "rft"),
        ("src.assembly.python_bindings.optimized_rft", "rft"),
    ]
    for modname, symbol in candidates:
        try:
            mod = __import__(modname, fromlist=[symbol])
            fn = getattr(mod, symbol)
            return fn, f"imported {symbol} from {modname}"
        except Exception:
            continue
    # last-ditch: try top-level name 'rft'
    try:
        from rft import rft as rftfn
        return rftfn, "imported rft from top-level rft"
    except Exception:
        pass
    return None, None


def unitary_dft_matrix(N: int) -> np.ndarray:
    """Return the unitary DFT matrix (1/sqrt(N) normalization)."""
    I = np.eye(N, dtype=complex)
    F = np.fft.fft(I, axis=0) / math.sqrt(N)
    return F


def build_transform_matrix(rft_fn, N: int) -> np.ndarray:
    """Build matrix T where column k = rft(e_k)."""
    T = np.zeros((N, N), dtype=complex)
    for k in range(N):
        e = np.zeros(N, dtype=complex)
        e[k] = 1.0
        y = rft_fn(e)
        y = np.asarray(y, dtype=complex).reshape(N,)
        T[:, k] = y
    return T


def column_match_analysis(T: np.ndarray, F: np.ndarray, tol: float = 1e-9):
    N = T.shape[1]
    matches = [-1] * N
    scalars = [0j] * N
    scores = [0.0] * N
    residuals = [float('inf')] * N
    used = set()
    for col_i in range(N):
        tcol = T[:, col_i]
        best_j = -1
        best_score = -1.0
        best_scalar = 0+0j
        best_res = float('inf')
        for j in range(N):
            if j in used:
                # allow reuse but prefer unique mapping; still test reuse
                pass
            fcol = F[:, j]
            # compute scalar s that minimizes ||tcol - s*fcol|| in least squares
            denom = np.vdot(fcol, fcol)
            if abs(denom) < 1e-18:
                continue
            s = np.vdot(fcol, tcol) / denom
            res = np.linalg.norm(tcol - s * fcol)
            # normalized match score via cosine of vectors
            score = abs(np.vdot(tcol, fcol)) / (np.linalg.norm(tcol) * np.linalg.norm(fcol) + 1e-30)
            if score > best_score:
                best_score = score
                best_j = j
                best_scalar = s
                best_res = res
        matches[col_i] = best_j
        scalars[col_i] = best_scalar
        scores[col_i] = float(best_score)
        residuals[col_i] = float(best_res)
        used.add(best_j)
    return {
        'matches': matches,
        'scalars': [complex(s) for s in scalars],
        'scores': scores,
        'residuals': residuals,
        'match_rate': sum(1 for sc in scores if sc > 1 - 1e-9) / float(len(scores))
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=32)
    parser.add_argument('--tol', type=float, default=1e-6)
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()

    rft_fn, note = find_rft()
    if rft_fn is None:
        print("Could not find an RFT implementation in expected locations.")
        print("Try running from repo root and ensure the package imports work, or pass an RFT callable.")
        sys.exit(3)
    print(f"Using RFT implementation: {note}")

    N = args.n
    print(f"Building transform matrix T for N={N} (this will call RFT N times)")
    t0 = perf_counter()
    T = build_transform_matrix(rft_fn, N)
    t1 = perf_counter()
    F = unitary_dft_matrix(N)

    print("Running column match analysis (match columns of T to columns of F up to scalar)")
    analysis = column_match_analysis(T, F, tol=args.tol)

    # aggregate
    median_res = float(np.median(analysis['residuals']))
    mean_score = float(np.mean(analysis['scores']))
    out = {
        'N': N,
        'rft_import_note': note,
        'build_time_s': t1 - t0,
        'median_residual': median_res,
        'mean_score': mean_score,
        'match_rate': analysis['match_rate'],
        'per_column': {
            'matches': analysis['matches'],
            'scores': analysis['scores'],
            'residuals': analysis['residuals'],
            'scalars': [str(s) for s in analysis['scalars']]
        }
    }

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"Wrote results to {args.out}")
    else:
        print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
