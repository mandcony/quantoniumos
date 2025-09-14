"""
Complexity sweep: measure runtime of RFT vs FFT across N in powers of two (64..4096).
Outputs JSON with raw times, median across trials, and normalized time t/(n*log2(n)).

Usage:
  python tests/analysis/transform_complexity_sweep.py --out results/complexity_sweep.json --min 64 --max 4096

Requires numpy; matplotlib optional for quick plot.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
import time
import numpy as np
from statistics import median


def find_rft():
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
    try:
        from rft import rft as rftfn
        return rftfn, "imported rft from top-level rft"
    except Exception:
        pass
    return None, None


def time_fn(fn, x, trials=5):
    times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        y = fn(x)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return times


def fft_fn(x):
    return np.fft.fft(x)


def run_sweep(rft_fn, Ns, trials=5):
    results = []
    for N in Ns:
        x = np.random.randn(N) + 1j*np.random.randn(N)
        # ensure warmup
        dummy = fft_fn(x)
        if rft_fn is not None:
            try:
                dummy2 = rft_fn(x)
            except Exception as e:
                print(f"RFT failed for N={N} with error: {e}")
                rft_times = None
            else:
                rft_times = time_fn(rft_fn, x, trials=trials)
        else:
            rft_times = None
        fft_times = time_fn(fft_fn, x, trials=trials)

        entry = {
            'N': N,
            'fft_times': fft_times,
            'rft_times': rft_times,
            'fft_median': median(fft_times),
            'rft_median': median(rft_times) if rft_times is not None else None,
            'fft_norm': median(fft_times) / (N * math.log2(max(2, N))),
            'rft_norm': (median(rft_times) / (N * math.log2(max(2, N)))) if rft_times is not None else None,
        }
        results.append(entry)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--min', type=int, default=64)
    parser.add_argument('--max', type=int, default=4096)
    parser.add_argument('--trials', type=int, default=7)
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()

    Ns = []
    n = args.min
    while n <= args.max:
        Ns.append(n)
        n *= 2

    rft_fn, note = find_rft()
    if rft_fn is None:
        print("Warning: could not find RFT implementation; sweep will run only FFT.")
    else:
        print(f"Using RFT implementation: {note}")

    print(f"Running sweep for Ns: {Ns} with {args.trials} trials each")
    results = run_sweep(rft_fn, Ns, trials=args.trials)

    out = {
        'created_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'note': note if rft_fn is not None else 'RFT not found',
        'results': results
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
