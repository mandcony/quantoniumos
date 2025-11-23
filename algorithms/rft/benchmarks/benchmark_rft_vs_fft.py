#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""Compare RFT vs NumPy FFT accuracy and throughput on synthetic data."""
import time
import numpy as np

from algorithms.rft.core.canonical_true_rft import CanonicalTrueRFT


def run_once(n: int = 4096, trials: int = 3) -> dict:
    rng = np.random.default_rng(12345)
    x = rng.standard_normal(n) + 1j * rng.standard_normal(n)

    # FFT baseline
    t0 = time.perf_counter()
    for _ in range(trials):
        y_fft = np.fft.fft(x)
    fft_elapsed = time.perf_counter() - t0

    # RFT
    rft = CanonicalTrueRFT(n)
    t1 = time.perf_counter()
    for _ in range(trials):
        y_rft = rft.forward_transform(x)
    rft_elapsed = time.perf_counter() - t1

    # Accuracy (up to scaling differences)
    # Compare normalized magnitude spectra
    a = np.abs(y_fft) / np.linalg.norm(y_fft)
    b = np.abs(y_rft) / np.linalg.norm(y_rft)
    max_abs_diff = float(np.max(np.abs(a - b)))

    return {
        "n": n,
        "trials": trials,
        "fft_s": fft_elapsed,
        "rft_s": rft_elapsed,
        "speed_ratio_rft_over_fft": rft_elapsed / fft_elapsed if fft_elapsed > 0 else float("inf"),
        "max_abs_diff": max_abs_diff,
    }


if __name__ == "__main__":
    result = run_once()
    print(result)
