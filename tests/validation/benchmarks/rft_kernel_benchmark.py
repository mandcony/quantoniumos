#!/usr/bin/env python3
"""Benchmark the UnitaryRFT native kernel across multiple sizes."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from unitary_rft import UnitaryRFT, RFT_FLAG_QUANTUM_SAFE

RESULT_PATH = Path(__file__).resolve().parents[1] / "results" / "rft_kernel_benchmark.json"


def benchmark_size(size: int, runs: int = 10) -> dict[str, float]:
    engine = UnitaryRFT(size, RFT_FLAG_QUANTUM_SAFE)
    signal = np.random.default_rng(2025).standard_normal(size).astype(np.float64)

    times_forward: list[float] = []
    times_inverse: list[float] = []

    for _ in range(runs):
        start = time.perf_counter()
        coeffs = engine.forward(signal)
        times_forward.append(time.perf_counter() - start)

        start = time.perf_counter()
        engine.inverse(coeffs)
        times_inverse.append(time.perf_counter() - start)

    return {
        "size": size,
        "runs": runs,
        "forward_avg_ms": 1000 * float(np.mean(times_forward)),
        "forward_std_ms": 1000 * float(np.std(times_forward, ddof=1)),
        "inverse_avg_ms": 1000 * float(np.mean(times_inverse)),
        "inverse_std_ms": 1000 * float(np.std(times_inverse, ddof=1)),
    }


def main() -> None:
    sizes = [256, 512, 1024, 2048, 4096]
    results = {
        "timestamp": int(time.time()),
        "benchmarks": [benchmark_size(size) for size in sizes],
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"Benchmark results saved to {RESULT_PATH}")


if __name__ == "__main__":
    main()
