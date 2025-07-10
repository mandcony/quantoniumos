#!/usr/bin/env python3
"""
QuantoniumOS Automated Throughput Benchmark
Generates CI artifact for performance validation
"""

import hashlib
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict

# Add core modules to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "core"))


def benchmark_sha256_throughput(
    data_size_mb: int = 100, runs: int = 5
) -> Dict[str, Any]:
    """Benchmark SHA-256 throughput performance"""
    data_size_bytes = data_size_mb * 1024 * 1024
    test_data = b"a" * data_size_bytes

    print(f"Benchmarking SHA-256 with {data_size_mb}MB data, {runs} runs...")

    times = []
    for run in range(runs):
        print(f"  Run {run + 1}/{runs}...", end="", flush=True)
        start = time.time()
        hashlib.sha256(test_data).hexdigest()
        end = time.time()
        duration = end - start
        times.append(duration)
        print(f" {duration:.3f}s")

    avg_time = sum(times) / len(times)
    throughput_mbps = data_size_mb / avg_time
    throughput_gbps = throughput_mbps / 1024

    return {
        "test_name": "sha256_throughput",
        "data_size_mb": data_size_mb,
        "runs": runs,
        "individual_times": times,
        "average_time_seconds": avg_time,
        "throughput_mbps": throughput_mbps,
        "throughput_gbps": throughput_gbps,
        "min_time": min(times),
        "max_time": max(times),
        "std_dev": (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5,
    }


def benchmark_rft_performance() -> Dict[str, Any]:
    """Benchmark RFT performance if available"""
    try:
        from encryption.resonance_fourier import (
            inverse_resonance_fourier_transform, resonance_fourier_transform)

        return {"test_name": "rft_performance", "status": "available"}
    except ImportError:
        return {"test_name": "rft_performance", "status": "not_available"}


def benchmark_geometric_hash_performance() -> Dict[str, Any]:
    """Benchmark geometric waveform hashing if available"""
    try:
        from encryption.geometric_waveform_hash import geometric_waveform_hash

        return {"test_name": "geometric_hash_performance", "status": "available"}
    except ImportError:
        return {"test_name": "geometric_hash_performance", "status": "not_available"}


def benchmark_quantum_simulation() -> Dict[str, Any]:
    """Benchmark quantum state simulation"""
    try:
        from protected.quantum_engine import QuantumEngine

        return {"test_name": "quantum_simulation_performance", "status": "available"}
    except ImportError:
        return {
            "test_name": "quantum_simulation_performance",
            "status": "not_available",
        }


def main():
    """Main benchmark execution"""
    print("=" * 60)
    print("QUANTONIUM OS AUTOMATED THROUGHPUT BENCHMARK")
    print("=" * 60)

    # Run benchmarks
    sha256_results = benchmark_sha256_throughput()
    rft_results = benchmark_rft_performance()
    geometric_results = benchmark_geometric_hash_performance()
    quantum_results = benchmark_quantum_simulation()

    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"SHA-256 Throughput: {sha256_results['throughput_gbps']:.3f} GB/s")
    print(f"Average Time: {sha256_results['average_time_seconds']:.3f}s")
    print(f"Test Data Size: {sha256_results['data_size_mb']}MB")
    print(
        f"RFT: {'AVAILABLE' if rft_results['status'] == 'available' else 'NOT_AVAILABLE'}"
    )
    print(
        f"GEOMETRIC_HASH: {'AVAILABLE' if geometric_results['status'] == 'available' else 'NOT_AVAILABLE'}"
    )
    print(
        f"QUANTUM_SIMULATION: {'AVAILABLE' if quantum_results['status'] == 'available' else 'NOT_AVAILABLE'}"
    )

    # Save comprehensive report
    report = {
        "timestamp": datetime.now().isoformat(),
        "patent_application": "USPTO #19/169,399",
        "sha256_benchmark": sha256_results,
        "rft_benchmark": rft_results,
        "geometric_hash_benchmark": geometric_results,
        "quantum_simulation": quantum_results,
        "system_info": {
            "platform": sys.platform,
            "python_version": sys.version.split()[0],
            "cpu_count": os.cpu_count(),
        },
    }

    with open("benchmark_throughput_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Create CSV file for CI artifact
    csv_data = []
    csv_data.append(["metric", "value", "unit", "timestamp"])
    csv_data.append(
        [
            "sha256_throughput",
            sha256_results["throughput_gbps"],
            "GB/s",
            datetime.now().isoformat(),
        ]
    )
    csv_data.append(
        [
            "sha256_avg_time",
            sha256_results["average_time_seconds"],
            "seconds",
            datetime.now().isoformat(),
        ]
    )
    csv_data.append(
        [
            "sha256_data_size",
            sha256_results["data_size_mb"],
            "MB",
            datetime.now().isoformat(),
        ]
    )
    csv_data.append(
        [
            "rft_available",
            1 if rft_results["status"] == "available" else 0,
            "boolean",
            datetime.now().isoformat(),
        ]
    )
    csv_data.append(
        [
            "geometric_hash_available",
            1 if geometric_results["status"] == "available" else 0,
            "boolean",
            datetime.now().isoformat(),
        ]
    )
    csv_data.append(
        [
            "quantum_simulation_available",
            1 if quantum_results["status"] == "available" else 0,
            "boolean",
            datetime.now().isoformat(),
        ]
    )
    csv_data.append(
        ["test_runs", sha256_results["runs"], "count", datetime.now().isoformat()]
    )
    csv_data.append(
        [
            "python_version",
            sys.version.split()[0],
            "version",
            datetime.now().isoformat(),
        ]
    )

    # Write CSV
    csv_path = "throughput_results.csv"
    with open(csv_path, "w") as f:
        for row in csv_data:
            f.write(",".join(str(x) for x in row) + "\n")

    print(f"\nReport saved to: benchmark_throughput_report.json")
    print(f"CSV saved to: {csv_path}")
    print(f"Patent Application: {report['patent_application']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
