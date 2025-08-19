"""
QuantoniumOS - System Benchmark

This module provides comprehensive system benchmarking capabilities to evaluate
the performance of QuantoniumOS across multiple dimensions.
"""

import os
import sys
import time
import logging
import hashlib
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import QuantoniumOS modules if available
try:
    from core.encryption.geometric_waveform_hash import geometric_hash
    from core.encryption.wave_entropy_engine import get_quantum_random_bytes
except ImportError:
    # Define fallback functions for demonstration
    def geometric_hash(data):
        return hashlib.sha256(data).digest()

    def get_quantum_random_bytes(n):
        return os.urandom(n)

logger = logging.getLogger(__name__)

# CPU Benchmark
def cpu_benchmark(n):
    """
    Benchmark CPU performance by performing intensive computations.

    Args:
        n: Number of iterations

    Returns:
        Time taken in seconds
    """
    start = time.time()
    for i in range(n):
        [x**2 for x in range(1000)]
    return time.time() - start

# Memory Benchmark
def memory_benchmark(n):
    """
    Benchmark memory allocation and access performance.

    Args:
        n: Number of arrays to allocate

    Returns:
        Time taken in seconds
    """
    start = time.time()
    large_array = [np.random.random((1000,1000)) for _ in range(n)]
    duration = time.time() - start
    del large_array
    return duration

# Encryption Benchmark
def encryption_benchmark(n, algorithm="standard"):
    """
    Benchmark encryption performance.

    Args:
        n: Number of iterations
        algorithm: Algorithm to use ("standard" or "quantum")

    Returns:
        Time taken in seconds
    """
    data = b"benchmark" * 1000

    if algorithm == "quantum":
        start = time.time()
        for _ in range(n):
            geometric_hash(data)
        return time.time() - start
    else:
        start = time.time()
        for _ in range(n):
            hashlib.sha256(data).digest()
        return time.time() - start

def entropy_generation_benchmark(n, size=1024):
    """
    Benchmark entropy generation performance.

    Args:
        n: Number of iterations
        size: Size of entropy to generate in bytes

    Returns:
        Time taken in seconds and entropy quality assessment
    """
    start = time.time()
    for _ in range(n):
        entropy = get_quantum_random_bytes(size)
    duration = time.time() - start

    # Assess entropy quality using simple histogram analysis
    histogram = np.zeros(256, dtype=int)
    for b in entropy:
        histogram[b] += 1

    # Calculate chi-squared statistic
    expected = len(entropy) / 256
    chi_squared = sum((obs - expected) ** 2 / expected for obs in histogram)

    return duration, chi_squared

def run_system_benchmark(iterations=[10, 50, 100, 200, 500]):
    """
    Run a comprehensive system benchmark and return the results.

    Args:
        iterations: List of iteration counts to test

    Returns:
        Dictionary containing benchmark results
    """
    logger.info("Starting system benchmark")

    results = {
        "iterations": iterations,
        "cpu": [],
        "memory": [],
        "encryption_standard": [],
        "encryption_quantum": [],
        "entropy_generation": []
    }

    for i in iterations:
        logger.info(f"Running benchmarks with {i} iterations")

        # CPU benchmark
        cpu_time = cpu_benchmark(i)
        results["cpu"].append(cpu_time)
        logger.info(f"CPU benchmark: {cpu_time:.4f}s")

        # Memory benchmark (scale down for larger iterations)
        mem_time = memory_benchmark(max(1, i//10))
        results["memory"].append(mem_time)
        logger.info(f"Memory benchmark: {mem_time:.4f}s")

        # Standard encryption benchmark
        enc_std_time = encryption_benchmark(i*10, algorithm="standard")
        results["encryption_standard"].append(enc_std_time)
        logger.info(f"Standard encryption benchmark: {enc_std_time:.4f}s")

        # Quantum encryption benchmark
        enc_q_time = encryption_benchmark(i*10, algorithm="quantum")
        results["encryption_quantum"].append(enc_q_time)
        logger.info(f"Quantum encryption benchmark: {enc_q_time:.4f}s")

        # Entropy generation benchmark
        ent_time, ent_quality = entropy_generation_benchmark(i)
        results["entropy_generation"].append({"time": ent_time, "quality": ent_quality})
        logger.info(f"Entropy generation benchmark: {ent_time:.4f}s, quality: {ent_quality:.2f}")

    return results

def plot_benchmark_results(results, save_path=None):
    """
    Create a 3D visualization of benchmark results.

    Args:
        results: Benchmark results from run_system_benchmark
        save_path: Path to save the plot image

    Returns:
        Matplotlib figure object
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    iterations = results["iterations"]

    # Plot CPU benchmark
    ax.plot(iterations, [1]*len(iterations), results["cpu"], label='CPU Benchmark', marker='o')

    # Plot memory benchmark
    ax.plot(iterations, [2]*len(iterations), results["memory"], label='Memory Benchmark', marker='s')

    # Plot standard encryption benchmark
    ax.plot(iterations, [3]*len(iterations), results["encryption_standard"],
            label='Standard Encryption', marker='^')

    # Plot quantum encryption benchmark
    ax.plot(iterations, [4]*len(iterations), results["encryption_quantum"],
            label='Quantum Encryption', marker='*')

    # Plot entropy generation benchmark (time only)
    entropy_times = [entry["time"] for entry in results["entropy_generation"]]
    ax.plot(iterations, [5]*len(iterations), entropy_times,
            label='Entropy Generation', marker='D')

    # Annotations
    for i, iter_count in enumerate(iterations):
        ax.text(iter_count, 1, results["cpu"][i], f'{results["cpu"][i]:.2f}s',
                fontsize=8, ha='center', va='bottom')
        ax.text(iter_count, 2, results["memory"][i], f'{results["memory"][i]:.2f}s',
                fontsize=8, ha='center', va='bottom')
        ax.text(iter_count, 3, results["encryption_standard"][i], f'{results["encryption_standard"][i]:.2f}s',
                fontsize=8, ha='center', va='bottom')
        ax.text(iter_count, 4, results["encryption_quantum"][i], f'{results["encryption_quantum"][i]:.2f}s',
                fontsize=8, ha='center', va='bottom')
        ax.text(iter_count, 5, entropy_times[i], f'{entropy_times[i]:.2f}s',
                fontsize=8, ha='center', va='bottom')

    ax.set_xlabel('Iterations')
    ax.set_ylabel('Benchmark Type')
    ax.set_zlabel('Time (seconds)')
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(['CPU', 'Memory', 'Std Encryption', 'Quantum Encryption', 'Entropy'])

    plt.title('3D Visualization of QuantoniumOS Benchmark Performance')
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        logger.info(f"Benchmark plot saved to {save_path}")

    return fig

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    iterations = [10, 50, 100, 200, 500]
    results = run_system_benchmark(iterations)
    plot_benchmark_results(results)
    plt.show()
