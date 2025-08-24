#!/usr/bin/env python3
"""
QuantoniumOS Memory Profiler

This script provides detailed memory profiling for QuantoniumOS cryptographic operations.
It helps identify memory usage patterns, potential leaks, and optimization opportunities.
"""

import os
import sys
import time
import tracemalloc
from datetime import datetime
from typing import Any, Callable, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import psutil

# Fix import paths - add the project root to Python's module search path current_dir = os.path.dirname(os.path.abspath(__file__)) project_root = os.path.abspath(os.path.join(current_dir, '..', '..')) # Import core components try: from core.encryption.resonance_encrypt import encrypt as resonance_encrypt, decrypt_data from core.encryption.geometric_waveform_hash import GeometricWaveformHash from core.encryption.wave_entropy_engine import WaveformEntropyEngine # Create a wrapper class to maintain compatibility class ResonanceEncryption: """ Wrapper class for resonance encryption functions to provide an object-oriented interface """ def encrypt(self, data, key): """Encrypt data using resonance encryption""" if isinstance(data, bytes): data = data.decode('utf-8', errors='surrogate') if isinstance(key, bytes): key = key.decode('utf-8', errors='surrogate') result = resonance_encrypt(data, key) return result.get('ciphertext', '').encode('utf-8') def decrypt(self, data, key): """Decrypt data using resonance encryption""" if isinstance(data, bytes): data = data.decode('utf-8', errors='surrogate') if isinstance(key, bytes): key = key.decode('utf-8', errors='surrogate') return decrypt_data(data, key).encode('utf-8') except ImportError as e: print(f"Error importing QuantoniumOS modules: {e}") print("Make sure you're running this script from the project root or the correct modules are installed.")
    sys.exit(1)

class MemoryProfiler:
    """Memory profiler for QuantoniumOS cryptographic operations"""

    def __init__(self, output_dir: str = "memory_profiles"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results = {}

    def profile_function(self, func: Callable, func_name: str,
                        input_sizes: List[int],
                        iterations: int = 10,
                        **kwargs) -> Dict[str, Any]:
        """
        Profile memory usage of a function with varying input sizes

        Args:
            func: Function to profile
            func_name: Name of the function (for reporting)
            input_sizes: List of input sizes to test (in bytes)
            iterations: Number of iterations per size
            **kwargs: Additional arguments to pass to the function

        Returns:
            Dictionary with profiling results
        """
        print(f"Profiling {func_name}...")

        # Initialize results
        size_results = {}

        for size in input_sizes:
            print(f" Testing input size: {size} bytes")

            # Generate input data
            if 'key' in kwargs and kwargs['key'] is None:
                # Generate random key if needed
                kwargs['key'] = os.urandom(32)  # Default 32-byte key

            peak_memory_list = []
            allocated_memory_list = []
            execution_times = []

            for i in range(iterations):
                # Generate random input data for this iteration
                input_data = os.urandom(size)

                # Start memory tracking
                tracemalloc.start()
                process = psutil.Process()

                # Measure execution time
                start_time = time.time()

                # Execute function
                func(input_data, **kwargs)

                # Record execution time
                execution_time = time.time() - start_time
                execution_times.append(execution_time)

                # Get memory usage
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                # Convert to KB
                current_kb = current / 1024
                peak_kb = peak / 1024

                peak_memory_list.append(peak_kb)
                allocated_memory_list.append(current_kb)

            # Calculate statistics
            size_results[size] = {
                "mean_peak_memory_kb": np.mean(peak_memory_list),
                "std_peak_memory_kb": np.std(peak_memory_list),
                "min_peak_memory_kb": np.min(peak_memory_list),
                "max_peak_memory_kb": np.max(peak_memory_list),
                "mean_allocated_memory_kb": np.mean(allocated_memory_list),
                "std_allocated_memory_kb": np.std(allocated_memory_list),
                "mean_execution_time_s": np.mean(execution_times),
                "std_execution_time_s": np.std(execution_times),
                "raw_peak_memory_kb": peak_memory_list,
                "raw_allocated_memory_kb": allocated_memory_list,
                "raw_execution_times_s": execution_times
            }

        # Store results
        self.results[func_name] = {
            "sizes": input_sizes,
            "size_results": size_results,
            "parameters": kwargs
        }

        return self.results[func_name]

    def profile_memory_growth(self, func: Callable, func_name: str,
                            input_size: int, iterations: int = 100,
                            **kwargs) -> Dict[str, Any]:
        """
        Profile memory growth over repeated iterations

        Args:
            func: Function to profile
            func_name: Name of the function (for reporting)
            input_size: Input size to test (in bytes)
            iterations: Number of iterations
            **kwargs: Additional arguments to pass to the function

        Returns:
            Dictionary with profiling results
        """
        print(f"Profiling memory growth for {func_name} over {iterations} iterations...")

        # Generate input data
        if 'key' in kwargs and kwargs['key'] is None:
            # Generate random key if needed
            kwargs['key'] = os.urandom(32)  # Default 32-byte key

        # Start memory tracking
        tracemalloc.start()
        process = psutil.Process()

        peak_memory_list = []
        allocated_memory_list = []
        process_memory_list = []
        execution_times = []

        for i in range(iterations):
            # Generate random input data for this iteration
            input_data = os.urandom(input_size)

            # Measure execution time
            start_time = time.time()

            # Execute function
            func(input_data, **kwargs)

            # Record execution time
            execution_time = time.time() - start_time
            execution_times.append(execution_time)

            # Get memory usage
            current, peak = tracemalloc.get_traced_memory()
            process_memory = process.memory_info().rss / 1024  # KB

            # Convert to KB
            current_kb = current / 1024
            peak_kb = peak / 1024

            peak_memory_list.append(peak_kb)
            allocated_memory_list.append(current_kb)
            process_memory_list.append(process_memory)

            if i % 10 == 0:
                print(f" Iteration {i}: Peak Memory: {peak_kb:.2f} KB, RSS: {process_memory:.2f} KB")

        tracemalloc.stop()

        # Store results
        growth_results = {
            "iterations": list(range(iterations)),
            "peak_memory_kb": peak_memory_list,
            "allocated_memory_kb": allocated_memory_list,
            "process_memory_kb": process_memory_list,
            "execution_times_s": execution_times,
            "input_size": input_size,
            "parameters": kwargs
        }

        self.results[f"{func_name}_growth"] = growth_results

        return growth_results

    def plot_memory_usage(self, func_name: str):
        """
        Plot memory usage for a function across different input sizes

        Args:
            func_name: Name of the function to plot
        """
        if func_name not in self.results:
            print(f"No results found for {func_name}")
            return

        results = self.results[func_name]

        if "sizes" not in results:
            print(f"Results for {func_name} do not contain size data")
            return

        sizes = results["sizes"]
        peak_memory = [results["size_results"][size]["mean_peak_memory_kb"] for size in sizes]
        allocated_memory = [results["size_results"][size]["mean_allocated_memory_kb"] for size in sizes]

        # Convert sizes to KB for better readability
        sizes_kb = [s / 1024 for s in sizes]

        plt.figure(figsize=(10, 6))
        plt.plot(sizes_kb, peak_memory, 'o-', label="Peak Memory")
        plt.plot(sizes_kb, allocated_memory, 's-', label="Allocated Memory")
        plt.xlabel("Input Size (KB)")
        plt.ylabel("Memory Usage (KB)")
        plt.title(f"Memory Usage for {func_name}")
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Add best-fit line to see growth pattern
        from scipy import stats

        # Linear fit for peak memory
        slope, intercept, r_value, p_value, std_err = stats.linregress(sizes_kb, peak_memory)
        plt.plot(sizes_kb, [slope * x + intercept for x in sizes_kb], '--',
                label=f"Peak Memory Growth (slope={slope:.2f})")

        # Linear fit for allocated memory
        slope, intercept, r_value, p_value, std_err = stats.linregress(sizes_kb, allocated_memory)
        plt.plot(sizes_kb, [slope * x + intercept for x in sizes_kb], '--',
                label=f"Allocated Memory Growth (slope={slope:.2f})")

        plt.legend()

        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"memory_profile_{func_name}_{timestamp}.png"
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()

        print(f"Memory usage plot saved to {os.path.join(self.output_dir, filename)}")

    def plot_memory_growth(self, func_name: str):
        """
        Plot memory growth for a function over repeated iterations

        Args:
            func_name: Name of the function to plot
        """
        growth_name = f"{func_name}_growth"

        if growth_name not in self.results:
            print(f"No growth results found for {func_name}")
            return

        results = self.results[growth_name]

        iterations = results["iterations"]
        peak_memory = results["peak_memory_kb"]
        allocated_memory = results["allocated_memory_kb"]
        process_memory = results["process_memory_kb"]

        plt.figure(figsize=(12, 8))

        # Plot memory usage
        plt.subplot(2, 1, 1)
        plt.plot(iterations, peak_memory, 'o-', label="Peak Memory", markersize=3)
        plt.plot(iterations, allocated_memory, 's-', label="Allocated Memory", markersize=3)
        plt.plot(iterations, process_memory, '^-', label="Process Memory (RSS)", markersize=3)
        plt.xlabel("Iteration")
        plt.ylabel("Memory Usage (KB)")
        plt.title(f"Memory Growth for {func_name} ({results['input_size']} bytes)")
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Plot memory growth rate
        plt.subplot(2, 1, 2)

        # Calculate growth rates (percentage change from initial value)
        if peak_memory[0] > 0:
            peak_growth = [(m / peak_memory[0]) * 100 - 100 for m in peak_memory]
            plt.plot(iterations, peak_growth, 'o-', label="Peak Memory Growth %", markersize=3)

        if allocated_memory[0] > 0:
            allocated_growth = [(m / allocated_memory[0]) * 100 - 100 for m in allocated_memory]
            plt.plot(iterations, allocated_growth, 's-', label="Allocated Memory Growth %", markersize=3)

        if process_memory[0] > 0:
            process_growth = [(m / process_memory[0]) * 100 - 100 for m in process_memory]
            plt.plot(iterations, process_growth, '^-', label="Process Memory Growth %", markersize=3)

        plt.xlabel("Iteration")
        plt.ylabel("Memory Growth (%)")
        plt.title(f"Memory Growth Rate for {func_name}")
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.tight_layout()

        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"memory_growth_{func_name}_{timestamp}.png"
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()

        print(f"Memory growth plot saved to {os.path.join(self.output_dir, filename)}")

    def generate_report(self):
        """Generate a comprehensive memory profiling report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.output_dir, f"memory_profile_report_{timestamp}.md")

        with open(report_file, 'w') as f:
            f.write("# QuantoniumOS Memory Profiling Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## Summary\n\n")
            f.write("| Function | Input Size Range (KB) | Mean Peak Memory (KB) | Memory Growth Pattern |\n")
            f.write("|----------|----------------------|----------------------|----------------------|\n")

            for func_name, results in self.results.items():
                if "_growth" in func_name:
                    continue

                sizes = results["sizes"]
                min_size = min(sizes) / 1024  # Convert to KB
                max_size = max(sizes) / 1024  # Convert to KB

                # Calculate average peak memory across all sizes
                mean_peak = np.mean([results["size_results"][size]["mean_peak_memory_kb"] for size in sizes])

                # Determine memory growth pattern
                x = np.array([s / 1024 for s in sizes])
                y = np.array([results["size_results"][size]["mean_peak_memory_kb"] for size in sizes])

                from scipy import stats
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

                if slope < 0.1:
                    pattern = "Constant (O(1))"
                elif slope < 1.5:
                    pattern = "Linear (O(n))"
                elif slope < 2.5:
                    pattern = "Quadratic (O(n^2))"
                else:
                    pattern = f"Super-linear (slope={slope:.2f})"

                f.write(f"| {func_name} | {min_size:.1f} - {max_size:.1f} | {mean_peak:.1f} | {pattern} |\n")

            # Detailed results for each function
            for func_name, results in self.results.items():
                if "_growth" not in func_name:
                    f.write(f"\n## {func_name}\n\n")

                    # Memory usage vs input size
                    f.write("### Memory Usage vs Input Size\n\n")
                    f.write("| Input Size (KB) | Peak Memory (KB) | Allocated Memory (KB) | Execution Time (ms) |\n")
                    f.write("|----------------|-----------------|----------------------|--------------------|\n")

                    for size in results["sizes"]:
                        size_kb = size / 1024
                        peak = results["size_results"][size]["mean_peak_memory_kb"]
                        allocated = results["size_results"][size]["mean_allocated_memory_kb"]
                        exec_time = results["size_results"][size]["mean_execution_time_s"] * 1000  # Convert to ms

                        f.write(f"| {size_kb:.1f} | {peak:.1f} | {allocated:.1f} | {exec_time:.1f} |\n")

                    # Memory growth analysis
                    f.write("\n### Memory Growth Analysis\n\n")

                    # Fit a linear model to predict memory usage based on input size
                    x = np.array([s / 1024 for s in results["sizes"]])
                    y = np.array([results["size_results"][size]["mean_peak_memory_kb"] for size in results["sizes"]])

                    from scipy import stats
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

                    f.write(f"Peak memory usage model: {slope:.2f} × input_size_kb + {intercept:.2f} KB\n\n")
                    f.write(f"- R^2 value: {r_value**2:.4f}\n")
                    f.write(f"- Standard error: {std_err:.4f}\n\n")

                    # Memory growth pattern
                    if slope < 0.1:
                        f.write("Memory growth pattern: **Constant (O(1))**\n\n")
                        f.write("This indicates that memory usage is effectively independent of input size,\n")
                        f.write("suggesting efficient memory management and no significant memory leaks.\n")
                    elif slope < 1.5:
                        f.write("Memory growth pattern: **Linear (O(n))**\n\n")
                        f.write("This indicates that memory usage grows linearly with input size,\n")
                        f.write("which is expected for algorithms that need to process each input element.\n")
                    elif slope < 2.5:
                        f.write("Memory growth pattern: **Quadratic (O(n^2))**\n\n")
                        f.write("This indicates that memory usage grows quadratically with input size,\n")
                        f.write("which might indicate inefficient algorithms or data structures for large inputs.\n")
                    else:
                        f.write(f"Memory growth pattern: **Super-linear (slope={slope:.2f})**\n\n")
                        f.write("This indicates that memory usage grows faster than linearly with input size,\n")
                        f.write("which might indicate potential memory efficiency issues for large inputs.\n")

                    # Add reference to plots
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    f.write(f"\n![Memory Usage Plot](memory_profile_{func_name}_{timestamp}.png)\n")

            # Growth over iterations
            for func_name, results in self.results.items():
                if "_growth" in func_name:
                    base_name = func_name.replace("_growth", "")
                    f.write(f"\n## {base_name} (Memory Growth Over Iterations)\n\n")

                    input_size = results["input_size"]

                    f.write(f"Input size: {input_size} bytes\n\n")

                    # Calculate memory growth statistics
                    peak_memory = results["peak_memory_kb"]
                    initial_peak = peak_memory[0]
                    final_peak = peak_memory[-1]
                    peak_growth = ((final_peak / initial_peak) - 1) * 100 if initial_peak > 0 else 0

                    process_memory = results["process_memory_kb"]
                    initial_process = process_memory[0]
                    final_process = process_memory[-1]
                    process_growth = ((final_process / initial_process) - 1) * 100 if initial_process > 0 else 0

                    f.write("### Memory Growth Statistics\n\n")
                    f.write("| Metric | Initial (KB) | Final (KB) | Growth (%) |\n")
                    f.write("|--------|-------------|------------|------------|\n")
                    f.write(f"| Peak Memory | {initial_peak:.1f} | {final_peak:.1f} | {peak_growth:.1f} |\n")
                    f.write(f"| Process Memory | {initial_process:.1f} | {final_process:.1f} | {process_growth:.1f} |\n\n")

                    if peak_growth < 1.0:
                        f.write("**Analysis**: No significant memory growth detected over repeated iterations.\n")
                        f.write("This suggests that the function is not leaking memory and has consistent memory usage.\n\n")
                    elif peak_growth < 5.0:
                        f.write("**Analysis**: Slight memory growth detected over repeated iterations.\n")
                        f.write("This might indicate minor memory accumulation, but is likely within acceptable bounds.\n\n")
                    else:
                        f.write("**Analysis**: Significant memory growth detected over repeated iterations.\n")
                        f.write("This might indicate a memory leak or accumulation that should be investigated.\n\n")

                    # Add reference to plots
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    f.write(f"\n![Memory Growth Plot](memory_growth_{base_name}_{timestamp}.png)\n")

        print(f"Memory profiling report saved to {report_file}")

        return report_file

def profile_quantoniumos():
    """Run memory profiling on QuantoniumOS cryptographic components"""
    print("="*80)
    print("QuantoniumOS Memory Profiling")
    print("="*80)

    # Create profiler
    profiler = MemoryProfiler()

    # Initialize components
    encryption = ResonanceEncryption()
    hash_function = GeometricWaveformHash()
    entropy_engine = WaveformEntropyEngine()

    # Define input sizes to test (exponential range)
    input_sizes = [1024 * 2**i for i in range(8)]  # 1KB to 128KB

    # Profile encryption
    profiler.profile_function(
        encryption.encrypt,
        "ResonanceEncryption",
        input_sizes,
        key=os.urandom(32)
    )
    profiler.plot_memory_usage("ResonanceEncryption")

    # Profile hash function
    profiler.profile_function(
        hash_function.hash,
        "GeometricWaveformHash",
        input_sizes
    )
    profiler.plot_memory_usage("GeometricWaveformHash")

    # Profile entropy generation
    profiler.profile_function(
        entropy_engine.generate_bytes,
        "WaveEntropyEngine",
        [s // 8 for s in input_sizes]  # Smaller sizes for entropy generation
    )
    profiler.plot_memory_usage("WaveEntropyEngine")

    # Test for memory leaks with repeated calls
    profiler.profile_memory_growth(
        encryption.encrypt,
        "ResonanceEncryption",
        8192,  # 8KB input
        iterations=100,
        key=os.urandom(32)
    )
    profiler.plot_memory_growth("ResonanceEncryption")

    profiler.profile_memory_growth(
        hash_function.hash,
        "GeometricWaveformHash",
        8192,  # 8KB input
        iterations=100
    )
    profiler.plot_memory_growth("GeometricWaveformHash")

    # Generate report
    report_file = profiler.generate_report()

    print("\nMemory profiling completed!")
    print(f"Report saved to: {report_file}")

if __name__ == "__main__":
    profile_quantoniumos()
