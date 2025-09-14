#!/usr/bin/env python3
"""
REAL TIMING CURVES TEST
=======================
Replace "0.00 ms" logs with averaged timings over 1000 runs. Show actual μs/element scaling.
"""

import sys
import os
import time
import statistics
import numpy as np

# Add the src path to import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from assembly.python_bindings.quantum_symbolic_engine import QuantumSymbolicEngine

def benchmark_timing_curves():
    """Generate precise timing curves for quantum operations"""
    
    print("⏱️ REAL TIMING CURVES TEST")
    print("=" * 80)
    print("Measuring precise timings over 1000 runs per size")
    print("Target: Show actual μs/element scaling")
    print("=" * 80)
    
    # Test sizes
    test_sizes = [8, 16, 32, 64, 128, 256, 512]
    timing_results = []
    
    # Initialize assembly engine
    engine = QuantumSymbolicEngine()
    
    for size in test_sizes:
        print(f"\nBenchmarking {size} qubits (1000 runs)...")
        
        # Timing arrays
        init_times = []
        compress_times = []
        cleanup_times = []
        
        # Run 1000 iterations for statistical accuracy
        for run in range(1000):
            if run % 100 == 0:
                print(f"  Progress: {run}/1000 runs...")
            
            # Time initialization
            start = time.perf_counter()
            init_success = engine.initialize_state(size)
            init_time = (time.perf_counter() - start) * 1e6  # Convert to microseconds
            init_times.append(init_time)
            
            if not init_success:
                print(f"    ⚠️ Initialization failed for run {run}")
                continue
            
            # Time compression (using the existing compress_million_qubits method)
            start = time.perf_counter()
            success, result = engine.compress_million_qubits(size)
            compress_time = (time.perf_counter() - start) * 1e6  # Convert to microseconds
            compress_times.append(compress_time)
            
            # Time cleanup
            start = time.perf_counter()
            engine.cleanup()
            cleanup_time = (time.perf_counter() - start) * 1e6  # Convert to microseconds
            cleanup_times.append(cleanup_time)
        
        # Calculate statistics
        init_stats = {
            'mean': statistics.mean(init_times),
            'median': statistics.median(init_times),
            'stdev': statistics.stdev(init_times) if len(init_times) > 1 else 0,
            'min': min(init_times),
            'max': max(init_times)
        }
        
        compress_stats = {
            'mean': statistics.mean(compress_times),
            'median': statistics.median(compress_times),
            'stdev': statistics.stdev(compress_times) if len(compress_times) > 1 else 0,
            'min': min(compress_times),
            'max': max(compress_times)
        }
        
        cleanup_stats = {
            'mean': statistics.mean(cleanup_times),
            'median': statistics.median(cleanup_times),
            'stdev': statistics.stdev(cleanup_times) if len(cleanup_times) > 1 else 0,
            'min': min(cleanup_times),
            'max': max(cleanup_times)
        }
        
        # Calculate per-element timings
        per_element_init = init_stats['mean'] / size
        per_element_compress = compress_stats['mean'] / size
        per_element_total = (init_stats['mean'] + compress_stats['mean'] + cleanup_stats['mean']) / size
        
        timing_results.append({
            'size': size,
            'init_stats': init_stats,
            'compress_stats': compress_stats,
            'cleanup_stats': cleanup_stats,
            'per_element_init': per_element_init,
            'per_element_compress': per_element_compress,
            'per_element_total': per_element_total
        })
        
        print(f"  ✅ Init: {init_stats['mean']:.2f} ± {init_stats['stdev']:.2f} μs")
        print(f"  ✅ Compress: {compress_stats['mean']:.2f} ± {compress_stats['stdev']:.2f} μs")
        print(f"  ✅ Cleanup: {cleanup_stats['mean']:.2f} ± {cleanup_stats['stdev']:.2f} μs")
        print(f"  📊 Per-element: {per_element_total:.3f} μs/qubit")
    
    # Generate timing curves report
    print("\n" + "=" * 80)
    print("TIMING CURVES ANALYSIS")
    print("=" * 80)
    
    print("\nScaling Analysis:")
    print("Size    | Init (μs) | Compress (μs) | Total (μs) | μs/qubit")
    print("-" * 60)
    
    for result in timing_results:
        total_time = result['init_stats']['mean'] + result['compress_stats']['mean'] + result['cleanup_stats']['mean']
        print(f"{result['size']:7d} | {result['init_stats']['mean']:8.2f} | "
              f"{result['compress_stats']['mean']:12.2f} | {total_time:9.2f} | "
              f"{result['per_element_total']:7.3f}")
    
    # Check for linear scaling
    print("\nScaling Validation:")
    if len(timing_results) >= 2:
        first_per_element = timing_results[0]['per_element_total']
        last_per_element = timing_results[-1]['per_element_total']
        scaling_ratio = last_per_element / first_per_element
        
        if scaling_ratio < 2.0:  # Less than 2x increase suggests good scaling
            print(f"✅ Excellent scaling: {scaling_ratio:.2f}x per-element increase from {timing_results[0]['size']} to {timing_results[-1]['size']} qubits")
        elif scaling_ratio < 5.0:
            print(f"⚠️ Moderate scaling: {scaling_ratio:.2f}x per-element increase")
        else:
            print(f"❌ Poor scaling: {scaling_ratio:.2f}x per-element increase")
    
    return timing_results

if __name__ == "__main__":
    results = benchmark_timing_curves()
    print("\n🎉 TIMING CURVES: ✅ COMPLETE")
    print("Real timing data collected with statistical precision!")
