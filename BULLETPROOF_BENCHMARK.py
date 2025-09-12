#!/usr/bin/env python3
"""
BULLETPROOF SCALING BENCHMARK
Addresses all timing and measurement issues identified
"""

import os
import sys
import time
import gc
import tracemalloc
import threading
import statistics
import numpy as np
from typing import List, Dict, Tuple, Any
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def pin_threads():
    """Pin to single thread to avoid threading variability"""
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMBA_NUM_THREADS'] = '1'
    if hasattr(np, 'seterr'):
        np.seterr(all='ignore')  # Suppress warnings during benchmark

class BulletproofBenchmark:
    """Bulletproof benchmark with proper init/compute separation"""
    
    def __init__(self):
        pin_threads()
        self.results = {}
        self.min_runtime_ms = 50  # Force minimum 50ms per measurement
        
    def benchmark_core_rft_proper(self) -> Dict[str, Any]:
        """Properly benchmark Core RFT with init/compute separation"""
        print("🎯 BULLETPROOF Core RFT Benchmark")
        print("=" * 50)
        
        try:
            from src.core.canonical_true_rft import CanonicalTrueRFT
        except ImportError as e:
            return {"error": f"Cannot import Core RFT: {e}"}
        
        sizes = [16, 32, 64, 128, 256, 512, 1024]
        results = []
        
        for size in sizes:
            print(f"  Testing size {size}...")
            
            # === PHASE 1: INITIALIZATION (NOT TIMED) ===
            gc.collect()  # Clean slate
            
            # Initialize RFT (this is setup, not the algorithm)
            rft = CanonicalTrueRFT(size)
            
            # Create test data
            test_signal = np.random.randn(size) + 1j * np.random.randn(size)
            test_signal = test_signal / np.linalg.norm(test_signal)
            
            # Warm-up runs (NOT TIMED)
            for _ in range(3):
                _ = rft.forward_transform(test_signal)
                _ = rft.inverse_transform(_)
            
            gc.collect()  # Clean before measurement
            
            # === PHASE 2: COMPUTE TIMING (ONLY ALGORITHM) ===
            
            # Calculate iterations to get minimum runtime
            single_run_time = self._time_single_operation(
                lambda: rft.forward_transform(test_signal)
            )
            iterations = max(1, int(self.min_runtime_ms / 1000 / single_run_time))
            
            # Memory measurement
            tracemalloc.start()
            baseline = tracemalloc.get_traced_memory()[0]
            
            # Multiple timed runs for statistics
            forward_times = []
            inverse_times = []
            
            for _ in range(5):  # 5 statistical samples
                # Forward timing
                start = time.perf_counter()
                for _ in range(iterations):
                    spectrum = rft.forward_transform(test_signal)
                end = time.perf_counter()
                forward_times.append((end - start) / iterations)
                
                # Inverse timing  
                start = time.perf_counter()
                for _ in range(iterations):
                    reconstructed = rft.inverse_transform(spectrum)
                end = time.perf_counter()
                inverse_times.append((end - start) / iterations)
            
            # Memory delta
            peak_memory, current_memory = tracemalloc.get_traced_memory()
            memory_delta = peak_memory - baseline
            tracemalloc.stop()
            
            # Validation
            error = np.linalg.norm(test_signal - reconstructed)
            
            result = {
                'size': size,
                'forward_time_ms': statistics.median(forward_times) * 1000,
                'forward_time_std': statistics.stdev(forward_times) * 1000 if len(forward_times) > 1 else 0,
                'inverse_time_ms': statistics.median(inverse_times) * 1000,
                'inverse_time_std': statistics.stdev(inverse_times) * 1000 if len(inverse_times) > 1 else 0,
                'memory_delta_bytes': memory_delta,
                'reconstruction_error': error,
                'iterations_per_sample': iterations,
                'total_runtime_ms': statistics.median(forward_times) * iterations * 1000
            }
            
            results.append(result)
            print(f"    Forward: {result['forward_time_ms']:.3f} ± {result['forward_time_std']:.3f} ms")
            print(f"    Memory: {result['memory_delta_bytes']:,} bytes")
            
        # Scaling analysis
        scaling_analysis = self._analyze_scaling_proper(results, 'forward_time_ms')
        memory_scaling = self._analyze_scaling_proper(results, 'memory_delta_bytes')
        
        return {
            'results': results,
            'time_scaling': scaling_analysis,
            'memory_scaling': memory_scaling,
            'methodology': 'Bulletproof: init/compute separated, warm-up, statistical sampling'
        }
    
    def benchmark_vertex_rft_proper(self) -> Dict[str, Any]:
        """Properly benchmark Vertex RFT with init/compute separation"""
        print("\n🎯 BULLETPROOF Vertex RFT Benchmark") 
        print("=" * 50)
        
        try:
            from src.assembly.python_bindings.vertex_quantum_rft import EnhancedVertexQuantumRFT
        except ImportError as e:
            return {"error": f"Cannot import Vertex RFT: {e}"}
        
        data_sizes = [16, 32, 64, 128, 256, 512]  # Data sizes, not vertex count
        vertex_count = 1000  # Fixed vertex count
        results = []
        
        for data_size in data_sizes:
            print(f"  Testing data size {data_size} (1000 vertices)...")
            
            # === PHASE 1: INITIALIZATION (NOT TIMED) ===
            gc.collect()
            
            # Initialize Vertex RFT (heavy setup)
            vrft = EnhancedVertexQuantumRFT(data_size, vertex_qubits=vertex_count)
            
            # Create test data
            test_data = np.random.randn(data_size)
            test_data = test_data / np.linalg.norm(test_data)
            
            # Warm-up runs (NOT TIMED)
            for _ in range(3):
                _ = vrft.enhanced_geometric_waveform_encode(test_data)
            
            gc.collect()
            
            # === PHASE 2: COMPUTE TIMING (ONLY ALGORITHM) ===
            
            # Calculate iterations for minimum runtime
            single_run_time = self._time_single_operation(
                lambda: vrft.enhanced_geometric_waveform_encode(test_data)
            )
            iterations = max(1, int(self.min_runtime_ms / 1000 / single_run_time))
            
            # Memory measurement
            tracemalloc.start()
            baseline = tracemalloc.get_traced_memory()[0]
            
            # Multiple timed runs
            encode_times = []
            
            for _ in range(5):  # 5 statistical samples
                start = time.perf_counter()
                for _ in range(iterations):
                    encoding = vrft.enhanced_geometric_waveform_encode(test_data)
                end = time.perf_counter()
                encode_times.append((end - start) / iterations)
            
            # Memory delta
            peak_memory, current_memory = tracemalloc.get_traced_memory()
            memory_delta = peak_memory - baseline
            tracemalloc.stop()
            
            result = {
                'data_size': data_size,
                'vertex_count': vertex_count,
                'encode_time_ms': statistics.median(encode_times) * 1000,
                'encode_time_std': statistics.stdev(encode_times) * 1000 if len(encode_times) > 1 else 0,
                'memory_delta_bytes': memory_delta,
                'iterations_per_sample': iterations,
                'phi_resonance': encoding.get('phi_resonance', 0),
                'total_runtime_ms': statistics.median(encode_times) * iterations * 1000
            }
            
            results.append(result)
            print(f"    Encode: {result['encode_time_ms']:.3f} ± {result['encode_time_std']:.3f} ms")
            print(f"    Memory: {result['memory_delta_bytes']:,} bytes")
        
        # Scaling analysis
        time_scaling = self._analyze_scaling_proper(results, 'encode_time_ms', size_key='data_size')
        memory_scaling = self._analyze_scaling_proper(results, 'memory_delta_bytes', size_key='data_size')
        
        return {
            'results': results,
            'time_scaling': time_scaling,
            'memory_scaling': memory_scaling,
            'methodology': 'Bulletproof: init/compute separated, fixed vertex count, statistical sampling'
        }
    
    def _time_single_operation(self, operation_func) -> float:
        """Time a single operation for iteration calculation"""
        start = time.perf_counter()
        operation_func()
        end = time.perf_counter()
        return end - start
    
    def _analyze_scaling_proper(self, results: List[Dict], metric_key: str, size_key: str = 'size') -> Dict[str, Any]:
        """Proper scaling analysis with statistical methods"""
        if len(results) < 2:
            return {"error": "Need at least 2 data points"}
        
        sizes = [r[size_key] for r in results]
        values = [r[metric_key] for r in results]
        
        # Calculate empirical scaling exponent using log-log regression
        log_sizes = np.log(sizes)
        log_values = np.log(values)
        
        # Linear regression on log-log plot: log(y) = a*log(x) + b
        # Slope 'a' is the scaling exponent
        n = len(sizes)
        sum_x = np.sum(log_sizes)
        sum_y = np.sum(log_values)
        sum_xy = np.sum(log_sizes * log_values)
        sum_x2 = np.sum(log_sizes ** 2)
        
        # Calculate slope (scaling exponent)
        if n * sum_x2 - sum_x ** 2 != 0:
            scaling_exponent = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
            intercept = (sum_y - scaling_exponent * sum_x) / n
        else:
            scaling_exponent = float('nan')
            intercept = float('nan')
        
        # R-squared calculation
        mean_log_y = sum_y / n
        ss_tot = np.sum((log_values - mean_log_y) ** 2)
        ss_res = np.sum((log_values - (scaling_exponent * log_sizes + intercept)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Classify scaling
        if scaling_exponent < 0.5:
            scaling_class = "Sub-linear (better than O(n))"
        elif 0.5 <= scaling_exponent < 1.2:
            scaling_class = "Linear O(n)"
        elif 1.2 <= scaling_exponent < 1.8:
            scaling_class = "O(n log n)"
        elif 1.8 <= scaling_exponent < 2.2:
            scaling_class = "Quadratic O(n²)"
        else:
            scaling_class = f"Polynomial O(n^{scaling_exponent:.1f})"
        
        return {
            'scaling_exponent': scaling_exponent,
            'scaling_class': scaling_class,
            'r_squared': r_squared,
            'first_value': values[0],
            'last_value': values[-1],
            'size_ratio': sizes[-1] / sizes[0],
            'value_ratio': values[-1] / values[0],
            'data_points': len(results),
            'raw_data': list(zip(sizes, values))
        }
    
    def run_bulletproof_benchmarks(self) -> Dict[str, Any]:
        """Run all bulletproof benchmarks"""
        print("🔬 BULLETPROOF SCALING BENCHMARK")
        print("Methodology: Init/compute separated, warm-up, statistical sampling")
        print("=" * 80)
        
        start_time = time.time()
        
        # Run benchmarks
        core_rft_results = self.benchmark_core_rft_proper()
        vertex_rft_results = self.benchmark_vertex_rft_proper()
        
        end_time = time.time()
        
        # Summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_benchmark_time': end_time - start_time,
            'methodology': 'Bulletproof: separated init/compute, warm-up, statistical sampling',
            'core_rft': core_rft_results,
            'vertex_rft': vertex_rft_results
        }
        
        # Print summary
        print("\n" + "=" * 80)
        print("BULLETPROOF BENCHMARK RESULTS SUMMARY")
        print("=" * 80)
        
        if 'time_scaling' in core_rft_results:
            core_scaling = core_rft_results['time_scaling']
            print(f"Core RFT Time Scaling: {core_scaling['scaling_class']}")
            print(f"  Exponent: {core_scaling['scaling_exponent']:.3f}")
            print(f"  R²: {core_scaling['r_squared']:.3f}")
            print(f"  Size ratio: {core_scaling['size_ratio']:.1f}x")
            print(f"  Time ratio: {core_scaling['value_ratio']:.1f}x")
        
        if 'time_scaling' in vertex_rft_results:
            vertex_scaling = vertex_rft_results['time_scaling']
            print(f"\nVertex RFT Time Scaling: {vertex_scaling['scaling_class']}")
            print(f"  Exponent: {vertex_scaling['scaling_exponent']:.3f}")
            print(f"  R²: {vertex_scaling['r_squared']:.3f}")
            print(f"  Size ratio: {vertex_scaling['size_ratio']:.1f}x")
            print(f"  Time ratio: {vertex_scaling['value_ratio']:.1f}x")
        
        print(f"\nTotal benchmark time: {end_time - start_time:.1f} seconds")
        print("Methodology: Bulletproof measurement with proper controls")
        
        return summary

if __name__ == "__main__":
    benchmark = BulletproofBenchmark()
    results = benchmark.run_bulletproof_benchmarks()
    
    # Save results
    with open('BULLETPROOF_BENCHMARK_RESULTS.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✅ Results saved to BULLETPROOF_BENCHMARK_RESULTS.json")
