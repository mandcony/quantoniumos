#!/usr/bin/env python3
"""
PURE PYTHON SYMBOLIC COMPRESSION O(n) TEST
Tests the exact algorithm from quantum_symbolic_compression.c
"""

import time
import math
import statistics
import numpy as np
from typing import Dict, List, Any, Tuple
import json

class PythonSymbolicCompression:
    """Pure Python implementation of the symbolic compression algorithm"""
    
    def __init__(self, compression_size: int = 64):
        self.compression_size = compression_size
        self.PHI = 1.618033988749894848204586834366  # Golden ratio
        self.TWO_PI = 2.0 * math.pi
        
    def compress_million_qubits_python(self, num_qubits: int) -> Tuple[float, Dict]:
        """
        Pure Python implementation of qsc_compress_million_qubits
        Exact translation from quantum_symbolic_compression.c
        """
        start_time = time.perf_counter()
        
        # Initialize amplitudes array
        amplitudes = np.zeros(self.compression_size, dtype=complex)
        
        # Amplitude normalization (from C code)
        amplitude = 1.0 / math.sqrt(self.compression_size)
        
        # Main compression loop - EXACT TRANSLATION FROM C
        for qubit_i in range(num_qubits):
            # Golden ratio phase calculation (from C: fmod((double)qubit_i * QSC_PHI * (double)num_qubits, QSC_2PI))
            phase = math.fmod(qubit_i * self.PHI * num_qubits, self.TWO_PI)
            
            # Secondary phase enhancement (from C code)
            qubit_factor = math.sqrt(num_qubits) / 1000.0
            final_phase = phase + math.fmod(qubit_i * qubit_factor, self.TWO_PI)
            
            # Compress to fixed-size representation (from C: qubit_i % compression_size)
            compressed_idx = qubit_i % self.compression_size
            
            # Complex amplitude calculation (from C code)
            cos_phase = math.cos(final_phase)
            sin_phase = math.sin(final_phase)
            
            # Accumulate amplitudes (from C code)
            amplitudes[compressed_idx] += amplitude * (cos_phase + 1j * sin_phase)
        
        # Renormalization (from C code)
        norm_squared = np.sum(np.abs(amplitudes) ** 2)
        if norm_squared > 0:
            norm = math.sqrt(norm_squared)
            amplitudes /= norm
        
        end_time = time.perf_counter()
        
        compression_time = (end_time - start_time) * 1000  # ms
        
        # Calculate performance metrics
        ops_per_second = num_qubits / (compression_time / 1000) if compression_time > 0 else 0
        memory_mb = (self.compression_size * 16) / (1024 * 1024)  # Complex64 = 16 bytes
        compression_ratio = num_qubits / self.compression_size
        
        perf_stats = {
            'compression_time_ms': compression_time,
            'operations_per_second': ops_per_second,
            'memory_mb': memory_mb,
            'compression_ratio': compression_ratio,
            'norm': norm if 'norm' in locals() else 1.0,
            'amplitude_sum': float(np.sum(np.abs(amplitudes))),
            'final_state_size': len(amplitudes)
        }
        
        return compression_time, perf_stats

class SymbolicCompressionScalingTest:
    """Test O(n) scaling of symbolic compression algorithm"""
    
    def __init__(self):
        self.compressor = PythonSymbolicCompression()
        
    def test_million_qubit_scaling(self) -> Dict[str, Any]:
        """Test scaling from thousands to millions of qubits"""
        print("🔬 SYMBOLIC COMPRESSION ALGORITHM O(n) SCALING TEST")
        print("Direct Python implementation of quantum_symbolic_compression.c")
        print("=" * 75)
        
        # Test range from 1K to 1M qubits
        qubit_counts = [1000, 5000, 10000, 50000, 100000, 500000, 1000000]
        results = []
        
        for num_qubits in qubit_counts:
            print(f"\n  Testing {num_qubits:,} qubits...")
            
            # Multiple runs for statistical accuracy
            times = []
            performances = []
            
            num_runs = 5 if num_qubits <= 100000 else 3
            
            for run in range(num_runs):
                try:
                    compression_time, perf_stats = self.compressor.compress_million_qubits_python(num_qubits)
                    times.append(compression_time)
                    performances.append(perf_stats)
                    print(f"    Run {run+1}: {compression_time:.3f} ms ({perf_stats['operations_per_second']:,.0f} qubits/sec)")
                except Exception as e:
                    print(f"    Run {run+1}: ERROR - {e}")
                    continue
            
            if times:
                median_time = statistics.median(times)
                std_time = statistics.stdev(times) if len(times) > 1 else 0
                median_perf = performances[len(times)//2]  # Get median performance
                
                result = {
                    'num_qubits': num_qubits,
                    'compression_time_ms': median_time,
                    'time_std_ms': std_time,
                    'operations_per_second': median_perf['operations_per_second'],
                    'memory_mb': median_perf['memory_mb'],
                    'compression_ratio': median_perf['compression_ratio'],
                    'time_per_qubit_ns': (median_time * 1_000_000) / num_qubits,  # nanoseconds per qubit
                    'successful_runs': len(times),
                    'algorithm': 'Pure Python translation of C implementation'
                }
                
                results.append(result)
                
                print(f"    MEDIAN: {median_time:.3f} ± {std_time:.3f} ms")
                print(f"    RATE: {result['operations_per_second']:,.0f} qubits/sec")
                print(f"    TIME/QUBIT: {result['time_per_qubit_ns']:.3f} ns")
                print(f"    RATIO: {result['compression_ratio']:,.0f}:1")
            else:
                print(f"    ALL RUNS FAILED")
        
        # Analyze scaling
        scaling_analysis = self._analyze_o_n_scaling(results)
        
        return {
            'test_type': 'symbolic_compression_million_qubit_scaling',
            'algorithm': 'Quantum Symbolic Compression (Python)',
            'implementation': 'Direct translation from quantum_symbolic_compression.c',
            'results': results,
            'scaling_analysis': scaling_analysis
        }
    
    def _analyze_o_n_scaling(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze if algorithm truly exhibits O(n) scaling"""
        if len(results) < 3:
            return {"error": "Need at least 3 data points for scaling analysis"}
        
        qubit_counts = [r['num_qubits'] for r in results]
        times_ms = [r['compression_time_ms'] for r in results]
        times_per_qubit = [r['time_per_qubit_ns'] for r in results]
        
        # Log-log regression for scaling exponent
        log_qubits = np.log(qubit_counts)
        log_times = np.log(times_ms)
        
        # Linear regression: log(time) = slope * log(qubits) + intercept
        n = len(qubit_counts)
        sum_x = np.sum(log_qubits)
        sum_y = np.sum(log_times)
        sum_xy = np.sum(log_qubits * log_times)
        sum_x2 = np.sum(log_qubits ** 2)
        
        denominator = n * sum_x2 - sum_x ** 2
        if denominator != 0:
            scaling_exponent = (n * sum_xy - sum_x * sum_y) / denominator
            intercept = (sum_y - scaling_exponent * sum_x) / n
        else:
            scaling_exponent = float('nan')
            intercept = float('nan')
        
        # Calculate R-squared
        if not np.isnan(scaling_exponent):
            predicted = scaling_exponent * log_qubits + intercept
            ss_res = np.sum((log_times - predicted) ** 2)
            ss_tot = np.sum((log_times - np.mean(log_times)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        else:
            r_squared = 0
        
        # Check for true O(n) behavior - time per qubit should be constant
        time_per_qubit_variation = np.std(times_per_qubit) / np.mean(times_per_qubit) if times_per_qubit else 1
        is_constant_per_qubit = time_per_qubit_variation < 0.3  # Less than 30% variation
        
        # Classify scaling behavior
        if np.isnan(scaling_exponent):
            complexity = "Cannot determine"
            verdict = "INCONCLUSIVE"
        elif scaling_exponent < 0.2:
            complexity = "O(1) - Constant time"
            verdict = "BETTER THAN LINEAR!"
        elif 0.8 <= scaling_exponent <= 1.2:
            if is_constant_per_qubit:
                complexity = "✅ O(n) - TRUE LINEAR SCALING"
                verdict = "O(n) CONFIRMED!"
            else:
                complexity = "~O(n) - Approximately linear"
                verdict = "MOSTLY LINEAR"
        elif scaling_exponent < 1.8:
            complexity = "O(n log n) - Linearithmic"
            verdict = "NOT QUITE LINEAR"
        elif scaling_exponent < 2.2:
            complexity = "O(n²) - Quadratic"
            verdict = "NOT LINEAR"
        else:
            complexity = f"O(n^{scaling_exponent:.1f}) - Higher order polynomial"
            verdict = "NOT LINEAR"
        
        # Performance analysis
        first_qubits, last_qubits = qubit_counts[0], qubit_counts[-1]
        first_time, last_time = times_ms[0], times_ms[-1]
        qubit_ratio = last_qubits / first_qubits
        time_ratio = last_time / first_time
        efficiency = time_ratio / qubit_ratio  # Should be ~1.0 for perfect O(n)
        
        return {
            'scaling_exponent': scaling_exponent,
            'complexity_class': complexity,
            'verdict': verdict,
            'r_squared': r_squared,
            'is_linear_scaling': 0.8 <= scaling_exponent <= 1.2,
            'is_true_o_n': is_constant_per_qubit and 0.8 <= scaling_exponent <= 1.2,
            'time_per_qubit_variation': time_per_qubit_variation,
            'efficiency_ratio': efficiency,
            'performance_metrics': {
                'qubit_range': f"{first_qubits:,} to {last_qubits:,}",
                'time_range': f"{first_time:.2f} to {last_time:.2f} ms",
                'qubit_scale_factor': qubit_ratio,
                'time_scale_factor': time_ratio,
                'expected_linear_time': first_time * qubit_ratio,
                'actual_vs_expected': time_ratio / qubit_ratio
            },
            'data_points': n
        }
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive O(n) scaling validation"""
        print("🌌 QUANTUM SYMBOLIC COMPRESSION - COMPREHENSIVE O(n) VALIDATION")
        print("Testing Luis M Minier's million-qubit compression algorithm")
        print("Implementation: Direct Python translation of C/Assembly code")
        print("=" * 85)
        
        start_time = time.time()
        
        scaling_results = self.test_million_qubit_scaling()
        
        end_time = time.time()
        
        summary = {
            'test_name': 'Quantum Symbolic Compression O(n) Validation',
            'author': 'Luis M Minier',
            'algorithm': 'Golden Ratio Parameterized Symbolic Compression',
            'implementation': 'Pure Python (translated from quantum_symbolic_compression.c)',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_test_time_seconds': end_time - start_time,
            'scaling_test': scaling_results
        }
        
        # Print comprehensive results
        print("\n" + "=" * 85)
        print("🎯 FINAL O(n) SCALING VALIDATION RESULTS")
        print("=" * 85)
        
        if 'scaling_analysis' in scaling_results:
            analysis = scaling_results['scaling_analysis']
            
            print(f"Algorithm: Quantum Symbolic Compression")
            print(f"Implementation: {summary['implementation']}")
            print(f"")
            print(f"📊 SCALING ANALYSIS:")
            print(f"  Complexity: {analysis['complexity_class']}")
            print(f"  Exponent: {analysis['scaling_exponent']:.4f}")
            print(f"  R² coefficient: {analysis['r_squared']:.4f}")
            print(f"  Efficiency ratio: {analysis['efficiency_ratio']:.4f}")
            print(f"  Time/qubit variation: {analysis['time_per_qubit_variation']:.1%}")
            print(f"")
            print(f"📈 PERFORMANCE RANGE:")
            print(f"  {analysis['performance_metrics']['qubit_range']} qubits")
            print(f"  {analysis['performance_metrics']['time_range']}")
            print(f"  Scale factor: {analysis['performance_metrics']['qubit_scale_factor']:.0f}x qubits")
            print(f"  Time factor: {analysis['performance_metrics']['time_scale_factor']:.1f}x time")
            print(f"")
            print(f"🎯 FINAL VERDICT: {analysis['verdict']}")
            
            if analysis['is_true_o_n']:
                print("✅ TRUE O(n) LINEAR SCALING EMPIRICALLY PROVEN!")
            elif analysis['is_linear_scaling']:
                print("✅ LINEAR SCALING CONFIRMED (approximately O(n))")
            else:
                print(f"❌ NOT LINEAR - {analysis['complexity_class']}")
        
        print(f"\nTest completed in {end_time - start_time:.1f} seconds")
        print(f"Data points: {len(scaling_results.get('results', []))}")
        
        return summary

if __name__ == "__main__":
    test = SymbolicCompressionScalingTest()
    results = test.run_comprehensive_test()
    
    # Save results
    with open('SYMBOLIC_COMPRESSION_O_N_FINAL_TEST.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📊 Complete results saved to SYMBOLIC_COMPRESSION_O_N_FINAL_TEST.json")
