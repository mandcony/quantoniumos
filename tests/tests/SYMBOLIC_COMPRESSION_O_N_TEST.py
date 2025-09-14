#!/usr/bin/env python3
"""
SYMBOLIC COMPRESSION O(n) SCALING TEST
Tests the actual quantum symbolic compression algorithm
"""

import os
import sys
import time
import statistics
import numpy as np
from typing import Dict, List, Any
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

class SymbolicCompressionTest:
    """Test the actual O(n) symbolic compression algorithm"""
    
    def __init__(self):
        self.results = []
        
    def test_symbolic_compression_scaling(self) -> Dict[str, Any]:
        """Test the core O(n) symbolic compression algorithm"""
        print("🔬 SYMBOLIC COMPRESSION O(n) SCALING TEST")
        print("Testing the ACTUAL million-qubit compression algorithm")
        print("=" * 70)
        
        try:
            from src.assembly.python_bindings.quantum_symbolic_engine import QuantumSymbolicEngine
        except ImportError as e:
            return {"error": f"Cannot import Symbolic Engine: {e}"}
        
        # Test scaling from thousands to millions of qubits
        qubit_counts = [1000, 10000, 100000, 500000, 1000000]
        results = []
        
        for num_qubits in qubit_counts:
            print(f"\n  Testing {num_qubits:,} qubits...")
            
            # Initialize engine
            engine = QuantumSymbolicEngine(
                compression_size=64,  # Fixed compression size
                use_assembly=True,    # Use optimized assembly
                use_simd=True
            )
            
            # Warm-up run
            try:
                success, _ = engine.compress_million_qubits(min(num_qubits, 1000))
                if not success:
                    print(f"    Warm-up failed for {num_qubits} qubits")
                    continue
            except Exception as e:
                print(f"    Warm-up error: {e}")
                continue
            
            # Multiple timed runs for statistics
            compression_times = []
            operation_rates = []
            memory_usage = []
            
            num_runs = 5 if num_qubits <= 100000 else 3  # Fewer runs for large tests
            
            for run in range(num_runs):
                try:
                    # Create fresh engine for each run
                    test_engine = QuantumSymbolicEngine(
                        compression_size=64,
                        use_assembly=True,
                        use_simd=True
                    )
                    
                    start_time = time.perf_counter()
                    success, perf_stats = test_engine.compress_million_qubits(num_qubits)
                    end_time = time.perf_counter()
                    
                    if success and perf_stats:
                        total_time_ms = (end_time - start_time) * 1000
                        compression_times.append(total_time_ms)
                        
                        # Calculate operations per second
                        ops_per_sec = num_qubits / (total_time_ms / 1000)
                        operation_rates.append(ops_per_sec)
                        
                        # Memory usage
                        memory_mb = perf_stats.get('memory_mb', 0)
                        memory_usage.append(memory_mb)
                        
                        print(f"    Run {run+1}: {total_time_ms:.2f} ms, {ops_per_sec:,.0f} qubits/sec")
                    else:
                        print(f"    Run {run+1}: FAILED")
                        
                    # Cleanup
                    test_engine.cleanup()
                    
                except Exception as e:
                    print(f"    Run {run+1}: ERROR - {e}")
                    continue
            
            # Calculate statistics
            if compression_times:
                median_time = statistics.median(compression_times)
                std_time = statistics.stdev(compression_times) if len(compression_times) > 1 else 0
                median_ops = statistics.median(operation_rates)
                median_memory = statistics.median(memory_usage) if memory_usage else 0
                
                result = {
                    'num_qubits': num_qubits,
                    'compression_time_ms': median_time,
                    'time_std_ms': std_time,
                    'operations_per_second': median_ops,
                    'memory_mb': median_memory,
                    'compression_ratio': num_qubits / 64,  # 64 is compression size
                    'time_per_qubit_ns': (median_time * 1000000) / num_qubits,  # nanoseconds per qubit
                    'successful_runs': len(compression_times),
                    'total_runs': num_runs
                }
                
                results.append(result)
                
                print(f"    MEDIAN: {median_time:.2f} ± {std_time:.2f} ms")
                print(f"    RATE: {median_ops:,.0f} qubits/sec") 
                print(f"    MEMORY: {median_memory:.2f} MB")
                print(f"    TIME/QUBIT: {result['time_per_qubit_ns']:.2f} ns")
            else:
                print(f"    ALL RUNS FAILED for {num_qubits} qubits")
        
        # Analyze scaling
        scaling_analysis = self._analyze_compression_scaling(results)
        
        return {
            'test_type': 'symbolic_compression_scaling',
            'algorithm': 'Quantum Symbolic Compression O(n)',
            'results': results,
            'scaling_analysis': scaling_analysis,
            'methodology': 'Direct C/Assembly backend testing'
        }
    
    def _analyze_compression_scaling(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze symbolic compression scaling"""
        if len(results) < 2:
            return {"error": "Need at least 2 data points"}
        
        qubit_counts = [r['num_qubits'] for r in results]
        times_ms = [r['compression_time_ms'] for r in results]
        times_per_qubit = [r['time_per_qubit_ns'] for r in results]
        
        # Calculate scaling exponent using log-log regression
        log_qubits = np.log(qubit_counts)
        log_times = np.log(times_ms)
        
        n = len(qubit_counts)
        sum_x = np.sum(log_qubits)
        sum_y = np.sum(log_times)
        sum_xy = np.sum(log_qubits * log_times)
        sum_x2 = np.sum(log_qubits ** 2)
        
        if n * sum_x2 - sum_x ** 2 != 0:
            scaling_exponent = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
            intercept = (sum_y - scaling_exponent * sum_x) / n
        else:
            scaling_exponent = float('nan')
            intercept = float('nan')
        
        # R-squared
        if not np.isnan(scaling_exponent):
            predicted = scaling_exponent * log_qubits + intercept
            ss_res = np.sum((log_times - predicted) ** 2)
            ss_tot = np.sum((log_times - np.mean(log_times)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        else:
            r_squared = 0
        
        # Classify scaling
        if np.isnan(scaling_exponent):
            complexity = "Cannot determine"
        elif scaling_exponent < 0.2:
            complexity = "O(1) - Constant time"
        elif scaling_exponent < 0.8:
            complexity = "O(√n) - Sub-linear"
        elif scaling_exponent < 1.2:
            complexity = "✅ O(n) - LINEAR SCALING" 
        elif scaling_exponent < 1.8:
            complexity = "O(n log n) - Linearithmic"
        elif scaling_exponent < 2.2:
            complexity = "O(n²) - Quadratic"
        else:
            complexity = f"O(n^{scaling_exponent:.1f}) - Polynomial"
        
        # Check if time per qubit is roughly constant (true O(n) behavior)
        time_per_qubit_ratio = max(times_per_qubit) / min(times_per_qubit) if times_per_qubit else float('inf')
        is_true_linear = time_per_qubit_ratio < 2.0  # Within 2x variation
        
        # Performance metrics
        first_qubits = qubit_counts[0]
        last_qubits = qubit_counts[-1]
        first_time = times_ms[0]
        last_time = times_ms[-1]
        
        qubit_ratio = last_qubits / first_qubits
        time_ratio = last_time / first_time
        efficiency = time_ratio / qubit_ratio  # Should be ~1.0 for O(n)
        
        return {
            'scaling_exponent': scaling_exponent,
            'complexity_class': complexity,
            'r_squared': r_squared,
            'is_linear_scaling': 0.8 <= scaling_exponent <= 1.2,
            'is_true_linear': is_true_linear,
            'time_per_qubit_consistency': time_per_qubit_ratio,
            'efficiency_ratio': efficiency,
            'qubit_range': f"{first_qubits:,} to {last_qubits:,}",
            'time_range': f"{first_time:.2f} to {last_time:.2f} ms",
            'performance_data': {
                'qubit_ratio': qubit_ratio,
                'time_ratio': time_ratio,
                'expected_linear_ratio': qubit_ratio,
                'actual_vs_expected': efficiency
            }
        }
    
    def run_comprehensive_symbolic_test(self) -> Dict[str, Any]:
        """Run comprehensive symbolic compression scaling test"""
        print("🌌 QUANTUM SYMBOLIC COMPRESSION - O(n) VALIDATION")
        print("Testing Luis M Minier's million-qubit symbolic compression algorithm")
        print("=" * 80)
        
        start_time = time.time()
        
        compression_results = self.test_symbolic_compression_scaling()
        
        end_time = time.time()
        
        summary = {
            'test_name': 'Quantum Symbolic Compression O(n) Validation',
            'algorithm': 'Golden Ratio Parameterized Symbolic Compression',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_test_time': end_time - start_time,
            'symbolic_compression': compression_results
        }
        
        # Print results
        print("\n" + "=" * 80)
        print("SYMBOLIC COMPRESSION SCALING RESULTS")
        print("=" * 80)
        
        if 'scaling_analysis' in compression_results:
            analysis = compression_results['scaling_analysis']
            print(f"Algorithm: Quantum Symbolic Compression")
            print(f"Scaling: {analysis['complexity_class']}")
            print(f"Exponent: {analysis['scaling_exponent']:.3f}")
            print(f"R²: {analysis['r_squared']:.3f}")
            print(f"Linear scaling: {analysis['is_linear_scaling']}")
            print(f"True O(n): {analysis['is_true_linear']}")
            print(f"Efficiency: {analysis['efficiency_ratio']:.3f} (1.0 = perfect)")
            print(f"Range: {analysis['qubit_range']}")
            print(f"Time range: {analysis['time_range']}")
            
            # Verdict
            if analysis['is_linear_scaling'] and analysis['is_true_linear']:
                print("\n🎯 VERDICT: TRUE O(n) LINEAR SCALING CONFIRMED!")
            elif analysis['is_linear_scaling']:
                print("\n✅ VERDICT: LINEAR SCALING CONFIRMED")
            else:
                print(f"\n❌ VERDICT: NOT LINEAR - {analysis['complexity_class']}")
        
        print(f"\nTest completed in {end_time - start_time:.1f} seconds")
        
        return summary

if __name__ == "__main__":
    test = SymbolicCompressionTest()
    results = test.run_comprehensive_symbolic_test()
    
    # Save results
    with open('SYMBOLIC_COMPRESSION_O_N_TEST.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📊 Results saved to SYMBOLIC_COMPRESSION_O_N_TEST.json")
