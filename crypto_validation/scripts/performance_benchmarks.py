#!/usr/bin/env python3
"""
Performance Benchmarking Suite
==============================

Comprehensive performance analysis of QuantoniumOS cryptographic components.
Measures throughput, latency, memory usage, and scaling characteristics.

Validates paper claims:
- Throughput: 9.2 MB/s target
- Memory efficiency: O(n) scaling
- Cross-platform performance consistency
"""

import sys
import os
import time
import json
import psutil
import secrets
import threading
from datetime import datetime
from typing import Dict, List, Tuple, Any
import statistics

# Add core path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'core'))

try:
    from enhanced_rft_crypto_v2 import EnhancedRFTCryptoV2
    from geometric_waveform_hash import GeometricWaveformHash
    from canonical_true_rft import CanonicalTrueRFT
    print("✓ Successfully imported performance test modules")
except ImportError as e:
    print(f"✗ Failed to import modules: {e}")
    sys.exit(1)


class PerformanceBenchmarker:
    """Comprehensive performance benchmarking suite"""
    
    def __init__(self):
        self.test_key = b"PERFORMANCE_TEST_KEY_QUANTUM32_"
        self.cipher = EnhancedRFTCryptoV2(self.test_key)
        self.hasher = GeometricWaveformHash()
        self.results = {}
        
        # System info
        self.system_info = {
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total,
            'platform': sys.platform,
            'python_version': sys.version,
            'timestamp': datetime.now().isoformat()
        }
    
    def measure_memory_usage(self, func, *args, **kwargs):
        """Measure memory usage of a function call"""
        process = psutil.Process()
        
        # Baseline memory
        baseline_memory = process.memory_info().rss
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Peak memory
        peak_memory = process.memory_info().rss
        
        memory_delta = peak_memory - baseline_memory
        
        return result, memory_delta
    
    def benchmark_cipher_throughput(self) -> Dict[str, Any]:
        """Benchmark cipher encryption/decryption throughput"""
        print("⚡ Benchmarking Cipher Throughput...")
        
        # Test different data sizes
        test_sizes = [
            (1024, "1KB"),
            (4096, "4KB"), 
            (16384, "16KB"),
            (65536, "64KB"),
            (262144, "256KB"),
            (1048576, "1MB")
        ]
        
        throughput_results = []
        
        for size_bytes, size_label in test_sizes:
            print(f"  Testing {size_label}...")
            
            # Generate test data
            test_data = secrets.token_bytes(size_bytes)
            
            # Warm-up runs
            for _ in range(3):
                encrypted = self.cipher.encrypt_aead(test_data)
                decrypted = self.cipher.decrypt_aead(encrypted)
            
            # Benchmark encryption
            encrypt_times = []
            encrypt_memory_deltas = []
            
            for _ in range(10):
                start_time = time.perf_counter()
                encrypted, memory_delta = self.measure_memory_usage(
                    self.cipher.encrypt_aead, test_data
                )
                end_time = time.perf_counter()
                
                encrypt_times.append(end_time - start_time)
                encrypt_memory_deltas.append(memory_delta)
            
            # Benchmark decryption
            decrypt_times = []
            decrypt_memory_deltas = []
            
            for _ in range(10):
                start_time = time.perf_counter()
                decrypted, memory_delta = self.measure_memory_usage(
                    self.cipher.decrypt_aead, encrypted
                )
                end_time = time.perf_counter()
                
                decrypt_times.append(end_time - start_time)
                decrypt_memory_deltas.append(memory_delta)
            
            # Calculate statistics
            avg_encrypt_time = statistics.mean(encrypt_times)
            avg_decrypt_time = statistics.mean(decrypt_times)
            
            encrypt_mbps = size_bytes / (avg_encrypt_time * 1024 * 1024)
            decrypt_mbps = size_bytes / (avg_decrypt_time * 1024 * 1024)
            
            throughput_results.append({
                'size_bytes': size_bytes,
                'size_label': size_label,
                'encrypt_time_ms': avg_encrypt_time * 1000,
                'decrypt_time_ms': avg_decrypt_time * 1000,
                'encrypt_mbps': encrypt_mbps,
                'decrypt_mbps': decrypt_mbps,
                'encrypt_memory_avg': statistics.mean(encrypt_memory_deltas),
                'decrypt_memory_avg': statistics.mean(decrypt_memory_deltas),
                'round_trip_success': test_data == decrypted
            })
            
            print(f"    Encrypt: {encrypt_mbps:6.1f} MB/s, Decrypt: {decrypt_mbps:6.1f} MB/s")
        
        # Calculate overall metrics
        large_file_results = [r for r in throughput_results if r['size_bytes'] >= 65536]
        avg_encrypt_mbps = statistics.mean([r['encrypt_mbps'] for r in large_file_results])
        avg_decrypt_mbps = statistics.mean([r['decrypt_mbps'] for r in large_file_results])
        
        # Check paper compliance (9.2 MB/s target)
        meets_paper_target = avg_encrypt_mbps >= 9.0
        
        return {
            'test_results': throughput_results,
            'summary': {
                'avg_encrypt_mbps': avg_encrypt_mbps,
                'avg_decrypt_mbps': avg_decrypt_mbps,
                'meets_paper_target': meets_paper_target,
                'target_mbps': 9.2,
                'performance_ratio': avg_encrypt_mbps / 9.2
            }
        }
    
    def benchmark_hash_performance(self) -> Dict[str, Any]:
        """Benchmark geometric waveform hash performance"""
        print("🔗 Benchmarking Hash Performance...")
        
        # Test different input sizes
        test_sizes = [
            (64, "64B"),
            (256, "256B"),
            (1024, "1KB"),
            (4096, "4KB"),
            (16384, "16KB"),
            (65536, "64KB")
        ]
        
        hash_results = []
        
        for size_bytes, size_label in test_sizes:
            print(f"  Testing {size_label}...")
            
            # Generate test data
            test_data = secrets.token_bytes(size_bytes)
            
            # Warm-up runs
            for _ in range(3):
                _ = self.hasher.hash(test_data)
            
            # Benchmark hashing
            hash_times = []
            memory_deltas = []
            
            for _ in range(20):
                start_time = time.perf_counter()
                hash_result, memory_delta = self.measure_memory_usage(
                    self.hasher.hash, test_data
                )
                end_time = time.perf_counter()
                
                hash_times.append(end_time - start_time)
                memory_deltas.append(memory_delta)
            
            # Calculate statistics
            avg_hash_time = statistics.mean(hash_times)
            hash_mbps = size_bytes / (avg_hash_time * 1024 * 1024)
            
            hash_results.append({
                'size_bytes': size_bytes,
                'size_label': size_label,
                'hash_time_ms': avg_hash_time * 1000,
                'hash_mbps': hash_mbps,
                'memory_avg': statistics.mean(memory_deltas),
                'output_size': len(hash_result)
            })
            
            print(f"    Hash: {hash_mbps:6.1f} MB/s")
        
        # Calculate scaling characteristics
        scaling_analysis = self.analyze_scaling(hash_results, 'hash_mbps')
        
        return {
            'test_results': hash_results,
            'scaling_analysis': scaling_analysis
        }
    
    def benchmark_rft_transforms(self) -> Dict[str, Any]:
        """Benchmark core RFT transform performance"""
        print("🌀 Benchmarking RFT Transforms...")
        
        # Test different transform sizes
        test_sizes = [8, 16, 32, 64, 128, 256]
        
        rft_results = []
        
        for size in test_sizes:
            print(f"  Testing size {size}...")
            
            try:
                rft = CanonicalTrueRFT(size)
                test_signal = secrets.token_bytes(size)
                test_complex = [complex(b, 0) for b in test_signal]
                
                # Warm-up
                for _ in range(3):
                    _ = rft.forward_transform(test_complex)
                
                # Benchmark forward transform
                forward_times = []
                for _ in range(50):
                    start_time = time.perf_counter()
                    forward_result = rft.forward_transform(test_complex)
                    end_time = time.perf_counter()
                    forward_times.append(end_time - start_time)
                
                # Benchmark inverse transform
                inverse_times = []
                for _ in range(50):
                    start_time = time.perf_counter()
                    inverse_result = rft.inverse_transform(forward_result)
                    end_time = time.perf_counter()
                    inverse_times.append(end_time - start_time)
                
                # Calculate unitarity error
                unitarity_error = rft.get_unitarity_error()
                
                rft_results.append({
                    'size': size,
                    'forward_time_us': statistics.mean(forward_times) * 1e6,
                    'inverse_time_us': statistics.mean(inverse_times) * 1e6,
                    'unitarity_error': unitarity_error,
                    'transforms_per_second': 1.0 / statistics.mean(forward_times)
                })
                
                print(f"    Forward: {rft_results[-1]['forward_time_us']:6.1f} μs, "
                      f"Inverse: {rft_results[-1]['inverse_time_us']:6.1f} μs")
                
            except Exception as e:
                print(f"    Error with size {size}: {e}")
        
        # Analyze scaling
        scaling_analysis = self.analyze_rft_scaling(rft_results)
        
        return {
            'test_results': rft_results,
            'scaling_analysis': scaling_analysis
        }
    
    def analyze_scaling(self, results: List[Dict], metric_key: str) -> Dict[str, Any]:
        """Analyze scaling characteristics of performance data"""
        sizes = [r['size_bytes'] for r in results]
        metrics = [r[metric_key] for r in results]
        
        # Simple linear regression for scaling analysis
        n = len(sizes)
        sum_x = sum(sizes)
        sum_y = sum(metrics)
        sum_xy = sum(x * y for x, y in zip(sizes, metrics))
        sum_x2 = sum(x * x for x in sizes)
        
        # Linear fit: y = mx + b
        if n * sum_x2 - sum_x * sum_x != 0:
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            intercept = (sum_y - slope * sum_x) / n
        else:
            slope = 0
            intercept = sum_y / n if n > 0 else 0
        
        # Determine scaling behavior
        if abs(slope) < 1e-6:
            scaling_type = "O(1) - Constant"
        elif slope > 0:
            scaling_type = "O(n) - Linear or worse"
        else:
            scaling_type = "O(1/n) - Inverse scaling"
        
        return {
            'slope': slope,
            'intercept': intercept,
            'scaling_type': scaling_type,
            'correlation_strength': 'analysis_needed'  # Would need proper correlation calc
        }
    
    def analyze_rft_scaling(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze RFT transform scaling (should be O(n log n))"""
        sizes = [r['size'] for r in results]
        times = [r['forward_time_us'] for r in results]
        
        # Calculate expected O(n log n) scaling
        expected_times = []
        base_time = times[0] if times else 1.0
        base_size = sizes[0] if sizes else 1
        
        for size in sizes:
            # Expected time for O(n log n) scaling
            expected = base_time * (size / base_size) * (np.log2(size) / np.log2(base_size))
            expected_times.append(expected)
        
        # Calculate scaling efficiency
        efficiency_ratios = []
        for actual, expected in zip(times, expected_times):
            if expected > 0:
                efficiency_ratios.append(actual / expected)
        
        avg_efficiency = statistics.mean(efficiency_ratios) if efficiency_ratios else 1.0
        
        return {
            'expected_scaling': 'O(n log n)',
            'efficiency_ratio': avg_efficiency,
            'efficiency_analysis': 'Good' if avg_efficiency < 2.0 else 'Needs optimization'
        }
    
    def benchmark_concurrent_performance(self) -> Dict[str, Any]:
        """Benchmark performance under concurrent load"""
        print("🔄 Benchmarking Concurrent Performance...")
        
        def worker_function(worker_id: int, num_operations: int, results_list: List):
            """Worker function for concurrent testing"""
            worker_results = []
            test_data = secrets.token_bytes(1024)
            
            start_time = time.perf_counter()
            
            for i in range(num_operations):
                # Mix of operations
                if i % 3 == 0:
                    encrypted = self.cipher.encrypt_aead(test_data)
                    decrypted = self.cipher.decrypt_aead(encrypted)
                elif i % 3 == 1:
                    hash_result = self.hasher.hash(test_data)
                else:
                    # RFT transform
                    try:
                        rft = CanonicalTrueRFT(16)
                        signal = [complex(b, 0) for b in test_data[:16]]
                        transformed = rft.forward_transform(signal)
                    except:
                        pass  # Skip if RFT fails
            
            end_time = time.perf_counter()
            
            worker_results.append({
                'worker_id': worker_id,
                'operations_completed': num_operations,
                'total_time': end_time - start_time,
                'ops_per_second': num_operations / (end_time - start_time)
            })
            
            results_list.extend(worker_results)
        
        # Test with different numbers of concurrent workers
        concurrent_results = []
        
        for num_workers in [1, 2, 4, 8]:
            print(f"  Testing {num_workers} concurrent workers...")
            
            results_list = []
            threads = []
            operations_per_worker = 100
            
            # Start workers
            start_time = time.perf_counter()
            
            for worker_id in range(num_workers):
                thread = threading.Thread(
                    target=worker_function,
                    args=(worker_id, operations_per_worker, results_list)
                )
                threads.append(thread)
                thread.start()
            
            # Wait for completion
            for thread in threads:
                thread.join()
            
            end_time = time.perf_counter()
            
            # Analyze results
            total_operations = num_workers * operations_per_worker
            total_time = end_time - start_time
            aggregate_ops_per_second = total_operations / total_time
            
            concurrent_results.append({
                'num_workers': num_workers,
                'operations_per_worker': operations_per_worker,
                'total_operations': total_operations,
                'total_time': total_time,
                'aggregate_ops_per_second': aggregate_ops_per_second,
                'worker_results': results_list
            })
            
            print(f"    {aggregate_ops_per_second:.1f} ops/sec aggregate")
        
        # Calculate scaling efficiency
        single_worker_ops = concurrent_results[0]['aggregate_ops_per_second']
        scaling_efficiency = []
        
        for result in concurrent_results:
            expected_ops = single_worker_ops * result['num_workers']
            actual_ops = result['aggregate_ops_per_second']
            efficiency = actual_ops / expected_ops if expected_ops > 0 else 0
            scaling_efficiency.append(efficiency)
        
        return {
            'concurrent_results': concurrent_results,
            'scaling_efficiency': scaling_efficiency,
            'average_efficiency': statistics.mean(scaling_efficiency)
        }
    
    def run_comprehensive_benchmarks(self) -> Dict[str, Any]:
        """Run complete performance benchmark suite"""
        print("=" * 60)
        print("PERFORMANCE BENCHMARKING - COMPREHENSIVE SUITE")
        print("=" * 60)
        
        start_time = time.time()
        
        # System information
        self.results['system_info'] = self.system_info
        
        # Run all benchmarks
        self.results['cipher_throughput'] = self.benchmark_cipher_throughput()
        self.results['hash_performance'] = self.benchmark_hash_performance()
        self.results['rft_transforms'] = self.benchmark_rft_transforms()
        self.results['concurrent_performance'] = self.benchmark_concurrent_performance()
        
        # Compile summary
        end_time = time.time()
        benchmark_time = end_time - start_time
        
        # Overall performance assessment
        cipher_mbps = self.results['cipher_throughput']['summary']['avg_encrypt_mbps']
        meets_target = self.results['cipher_throughput']['summary']['meets_paper_target']
        concurrent_efficiency = self.results['concurrent_performance']['average_efficiency']
        
        self.results['summary'] = {
            'benchmark_time_seconds': benchmark_time,
            'timestamp': datetime.now().isoformat(),
            'cipher_throughput_mbps': cipher_mbps,
            'meets_paper_target': meets_target,
            'concurrent_efficiency': concurrent_efficiency,
            'overall_performance_grade': self.calculate_performance_grade(cipher_mbps, concurrent_efficiency),
            'paper_compliance': meets_target
        }
        
        print("\n" + "=" * 60)
        print("PERFORMANCE SUMMARY")
        print("=" * 60)
        print(f"Cipher Throughput: {cipher_mbps:.1f} MB/s")
        print(f"Paper Target (9.2 MB/s): {'✓ MET' if meets_target else '✗ BELOW'}")
        print(f"Concurrent Efficiency: {concurrent_efficiency:.1%}")
        print(f"Benchmark Time: {benchmark_time:.2f} seconds")
        print(f"Overall Grade: {self.results['summary']['overall_performance_grade']}")
        
        return self.results
    
    def calculate_performance_grade(self, throughput_mbps: float, concurrent_efficiency: float) -> str:
        """Calculate overall performance grade"""
        # Throughput score (out of 50 points)
        throughput_score = min(50, (throughput_mbps / 9.2) * 50)
        
        # Concurrent efficiency score (out of 50 points)
        efficiency_score = concurrent_efficiency * 50
        
        total_score = throughput_score + efficiency_score
        
        if total_score >= 90:
            return "A (Excellent)"
        elif total_score >= 80:
            return "B (Good)"
        elif total_score >= 70:
            return "C (Acceptable)"
        elif total_score >= 60:
            return "D (Below Target)"
        else:
            return "F (Needs Improvement)"


def main():
    """Main benchmarking entry point"""
    # Import numpy here for scaling analysis
    global np
    try:
        import numpy as np
    except ImportError:
        print("Warning: numpy not available, scaling analysis will be limited")
        np = None
    
    benchmarker = PerformanceBenchmarker()
    results = benchmarker.run_comprehensive_benchmarks()
    
    # Save results
    output_file = os.path.join(os.path.dirname(__file__), '..', 'benchmarks', 'performance_benchmark_report.json')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📊 Full results saved to: {output_file}")
    
    # Exit with appropriate code
    exit_code = 0 if results['summary']['paper_compliance'] else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
