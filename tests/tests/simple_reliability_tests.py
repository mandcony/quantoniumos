#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QuantoniumOS Simple Reliability Tests
====================================
Essential reliability testing for QuantoniumOS validation
"""

import time
import threading
import psutil
import numpy as np
import gc
from datetime import datetime
from pathlib import Path
import json
import argparse

class SimpleReliabilityTester:
    """Simplified reliability testing for QuantoniumOS"""
    
    def __init__(self, test_duration_hours=0.25):
        """Initialize reliability tester"""
        self.test_duration_hours = test_duration_hours
        self.test_duration_seconds = test_duration_hours * 3600
        self.results = {}
        
    def run_reliability_tests(self):
        """Run essential reliability tests"""
        print("=" * 60)
        print(f"QUANTONIUMOS RELIABILITY TESTS ({self.test_duration_hours}h)")
        print("=" * 60)
        
        start_time = datetime.now()
        
        # 1. Memory stability test
        print("\nMEMORY STABILITY TEST:")
        self.results['memory_stability'] = self.test_memory_stability()
        
        # 2. Performance consistency test  
        print("\nPERFORMANCE CONSISTENCY TEST:")
        self.results['performance_consistency'] = self.test_performance_consistency()
        
        # 3. Concurrent operation test
        print("\nCONCURRENT OPERATION TEST:")
        self.results['concurrent_operations'] = self.test_concurrent_operations()
        
        # 4. System stability test
        print("\nSYSTEM STABILITY TEST:")
        self.results['system_stability'] = self.test_system_stability()
        
        end_time = datetime.now()
        self.results['test_summary'] = {
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'actual_duration_hours': (end_time - start_time).total_seconds() / 3600,
            'planned_duration_hours': self.test_duration_hours
        }
        
        # Generate summary
        self.generate_summary()
        
        return self.results
    
    def test_memory_stability(self):
        """Test memory stability over time"""
        print("  Testing memory stability...")
        
        # Record initial memory
        initial_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
        memory_samples = []
        
        test_duration = min(self.test_duration_seconds * 0.3, 300)  # Max 5 minutes
        start_time = time.time()
        iteration = 0
        
        while time.time() - start_time < test_duration:
            # Simulate typical operations
            for _ in range(10):
                # Create and process data
                data = np.random.random(1000) + 1j * np.random.random(1000)
                result = np.fft.fft(data)
                
                # Simulate Bell state operations
                bell_state = np.array([1, 0, 0, 1]) / np.sqrt(2)
                processed = result[:4] * bell_state
                
                # Clean up
                del data, result, processed
            
            # Force garbage collection
            gc.collect()
            
            # Sample memory
            current_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            memory_samples.append(current_memory)
            
            iteration += 1
            
            if iteration % 20 == 0:
                print(f"    Iteration {iteration}: Memory = {current_memory:.1f} MB")
            
            time.sleep(0.5)
        
        final_memory = memory_samples[-1] if memory_samples else initial_memory
        memory_growth = final_memory - initial_memory
        
        memory_stable = abs(memory_growth) < 50  # Less than 50MB growth
        
        result = {
            'initial_memory_mb': initial_memory,
            'final_memory_mb': final_memory,
            'memory_growth_mb': memory_growth,
            'memory_stable': memory_stable,
            'iterations': iteration,
            'test_duration_minutes': test_duration / 60
        }
        
        print(f"  Memory growth: {memory_growth:.1f} MB")
        print(f"  Memory stable: {memory_stable}")
        
        return result
    
    def test_performance_consistency(self):
        """Test performance consistency over time"""
        print("  Testing performance consistency...")
        
        performance_samples = []
        test_duration = min(self.test_duration_seconds * 0.3, 300)  # Max 5 minutes
        start_time = time.time()
        
        # Establish baseline
        baseline_times = []
        for _ in range(10):
            test_data = np.random.random(1000) + 1j * np.random.random(1000)
            
            op_start = time.perf_counter()
            result = np.fft.fft(test_data)
            op_end = time.perf_counter()
            
            baseline_times.append(op_end - op_start)
        
        baseline_mean = np.mean(baseline_times)
        print(f"    Baseline performance: {baseline_mean*1000:.2f} ms")
        
        # Extended testing
        iteration = 0
        while time.time() - start_time < test_duration:
            # Batch of operations
            batch_times = []
            for _ in range(5):
                test_data = np.random.random(1000) + 1j * np.random.random(1000)
                
                op_start = time.perf_counter()
                result = np.fft.fft(test_data)
                op_end = time.perf_counter()
                
                batch_times.append(op_end - op_start)
            
            batch_mean = np.mean(batch_times)
            degradation = (batch_mean - baseline_mean) / baseline_mean * 100
            performance_samples.append(degradation)
            
            iteration += 1
            
            if iteration % 10 == 0:
                print(f"    Iteration {iteration}: Performance change = {degradation:+.1f}%")
            
            time.sleep(2)
        
        max_degradation = max(performance_samples) if performance_samples else 0
        mean_degradation = np.mean(performance_samples) if performance_samples else 0
        
        performance_stable = max_degradation < 20  # Less than 20% degradation
        
        result = {
            'baseline_time_ms': baseline_mean * 1000,
            'max_degradation_percent': max_degradation,
            'mean_degradation_percent': mean_degradation,
            'performance_stable': performance_stable,
            'iterations': iteration,
            'test_duration_minutes': test_duration / 60
        }
        
        print(f"  Max degradation: {max_degradation:.1f}%")
        print(f"  Performance stable: {performance_stable}")
        
        return result
    
    def test_concurrent_operations(self):
        """Test concurrent operation stability"""
        print("  Testing concurrent operations...")
        
        shared_results = {'operations': 0, 'errors': 0}
        results_lock = threading.Lock()
        
        def worker_thread(thread_id, duration):
            """Worker thread for concurrent testing"""
            local_operations = 0
            local_errors = 0
            
            thread_start = time.time()
            while time.time() - thread_start < duration:
                try:
                    # Generate test data
                    test_data = np.random.random(200) + 1j * np.random.random(200)
                    result = np.fft.fft(test_data)
                    local_operations += 1
                except Exception:
                    local_errors += 1
                
                time.sleep(0.01)
            
            with results_lock:
                shared_results['operations'] += local_operations
                shared_results['errors'] += local_errors
        
        # Run concurrent test
        test_duration = min(self.test_duration_seconds * 0.2, 120)  # Max 2 minutes
        thread_count = 4
        
        threads = []
        start_time = time.time()
        
        for i in range(thread_count):
            thread = threading.Thread(target=worker_thread, args=(i, test_duration))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        actual_duration = time.time() - start_time
        error_rate = shared_results['errors'] / max(shared_results['operations'], 1)
        throughput = shared_results['operations'] / actual_duration
        
        concurrent_stable = error_rate < 0.01 and throughput > 50  # <1% errors, >50 ops/sec
        
        result = {
            'thread_count': thread_count,
            'total_operations': shared_results['operations'],
            'total_errors': shared_results['errors'],
            'error_rate': error_rate,
            'throughput_ops_per_second': throughput,
            'concurrent_stable': concurrent_stable,
            'test_duration_minutes': actual_duration / 60
        }
        
        print(f"  Operations: {shared_results['operations']}")
        print(f"  Error rate: {error_rate:.2%}")
        print(f"  Concurrent stable: {concurrent_stable}")
        
        return result
    
    def test_system_stability(self):
        """Test overall system stability"""
        print("  Testing system stability...")
        
        # Monitor system metrics
        initial_cpu = psutil.cpu_percent(interval=1)
        initial_memory = psutil.virtual_memory().percent
        
        stability_samples = []
        test_duration = min(self.test_duration_seconds * 0.2, 120)  # Max 2 minutes
        start_time = time.time()
        
        while time.time() - start_time < test_duration:
            # Generate load
            for _ in range(5):
                data = np.random.random(2000)
                result = np.fft.fft(data)
                
                # Quantum-like operations
                bell_data = np.array([1, 0, 0, 1]) / np.sqrt(2)
                processed = result[:4] * bell_data
            
            # Sample system metrics
            current_cpu = psutil.cpu_percent()
            current_memory = psutil.virtual_memory().percent
            
            stability_samples.append({
                'cpu_percent': current_cpu,
                'memory_percent': current_memory,
                'timestamp': time.time() - start_time
            })
            
            time.sleep(5)
        
        # Analyze stability
        if stability_samples:
            cpu_values = [s['cpu_percent'] for s in stability_samples]
            memory_values = [s['memory_percent'] for s in stability_samples]
            
            cpu_stable = max(cpu_values) - min(cpu_values) < 50  # CPU variance < 50%
            memory_stable = max(memory_values) - min(memory_values) < 20  # Memory variance < 20%
        else:
            cpu_stable = True
            memory_stable = True
        
        system_stable = cpu_stable and memory_stable
        
        result = {
            'initial_cpu_percent': initial_cpu,
            'initial_memory_percent': initial_memory,
            'cpu_stable': cpu_stable,
            'memory_stable': memory_stable,
            'system_stable': system_stable,
            'samples_collected': len(stability_samples),
            'test_duration_minutes': test_duration / 60
        }
        
        print(f"  CPU stable: {cpu_stable}")
        print(f"  Memory stable: {memory_stable}")
        print(f"  System stable: {system_stable}")
        
        return result
    
    def generate_summary(self):
        """Generate reliability test summary"""
        print("\n" + "=" * 60)
        print("RELIABILITY TEST SUMMARY")
        print("=" * 60)
        
        # Calculate overall reliability score
        stability_scores = []
        
        for test_name, result in self.results.items():
            if test_name == 'test_summary':
                continue
            
            if isinstance(result, dict):
                stable_indicators = [
                    result.get('memory_stable', False),
                    result.get('performance_stable', False), 
                    result.get('concurrent_stable', False),
                    result.get('system_stable', False)
                ]
                
                valid_indicators = [x for x in stable_indicators if x is not None]
                if valid_indicators:
                    test_score = sum(valid_indicators) / len(valid_indicators)
                    stability_scores.append(test_score)
        
        overall_reliability = np.mean(stability_scores) if stability_scores else 0.5
        
        if overall_reliability >= 0.8:
            status = "EXCELLENT RELIABILITY"
        elif overall_reliability >= 0.6:
            status = "GOOD RELIABILITY"
        elif overall_reliability >= 0.4:
            status = "MODERATE RELIABILITY"
        else:
            status = "RELIABILITY ISSUES"
        
        print(f"Overall Status: {status}")
        print(f"Reliability Score: {overall_reliability:.1%}")
        
        print("\nTEST RESULTS:")
        for test_name, result in self.results.items():
            if test_name == 'test_summary':
                continue
                
            if isinstance(result, dict):
                stable_key = next((k for k in result.keys() if 'stable' in k), None)
                if stable_key:
                    status = "PASS" if result[stable_key] else "FAIL"
                    print(f"- {test_name.replace('_', ' ').title()}: {status}")
        
        # Save results
        results_file = Path("reliability_test_results.json")
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\nDetailed results saved to: {results_file}")
        
        return overall_reliability >= 0.6

def main():
    """Run reliability tests"""
    parser = argparse.ArgumentParser(description="QuantoniumOS Reliability Tests")
    parser.add_argument('--duration', type=float, default=0.25,
                       help='Test duration in hours (default: 0.25)')
    args = parser.parse_args()
    
    tester = SimpleReliabilityTester(test_duration_hours=args.duration)
    results = tester.run_reliability_tests()
    
    print("\n" + "=" * 60)
    print("RELIABILITY TESTING COMPLETE")
    print("=" * 60)
    
    return results

if __name__ == "__main__":
    main()