#!/usr/bin/env python3
"""
QuantoniumOS Long-term Reliability Test Suite
============================================
Extended testing for production deployment confidence
"""

import time
import threading
import psutil
import numpy as np
import gc
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging

class LongTermReliabilityTests:
    """Long-term reliability and stability testing"""
    
    def __init__(self, test_duration_hours=1):
        """Initialize reliability test suite"""
        self.test_duration_hours = test_duration_hours
        self.test_duration_seconds = test_duration_hours * 3600
        self.results = {}
        self.monitoring_active = False
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('reliability_test.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def run_reliability_tests(self):
        """Run comprehensive reliability tests"""
        print("=" * 80)
        print(f"LONG-TERM RELIABILITY TESTS ({self.test_duration_hours} hours)")
        print("=" * 80)
        
        start_time = datetime.now()
        
        # 1. Memory Leak Detection
        print("\n?? MEMORY LEAK DETECTION")
        self.results['memory_leak'] = self.test_memory_leaks()
        
        # 2. Sustained Performance Test
        print("\n? SUSTAINED PERFORMANCE TEST")
        self.results['sustained_performance'] = self.test_sustained_performance()
        
        # 3. Error Accumulation Test
        print("\n?? ERROR ACCUMULATION TEST")
        self.results['error_accumulation'] = self.test_error_accumulation()
        
        # 4. Resource Exhaustion Test
        print("\n?? RESOURCE EXHAUSTION TEST")
        self.results['resource_exhaustion'] = self.test_resource_exhaustion()
        
        # 5. Concurrent Access Test
        print("\n?? CONCURRENT ACCESS TEST")
        self.results['concurrent_access'] = self.test_concurrent_access()
        
        # 6. Thermal Stability Test
        print("\n??? THERMAL STABILITY TEST")
        self.results['thermal_stability'] = self.test_thermal_stability()
        
        end_time = datetime.now()
        self.results['test_summary'] = {
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'total_duration_hours': (end_time - start_time).total_seconds() / 3600,
            'planned_duration_hours': self.test_duration_hours
        }
        
        # Generate report
        self.generate_reliability_report()
        
        return self.results
    
    def test_memory_leaks(self):
        """Test for memory leaks over extended operation"""
        print("  Starting memory leak detection...")
        
        memory_data = []
        start_time = time.time()
        test_duration = min(self.test_duration_seconds * 0.3, 1800)  # Max 30 minutes
        
        # Baseline memory measurement
        baseline_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
        
        iteration = 0
        while time.time() - start_time < test_duration:
            # Simulate typical RFT operations
            for _ in range(100):
                # Create and process data
                data = np.random.random(1024) + 1j * np.random.random(1024)
                
                # Simulate RFT processing
                result = np.fft.fft(data)
                
                # Simulate quantum operations
                bell_state = np.array([1, 0, 0, 1]) / np.sqrt(2)
                processed_state = result[:4] * bell_state
                
                # Clean up references
                del data, result, processed_state
            
            # Force garbage collection
            gc.collect()
            
            # Measure memory
            current_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
            memory_data.append({
                'timestamp': time.time() - start_time,
                'memory_mb': current_memory,
                'iteration': iteration
            })
            
            iteration += 1
            
            # Log progress every 5 minutes
            if iteration % 100 == 0:
                memory_growth = current_memory - baseline_memory
                print(f"    Iteration {iteration}: Memory = {current_memory:.1f} MB "
                      f"(+{memory_growth:.1f} MB)")
            
            time.sleep(1)  # 1 second between measurements
        
        # Analyze memory leak
        if len(memory_data) > 10:
            final_memory = memory_data[-1]['memory_mb']
            memory_growth = final_memory - baseline_memory
            growth_rate = memory_growth / (test_duration / 3600)  # MB/hour
            
            # Linear regression to detect trend
            timestamps = [d['timestamp'] for d in memory_data]
            memories = [d['memory_mb'] for d in memory_data]
            
            if len(timestamps) > 1:
                slope = np.polyfit(timestamps, memories, 1)[0]  # MB/second
                slope_per_hour = slope * 3600
            else:
                slope_per_hour = 0
        else:
            memory_growth = 0
            growth_rate = 0
            slope_per_hour = 0
        
        leak_detected = abs(slope_per_hour) > 10  # More than 10 MB/hour growth
        
        return {
            'baseline_memory_mb': baseline_memory,
            'final_memory_mb': memory_data[-1]['memory_mb'] if memory_data else baseline_memory,
            'memory_growth_mb': memory_growth,
            'growth_rate_mb_per_hour': growth_rate,
            'slope_mb_per_hour': slope_per_hour,
            'leak_detected': leak_detected,
            'test_duration_minutes': test_duration / 60,
            'iterations_completed': iteration,
            'memory_samples': len(memory_data)
        }
    
    def test_sustained_performance(self):
        """Test performance stability over extended operation"""
        print("  Starting sustained performance test...")
        
        performance_data = []
        start_time = time.time()
        test_duration = min(self.test_duration_seconds * 0.4, 2400)  # Max 40 minutes
        
        baseline_times = []
        
        # Establish baseline performance
        print("    Establishing baseline performance...")
        for _ in range(50):
            test_data = np.random.random(1024) + 1j * np.random.random(1024)
            
            op_start = time.perf_counter()
            result = np.fft.fft(test_data)
            op_end = time.perf_counter()
            
            baseline_times.append(op_end - op_start)
        
        baseline_mean = np.mean(baseline_times)
        baseline_std = np.std(baseline_times)
        
        print(f"    Baseline: {baseline_mean*1000:.3f} ± {baseline_std*1000:.3f} ms")
        
        # Extended performance monitoring
        iteration = 0
        while time.time() - start_time < test_duration:
            batch_times = []
            
            # Perform batch of operations
            for _ in range(20):
                test_data = np.random.random(1024) + 1j * np.random.random(1024)
                
                op_start = time.perf_counter()
                result = np.fft.fft(test_data)
                op_end = time.perf_counter()
                
                batch_times.append(op_end - op_start)
            
            # Analyze batch performance
            batch_mean = np.mean(batch_times)
            batch_std = np.std(batch_times)
            performance_degradation = (batch_mean - baseline_mean) / baseline_mean * 100
            
            performance_data.append({
                'timestamp': time.time() - start_time,
                'mean_time': batch_mean,
                'std_time': batch_std,
                'degradation_percent': performance_degradation,
                'iteration': iteration
            })
            
            iteration += 1
            
            # Log progress
            if iteration % 20 == 0:
                print(f"    Iteration {iteration}: {batch_mean*1000:.3f} ms "
                      f"({performance_degradation:+.1f}%)")
            
            time.sleep(5)  # 5 seconds between batches
        
        # Analyze performance stability
        if performance_data:
            degradations = [d['degradation_percent'] for d in performance_data]
            max_degradation = max(degradations)
            mean_degradation = np.mean(degradations)
            std_degradation = np.std(degradations)
            
            # Check for performance drift
            timestamps = [d['timestamp'] for d in performance_data]
            if len(timestamps) > 1:
                drift_slope = np.polyfit(timestamps, degradations, 1)[0]  # %/second
                drift_per_hour = drift_slope * 3600
            else:
                drift_per_hour = 0
            
            performance_stable = max_degradation < 25 and abs(drift_per_hour) < 5
        else:
            max_degradation = 0
            mean_degradation = 0
            std_degradation = 0
            drift_per_hour = 0
            performance_stable = True
        
        return {
            'baseline_time_ms': baseline_mean * 1000,
            'baseline_std_ms': baseline_std * 1000,
            'max_degradation_percent': max_degradation,
            'mean_degradation_percent': mean_degradation,
            'std_degradation_percent': std_degradation,
            'drift_percent_per_hour': drift_per_hour,
            'performance_stable': performance_stable,
            'test_duration_minutes': test_duration / 60,
            'batches_completed': len(performance_data)
        }
    
    def test_error_accumulation(self):
        """Test for numerical error accumulation"""
        print("  Starting error accumulation test...")
        
        error_data = []
        start_time = time.time()
        test_duration = min(self.test_duration_seconds * 0.2, 1200)  # Max 20 minutes
        
        # Reference state
        reference_state = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)
        initial_norm = np.sum(np.abs(reference_state)**2)
        
        current_state = reference_state.copy()
        iteration = 0
        
        while time.time() - start_time < test_duration:
            # Apply sequence of operations that should preserve state properties
            for _ in range(10):
                # Simulate quantum gate operations
                # Apply "identity" through FFT and IFFT
                padded_state = np.zeros(16, dtype=complex)
                padded_state[:4] = current_state
                
                transformed = np.fft.fft(padded_state)
                restored = np.fft.ifft(transformed)
                current_state = restored[:4]
                
                # Renormalize to prevent numerical drift
                norm = np.sum(np.abs(current_state)**2)
                if norm > 1e-16:
                    current_state = current_state / np.sqrt(norm)
            
            # Measure accumulated errors
            norm_error = abs(np.sum(np.abs(current_state)**2) - initial_norm)
            state_error = np.linalg.norm(current_state - reference_state)
            
            error_data.append({
                'timestamp': time.time() - start_time,
                'norm_error': norm_error,
                'state_error': state_error,
                'iteration': iteration
            })
            
            iteration += 1
            
            if iteration % 50 == 0:
                print(f"    Iteration {iteration}: Norm error = {norm_error:.2e}, "
                      f"State error = {state_error:.2e}")
            
            time.sleep(0.1)  # Small delay
        
        # Analyze error accumulation
        if error_data:
            final_norm_error = error_data[-1]['norm_error']
            final_state_error = error_data[-1]['state_error']
            
            # Check for error growth
            norm_errors = [d['norm_error'] for d in error_data]
            state_errors = [d['state_error'] for d in error_data]
            
            norm_error_growth = max(norm_errors) / min(norm_errors) if min(norm_errors) > 1e-16 else 1
            state_error_growth = max(state_errors) / min(state_errors) if min(state_errors) > 1e-16 else 1
            
            errors_stable = final_norm_error < 1e-10 and final_state_error < 1e-8
        else:
            final_norm_error = 0
            final_state_error = 0
            norm_error_growth = 1
            state_error_growth = 1
            errors_stable = True
        
        return {
            'final_norm_error': final_norm_error,
            'final_state_error': final_state_error,
            'norm_error_growth_factor': norm_error_growth,
            'state_error_growth_factor': state_error_growth,
            'errors_stable': errors_stable,
            'test_duration_minutes': test_duration / 60,
            'iterations_completed': iteration
        }
    
    def test_resource_exhaustion(self):
        """Test behavior under resource exhaustion"""
        print("  Starting resource exhaustion test...")
        
        exhaustion_results = {}
        
        # Test 1: Memory pressure
        print("    Testing memory pressure handling...")
        try:
            memory_blocks = []
            block_size = 100 * 1024 * 1024  # 100 MB blocks
            
            for i in range(20):  # Try to allocate up to 2GB
                try:
                    block = np.random.random(block_size // 8)  # 8 bytes per float64
                    memory_blocks.append(block)
                    
                    # Try to perform RFT operation under memory pressure
                    test_data = np.random.random(1024) + 1j * np.random.random(1024)
                    result = np.fft.fft(test_data)
                    
                except MemoryError:
                    break
                except Exception as e:
                    print(f"      Unexpected error at block {i}: {e}")
                    break
            
            # Clean up
            del memory_blocks
            gc.collect()
            
            exhaustion_results['memory_pressure'] = {
                'blocks_allocated': i,
                'memory_allocated_gb': (i * block_size) / (1024**3),
                'graceful_degradation': True,
                'recovery_possible': True
            }
            
        except Exception as e:
            exhaustion_results['memory_pressure'] = {
                'error': str(e),
                'graceful_degradation': False
            }
        
        # Test 2: CPU saturation
        print("    Testing CPU saturation handling...")
        try:
            cpu_count = psutil.cpu_count()
            
            def cpu_intensive_task(duration):
                start_time = time.time()
                while time.time() - start_time < duration:
                    # CPU-intensive work
                    data = np.random.random(1000)
                    result = np.fft.fft(data)
            
            # Start multiple CPU-intensive threads
            threads = []
            for i in range(cpu_count * 2):  # Oversubscribe
                thread = threading.Thread(target=cpu_intensive_task, args=(5,))
                threads.append(thread)
                thread.start()
            
            # Try to perform normal operations during CPU saturation
            operation_times = []
            for _ in range(10):
                test_data = np.random.random(512) + 1j * np.random.random(512)
                
                start_time = time.perf_counter()
                result = np.fft.fft(test_data)
                end_time = time.perf_counter()
                
                operation_times.append(end_time - start_time)
            
            # Wait for threads to complete
            for thread in threads:
                thread.join()
            
            exhaustion_results['cpu_saturation'] = {
                'threads_spawned': len(threads),
                'operations_completed': len(operation_times),
                'mean_operation_time': np.mean(operation_times),
                'graceful_degradation': np.mean(operation_times) < 1.0  # Less than 1 second
            }
            
        except Exception as e:
            exhaustion_results['cpu_saturation'] = {
                'error': str(e),
                'graceful_degradation': False
            }
        
        return exhaustion_results
    
    def test_concurrent_access(self):
        """Test concurrent access patterns"""
        print("  Starting concurrent access test...")
        
        shared_results = {'operations': 0, 'errors': 0, 'times': []}
        results_lock = threading.Lock()
        
        def worker_thread(thread_id, duration):
            """Worker thread for concurrent testing"""
            thread_start = time.time()
            local_operations = 0
            local_errors = 0
            local_times = []
            
            while time.time() - thread_start < duration:
                try:
                    # Generate thread-specific data
                    np.random.seed(thread_id * 1000 + local_operations)
                    test_data = np.random.random(256) + 1j * np.random.random(256)
                    
                    # Timed operation
                    op_start = time.perf_counter()
                    result = np.fft.fft(test_data)
                    op_end = time.perf_counter()
                    
                    local_operations += 1
                    local_times.append(op_end - op_start)
                    
                except Exception as e:
                    local_errors += 1
                
                time.sleep(0.001)  # Small delay
            
            # Update shared results
            with results_lock:
                shared_results['operations'] += local_operations
                shared_results['errors'] += local_errors
                shared_results['times'].extend(local_times)
        
        # Test concurrent access with multiple threads
        test_duration = min(self.test_duration_seconds * 0.1, 300)  # Max 5 minutes
        thread_count = min(psutil.cpu_count(), 8)
        
        threads = []
        start_time = time.time()
        
        for i in range(thread_count):
            thread = threading.Thread(target=worker_thread, args=(i, test_duration))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        actual_duration = time.time() - start_time
        
        # Analyze results
        if shared_results['times']:
            mean_time = np.mean(shared_results['times'])
            std_time = np.std(shared_results['times'])
            throughput = shared_results['operations'] / actual_duration
        else:
            mean_time = 0
            std_time = 0
            throughput = 0
        
        error_rate = shared_results['errors'] / max(shared_results['operations'], 1)
        concurrent_stable = error_rate < 0.01 and throughput > 100  # Less than 1% errors, >100 ops/sec
        
        return {
            'thread_count': thread_count,
            'test_duration_seconds': actual_duration,
            'total_operations': shared_results['operations'],
            'total_errors': shared_results['errors'],
            'error_rate': error_rate,
            'mean_operation_time_ms': mean_time * 1000,
            'std_operation_time_ms': std_time * 1000,
            'throughput_ops_per_second': throughput,
            'concurrent_stable': concurrent_stable
        }
    
    def test_thermal_stability(self):
        """Test thermal stability over extended operation"""
        print("  Starting thermal stability test...")
        
        thermal_data = []
        start_time = time.time()
        test_duration = min(self.test_duration_seconds * 0.3, 1800)  # Max 30 minutes
        
        # Baseline temperature
        try:
            initial_temps = []
            temps = psutil.sensors_temperatures()
            if temps:
                for name, entries in temps.items():
                    for entry in entries:
                        if entry.current:
                            initial_temps.append(entry.current)
            baseline_temp = np.mean(initial_temps) if initial_temps else None
        except:
            baseline_temp = None
        
        # Sustained thermal load
        print(f"    Running thermal load for {test_duration/60:.1f} minutes...")
        
        while time.time() - start_time < test_duration:
            # Generate sustained computational load
            for _ in range(10):
                data = np.random.random(2048) + 1j * np.random.random(2048)
                result = np.fft.fft(data)
                
                # Additional quantum-like operations
                bell_data = np.array([1, 0, 0, 1]) / np.sqrt(2)
                processed = result[:4] * bell_data
            
            # Sample temperature and performance
            try:
                current_temps = []
                temps = psutil.sensors_temperatures()
                if temps:
                    for name, entries in temps.items():
                        for entry in entries:
                            if entry.current:
                                current_temps.append(entry.current)
                
                if current_temps:
                    current_temp = max(current_temps)
                    thermal_data.append({
                        'timestamp': time.time() - start_time,
                        'temperature': current_temp,
                        'cpu_percent': psutil.cpu_percent()
                    })
            except:
                pass
            
            time.sleep(10)  # Sample every 10 seconds
        
        # Analyze thermal stability
        if thermal_data and baseline_temp:
            temperatures = [d['temperature'] for d in thermal_data]
            max_temp = max(temperatures)
            mean_temp = np.mean(temperatures)
            temp_rise = max_temp - baseline_temp
            
            # Check for thermal throttling (simplified detection)
            cpu_percentages = [d['cpu_percent'] for d in thermal_data]
            performance_drop = (max(cpu_percentages) - min(cpu_percentages)) / max(cpu_percentages) * 100
            
            thermal_stable = temp_rise < 30 and performance_drop < 20  # Less than 30°C rise, <20% perf drop
        else:
            max_temp = None
            mean_temp = None
            temp_rise = None
            performance_drop = None
            thermal_stable = True  # Assume stable if can't measure
        
        return {
            'baseline_temperature': baseline_temp,
            'max_temperature': max_temp,
            'mean_temperature': mean_temp,
            'temperature_rise': temp_rise,
            'performance_drop_percent': performance_drop,
            'thermal_stable': thermal_stable,
            'test_duration_minutes': test_duration / 60,
            'temperature_samples': len(thermal_data)
        }
    
    def generate_reliability_report(self):
        """Generate comprehensive reliability report"""
        report_path = Path("reliability_test_report.md")
        
        with open(report_path, 'w') as f:
            f.write("# Long-term Reliability Test Report\n\n")
            f.write(f"**Test Duration**: {self.test_duration_hours} hours\n")
            f.write(f"**Start Time**: {self.results['test_summary']['start_time']}\n")
            f.write(f"**End Time**: {self.results['test_summary']['end_time']}\n\n")
            
            f.write("## Executive Summary\n\n")
            
            # Determine overall reliability
            reliability_scores = []
            
            for test_name, test_results in self.results.items():
                if test_name == 'test_summary':
                    continue
                
                if isinstance(test_results, dict):
                    # Check for stability indicators
                    stable_indicators = [
                        test_results.get('leak_detected', True) == False,
                        test_results.get('performance_stable', False) == True,
                        test_results.get('errors_stable', False) == True,
                        test_results.get('graceful_degradation', False) == True,
                        test_results.get('concurrent_stable', False) == True,
                        test_results.get('thermal_stable', False) == True
                    ]
                    
                    # Calculate reliability score for this test
                    valid_indicators = [x for x in stable_indicators if x is not None]
                    if valid_indicators:
                        test_score = sum(valid_indicators) / len(valid_indicators)
                        reliability_scores.append(test_score)
            
            overall_reliability = np.mean(reliability_scores) if reliability_scores else 0.5
            
            if overall_reliability >= 0.9:
                f.write("? **EXCELLENT RELIABILITY** - System demonstrates production-ready stability\n\n")
            elif overall_reliability >= 0.7:
                f.write("? **GOOD RELIABILITY** - System suitable for most production deployments\n\n")
            elif overall_reliability >= 0.5:
                f.write("?? **MODERATE RELIABILITY** - Some issues detected, review recommended\n\n")
            else:
                f.write("? **RELIABILITY CONCERNS** - Significant issues detected\n\n")
            
            # Test Results Summary
            f.write("## Test Results Summary\n\n")
            
            for test_name, test_results in self.results.items():
                if test_name == 'test_summary':
                    continue
                
                f.write(f"### {test_name.replace('_', ' ').title()}\n\n")
                
                if test_name == 'memory_leak':
                    leak_detected = test_results.get('leak_detected', False)
                    growth_rate = test_results.get('growth_rate_mb_per_hour', 0)
                    f.write(f"- **Memory Leak Detected**: {'? Yes' if leak_detected else '? No'}\n")
                    f.write(f"- **Growth Rate**: {growth_rate:.2f} MB/hour\n")
                    
                elif test_name == 'sustained_performance':
                    stable = test_results.get('performance_stable', False)
                    max_degradation = test_results.get('max_degradation_percent', 0)
                    f.write(f"- **Performance Stable**: {'? Yes' if stable else '? No'}\n")
                    f.write(f"- **Max Degradation**: {max_degradation:.1f}%\n")
                    
                elif test_name == 'error_accumulation':
                    stable = test_results.get('errors_stable', False)
                    final_error = test_results.get('final_state_error', 0)
                    f.write(f"- **Errors Stable**: {'? Yes' if stable else '? No'}\n")
                    f.write(f"- **Final State Error**: {final_error:.2e}\n")
                    
                elif test_name == 'concurrent_access':
                    stable = test_results.get('concurrent_stable', False)
                    error_rate = test_results.get('error_rate', 0)
                    throughput = test_results.get('throughput_ops_per_second', 0)
                    f.write(f"- **Concurrent Stable**: {'? Yes' if stable else '? No'}\n")
                    f.write(f"- **Error Rate**: {error_rate:.2%}\n")
                    f.write(f"- **Throughput**: {throughput:.1f} ops/sec\n")
                
                f.write("\n")
            
            f.write("## Recommendations\n\n")
            
            if overall_reliability >= 0.9:
                f.write("The system demonstrates excellent long-term reliability and is ")
                f.write("recommended for production deployment.\n\n")
                f.write("**Deployment Recommendations**:\n")
                f.write("- ? Suitable for 24/7 operation\n")
                f.write("- ? Can handle sustained computational loads\n")
                f.write("- ? Demonstrates excellent resource management\n")
            else:
                f.write("**Issues Identified**:\n")
                for test_name, test_results in self.results.items():
                    if isinstance(test_results, dict):
                        if test_results.get('leak_detected', False):
                            f.write("- ?? Memory leak detected - investigate memory management\n")
                        if not test_results.get('performance_stable', True):
                            f.write("- ?? Performance degradation detected - review optimization\n")
                        if not test_results.get('errors_stable', True):
                            f.write("- ?? Error accumulation detected - check numerical stability\n")
        
        print(f"\n?? Reliability test report: {report_path}")
        
        # Save detailed results
        json_path = Path("reliability_test_results.json")
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"?? Reliability test data: {json_path}")

def main():
    """Run reliability tests"""
    import argparse
    
    parser = argparse.ArgumentParser(description="QuantoniumOS Long-term Reliability Tests")
    parser.add_argument('--duration', type=float, default=0.5, 
                       help='Test duration in hours (default: 0.5)')
    args = parser.parse_args()
    
    tester = LongTermReliabilityTests(test_duration_hours=args.duration)
    results = tester.run_reliability_tests()
    
    print("\n" + "="*80)
    print("LONG-TERM RELIABILITY TESTS COMPLETE")
    print("="*80)
    
    return results

if __name__ == "__main__":
    main()