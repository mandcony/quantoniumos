"""
QuantoniumOS Assembly Performance Analyzer
==========================================
Analyzes and profiles assembly code for optimization opportunities
"""

import os
import sys
import time
import psutil
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import subprocess
import json

class AssemblyPerformanceAnalyzer:
    """Analyzes assembly code performance and suggests optimizations."""
    
    def __init__(self):
        """Initialize the performance analyzer."""
        self.results = {}
        self.baseline_metrics = {}
        self.optimization_suggestions = []
        
    def analyze_rft_performance(self, sizes: List[int] = None) -> Dict:
        """Analyze RFT transform performance across different sizes.
        
        Args:
            sizes: List of transform sizes to test
            
        Returns:
            Performance analysis results
        """
        if sizes is None:
            sizes = [16, 32, 64, 128, 256, 512, 1024]
        
        print("=== RFT Performance Analysis ===")
        results = {}
        
        for size in sizes:
            print(f"Testing size {size}...")
            
            # Test optimized version if available
            optimized_time = self._test_optimized_rft(size)
            
            # Test fallback version
            fallback_time = self._test_fallback_rft(size)
            
            # Calculate speedup
            speedup = fallback_time / optimized_time if optimized_time > 0 else 0
            
            results[size] = {
                'optimized_time': optimized_time,
                'fallback_time': fallback_time,
                'speedup': speedup,
                'efficiency': 100 * (1 - optimized_time / fallback_time) if fallback_time > 0 else 0
            }
            
            print(f"  Optimized: {optimized_time:.4f}s")
            print(f"  Fallback:  {fallback_time:.4f}s")
            print(f"  Speedup:   {speedup:.2f}x")
        
        return results
    
    def _test_optimized_rft(self, size: int, iterations: int = 100) -> float:
        """Test optimized RFT implementation."""
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ASSEMBLY", "python_bindings"))
            from optimized_rft import OptimizedRFTProcessor
            
            processor = OptimizedRFTProcessor(size)
            test_data = np.random.random(size) + 1j * np.random.random(size)
            
            # Warm up
            for _ in range(10):
                processor.forward_optimized(test_data)
            
            # Measure performance
            start_time = time.time()
            for _ in range(iterations):
                result = processor.forward_optimized(test_data)
            end_time = time.time()
            
            return (end_time - start_time) / iterations
            
        except Exception as e:
            print(f"Optimized RFT test failed: {e}")
            return 0.0
    
    def _test_fallback_rft(self, size: int, iterations: int = 100) -> float:
        """Test fallback RFT implementation."""
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ASSEMBLY", "python_bindings"))
            from unitary_rft import RFTProcessor
            
            processor = RFTProcessor(size)
            test_data = np.random.random(size) + 1j * np.random.random(size)
            
            # Warm up
            for _ in range(10):
                processor.process_quantum_field(test_data)
            
            # Measure performance
            start_time = time.time()
            for _ in range(iterations):
                result = processor.process_quantum_field(test_data)
            end_time = time.time()
            
            return (end_time - start_time) / iterations
            
        except Exception as e:
            print(f"Fallback RFT test failed: {e}")
            return 1.0  # Return 1.0 as baseline
    
    def analyze_memory_usage(self) -> Dict:
        """Analyze memory usage patterns."""
        print("\n=== Memory Usage Analysis ===")
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Test memory usage with different sizes
        memory_usage = {}
        sizes = [64, 128, 256, 512, 1024]
        
        for size in sizes:
            try:
                # Load RFT processor
                sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ASSEMBLY", "python_bindings"))
                from optimized_rft import OptimizedRFTProcessor
                
                processor = OptimizedRFTProcessor(size)
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_delta = current_memory - initial_memory
                
                memory_usage[size] = {
                    'total_memory_mb': current_memory,
                    'memory_delta_mb': memory_delta,
                    'memory_per_element': memory_delta * 1024 * 1024 / size if size > 0 else 0
                }
                
                print(f"Size {size}: {memory_delta:.2f} MB delta, {memory_usage[size]['memory_per_element']:.1f} bytes/element")
                
                # Clean up
                del processor
                
            except Exception as e:
                print(f"Memory test failed for size {size}: {e}")
        
        return memory_usage
    
    def analyze_cpu_utilization(self) -> Dict:
        """Analyze CPU utilization and identify bottlenecks."""
        print("\n=== CPU Utilization Analysis ===")
        
        # Get CPU info
        cpu_info = {
            'cpu_count': psutil.cpu_count(logical=False),
            'logical_cpu_count': psutil.cpu_count(logical=True),
            'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
        }
        
        print(f"Physical CPUs: {cpu_info['cpu_count']}")
        print(f"Logical CPUs: {cpu_info['logical_cpu_count']}")
        if cpu_info['cpu_freq']:
            print(f"CPU Frequency: {cpu_info['cpu_freq']['current']:.0f} MHz")
        
        # Monitor CPU usage during RFT operations
        cpu_usage_data = []
        
        def monitor_cpu():
            for _ in range(50):  # Monitor for 5 seconds
                cpu_usage_data.append(psutil.cpu_percent(interval=0.1))
        
        # Start monitoring in background
        import threading
        monitor_thread = threading.Thread(target=monitor_cpu)
        monitor_thread.start()
        
        # Perform intensive RFT operations
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ASSEMBLY", "python_bindings"))
            from optimized_rft import OptimizedRFTProcessor
            
            processor = OptimizedRFTProcessor(512)
            test_data = np.random.random(512) + 1j * np.random.random(512)
            
            # Intensive computation
            for _ in range(100):
                result = processor.forward_optimized(test_data)
        
        except Exception as e:
            print(f"CPU utilization test failed: {e}")
        
        monitor_thread.join()
        
        cpu_stats = {
            'max_usage': max(cpu_usage_data) if cpu_usage_data else 0,
            'avg_usage': sum(cpu_usage_data) / len(cpu_usage_data) if cpu_usage_data else 0,
            'min_usage': min(cpu_usage_data) if cpu_usage_data else 0,
            'usage_data': cpu_usage_data
        }
        
        print(f"CPU Usage - Max: {cpu_stats['max_usage']:.1f}%, Avg: {cpu_stats['avg_usage']:.1f}%")
        
        return {**cpu_info, 'usage_stats': cpu_stats}
    
    def detect_simd_capabilities(self) -> Dict:
        """Detect available SIMD instruction sets."""
        print("\n=== SIMD Capability Detection ===")
        
        capabilities = {
            'sse2': False,
            'sse3': False,
            'sse4_1': False,
            'sse4_2': False,
            'avx': False,
            'avx2': False,
            'avx512f': False,
            'avx512dq': False
        }
        
        try:
            # Try to get CPU info from /proc/cpuinfo on Linux
            if os.path.exists('/proc/cpuinfo'):
                with open('/proc/cpuinfo', 'r') as f:
                    cpuinfo = f.read().lower()
                    
                for cap in capabilities.keys():
                    if cap in cpuinfo:
                        capabilities[cap] = True
            
            # Alternative method using cpuinfo library
            try:
                import cpuinfo
                info = cpuinfo.get_cpu_info()
                flags = [f.lower() for f in info.get('flags', [])]
                
                for cap in capabilities.keys():
                    if cap in flags:
                        capabilities[cap] = True
                        
            except ImportError:
                print("cpuinfo library not available")
        
        except Exception as e:
            print(f"SIMD detection failed: {e}")
        
        # Print detected capabilities
        print("Detected SIMD support:")
        for cap, supported in capabilities.items():
            status = "?" if supported else "?"
            print(f"  {cap.upper()}: {status}")
        
        return capabilities
    
    def generate_optimization_suggestions(self, performance_data: Dict) -> List[str]:
        """Generate optimization suggestions based on analysis."""
        suggestions = []
        
        # Analyze performance data
        if 'rft_performance' in performance_data:
            rft_data = performance_data['rft_performance']
            
            # Check for poor speedups
            poor_speedups = [size for size, data in rft_data.items() 
                           if data['speedup'] < 2.0]
            
            if poor_speedups:
                suggestions.append(
                    f"Low speedup detected for sizes {poor_speedups}. "
                    "Consider implementing size-specific optimizations."
                )
            
            # Check for memory efficiency
            if 'memory_usage' in performance_data:
                mem_data = performance_data['memory_usage']
                high_memory = [size for size, data in mem_data.items()
                             if data['memory_per_element'] > 32]  # 32 bytes per element
                
                if high_memory:
                    suggestions.append(
                        f"High memory usage for sizes {high_memory}. "
                        "Consider memory pooling or in-place operations."
                    )
        
        # Check SIMD utilization
        if 'simd_capabilities' in performance_data:
            simd_data = performance_data['simd_capabilities']
            
            if simd_data.get('avx512f', False):
                suggestions.append(
                    "AVX-512 detected but may not be fully utilized. "
                    "Implement 512-bit vector operations for maximum performance."
                )
            elif simd_data.get('avx2', False):
                suggestions.append(
                    "AVX2 detected. Ensure 256-bit vector operations are used."
                )
            elif simd_data.get('avx', False):
                suggestions.append(
                    "AVX detected. Consider upgrading to AVX2 optimizations."
                )
        
        # Check CPU utilization
        if 'cpu_utilization' in performance_data:
            cpu_data = performance_data['cpu_utilization']
            
            if cpu_data['usage_stats']['max_usage'] < 50:
                suggestions.append(
                    "Low CPU utilization detected. Consider parallel processing "
                    "or more aggressive optimizations."
                )
            
            logical_cpus = cpu_data.get('logical_cpu_count', 1)
            if logical_cpus > 4:
                suggestions.append(
                    f"Multi-core system detected ({logical_cpus} cores). "
                    "Implement parallel RFT processing for large transforms."
                )
        
        return suggestions
    
    def run_complete_analysis(self) -> Dict:
        """Run complete performance analysis."""
        print("=" * 60)
        print("QuantoniumOS Assembly Performance Analysis")
        print("=" * 60)
        
        results = {}
        
        # 1. RFT Performance Analysis
        results['rft_performance'] = self.analyze_rft_performance()
        
        # 2. Memory Usage Analysis
        results['memory_usage'] = self.analyze_memory_usage()
        
        # 3. CPU Utilization Analysis
        results['cpu_utilization'] = self.analyze_cpu_utilization()
        
        # 4. SIMD Capability Detection
        results['simd_capabilities'] = self.detect_simd_capabilities()
        
        # 5. Generate Optimization Suggestions
        suggestions = self.generate_optimization_suggestions(results)
        results['optimization_suggestions'] = suggestions
        
        # Print summary
        print("\n" + "=" * 60)
        print("OPTIMIZATION RECOMMENDATIONS")
        print("=" * 60)
        
        if suggestions:
            for i, suggestion in enumerate(suggestions, 1):
                print(f"{i}. {suggestion}")
        else:
            print("No specific optimization recommendations at this time.")
        
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)
        
        return results
    
    def export_results(self, results: Dict, filename: str = "assembly_analysis.json"):
        """Export analysis results to JSON file."""
        try:
            # Convert numpy arrays to lists for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {key: convert_numpy(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                else:
                    return obj
            
            converted_results = convert_numpy(results)
            
            output_path = os.path.join(os.path.dirname(__file__), "..", "ASSEMBLY", filename)
            with open(output_path, 'w') as f:
                json.dump(converted_results, f, indent=2)
            
            print(f"Results exported to: {output_path}")
            
        except Exception as e:
            print(f"Failed to export results: {e}")

# Performance testing and optimization entry point
if __name__ == "__main__":
    analyzer = AssemblyPerformanceAnalyzer()
    
    # Run complete analysis
    results = analyzer.run_complete_analysis()
    
    # Export results
    analyzer.export_results(results)
    
    # Show performance summary
    print(f"\nPerformance Summary:")
    if 'rft_performance' in results:
        best_speedup = max(data['speedup'] for data in results['rft_performance'].values())
        avg_speedup = sum(data['speedup'] for data in results['rft_performance'].values()) / len(results['rft_performance'])
        print(f"  Best RFT Speedup: {best_speedup:.2f}x")
        print(f"  Average RFT Speedup: {avg_speedup:.2f}x")
    
    if 'simd_capabilities' in results:
        simd_support = sum(results['simd_capabilities'].values())
        print(f"  SIMD Instructions Supported: {simd_support}")
    
    print(f"  Optimization Suggestions: {len(results.get('optimization_suggestions', []))}")