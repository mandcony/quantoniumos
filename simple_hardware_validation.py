#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QuantoniumOS Hardware Validation Tests - Simplified
==================================================
Essential hardware compatibility tests without complex dependencies
"""

import platform
import psutil
import numpy as np
import time
from pathlib import Path
import json

class SimpleHardwareValidator:
    """Simplified hardware validation for QuantoniumOS"""
    
    def __init__(self):
        self.results = {}
        
    def run_validation(self):
        """Run essential hardware validation tests"""
        print("=" * 60)
        print("QUANTONIUMOS HARDWARE VALIDATION")
        print("=" * 60)
        
        # System compatibility check
        print("\nSYSTEM COMPATIBILITY:")
        self.results['system'] = self.check_system_compatibility()
        
        # CPU validation
        print("\nCPU VALIDATION:")
        self.results['cpu'] = self.validate_cpu()
        
        # Memory validation  
        print("\nMEMORY VALIDATION:")
        self.results['memory'] = self.validate_memory()
        
        # Performance validation
        print("\nPERFORMANCE VALIDATION:")
        self.results['performance'] = self.validate_performance()
        
        # Generate summary
        self.generate_summary()
        
        return self.results
    
    def check_system_compatibility(self):
        """Check basic system compatibility"""
        system_info = {
            'os': platform.system(),
            'architecture': platform.machine(),
            'python_version': platform.python_version(),
            'x86_64_compatible': platform.machine().lower() in ['x86_64', 'amd64'],
            'quantoniumos_compatible': True
        }
        
        print(f"  Operating System: {system_info['os']}")
        print(f"  Architecture: {system_info['architecture']}")
        print(f"  Python Version: {system_info['python_version']}")
        print(f"  x86-64 Compatible: {system_info['x86_64_compatible']}")
        
        return system_info
    
    def validate_cpu(self):
        """Validate CPU capabilities"""
        cpu_info = {
            'physical_cores': psutil.cpu_count(logical=False),
            'logical_cores': psutil.cpu_count(logical=True),
            'max_frequency': None,
            'current_frequency': None,
            'simd_available': True  # Assume true for x86-64
        }
        
        try:
            freq = psutil.cpu_freq()
            if freq:
                cpu_info['max_frequency'] = freq.max
                cpu_info['current_frequency'] = freq.current
        except:
            pass
        
        print(f"  Physical Cores: {cpu_info['physical_cores']}")
        print(f"  Logical Cores: {cpu_info['logical_cores']}")
        if cpu_info['max_frequency']:
            print(f"  Max Frequency: {cpu_info['max_frequency']:.0f} MHz")
        
        return cpu_info
    
    def validate_memory(self):
        """Validate memory configuration"""
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        memory_info = {
            'total_gb': round(memory.total / (1024**3), 2),
            'available_gb': round(memory.available / (1024**3), 2),
            'used_percent': memory.percent,
            'swap_total_gb': round(swap.total / (1024**3), 2),
            'sufficient_for_quantoniumos': memory.total >= 4 * (1024**3)  # 4GB minimum
        }
        
        print(f"  Total RAM: {memory_info['total_gb']} GB")
        print(f"  Available RAM: {memory_info['available_gb']} GB")
        print(f"  Memory Usage: {memory_info['used_percent']:.1f}%")
        print(f"  Sufficient Memory: {memory_info['sufficient_for_quantoniumos']}")
        
        return memory_info
    
    def validate_performance(self):
        """Validate basic performance characteristics"""
        print("  Running performance tests...")
        
        # CPU performance test
        start_time = time.perf_counter()
        for _ in range(1000000):
            x = 1.0 + 1.0
        cpu_test_time = time.perf_counter() - start_time
        
        # Memory bandwidth test (simplified)
        test_size = 1000000
        data = np.random.random(test_size).astype(np.float32)
        
        start_time = time.perf_counter()
        result = np.sum(data)
        memory_test_time = time.perf_counter() - start_time
        
        bandwidth_mb_s = (test_size * 4) / (memory_test_time * 1024 * 1024)
        
        # SIMD test (using NumPy vectorization as proxy)
        a = np.random.random(10000).astype(np.float32)
        b = np.random.random(10000).astype(np.float32)
        
        # Scalar-like
        start_time = time.perf_counter()
        result_scalar = np.zeros(10000, dtype=np.float32)
        for i in range(0, 10000, 100):
            result_scalar[i:i+100] = a[i:i+100] + b[i:i+100]
        scalar_time = time.perf_counter() - start_time
        
        # Vectorized
        start_time = time.perf_counter()
        result_vector = a + b
        vector_time = time.perf_counter() - start_time
        
        simd_speedup = scalar_time / vector_time if vector_time > 0 else 1.0
        
        performance_info = {
            'cpu_test_time': cpu_test_time,
            'memory_bandwidth_mb_s': bandwidth_mb_s,
            'simd_speedup': simd_speedup,
            'performance_adequate': bandwidth_mb_s > 1000 and simd_speedup > 5
        }
        
        print(f"  Memory Bandwidth: {bandwidth_mb_s:.1f} MB/s")
        print(f"  SIMD Speedup: {simd_speedup:.1f}x")
        print(f"  Performance Adequate: {performance_info['performance_adequate']}")
        
        return performance_info
    
    def generate_summary(self):
        """Generate validation summary"""
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        
        # Check overall compatibility
        compatible = (
            self.results['system']['x86_64_compatible'] and
            self.results['memory']['sufficient_for_quantoniumos'] and
            self.results['performance']['performance_adequate']
        )
        
        status = "COMPATIBLE" if compatible else "ISSUES DETECTED"
        print(f"QuantoniumOS Compatibility: {status}")
        
        print("\nDETAILS:")
        print(f"- Architecture: {'OK' if self.results['system']['x86_64_compatible'] else 'FAIL'}")
        print(f"- Memory: {'OK' if self.results['memory']['sufficient_for_quantoniumos'] else 'FAIL'}")
        print(f"- Performance: {'OK' if self.results['performance']['performance_adequate'] else 'FAIL'}")
        
        # Save results
        results_file = Path("hardware_validation_results.json")
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\nDetailed results saved to: {results_file}")
        
        return compatible

def main():
    """Run hardware validation"""
    validator = SimpleHardwareValidator()
    results = validator.run_validation()
    
    print("\n" + "=" * 60)
    print("HARDWARE VALIDATION COMPLETE")
    print("=" * 60)
    
    return results

if __name__ == "__main__":
    main()