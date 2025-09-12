#!/usr/bin/env python3
"""
CI-SAFE SCALING ANALYSIS
========================
Calculate scaling exponent without Unicode characters for CI compatibility.
"""

import numpy as np
import sys
import os

# Add the src path to import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from assembly.python_bindings.quantum_symbolic_engine import QuantumSymbolicEngine

def calculate_scaling_exponent():
    """Calculate formal scaling exponent from timing data"""
    
    print("SCALING EXPONENT ANALYSIS")
    print("=" * 50)
    print("Calculating formal O(n^alpha) scaling exponent")
    print("=" * 50)
    
    # Test sizes and collect timing data
    test_sizes = [8, 16, 32, 64, 128, 256, 512]
    engine = QuantumSymbolicEngine()
    
    times = []
    sizes = []
    
    for size in test_sizes:
        print(f"Testing {size} qubits...")
        
        # Single run for scaling analysis
        engine.initialize_state(size)
        
        import time
        start = time.perf_counter()
        success, result = engine.compress_million_qubits(size)
        compress_time = (time.perf_counter() - start) * 1e6  # μs
        
        if success:
            times.append(compress_time)
            sizes.append(size)
            print(f"  PASS: {compress_time:.2f} us")
        
        engine.cleanup()
    
    # Calculate scaling exponent using log-log regression
    log_sizes = np.log(sizes)
    log_times = np.log(times)
    
    # Linear fit: log(time) = α * log(size) + β
    coeffs = np.polyfit(log_sizes, log_times, 1)
    alpha = coeffs[0]  # Scaling exponent
    beta = coeffs[1]   # Constant factor
    
    # Calculate R² correlation
    predicted_log_times = alpha * log_sizes + beta
    ss_res = np.sum((log_times - predicted_log_times) ** 2)
    ss_tot = np.sum((log_times - np.mean(log_times)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    print(f"\nSCALING ANALYSIS RESULTS:")
    print(f"=" * 40)
    print(f"Scaling exponent (alpha): {alpha:.3f}")
    print(f"Correlation (R-squared):  {r_squared:.4f}")
    print(f"")
    print(f"FORMAL COMPLEXITY: O(n^{alpha:.3f})")
    
    if alpha < 1.0:
        print(f"SUB-LINEAR: Better than O(n) by {1-alpha:.3f}")
    elif alpha < 1.1:
        print(f"NEAR-LINEAR: Close to theoretical O(n)")
    elif alpha < 2.0:
        print(f"SUPER-LINEAR: {alpha:.3f}x worse than O(n)")
    else:
        print(f"QUADRATIC+: Needs optimization")
    
    return alpha, r_squared

if __name__ == "__main__":
    alpha, r2 = calculate_scaling_exponent()
    print(f"\nCONCLUSION: O(n^{alpha:.3f}) with R-squared={r2:.4f}")
