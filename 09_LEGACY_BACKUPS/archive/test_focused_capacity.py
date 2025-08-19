||#!/usr/bin/env python3
"""
Focused Quantum Simulator Capacity Test This test determines the exact qubit limit by testing incrementally and provides detailed memory and performance analysis.
"""
"""

import numpy as np
import time
import psutil
import os
import sys
import gc from typing
import Dict sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))) from quantoniumos.secure_core.quantum_entanglement
import QuantumSimulator
def get_memory_usage() -> float:
"""
"""
        Get current memory usage in MB
"""
"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
def test_single_qubit_count(n_qubits: int) -> Dict:
"""
"""
        Test a specific number of qubits
"""
        """ memory_before = get_memory_usage() gc.collect()
        try:
        print(f" Testing {n_qubits} qubits (2^{n_qubits} = {2**n_qubits:,} states)...") start_time = time.time()

        # Initialize simulator init_start = time.time() sim = QuantumSimulator(n_qubits) init_time = time.time() - init_start memory_after_init = get_memory_usage()

        # Apply some gates gates_start = time.time() sim.apply_hadamard(0)
        if n_qubits >= 2: sim.apply_cnot(0, 1) gates_time = time.time() - gates_start

        # Get probabilities prob_start = time.time() probs = sim.get_state_probabilities() prob_time = time.time() - prob_start

        # Measure measure_start = time.time() measurement = sim.measure_all() measure_time = time.time() - measure_start total_time = time.time() - start_time memory_after = get_memory_usage()

        # Test precision sim_test = QuantumSimulator(n_qubits) sim_test.initialize_random() norm_error = abs(1.0 - np.linalg.norm(sim_test.state)) result = { 'success': True, 'n_qubits': n_qubits, 'state_size': 2**n_qubits, 'memory_before': memory_before, 'memory_after_init': memory_after_init, 'memory_after_ops': memory_after, 'memory_used': memory_after - memory_before, 'init_time': init_time, 'gates_time': gates_time, 'prob_time': prob_time, 'measure_time': measure_time, 'total_time': total_time, 'norm_error': norm_error, 'measurement': measurement }
        print(f" ✅ Success: {total_time:.3f}s, {result['memory_used']:.1f}MB")
        return result except Exception as e: memory_after = get_memory_usage() result = { 'success': False, 'n_qubits': n_qubits, 'error': str(e), 'memory_used': memory_after - memory_before }
        print(f" ❌ Failed: {str(e)}")
        return result
def find_maximum_capacity(): """
        Find the maximum number of qubits we can simulate
"""
"""
        print("QUANTUM SIMULATOR MAXIMUM CAPACITY TEST")
        print("=" * 50)
        print(f"System memory: {psutil.virtual_memory().total / 1024**3:.1f} GB")
        print(f"Available memory: {psutil.virtual_memory().available / 1024**3:.1f} GB")
        print()

        # Start with known working range working_qubits = [] failed_qubits = []

        # Test small sizes first (1-10)
        print("Phase 1: Testing small qubit counts (1-10)")
        for n in range(1, 11): result = test_single_qubit_count(n)
        if result['success']: working_qubits.append(result)
        else: failed_qubits.append(result) break
        if not working_qubits:
        print("❌ Cannot simulate even 1 qubit!")
        return

        # Test medium sizes (11-15)
        print("\nPhase 2: Testing medium qubit counts (11-15)")
        for n in range(11, 16): result = test_single_qubit_count(n)
        if result['success']: working_qubits.append(result)
        else: failed_qubits.append(result) break

        # If we made it past 15, try larger sizes
        if working_qubits and working_qubits[-1]['n_qubits'] >= 15:
        print("\nPhase 3: Testing large qubit counts (16-20)")
        for n in range(16, 21): result = test_single_qubit_count(n)
        if result['success']: working_qubits.append(result)
        else: failed_qubits.append(result) break

        # Analyze results
        print("\n" + "=" * 60)
        print("FINAL CAPACITY ANALYSIS")
        print("=" * 60)
        if not working_qubits:
        print("❌ No successful simulations!")
        return max_qubits = working_qubits[-1]['n_qubits'] max_states = 2**max_qubits
        print(f" MAXIMUM CAPACITY: {max_qubits} qubits")
        print(f"🌐 STATE SPACE SIZE: 2^{max_qubits} = {max_states:,} quantum states")
        print(f"💾 PEAK MEMORY USAGE: {working_qubits[-1]['memory_used']:.1f} MB")
        print(f"⏱️ PERFORMANCE AT MAX: {working_qubits[-1]['total_time']:.3f} seconds")

        # Detailed performance breakdown
        print(f"||n PERFORMANCE SCALING:")
        print("Qubits | States | Memory(MB) | Time(s) | Init(ms) | Gates(ms) | Prob(ms)")
        print("-" * 75)
        for result in working_qubits[-5:]:

        # Show last 5 results n = result['n_qubits'] states = result['state_size'] memory = result['memory_used'] total_time = result['total_time'] init_time = result['init_time'] * 1000 gates_time = result['gates_time'] * 1000 prob_time = result['prob_time'] * 1000
        print(f"{n:6} | {states:10,} | {memory:9.1f} | {total_time:6.3f} | " f"{init_time:7.1f} | {gates_time:8.1f} ||| {prob_time:7.1f}")

        # Precision analysis precision_errors = [r['norm_error']
        for r in working_qubits] avg_error = np.mean(precision_errors) max_error = max(precision_errors)
        print(f"\n NUMERICAL PRECISION:")
        print(f" Average normalization error: {avg_error:.2e}")
        print(f" Maximum normalization error: {max_error:.2e}")
        if max_error < 1e-14: precision_grade = "EXCELLENT (machine precision)"
        el
        if max_error < 1e-12: precision_grade = "VERY GOOD"
        el
        if max_error < 1e-10: precision_grade = "GOOD"
        else: precision_grade = "POOR - numerical instability detected"
        print(f" Precision grade: {precision_grade}")

        # Real-world comparison
        print(f"\n🔬 COMPARISON WITH REAL QUANTUM SYSTEMS:")
        if max_qubits >= 50:
        print(" EXCEEDS most current quantum computers (IBM: ~127 qubits, but with noise)")
        print(" 🏆 Suitable for advanced quantum algorithm research")
        el
        if max_qubits >= 30:
        print(" COMPETITIVE with advanced superconducting quantum processors")
        print(" 🧪 Suitable for serious quantum algorithm development")
        el
        if max_qubits >= 20:
        print(" ✅ GOOD for quantum research and education")
        print(" 📚 Can simulate interesting quantum algorithms")
        el
        if max_qubits >= 15:
        print(" 📖 SUITABLE for educational purposes")
        print(" 🔬 Good for learning quantum algorithms")
        else:
        print(" 📝 BASIC quantum simulation capability")
        print(" 🎓 Suitable for introductory quantum computing")

        # Practical applications
        print(f"\n💼 PRACTICAL APPLICATIONS AT {max_qubits} QUBITS:") applications = [ (5, "Quantum error correction codes"), (8, "Quantum Fourier Transform demos"), (10, "Grover's algorithm on small databases"), (12, "Quantum chemistry simulations (small molecules)"), (15, "Quantum machine learning experiments"), (20, "Advanced quantum algorithms research"), (25, "Professional quantum software development") ] for req_qubits, application in applications:
        if max_qubits >= req_qubits:
        print(f" ✅ {application}")
        else:
        print(f" ❌ {application} (requires {req_qubits} qubits)")

        # Memory efficiency analysis last_result = working_qubits[-1] memory_per_state = last_result['memory_used'] * 1024 * 1024 / last_result['state_size'] # bytes per state complex128_size = 16 # bytes for complex128
        print(f"||n💾 MEMORY EFFICIENCY:")
        print(f" Memory per quantum state: {memory_per_state:.1f} bytes")
        print(f" Theoretical minimum (complex128): {complex128_size} bytes")
        print(f" Efficiency ratio: {complex128_size/memory_per_state:.2f}")
        if memory_per_state <= complex128_size * 2:
        print(" ✅ Excellent memory efficiency")
        el
        if memory_per_state <= complex128_size * 5:
        print(" ✅ Good memory efficiency")
        else:
        print(" ⚠️ Memory overhead detected")

if __name__ == "__main__": find_maximum_capacity()