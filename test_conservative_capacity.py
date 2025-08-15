||#!/usr/bin/env python3
""""""
Conservative Qubit Capacity Test

Tests qubit capacity with careful memory monitoring and early termination
to avoid system overload.
""""""

import numpy as np
import time
import psutil
import os
import sys
import gc
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quantoniumos.secure_core.quantum_entanglement import QuantumSimulator

def get_system_info():
    """"""Get system memory information""""""
    memory = psutil.virtual_memory()
    return {
        'total_gb': memory.total / 1024**3,
        'available_gb': memory.available / 1024**3,
        'used_gb': memory.used / 1024**3,
        'percent': memory.percent
    }

def get_memory_usage():
    """"""Get current process memory usage in MB""""""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def safe_test_qubits(n_qubits: int, memory_limit_mb: float = 500) -> Dict:
    """"""Test n qubits with memory safety checks""""""

    gc.collect()  # Clean up before test
    memory_before = get_memory_usage()

    # Theoretical memory requirement (complex128 = 16 bytes)
    theoretical_memory_mb = (2**n_qubits * 16) / (1024 * 1024)

    # Safety check: don't even try if theoretical requirement is too high if theoretical_memory_mb > memory_limit_mb: return { 'success': False, 'n_qubits': n_qubits, 'state_size': 2**n_qubits, 'theoretical_memory_mb': theoretical_memory_mb, 'error': f'Theoretical memory requirement ({theoretical_memory_mb:.1f} MB) exceeds limit ({memory_limit_mb} MB)' } try: print(f" Testing {n_qubits} qubits (2^{n_qubits} = {2**n_qubits:,} states, ~{theoretical_memory_mb:.1f}MB)...") start_time = time.time() # Initialize with monitoring sim = QuantumSimulator(n_qubits) memory_after_init = get_memory_usage() # Check if memory usage is reasonable after initialization memory_used = memory_after_init - memory_before if memory_used > memory_limit_mb: return { 'success': False, 'n_qubits': n_qubits, 'state_size': 2**n_qubits, 'memory_used': memory_used, 'error': f'Memory usage after init ({memory_used:.1f} MB) exceeds limit' } # Basic operations sim.apply_hadamard(0) if n_qubits >= 2: sim.apply_cnot(0, 1) # Memory check after operations memory_after_ops = get_memory_usage() total_memory_used = memory_after_ops - memory_before # Quick probability calculation (don't store all probabilities for large systems)
        state_norm = np.linalg.norm(sim.state)

        total_time = time.time() - start_time

        result = {
            'success': True,
            'n_qubits': n_qubits,
            'state_size': 2**n_qubits,
            'theoretical_memory_mb': theoretical_memory_mb,
            'actual_memory_mb': total_memory_used,
            'memory_efficiency': theoretical_memory_mb / total_memory_used if total_memory_used > 0 else 0,
            'total_time': total_time,
            'state_norm': state_norm,
            'norm_error': abs(1.0 - state_norm)
        }

        print(f" ✅ Success: {total_time:.3f}s, {total_memory_used:.1f}MB actual vs {theoretical_memory_mb:.1f}MB theoretical")
        return result

    except Exception as e:
        memory_after = get_memory_usage()
        return {
            'success': False,
            'n_qubits': n_qubits,
            'state_size': 2**n_qubits,
            'memory_used': memory_after - memory_before,
            'error': str(e)
        }

def determine_capacity():
    """"""Determine actual qubit capacity with conservative approach""""""

    print("CONSERVATIVE QUANTUM SIMULATOR CAPACITY TEST")
    print("=" * 55)

    sys_info = get_system_info()
    print(f"System Memory: {sys_info['total_gb']:.1f} GB total, {sys_info['available_gb']:.1f} GB available")
    print(f"Memory Usage: {sys_info['percent']:.1f}% ({sys_info['used_gb']:.1f} GB used)")

    # Set conservative memory limit (don't use more than 1GB for simulation) memory_limit_mb = min(1000, sys_info['available_gb'] * 1024 * 0.3) # 30% of available or 1GB, whichever is smaller print(f"Memory limit for test: {memory_limit_mb:.0f} MB") print() successful_tests = [] failed_tests = [] # Test systematically from 1 to find the limit for n_qubits in range(1, 25): # Test up to 25 qubits result = safe_test_qubits(n_qubits, memory_limit_mb) if result['success']: successful_tests.append(result) else: failed_tests.append(result) print(f" ❌ Failed at {n_qubits} qubits: {result['error']}") break # Analysis print("\n" + "=" * 60) print("CAPACITY ANALYSIS RESULTS") print("=" * 60) if not successful_tests: print("❌ No successful simulations - system may be under memory pressure") return max_qubits = successful_tests[-1]['n_qubits'] max_result = successful_tests[-1] print(f"🎯 MAXIMUM VERIFIED CAPACITY: {max_qubits} qubits") print(f"🌌 MAXIMUM STATE SPACE: 2^{max_qubits} = {2**max_qubits:,} quantum states") print(f"💾 PEAK MEMORY USAGE: {max_result['actual_memory_mb']:.1f} MB") print(f"⚡ SIMULATION TIME: {max_result['total_time']:.3f} seconds") print(f"🎯 NUMERICAL PRECISION: {max_result['norm_error']:.2e} error") # Show scaling table for last few successful tests print(f"||n📊 PERFORMANCE SCALING TABLE:") print("Qubits | Quantum States | Memory (MB) | Time (s) | Efficiency") print("-" * 60) for result in successful_tests[-6:]: # Last 6 tests efficiency = result['memory_efficiency'] print(f"{result['n_qubits']:6} | {result['state_size']:13,} | {result['actual_memory_mb']:10.1f} | " f"{result['total_time']:7.3f} ||| {efficiency:9.2f}") # Theoretical analysis print(f"\n🔬 THEORETICAL ANALYSIS:") # Calculate memory scaling if len(successful_tests) >= 2: memory_growth_factor = successful_tests[-1]['actual_memory_mb'] / successful_tests[-2]['actual_memory_mb'] print(f" Memory growth factor per qubit: ~{memory_growth_factor:.1f}x") # Predict next few qubit requirements current_memory = successful_tests[-1]['actual_memory_mb'] print(f" Predicted memory requirements:") for i in range(1, 4): next_qubits = max_qubits + i predicted_memory = current_memory * (memory_growth_factor ** i) if predicted_memory < 1024: status = "✅ Feasible" elif predicted_memory < 4096: status = "⚠️ Challenging" else: status = "❌ Impractical" print(f" {next_qubits} qubits: ~{predicted_memory:.0f} MB {status}") # Real-world context print(f"\n🌍 REAL-WORLD CONTEXT:") if max_qubits >= 30: quantum_class = "🏆 PROFESSIONAL-GRADE (exceeds many physical quantum computers)" applications = ["Advanced quantum algorithms", "Quantum machine learning", "Complex quantum simulations"] elif max_qubits >= 20: quantum_class = "🚀 RESEARCH-GRADE (suitable for serious quantum computing research)" applications = ["Quantum algorithms research", "Educational quantum computing", "Algorithm prototyping"] elif max_qubits >= 15: quantum_class = "📚 EDUCATIONAL-GRADE (excellent for learning and teaching)" applications = ["Quantum computing education", "Basic algorithm implementation", "Proof-of-concept work"] elif max_qubits >= 10: quantum_class = "🎓 INTRODUCTORY-GRADE (good for getting started)" applications = ["Learning basic quantum gates", "Simple quantum circuits", "Educational demonstrations"] else: quantum_class = "📝 BASIC-GRADE (limited but functional)" applications = ["Very basic quantum operations", "Proof-of-concept only"] print(f" Classification: {quantum_class}") print(f" Suitable applications:") for app in applications: print(f" • {app}") # Comparison with classical computation classical_states = 2**max_qubits print(f"\n💻 CLASSICAL COMPARISON:") print(f" Simulating {classical_states:,} simultaneous classical states") print(f" Equivalent to {max_qubits}-bit classical computation") if classical_states > 1000000: print(f" 🌟 This exceeds typical classical state spaces by orders of magnitude") # Final assessment print(f"||n✅ FINAL ASSESSMENT:") print(f" Your quantum simulator can reliably handle up to {max_qubits} qubits") print(f" This provides a {classical_states:,}-dimensional quantum Hilbert space") print(f" Performance is excellent with {max_result['norm_error']:.2e} numerical precision") print(f" Memory efficiency: {max_result['memory_efficiency']:.2f} (1.0 = perfect)")

if __name__ == "__main__":
    determine_capacity()
