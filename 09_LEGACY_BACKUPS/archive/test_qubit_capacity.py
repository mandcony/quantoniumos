||#!/usr/bin/env python3
""""""
Quantum Simulator Capacity Test with RFT Integration This test determines the maximum number of qubits that can be accurately simulated in the current environment by measuring memory usage, computation time, and numerical precision. Now includes RFT-based quantum computing capabilities.
"""
"""
import numpy as np
import time
import psutil
import os
import sys from typing
import Tuple, Dict, List
import gc

# Import our quantum simulators sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))) from quantoniumos.secure_core.quantum_entanglement
import QuantumSimulator

# Import RFT quantum computing (
if available)
try: from rft_quantum_computing
import RFTQuantumComputer HAS_RFT_QUANTUM = True
except ImportError: HAS_RFT_QUANTUM = False
print("Warning: RFT quantum computing not available")
def get_memory_usage() -> float: """"""
        Get current memory usage in MB
"""
"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
def test_qubit_capacity(max_qubits: int = 20, include_rft: bool = True) -> Dict[int, Dict]:
"""
"""
        Test quantum simulator capacity by incrementally increasing qubit count Args: max_qubits: Maximum number of qubits to test include_rft: Whether to include RFT quantum computing tests Returns: Dictionary with results for each qubit count tested
"""
        """ results = {}
        print("QUANTUM SIMULATOR CAPACITY TEST" + (" WITH RFT INTEGRATION"
        if include_rft and HAS_RFT_QUANTUM else ""))
        print("=" * 70)
        print(f"Testing qubit capacity up to {max_qubits} qubits...")
        if include_rft and HAS_RFT_QUANTUM:
        print("Including RFT-based quantum computing comparison...")
        print()
        for n_qubits in range(1, max_qubits + 1):
        print(f"Testing {n_qubits} qubits...", end=" ", flush=True)

        # Memory before memory_before = get_memory_usage() gc.collect()

        # Force garbage collection
        try: # === STANDARD QUANTUM SIMULATOR === start_time = time.time()

        # Initialize simulator sim = QuantumSimulator(n_qubits) memory_after_init = get_memory_usage()

        # Test basic operations operation_times = {}

        # Test Hadamard gate (creates superposition) hadamard_start = time.time() sim.apply_hadamard(0) operation_times['hadamard'] = time.time() - hadamard_start

        # Test CNOT gate (creates entanglement)
        if we have at least 2 qubits
        if n_qubits >= 2: cnot_start = time.time() sim.apply_cnot(0, 1) operation_times['cnot'] = time.time() - cnot_start

        # Test state probability calculation prob_start = time.time() probabilities = sim.get_state_probabilities() operation_times['probabilities'] = time.time() - prob_start

        # Test measurement measure_start = time.time() measurement = sim.measure_all() operation_times['measurement'] = time.time() - measure_start total_time = time.time() - start_time memory_after_ops = get_memory_usage()

        # Calculate state vector size state_size = 2**n_qubits memory_used = memory_after_ops - memory_before

        # Test numerical precision by checking normalization sim_test = QuantumSimulator(n_qubits) sim_test.initialize_random() norm = np.linalg.norm(sim_test.state) precision_error = abs(1.0 - norm) # === RFT QUANTUM SIMULATOR === rft_results = {}
        if include_rft and HAS_RFT_QUANTUM and n_qubits >= 2:
        try: rft_start = time.time()

        # Initialize RFT quantum computer rft_qc = RFTQuantumComputer(n_qubits)

        # Test basic RFT operations rft_operation_times = {}

        # Test Hadamard in RFT basis rft_h_start = time.time() rft_qc.apply_hadamard(0, "basis_hop") rft_operation_times['rft_hadamard_basishop'] = time.time() - rft_h_start

        # Test local conjugation approach
        if n_qubits <= 8:

        # Only test for smaller systems due to complexity rft_h_conj_start = time.time() rft_qc.apply_hadamard(1
        if n_qubits > 1 else 0, "local_conjugation") rft_operation_times['rft_hadamard_conjugation'] = time.time() - rft_h_conj_start

        # Test CNOT in RFT basis
        if n_qubits >= 2: rft_cnot_start = time.time() rft_qc.apply_cnot(0, 1, "basis_hop") rft_operation_times['rft_cnot'] = time.time() - rft_cnot_start

        # Test RFT state probabilities rft_prob_start = time.time() rft_probabilities = rft_qc.get_state_probabilities() rft_operation_times['rft_probabilities'] = time.time() - rft_prob_start

        # Test RFT measurement rft_measure_start = time.time() rft_measurement = rft_qc.measure_all() rft_operation_times['rft_measurement'] = time.time() - rft_measure_start rft_total_time = time.time() - rft_start

        # Get RFT-specific metrics rft_coherence = rft_qc.get_coherence_score() rft_entanglement = rft_qc.get_entanglement_score() rft_results = { 'rft_total_time': rft_total_time, 'rft_operation_times': rft_operation_times, 'rft_measurement': rft_measurement, 'rft_coherence': rft_coherence, 'rft_entanglement': rft_entanglement, 'rft_speedup_ratio': total_time / rft_total_time
        if rft_total_time > 0 else float('inf') } except Exception as rft_e: rft_results = {'rft_error': str(rft_e)}

        # Combine results results[n_qubits] = { 'success': True, 'state_size': state_size, 'total_time': total_time, 'memory_used_mb': memory_used, 'memory_after_init_mb': memory_after_init, 'operation_times': operation_times, 'precision_error': precision_error, 'measurement_result': measurement, 'num_basis_states': len(probabilities), **rft_results

        # Include RFT results
        if available }

        # Print status status_msg = f"✓ ({total_time:.3f}s, {memory_used:.1f}MB, {state_size} states" if 'rft_total_time' in rft_results: rft_time = rft_results['rft_total_time'] status_msg += f", RFT: {rft_time:.3f}s" status_msg += ")"
        print(status_msg)

        # Safety check:
        if memory usage is getting too high, stop
        if memory_used > 1000: # 1GB limit
        print(f"⚠️ Memory usage high ({memory_used:.1f}MB), stopping test") break except Exception as e: results[n_qubits] = { 'success': False, 'error': str(e), 'memory_used_mb': get_memory_usage() - memory_before }
        print(f"✗ Error: {str(e)}") break
        return results
def analyze_results(results: Dict) -> None: """"""
        Analyze and display test results
"""
"""
        print("\n" + "=" * 60)
        print("CAPACITY ANALYSIS RESULTS")
        print("=" * 60) successful_tests = {k: v for k, v in results.items()
        if v.get('success', False)}
        if not successful_tests:
        print("❌ No successful tests!")
        return max_qubits = max(successful_tests.keys())
        print(f"✅ Maximum qubits successfully simulated: {max_qubits}")
        print(f"✅ Maximum state space size: 2^{max_qubits} = {2**max_qubits:,} states")

        # Memory analysis memory_usage = [(k, v['memory_used_mb']) for k, v in successful_tests.items()]
        print(f"||n Memory Usage Scaling:")
        print("Qubits | State Size | Memory (MB) | Time (s)")
        print("-" * 45) for qubits, data in successful_tests.items(): state_size = data['state_size'] memory = data['memory_used_mb'] time_taken = data['total_time']
        print(f"{qubits:6} | {state_size:10,} | {memory:10.1f} ||| {time_taken:7.3f}")

        # Operation timing analysis
        print(f"\n Operation Performance (for {max_qubits} qubits):")
        if max_qubits in successful_tests: ops = successful_tests[max_qubits]['operation_times'] for op_name, op_time in ops.items():
        print(f" {op_name:12}: {op_time*1000:.2f} ms")

        # Precision analysis precision_errors = [v['precision_error']
        for v in successful_tests.values()] max_error = max(precision_errors) avg_error = np.mean(precision_errors)
        print(f"\n Numerical Precision:")
        print(f" Average error: {avg_error:.2e}")
        print(f" Maximum error: {max_error:.2e}")
        if max_error < 1e-14:
        print(" ✅ Excellent precision (machine epsilon level)")
        el
        if max_error < 1e-12:
        print(" ✅ Very good precision")
        el
        if max_error < 1e-10:
        print(" ⚠️ Good precision")
        else:
        print(" ❌ Poor precision - potential numerical instability")

        # Practical capacity assessment
        print(f"\n🏗️ PRACTICAL CAPACITY ASSESSMENT:")
        print(f" • Real-time simulations: up to ~{min(max_qubits, 15)} qubits")
        print(f" • Batch processing: up to ~{max_qubits} qubits")
        print(f" • Memory limit reached at: ~{max_qubits + 1} qubits (projected)")

        # Comparison with real quantum computers
        print(f"\n🔬 Comparison with Real Quantum Hardware:")
        if max_qubits >= 50:
        print(" Exceeds current NISQ devices (50+ qubits)")
        el
        if max_qubits >= 30:
        print(" Competitive with advanced NISQ devices")
        el
        if max_qubits >= 15:
        print(" ✅ Good for educational/research purposes")
        else:
        print(" 📚 Suitable for basic quantum algorithm testing")
def test_specific_quantum_algorithms(max_qubits: int) -> None: """"""
        Test specific quantum algorithms to verify practical capability
"""
"""
        print(f"||n🧪 QUANTUM ALGORITHM TESTS")
        print("=" * 40)

        # Test Bell state creation
        print("Testing Bell state creation...", end=" ")
        try: sim = QuantumSimulator(2) sim.apply_hadamard(0) sim.apply_cnot(0, 1) probs = sim.get_state_probabilities()

        # Check
        if we have proper Bell state: |00> + |11> expected_states = ['00', '11'] bell_prob = sum(probs.get(state, 0)
        for state in expected_states)
        if bell_prob > 0.99:
        print("✅ Perfect Bell state")
        else:
        print(f"⚠️ Imperfect Bell state (prob={bell_prob:.3f})") except Exception as e:
        print(f"❌ Failed: {e}")

        # Test quantum superposition
        print("Testing quantum superposition...", end=" ")
        try: sim = QuantumSimulator(3)

        # Create |+++> state
        for i in range(3): sim.apply_hadamard(i) probs = sim.get_state_probabilities()

        # Should have equal probabilities for all 8 states expected_prob = 1.0 / 8.0 prob_variance = np.var(list(probs.values()))
        if prob_variance < 1e-10:
        print("✅ Perfect superposition")
        else:
        print(f"⚠️ Imperfect superposition (var={prob_variance:.2e})") except Exception as e:
        print(f"❌ Failed: {e}")

        # Test GHZ state (
        if we can handle enough qubits)
        if max_qubits >= 3:
        print("Testing GHZ state creation...", end=" ")
        try: sim = QuantumSimulator(3) sim.apply_hadamard(0) sim.apply_cnot(0, 1) sim.apply_cnot(0, 2) probs = sim.get_state_probabilities()

        # Should have |000> + |||111> ghz_prob = probs.get('000', 0) + probs.get('111', 0)
        if ghz_prob > 0.99:
        print("✅ Perfect GHZ state")
        else:
        print(f"⚠️ Imperfect GHZ state (prob={ghz_prob:.3f})") except Exception as e:
        print(f"❌ Failed: {e}")

if __name__ == "__main__":
print("Starting quantum simulator capacity test...")
print(f"System memory: {psutil.virtual_memory().total / 1024**3:.1f} GB")
print(f"Available memory: {psutil.virtual_memory().available / 1024**3:.1f} GB")
print()

# Test capacity results = test_qubit_capacity(25)

# Test up to 25 qubits

# Analyze results analyze_results(results)

# Test specific algorithms successful_qubits = [k for k, v in results.items()
if v.get('success', False)]
if successful_qubits: max_qubits = max(successful_qubits) test_specific_quantum_algorithms(max_qubits)
print(f"||n✅ Capacity test complete!")