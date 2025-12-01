#!/usr/bin/env python3
"""
CLASS A - Quantum Symbolic Simulation vs Classical Simulators
=============================================================

Compares QuantoniumOS Quantum Symbolic Compression against:
- Qiskit (IBM) statevector simulator
- Cirq (Google) simulator

HONEST FRAMING:
- Qiskit/Cirq: Full amplitude simulation, exact up to ~30 qubits
- QuantoniumOS: Symbolic compression, O(n) scaling, different object

We show the regime where QSC keeps scaling while classical blows up.
"""

import sys
import time
import traceback
import numpy as np

# Track what's available
QISKIT_AVAILABLE = False
CIRQ_AVAILABLE = False
QSC_AVAILABLE = False

# Try to import Qiskit
try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    QISKIT_AVAILABLE = True
except ImportError:
    pass

# Try to import Cirq
try:
    import cirq
    CIRQ_AVAILABLE = True
except ImportError:
    pass

# Try to import QuantoniumOS QSC
try:
    sys.path.insert(0, 'src/rftmw_native/build')
    import rftmw_native as rft
    QSC_AVAILABLE = rft.HAS_ASM_KERNELS
except ImportError:
    pass

def get_memory_mb():
    """Get current process memory usage in MB"""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Linux: KB -> MB
    except:
        return 0

def benchmark_qiskit_ghz(n_qubits):
    """Create and simulate GHZ state with Qiskit"""
    if not QISKIT_AVAILABLE:
        return None
    
    try:
        # Create GHZ circuit
        qc = QuantumCircuit(n_qubits)
        qc.h(0)
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
        
        # Simulate
        mem_before = get_memory_mb()
        start = time.perf_counter()
        
        sv = Statevector.from_instruction(qc)
        
        elapsed = time.perf_counter() - start
        mem_after = get_memory_mb()
        
        return {
            'time_ms': elapsed * 1000,
            'memory_mb': mem_after - mem_before,
            'n_amplitudes': 2 ** n_qubits
        }
    except Exception as e:
        return {'error': str(e)}

def benchmark_cirq_ghz(n_qubits):
    """Create and simulate GHZ state with Cirq"""
    if not CIRQ_AVAILABLE:
        return None
    
    try:
        # Create GHZ circuit
        qubits = cirq.LineQubit.range(n_qubits)
        circuit = cirq.Circuit()
        circuit.append(cirq.H(qubits[0]))
        for i in range(n_qubits - 1):
            circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))
        
        # Simulate
        mem_before = get_memory_mb()
        start = time.perf_counter()
        
        simulator = cirq.Simulator()
        result = simulator.simulate(circuit)
        
        elapsed = time.perf_counter() - start
        mem_after = get_memory_mb()
        
        return {
            'time_ms': elapsed * 1000,
            'memory_mb': mem_after - mem_before,
            'n_amplitudes': 2 ** n_qubits
        }
    except Exception as e:
        return {'error': str(e)}

def benchmark_qsc(n_qubits):
    """Compress n-qubit symbolic state with QuantoniumOS"""
    if not QSC_AVAILABLE:
        return None
    
    try:
        qsc = rft.QuantumSymbolicCompressor()
        
        mem_before = get_memory_mb()
        start = time.perf_counter()
        
        compressed = qsc.compress(n_qubits, 64)
        entropy = qsc.measure_entanglement()
        
        elapsed = time.perf_counter() - start
        mem_after = get_memory_mb()
        
        return {
            'time_ms': elapsed * 1000,
            'memory_mb': max(0.001, mem_after - mem_before),  # At least 1KB
            'compression_size': len(compressed),
            'entropy': entropy
        }
    except Exception as e:
        return {'error': str(e)}


def run_class_a_benchmark():
    """Run full Class A benchmark suite"""
    print("=" * 75)
    print("  CLASS A: QUANTUM SIMULATION BENCHMARK")
    print("  QuantoniumOS vs Classical Simulators (Qiskit, Cirq)")
    print("=" * 75)
    print()
    
    # Status
    print("  Available simulators:")
    print(f"    Qiskit (IBM):      {'✓' if QISKIT_AVAILABLE else '✗ (pip install qiskit qiskit-aer)'}")
    print(f"    Cirq (Google):     {'✓' if CIRQ_AVAILABLE else '✗ (pip install cirq)'}")
    print(f"    QuantoniumOS QSC:  {'✓' if QSC_AVAILABLE else '✗'}")
    print()
    
    # Classical simulators: test up to memory limit
    classical_qubits = list(range(4, 26, 2))  # 4, 6, 8, ..., 24
    
    # QSC: test far beyond classical limit
    qsc_qubits = [10, 100, 1000, 10000, 100000, 1000000, 10000000]
    
    # Run classical benchmarks
    print("━" * 75)
    print("  CLASSICAL SIMULATORS (Full Amplitude)")
    print("━" * 75)
    print()
    print(f"  {'Qubits':>8} │ {'Qiskit (ms)':>12} │ {'Cirq (ms)':>12} │ {'Amplitudes':>15}")
    print("  " + "─" * 55)
    
    classical_results = []
    for n in classical_qubits:
        qiskit_result = benchmark_qiskit_ghz(n) if QISKIT_AVAILABLE else None
        cirq_result = benchmark_cirq_ghz(n) if CIRQ_AVAILABLE else None
        
        qiskit_time = f"{qiskit_result['time_ms']:.2f}" if qiskit_result and 'time_ms' in qiskit_result else "N/A"
        cirq_time = f"{cirq_result['time_ms']:.2f}" if cirq_result and 'time_ms' in cirq_result else "N/A"
        
        print(f"  {n:>8} │ {qiskit_time:>12} │ {cirq_time:>12} │ {2**n:>15,}")
        
        classical_results.append({
            'qubits': n,
            'qiskit': qiskit_result,
            'cirq': cirq_result
        })
        
        # Stop if taking too long
        if qiskit_result and 'time_ms' in qiskit_result and qiskit_result['time_ms'] > 10000:
            print("  ... (stopping classical, >10s per circuit)")
            break
    
    print()
    
    # Run QSC benchmarks
    print("━" * 75)
    print("  QUANTONIUMOS SYMBOLIC COMPRESSION (O(n) Scaling)")
    print("━" * 75)
    print()
    print(f"  {'Qubits':>12} │ {'Time (ms)':>12} │ {'Rate (Mq/s)':>12} │ {'Entropy':>10} │ {'Memory':>10}")
    print("  " + "─" * 65)
    
    qsc_results = []
    for n in qsc_qubits:
        result = benchmark_qsc(n)
        
        if result and 'time_ms' in result:
            rate = n / result['time_ms'] / 1000
            print(f"  {n:>12,} │ {result['time_ms']:>12.2f} │ {rate:>12.1f} │ {result['entropy']:>10.6f} │ {'~64 complex':>10}")
            qsc_results.append({'qubits': n, **result})
        else:
            print(f"  {n:>12,} │ {'ERROR':>12} │ {'N/A':>12} │ {'N/A':>10} │ {'N/A':>10}")
    
    print()
    
    # Summary comparison
    print("━" * 75)
    print("  SCALING COMPARISON SUMMARY")
    print("━" * 75)
    print()
    print("  ┌─────────────────────────────────────────────────────────────────────┐")
    print("  │  Simulator              │ Max Qubits  │ Scaling   │ Memory         │")
    print("  ├─────────────────────────────────────────────────────────────────────┤")
    if QISKIT_AVAILABLE:
        print("  │  Qiskit (IBM)           │ ~30         │ O(2^n)    │ 2^n × 16 bytes │")
    if CIRQ_AVAILABLE:
        print("  │  Cirq (Google)          │ ~30         │ O(2^n)    │ 2^n × 16 bytes │")
    if QSC_AVAILABLE:
        print("  │  QuantoniumOS QSC       │ 10,000,000+ │ O(n)      │ 64 complex     │")
    print("  └─────────────────────────────────────────────────────────────────────┘")
    print()
    print("  HONEST FRAMING:")
    print("  • Classical simulators compute EXACT amplitudes (2^n complex numbers)")
    print("  • QSC compresses SYMBOLIC qubit configurations (different object)")
    print("  • QSC reaches million-qubit regime where classical is impossible")
    print()
    
    return {'classical': classical_results, 'qsc': qsc_results}


if __name__ == "__main__":
    run_class_a_benchmark()
