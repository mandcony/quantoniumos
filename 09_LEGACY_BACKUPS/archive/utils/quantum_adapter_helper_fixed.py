"""
Quantum Adapter Helper - Fixed Implementation

This module provides working implementations using real AES-GCM crypto
instead of the missing quantum_engine_adapter.
"""

import base64
import json
import secrets
from typing import Dict, List
from utils.crypto_secure import encrypt_data as real_encrypt, decrypt_data as real_decrypt

def get_quantum_adapter():
    """
    Get the global quantum adapter instance (stub - returns None).

    Returns:
        None: No actual adapter implemented
    """
    return None

def encrypt_data(plaintext: str, key: str) -> str:
    """
    Encrypt text using AES-GCM (delegating to crypto_secure).

    Args:
        plaintext: Text to encrypt
        key: Encryption key

    Returns:
        Base64-encoded encrypted bundle as JSON string
    """
    result = real_encrypt(plaintext, key)
    return base64.b64encode(json.dumps(result).encode()).decode()

def decrypt_data(ciphertext: str, key: str) -> str:
    """
    Decrypt text using AES-GCM (delegating to crypto_secure).

    Args:
        ciphertext: Base64-encoded encrypted bundle
        key: Decryption key

    Returns:
        Decrypted plaintext
    """
    bundle = json.loads(base64.b64decode(ciphertext).decode())
    result = real_decrypt(bundle, key)
    return result.decode('utf-8')

def generate_quantum_entropy(amount: int = 32) -> str:
    """
    Generate cryptographically secure entropy.

    Args:
        amount: Amount of entropy to generate in bytes

    Returns:
        Base64-encoded entropy
    """
    entropy = secrets.token_bytes(amount)
    return base64.b64encode(entropy).decode()

def apply_quantum_rft(waveform: list) -> dict:
    """
    Apply basic FFT-like transform to a waveform (stub).

    Args:
        waveform: List of waveform values

    Returns:
        Dictionary with basic frequency analysis
    """
    import math
    N = len(waveform)
    freqs = [i * 1.0 / N for i in range(N//2)]
    # Simple magnitude calculation
    mags = [abs(sum(waveform[j] * math.cos(2*math.pi*i*j/N) for j in range(N))) for i in range(N//2)]
    phases = [0.0] * len(freqs)  # Simplified
    return {'frequencies': freqs, 'amplitudes': mags, 'phases': phases}

def unlock_quantum_container(waveform: list, container_hash: str, key: str) -> dict:
    """
    Stub for container unlocking (not implemented).

    Args:
        waveform: Waveform data
        container_hash: Container hash identifier
        key: Decryption key

    Returns:
        Empty result
    """
    return {'status': 'not_implemented', 'content': None}

def run_quantum_benchmark(max_qubits: int = 150, full_benchmark: bool = False) -> dict:
    """
    Run genuine quantum benchmarks using the QuantoniumOS quantum simulator.

    Args:
        max_qubits: Maximum number of qubits to test (limited to reasonable values)
        full_benchmark: Whether to run the full test suite

    Returns:
        Real benchmark results from quantum entanglement simulation
    """
    import time
    import sys
    import os

    # Add project root to path for imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    try:
        from quantoniumos.secure_core.quantum_entanglement import QuantumSimulator
    except ImportError:
        # Fallback if quantum simulator not available
        return {
            'status': 'quantum_simulator_unavailable',
            'max_qubits_tested': 0,
            'full_benchmark': full_benchmark,
            'results': 'Quantum simulator module not found - using classical crypto'
        }

    # Limit to reasonable qubit counts for simulation
    actual_max_qubits = min(max_qubits, 10 if full_benchmark else 6)

    results = {
        'status': 'completed',
        'max_qubits_tested': actual_max_qubits,
        'full_benchmark': full_benchmark,
        'benchmark_data': [],
        'timestamp': time.time()
    }

    # Test different qubit configurations
    test_qubits = [2, 3, 4] if not full_benchmark else list(range(2, actual_max_qubits + 1))

    for num_qubits in test_qubits:
        start_time = time.perf_counter()

        # Initialize quantum simulator
        sim = QuantumSimulator(num_qubits)

        # Run basic quantum operations
        operations_completed = 0

        # Create superposition states
        for i in range(num_qubits):
            sim.apply_hadamard(i)
            operations_completed += 1

        # Create entanglement pairs
        for i in range(num_qubits - 1):
            sim.apply_cnot(i, i + 1)
            operations_completed += 1

        # Measure entanglement
        entanglement_score = sim.get_entanglement_score()

        # Perform measurements
        measurement_results = sim.measure_all()

        end_time = time.perf_counter()

        qubit_result = {
            'qubits': num_qubits,
            'operations_completed': operations_completed,
            'execution_time': end_time - start_time,
            'entanglement_score': entanglement_score,
            'measurement_results': measurement_results,
            'state_vector_size': 2**num_qubits
        }

        results['benchmark_data'].append(qubit_result)

    # Calculate performance metrics
    if results['benchmark_data']:
        avg_time = sum(r['execution_time'] for r in results['benchmark_data']) / len(results['benchmark_data'])
        max_entanglement = max(r['entanglement_score'] for r in results['benchmark_data'])

        results['performance_summary'] = {
            'average_execution_time': avg_time,
            'maximum_entanglement_achieved': max_entanglement,
            'total_operations': sum(r['operations_completed'] for r in results['benchmark_data'])
        }

    return results
