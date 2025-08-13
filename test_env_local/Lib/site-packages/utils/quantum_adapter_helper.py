"""
Quantum Adapter Helper

Provides access to the quantum adapter for various QuantoniumOS applications.
"""

from encryption.quantum_engine_adapter import quantum_adapter

def get_quantum_adapter():
    """
    Get the global quantum adapter instance.
    
    Returns:
        QuantumEngineAdapter: The global quantum adapter instance
    """
    return quantum_adapter

def encrypt_data(plaintext: str, key: str) -> str:
    """
    Encrypt text using quantum-inspired algorithm.
    
    Args:
        plaintext: Text to encrypt
        key: Encryption key
        
    Returns:
        Base64-encoded encrypted data
    """
    return quantum_adapter.encrypt(plaintext, key)

def decrypt_data(ciphertext: str, key: str) -> str:
    """
    Decrypt text that was encrypted with quantum-inspired algorithm.
    
    Args:
        ciphertext: Base64-encoded encrypted data
        key: Decryption key
        
    Returns:
        Decrypted plaintext
    """
    return quantum_adapter.decrypt(ciphertext, key)

def generate_quantum_entropy(amount: int = 32) -> str:
    """
    Generate quantum-inspired entropy.
    
    Args:
        amount: Amount of entropy to generate in bytes
        
    Returns:
        Base64-encoded entropy
    """
    return quantum_adapter.generate_entropy(amount)

def apply_quantum_rft(waveform: list) -> dict:
    """
    Apply Resonance Fourier Transform to a waveform.
    
    Args:
        waveform: List of waveform values
        
    Returns:
        Dictionary with frequencies, amplitudes, and phases
    """
    return quantum_adapter.apply_rft(waveform)

def unlock_quantum_container(waveform: list, container_hash: str, key: str) -> dict:
    """
    Attempt to unlock a container using a waveform.
    
    Args:
        waveform: Waveform data
        container_hash: Container hash identifier
        key: Decryption key
        
    Returns:
        Container content if unlocked
    """
    return quantum_adapter.unlock_container(waveform, container_hash, key)

def run_quantum_benchmark(max_qubits: int = 150, full_benchmark: bool = False) -> dict:
    """
    Run a benchmark of the quantum engine capabilities.
    
    Args:
        max_qubits: Maximum number of qubits to test
        full_benchmark: Whether to run the 64-perturbation test
        
    Returns:
        Benchmark results
    """
    return quantum_adapter.run_benchmark(max_qubits, full_benchmark)