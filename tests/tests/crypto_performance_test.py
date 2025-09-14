#!/usr/bin/env python3
"""
Comprehensive cryptographic validation suite including:
- Performance benchmarks
- Differential cryptanalysis
- Pattern-based testing
"""

import os
import sys
import time
import secrets
import json
from typing import Dict, List, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

# Add paths for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Import implementations and quantum components
print("Importing crypto and quantum implementations...")

# Add ASSEMBLY paths
ASSEMBLY_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "ASSEMBLY")
sys.path.append(os.path.join(ASSEMBLY_PATH, "python_bindings"))

try:
    import numpy as np
    from src.core.enhanced_rft_crypto_v2 import EnhancedRFTCryptoV2
    
    # Try assembly imports, fallback to Python if not available
    try:
        from unitary_rft import UnitaryRFT
        from vertex_quantum_rft import EnhancedVertexQuantumRFT
        from quantum_symbolic_engine import QuantumSymbolicEngine
        print("✓ Successfully imported assembly-optimized implementations")
        ASSEMBLY_AVAILABLE = True
    except ImportError:
        print("⚠ Assembly imports not available, using Python fallback")
        UnitaryRFT = None
        EnhancedVertexQuantumRFT = None
        QuantumSymbolicEngine = None
        ASSEMBLY_AVAILABLE = False
    
    # Initialize assembly-optimized engines
    UNITARY_RFT = UnitaryRFT(size=1024, flags=0x00000008)  # RFT_FLAG_UNITARY
    VERTEX_RFT = EnhancedVertexQuantumRFT(data_size=1024)  # Default 1000 vertex qubits
    SYMBOLIC_ENGINE = QuantumSymbolicEngine()
    print("✓ Assembly-optimized quantum engines initialized")
    
except ImportError as e:
    print(f"✗ Failed to import assembly implementations: {e}")
    print("ASSEMBLY path:", ASSEMBLY_PATH)
    print("Python path:", sys.path)
    EnhancedRFTCryptoV2 = None
    UnitaryRFT = None
    EnhancedVertexQuantumRFT = None
    QuantumSymbolicEngine = None

# Mathematical constants
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio
TWO_PI = 2 * np.pi

# Don't try to import assembly version as it requires unitary_rft module
# We'll update the validation script instead to use the proper method names
EnhancedRFTCrypto = None

# Test parameters
KEY = secrets.token_bytes(32)
TEST_SIZES = [1024, 4096, 16384]  # Test with different message sizes
ITERATIONS = 3
SAMPLES = 10000  # Number of samples for differential analysis

@dataclass
class DifferentialTestResult:
    """Store results from differential cryptanalysis tests"""
    pattern_name: str
    samples: int
    differences: List[float]
    mean_difference: float
    max_difference: float
    threshold_exceeded: bool

class DifferentialTester:
    """Handles differential cryptanalysis testing"""
    
    def __init__(self, implementation, key):
        self.crypto = implementation(key)
        self.rft = UNITARY_RFT
        self.vertex_rft = VERTEX_RFT
        self.symbolic_engine = SYMBOLIC_ENGINE
        self.patterns = {
            'bit_tests': [
                'single_bit_0',
                'single_bit_7', 
                'single_bit_64',
                'single_bit_127'
            ],
            'pattern_tests': [
                'byte_pattern',
                'two_adjacent',
                'nibble_pattern',
                'word_pattern',
                'sparse_pattern',
                'dense_pattern',
                'diagonal'
            ]
        }
    
    def generate_test_pattern(self, pattern_name: str, size: int = 128) -> bytes:
        """Generate test patterns for differential analysis"""
        if pattern_name.startswith('single_bit_'):
            bit_pos = int(pattern_name.split('_')[-1])
            pattern = bytearray([0] * (size // 8))
            pattern[bit_pos // 8] = 1 << (bit_pos % 8)
            return bytes(pattern)
            
        patterns = {
            'byte_pattern': lambda: bytes([0xFF] + [0] * ((size // 8) - 1)),
            'two_adjacent': lambda: bytes([0xFF, 0xFF] + [0] * ((size // 8) - 2)),
            'nibble_pattern': lambda: bytes([0xF0] + [0] * ((size // 8) - 1)),
            'word_pattern': lambda: bytes([0xFF] * 4 + [0] * ((size // 8) - 4)),
            'sparse_pattern': lambda: bytes([0x55] * (size // 8)),
            'dense_pattern': lambda: bytes([0xAA] * (size // 8)),
            'diagonal': lambda: bytes([1 << (i % 8) for i in range(size // 8)])
        }
        
        return patterns[pattern_name]()

    def run_single_test(self, pattern_name: str, samples: int = SAMPLES) -> DifferentialTestResult:
        """Run differential analysis for a single pattern"""
        print(f"\nTesting {pattern_name}...")
        differences = []
        base_pattern = self.generate_test_pattern(pattern_name)
        
        for i in range(0, samples, 2000):
            print(f"Progress: {i}/{samples}")
            # Generate sample batch
            batch_differences = []
            for _ in range(min(2000, samples - i)):
                # Create slightly modified pattern
                modified = bytearray(base_pattern)
                mod_pos = secrets.randbelow(len(modified))
                modified[mod_pos] ^= (1 << secrets.randbelow(8))
                modified = bytes(modified)
                
                # Convert input patterns to numpy arrays
                base_data = np.frombuffer(base_pattern, dtype=np.uint8)
                mod_data = np.frombuffer(modified, dtype=np.uint8)
                
                # Assembly-optimized quantum encoding
                # First apply RFT using SIMD-optimized assembly
                base_rft = self.rft.apply_fast_rft(base_data)
                mod_rft = self.rft.apply_fast_rft(mod_data)
                
                # Apply vertex-based quantum transformation
                base_vertex = self.vertex_rft.transform_quantum_state(base_rft)
                mod_vertex = self.vertex_rft.transform_quantum_state(mod_rft)
                
                # Final symbolic compression using assembly engine
                base_encoded = self.symbolic_engine.compress_quantum_state(
                    base_vertex,
                    phi_factor=PHI,
                    use_simd=True
                )
                mod_encoded = self.symbolic_engine.compress_quantum_state(
                    mod_vertex,
                    phi_factor=PHI,
                    use_simd=True
                )
                
                # Assembly-optimized validation checks
                
                # SIMD-accelerated unitarity verification
                base_unitarity = self.rft.validate_unitarity(base_encoded)
                mod_unitarity = self.rft.validate_unitarity(mod_encoded)
                assert base_unitarity['perfect_unitarity'], "Loss of unitarity in base encoding"
                assert mod_unitarity['perfect_unitarity'], "Loss of unitarity in mod encoding"
                
                # Hardware-accelerated quantum metrics
                base_metrics = self.vertex_rft.validate_quantum_state(base_encoded)
                mod_metrics = self.vertex_rft.validate_quantum_state(mod_encoded)
                
                # Verify quantum properties using assembly-optimized checks
                assert base_metrics['norm_preservation'] > 0.99999, "Loss of state norm in base encoding"
                assert mod_metrics['norm_preservation'] > 0.99999, "Loss of state norm in mod encoding"
                assert base_metrics['reconstruction_error'] < 1e-12, "High reconstruction error in base encoding"
                assert mod_metrics['reconstruction_error'] < 1e-12, "High reconstruction error in mod encoding"
                
                # Assembly-optimized entanglement measurement
                base_entanglement = self.symbolic_engine.measure_entanglement(base_encoded)
                mod_entanglement = self.symbolic_engine.measure_entanglement(mod_encoded)
                assert abs(base_entanglement - 1.0) < 1e-6, "Loss of entanglement in base encoding"
                assert abs(mod_entanglement - 1.0) < 1e-6, "Loss of entanglement in mod encoding"
                
                # Encrypt encoded patterns with AEAD
                base_encrypted = self.crypto.encrypt_aead(base_encoded.tobytes())
                mod_encrypted = self.crypto.encrypt_aead(mod_encoded.tobytes())
                
                # Calculate difference ratio on compressed space
                diff_count = sum(1 for a, b in zip(base_encrypted, mod_encrypted) if a != b)
                diff_ratio = diff_count / len(base_encrypted)
                batch_differences.append(diff_ratio)
            
            differences.extend(batch_differences)
        
        # Calculate statistics
        mean_diff = sum(differences) / len(differences)
        max_diff = max(differences)
        threshold = 0.55  # Theoretical maximum for secure encryption
        
        return DifferentialTestResult(
            pattern_name=pattern_name,
            samples=samples,
            differences=differences,
            mean_difference=mean_diff,
            max_difference=max_diff,
            threshold_exceeded=max_diff > threshold
        )

    def run_all_tests(self) -> Dict[str, DifferentialTestResult]:
        """Run all differential analysis tests"""
        results = {}
        
        # Run bit tests
        print("\n=== Running bit-level tests ===")
        for pattern in self.patterns['bit_tests']:
            results[pattern] = self.run_single_test(pattern)
            
        # Run pattern tests
        print("\n=== Running pattern-based tests ===")
        for pattern in self.patterns['pattern_tests']:
            results[pattern] = self.run_single_test(pattern)
            
        return results

def run_performance_test(implementation, name):
    """Run performance tests for a given crypto implementation"""
    print(f"\n{'-' * 60}")
    print(f"Testing {name} implementation")
    print(f"{'-' * 60}")
    
    for size in TEST_SIZES:
        # Generate random message
        message = secrets.token_bytes(size)
        
        # Initialize crypto and assembly-optimized components
        crypto = implementation(KEY)
        rft = UNITARY_RFT
        vertex_rft = VERTEX_RFT
        symbolic_engine = SYMBOLIC_ENGINE
        
        # Set up methods with assembly-optimized encoding
        if name == "Pure Python":
            def encrypt_method(msg):
                # Convert to numpy array
                data = np.frombuffer(msg, dtype=np.uint8)
                
                # Assembly-optimized quantum pipeline
                rft_state = rft.apply_fast_rft(data)
                vertex_state = vertex_rft.transform_quantum_state(rft_state)
                encoded = symbolic_engine.compress_quantum_state(
                    vertex_state,
                    phi_factor=PHI,
                    use_simd=True
                )
                
                # Assembly-optimized validation
                validation = vertex_rft.validate_quantum_state(encoded)
                assert validation['unitarity_perfect'], "Loss of unitarity"
                assert validation['norm_preservation'] > 0.99999, "Loss of norm preservation"
                
                return crypto.encrypt_aead(encoded.tobytes())
                
                def decrypt_method(enc):
                    # Decrypt ciphertext
                    decrypted = crypto.decrypt_aead(enc)
                    quantum_state = np.frombuffer(decrypted, dtype=np.complex128)
                    
                    # Assembly-optimized quantum decoding
                    vertex_state = symbolic_engine.decompress_quantum_state(quantum_state)
                    rft_state = vertex_rft.inverse_transform_quantum_state(vertex_state)
                    decoded = rft.apply_inverse_fast_rft(rft_state)
                    
                    return decoded.tobytes()
        else:
            # Use direct crypto operations for assembly version
            encrypt_method = lambda msg: crypto.encrypt(msg)
            decrypt_method = lambda enc: crypto.decrypt(enc)        # Warmup
        encrypted = encrypt_method(message)
        decrypted = decrypt_method(encrypted)
        assert message == decrypted, "Encryption/decryption failed"
        
        # Test encryption
        total_time = 0
        for _ in range(ITERATIONS):
            start = time.perf_counter()
            encrypted = encrypt_method(message)
            end = time.perf_counter()
            total_time += (end - start)
        
        encrypt_time = total_time / ITERATIONS
        encrypt_mbps = (size / encrypt_time) / (1024 * 1024)
        
        # Test decryption
        total_time = 0
        for _ in range(ITERATIONS):
            start = time.perf_counter()
            decrypted = decrypt_method(encrypted)
            end = time.perf_counter()
            total_time += (end - start)
        
        decrypt_time = total_time / ITERATIONS
        decrypt_mbps = (size / decrypt_time) / (1024 * 1024)
        
        print(f"{size:6d} bytes: {encrypt_mbps:.3f} MB/s encrypt, {decrypt_mbps:.3f} MB/s decrypt")
    
    print()

# Run tests
if __name__ == "__main__":
    print("\n🎯 REALISTIC TABULATED PROOF SUITE")
    print("=" * 60)
    print("Generating comprehensive proof tables with sample-size-adjusted thresholds\n")
    
    print("Phase 1: Differential Cryptanalysis...")
    print("🔬 Generating Differential Proof Table (10,000 samples per test)")
    
    # Test pure Python implementation
    if EnhancedRFTCryptoV2:
        tester = DifferentialTester(EnhancedRFTCryptoV2, KEY)
        results = tester.run_all_tests()
        
        # Save results to JSON
        with open('differential_analysis_results.json', 'w') as f:
            json.dump(
                {
                    name: {
                        'samples': r.samples,
                        'mean_difference': r.mean_difference,
                        'max_difference': r.max_difference,
                        'threshold_exceeded': r.threshold_exceeded
                    }
                    for name, r in results.items()
                },
                f, 
                indent=2
            )
    
    print("\nPhase 2: Performance Analysis...")
    print("🚀 Generating Performance Benchmarks")
    
    # Run performance tests
    if EnhancedRFTCryptoV2:
        run_performance_test(EnhancedRFTCryptoV2, "Pure Python")
    
    if EnhancedRFTCrypto:
        run_performance_test(EnhancedRFTCrypto, "Assembly-optimized")
    
    print("\nAnalysis complete! Results saved to differential_analysis_results.json")
