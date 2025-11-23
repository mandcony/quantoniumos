#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
Test Vector Generator for QuantoniumOS Verilog Implementation
Generates reference outputs from Python to verify hardware correctness
"""

import sys
import numpy as np
import struct
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from algorithms.rft.core.canonical_true_rft import CanonicalTrueRFT
    from algorithms.crypto.crypto_benchmarks.rft_sis.rft_sis_hash_v31 import RFTSISHashV31, Point2D
    print("✓ Successfully imported QuantoniumOS modules")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Make sure you're running from the QuantoniumOS root directory")
    sys.exit(1)


def float_to_fixed_point(value, int_bits=16, frac_bits=16):
    """Convert float to Q16.16 fixed-point representation"""
    total_bits = int_bits + frac_bits
    scale = 2 ** frac_bits
    fixed = int(value * scale)
    
    # Handle overflow
    max_val = (1 << (total_bits - 1)) - 1
    min_val = -(1 << (total_bits - 1))
    fixed = max(min_val, min(max_val, fixed))
    
    # Convert to unsigned 32-bit representation
    if fixed < 0:
        fixed = (1 << total_bits) + fixed
    
    return fixed & 0xFFFFFFFF


def fixed_point_to_float(fixed, int_bits=16, frac_bits=16):
    """Convert Q16.16 fixed-point to float"""
    total_bits = int_bits + frac_bits
    scale = 2 ** frac_bits
    
    # Handle signed
    if fixed & (1 << (total_bits - 1)):
        fixed = fixed - (1 << total_bits)
    
    return fixed / scale


def generate_rft_test_vectors(output_file="test_vectors_rft.hex"):
    """Generate RFT transform test vectors"""
    print("\n" + "="*60)
    print("Generating RFT Test Vectors")
    print("="*60)
    
    N = 64  # Transform size
    rft = CanonicalTrueRFT(N)
    
    # Test cases
    test_cases = [
        {
            'name': 'DC Signal',
            'input': np.ones(N, dtype=complex)
        },
        {
            'name': 'Pure Frequency (k=1)',
            'input': np.exp(2j * np.pi * np.arange(N) / N)
        },
        {
            'name': 'Random Signal',
            'input': np.random.randn(N) + 1j * np.random.randn(N)
        },
        {
            'name': 'Impulse',
            'input': np.zeros(N, dtype=complex)
        }
    ]
    test_cases[3]['input'][0] = 1.0
    
    with open(output_file, 'w') as f:
        f.write("// RFT Test Vectors\n")
        f.write(f"// Transform size: {N}\n")
        f.write("// Format: test_id, input_real[0..N-1], input_imag[0..N-1], output_real[0..N-1], output_imag[0..N-1]\n\n")
        
        for test_id, test_case in enumerate(test_cases):
            print(f"\nTest {test_id}: {test_case['name']}")
            
            # Normalize input
            x = test_case['input'] / np.linalg.norm(test_case['input'])
            
            # Python RFT
            y = rft.forward_transform(x)
            
            # Verify round-trip
            x_reconstructed = rft.inverse_transform(y)
            error = np.linalg.norm(x - x_reconstructed)
            print(f"  Round-trip error: {error:.2e}")
            
            # Convert to fixed-point
            f.write(f"// Test {test_id}: {test_case['name']}\n")
            f.write(f"@{test_id:04X}\n")
            
            # Input (real part)
            for i in range(N):
                fixed = float_to_fixed_point(x[i].real)
                f.write(f"{fixed:08X} ")
                if (i + 1) % 8 == 0:
                    f.write("\n")
            
            # Input (imaginary part)
            for i in range(N):
                fixed = float_to_fixed_point(x[i].imag)
                f.write(f"{fixed:08X} ")
                if (i + 1) % 8 == 0:
                    f.write("\n")
            
            # Expected output (real part)
            for i in range(N):
                fixed = float_to_fixed_point(y[i].real)
                f.write(f"{fixed:08X} ")
                if (i + 1) % 8 == 0:
                    f.write("\n")
            
            # Expected output (imaginary part)
            for i in range(N):
                fixed = float_to_fixed_point(y[i].imag)
                f.write(f"{fixed:08X} ")
                if (i + 1) % 8 == 0:
                    f.write("\n")
            
            f.write("\n")
            
            # Print first few values for verification
            print(f"  Input[0]: {x[0].real:.6f} + {x[0].imag:.6f}j")
            print(f"  Output[0]: {y[0].real:.6f} + {y[0].imag:.6f}j")
            print(f"  Energy: {np.sum(np.abs(y)**2):.6f}")
    
    print(f"\n✓ RFT test vectors saved to {output_file}")


def generate_sis_hash_test_vectors(output_file="test_vectors_sis.hex"):
    """Generate RFT-SIS hash test vectors"""
    print("\n" + "="*60)
    print("Generating SIS Hash Test Vectors")
    print("="*60)
    
    hasher = RFTSISHashV31()
    
    # Test cases for avalanche effect
    test_cases = [
        Point2D(1.0, 2.0),
        Point2D(1.000001, 2.0),      # Tiny x change
        Point2D(1.0, 2.000001),      # Tiny y change
        Point2D(3.14159, 2.71828),   # π and e
        Point2D(0.0, 0.0),           # Origin
        Point2D(-100.5, 200.3),      # Negative coordinates
        Point2D(1e10, 1e-10),        # Large dynamic range
    ]
    
    with open(output_file, 'w') as f:
        f.write("// RFT-SIS Hash Test Vectors\n")
        f.write("// Format: test_id, coordinate_x (double), coordinate_y (double), hash_digest (256-bit)\n\n")
        
        previous_hash = None
        
        for test_id, point in enumerate(test_cases):
            print(f"\nTest {test_id}: ({point.x}, {point.y})")
            
            # Compute hash
            digest = hasher.hash_point(point)
            
            # Avalanche analysis
            if previous_hash is not None:
                bit_diff = sum(bin(a ^ b).count('1') for a, b in zip(digest, previous_hash))
                pct = bit_diff / 256 * 100
                print(f"  Bit difference from previous: {bit_diff}/256 ({pct:.1f}%)")
            
            previous_hash = digest
            
            # Write to file
            f.write(f"// Test {test_id}: ({point.x:.10f}, {point.y:.10f})\n")
            f.write(f"@{test_id:04X}\n")
            
            # Coordinates as double-precision bits
            x_bits = struct.pack('d', point.x)
            y_bits = struct.pack('d', point.y)
            
            f.write(f"// X coordinate: {x_bits.hex()}\n")
            f.write(f"{x_bits.hex()} ")
            
            f.write(f"// Y coordinate: {y_bits.hex()}\n")
            f.write(f"{y_bits.hex()} ")
            
            # Hash digest (256-bit = 32 bytes)
            f.write(f"// Expected hash digest:\n")
            f.write(f"{digest.hex()}\n\n")
            
            print(f"  Hash: {digest.hex()[:32]}...")
    
    print(f"\n✓ SIS hash test vectors saved to {output_file}")


def generate_feistel_test_vectors(output_file="test_vectors_feistel.hex"):
    """Generate Feistel-48 cipher test vectors"""
    print("\n" + "="*60)
    print("Generating Feistel-48 Test Vectors")
    print("="*60)
    
    try:
        from algorithms.rft.core.enhanced_rft_crypto_v2 import EnhancedRFTCryptoV2
    except ImportError:
        print("✗ Cannot import EnhancedRFTCryptoV2 - skipping Feistel tests")
        return
    
    # Test cases
    master_key = b'\x01\x23\x45\x67\x89\xAB\xCD\xEF' * 4  # 32 bytes
    
    test_cases = [
        b'\x00' * 16,                           # All zeros
        b'\xFF' * 16,                           # All ones
        b'\xDE\xAD\xBE\xEF\xCA\xFE\xBA\xBE' * 2,  # Repeated pattern
        bytes(range(16)),                       # Sequential
    ]
    
    cipher = EnhancedRFTCryptoV2(master_key)
    
    with open(output_file, 'w') as f:
        f.write("// Feistel-48 Cipher Test Vectors\n")
        f.write(f"// Master Key: {master_key.hex()}\n")
        f.write("// Format: test_id, plaintext (128-bit), ciphertext (128-bit)\n\n")
        
        for test_id, plaintext in enumerate(test_cases):
            print(f"\nTest {test_id}:")
            print(f"  Plaintext:  {plaintext.hex()}")
            
            # Encrypt
            ciphertext = cipher.encrypt(plaintext)
            print(f"  Ciphertext: {ciphertext.hex()}")
            
            # Decrypt (verify correctness)
            decrypted = cipher.decrypt(ciphertext)
            assert decrypted == plaintext, "Decryption failed!"
            
            # Write to file
            f.write(f"// Test {test_id}\n")
            f.write(f"@{test_id:04X}\n")
            f.write(f"// Plaintext:  {plaintext.hex()}\n")
            f.write(f"{plaintext.hex()}\n")
            f.write(f"// Ciphertext: {ciphertext.hex()}\n")
            f.write(f"{ciphertext.hex()}\n\n")
    
    print(f"\n✓ Feistel test vectors saved to {output_file}")


def generate_golden_ratio_constants(output_file="golden_ratio_constants.vh"):
    """Generate Verilog header with golden ratio constants"""
    print("\n" + "="*60)
    print("Generating Golden Ratio Constants")
    print("="*60)
    
    phi = (1 + np.sqrt(5)) / 2
    
    # Various fixed-point representations
    phi_q16_16 = float_to_fixed_point(phi, 16, 16)
    phi_q8_24 = float_to_fixed_point(phi, 8, 24)
    phi_64bit = int(phi * (2**63))  # Q1.63
    
    with open(output_file, 'w') as f:
        f.write("// Golden Ratio Constants for Verilog\n")
        f.write("// φ = (1 + √5) / 2 ≈ 1.618033988749894848\n\n")
        
        f.write(f"`define PHI_FLOAT {phi:.18f}\n")
        f.write(f"`define PHI_Q16_16 32'h{phi_q16_16:08X}  // Q16.16 fixed-point\n")
        f.write(f"`define PHI_Q8_24  32'h{phi_q8_24:08X}  // Q8.24 fixed-point\n")
        f.write(f"`define PHI_64BIT  64'h{phi_64bit:016X}  // Q1.63 fixed-point\n\n")
        
        # Golden ratio powers for key schedule
        f.write("// Powers of φ (for key derivation)\n")
        for i in range(1, 49):  # 48 rounds
            phi_power = phi ** i
            phi_power_fixed = float_to_fixed_point(phi_power, 16, 16)
            f.write(f"`define PHI_POWER_{i:02d} 32'h{phi_power_fixed:08X}  // φ^{i}\n")
        
        f.write("\n// Inverse golden ratio: 1/φ = φ - 1\n")
        inv_phi = 1.0 / phi
        inv_phi_fixed = float_to_fixed_point(inv_phi, 16, 16)
        f.write(f"`define INV_PHI_Q16_16 32'h{inv_phi_fixed:08X}  // 1/φ in Q16.16\n")
    
    print(f"φ = {phi:.18f}")
    print(f"φ (Q16.16) = 0x{phi_q16_16:08X}")
    print(f"\n✓ Golden ratio constants saved to {output_file}")


def main():
    """Generate all test vectors"""
    print("="*60)
    print("QuantoniumOS Hardware Test Vector Generator")
    print("="*60)
    
    # Create output directory
    output_dir = Path("hardware_test_vectors")
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Generate test vectors
    generate_golden_ratio_constants(output_dir / "golden_ratio_constants.vh")
    generate_rft_test_vectors(output_dir / "test_vectors_rft.hex")
    generate_sis_hash_test_vectors(output_dir / "test_vectors_sis.hex")
    generate_feistel_test_vectors(output_dir / "test_vectors_feistel.hex")
    
    print("\n" + "="*60)
    print("✓ All test vectors generated successfully!")
    print("="*60)
    print(f"\nTest vectors saved in: {output_dir}/")
    print("\nNext steps:")
    print("1. Copy test vectors to your Verilog simulation directory")
    print("2. Update testbench to read vectors from .hex files")
    print("3. Run simulation: make -f quantoniumos_engines_makefile sim")
    print("4. Compare hardware outputs against expected values")


if __name__ == "__main__":
    main()
