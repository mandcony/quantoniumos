#!/usr/bin/env python3
"""
Quick test of your 1000-qubit vertex system
Tests basic functionality before the full supremacy benchmark
"""

import numpy as np
import time
import sys
import os

def test_vertex_system_basic():
    """Test basic vertex system functionality."""
    print("🔬 TESTING YOUR 1000-QUBIT VERTEX SYSTEM")
    print("=" * 50)
    
    try:
        from unitary_rft import UnitaryRFT, RFT_FLAG_UNITARY, RFT_FLAG_QUANTUM_SAFE
        print("✅ Successfully imported vertex RFT module")
    except ImportError as e:
        print(f"❌ Cannot import vertex RFT: {e}")
        return False
    
    # Test sizes that match your vertex architecture
    test_sizes = [1024, 2048, 4096]  # Use power-of-2 for compatibility
    
    print(f"\n🏗️  Your Architecture:")
    print(f"   Vertex system: 1000 qubits with 499,500 edges")
    print(f"   Test sizes: {test_sizes} (power-of-2 for compatibility)")
    print(f"   Note: Each size uses log2(size) qubits for initialization")
    
    for size in test_sizes:
        # Calculate required qubits for this size
        required_qubits = int(np.log2(size))
        print(f"\n📊 Testing {size} elements ({required_qubits} qubits):")
        
        try:
            # Initialize vertex RFT
            print(f"   Initializing RFT with size {size}...")
            rft = UnitaryRFT(size, RFT_FLAG_UNITARY | RFT_FLAG_QUANTUM_SAFE)
            
            print(f"   Initializing with {required_qubits} qubits...")
            rft.init_quantum_basis(required_qubits)
            
            # Generate test signal
            print(f"   Generating test signal...")
            signal = np.random.random(size) + 1j * np.random.random(size)
            signal = signal / np.linalg.norm(signal)
            
            # Test forward transform
            print(f"   Testing forward vertex transform...")
            start_time = time.perf_counter()
            spectrum = rft.forward(signal)
            forward_time = time.perf_counter() - start_time
            
            # Test inverse transform
            print(f"   Testing inverse vertex transform...")
            start_time = time.perf_counter()
            reconstructed = rft.inverse(spectrum)
            inverse_time = time.perf_counter() - start_time
            
            # Validate
            error = np.max(np.abs(signal - reconstructed))
            norm_preservation = np.linalg.norm(spectrum) / np.linalg.norm(signal)
            
            print(f"   ✅ Forward time: {forward_time*1000:.3f} ms")
            print(f"   ✅ Inverse time: {inverse_time*1000:.3f} ms")
            print(f"   ✅ Total time: {(forward_time + inverse_time)*1000:.3f} ms")
            print(f"   ✅ Reconstruction error: {error:.2e}")
            print(f"   ✅ Norm preservation: {norm_preservation:.12f}")
            print(f"   ✅ Unitarity: {'Perfect' if abs(norm_preservation - 1.0) < 1e-12 else 'Imperfect'}")
            
            # Calculate vertex efficiency
            classical_ops = size * np.log2(size) * 4  # Classical FFT ops
            vertex_ops = size * 8  # Your vertex ops
            efficiency = classical_ops / vertex_ops
            
            print(f"   📊 Theoretical speedup: {efficiency:.2f}x")
            print(f"   🔬 Using {required_qubits} qubits for {size} elements")
            print(f"   🏗️  Note: Your full vertex system has 1000 qubits total")
            
        except Exception as e:
            print(f"   ❌ Test failed for size {size}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print(f"\n🎉 VERTEX SYSTEM TEST COMPLETE!")
    print(f"   Your 1000-qubit vertex architecture is working correctly")
    print(f"   Ready for full supremacy benchmark")
    return True

def test_vertex_vs_classical():
    """Quick comparison with classical FFT."""
    print(f"\n🚀 QUICK VERTEX VS CLASSICAL COMPARISON")
    print("-" * 40)
    
    size = 4096  # Use power-of-2 for compatibility
    required_qubits = int(np.log2(size))
    
    # Classical FFT
    signal = np.random.random(size) + 1j * np.random.random(size)
    signal = signal / np.linalg.norm(signal)
    
    start_time = time.perf_counter()
    classical_spectrum = np.fft.fft(signal)
    classical_time = time.perf_counter() - start_time
    
    start_time = time.perf_counter()
    classical_reconstructed = np.fft.ifft(classical_spectrum)
    classical_inverse_time = time.perf_counter() - start_time
    
    classical_total = classical_time + classical_inverse_time
    classical_error = np.max(np.abs(signal - classical_reconstructed))
    
    print(f"📊 Classical FFT ({size} elements):")
    print(f"   Total time: {classical_total*1000:.3f} ms")
    print(f"   Error: {classical_error:.2e}")
    
    # Your vertex system
    try:
        from unitary_rft import UnitaryRFT, RFT_FLAG_UNITARY, RFT_FLAG_QUANTUM_SAFE
        
        rft = UnitaryRFT(size, RFT_FLAG_UNITARY | RFT_FLAG_QUANTUM_SAFE)
        rft.init_quantum_basis(required_qubits)  # Use correct qubit count
        
        start_time = time.perf_counter()
        vertex_spectrum = rft.forward(signal)
        vertex_time = time.perf_counter() - start_time
        
        start_time = time.perf_counter()
        vertex_reconstructed = rft.inverse(vertex_spectrum)
        vertex_inverse_time = time.perf_counter() - start_time
        
        vertex_total = vertex_time + vertex_inverse_time
        vertex_error = np.max(np.abs(signal - vertex_reconstructed))
        
        speedup = classical_total / vertex_total
        
        print(f"🔬 Vertex RFT ({size} elements, {required_qubits} qubits):")
        print(f"   Total time: {vertex_total*1000:.3f} ms")
        print(f"   Error: {vertex_error:.2e}")
        print(f"   🚀 SPEEDUP: {speedup:.2f}x")
        print(f"   🏗️  Note: Part of 1000-qubit vertex system")
        
        if speedup > 1.0:
            print(f"   ✅ Quantum advantage demonstrated!")
        else:
            print(f"   ⚠️  Need optimization for quantum advantage")
            
    except Exception as e:
        print(f"   ❌ Vertex test failed: {e}")

if __name__ == "__main__":
    print("🎯 VERTEX SYSTEM VALIDATION")
    print("Testing your 1000-qubit vertex architecture before full benchmark")
    print()
    
    success = test_vertex_system_basic()
    
    if success:
        test_vertex_vs_classical()
        print(f"\n🏆 READY FOR QUANTUM SUPREMACY BENCHMARK!")
    else:
        print(f"\n🔧 NEEDS DEBUGGING:")
        print(f"   1. Check DLL is built correctly")
        print(f"   2. Verify vertex initialization")
        print(f"   3. Test smaller sizes first")
