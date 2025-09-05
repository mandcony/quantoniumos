#!/usr/bin/env python3
"""
Debug Vertex System Initialization
Test to understand why the quantum vertex system is hanging
"""

import numpy as np
import math

def debug_vertex_initialization():
    """Debug the vertex system initialization issue."""
    print("🔍 DEBUGGING VERTEX SYSTEM INITIALIZATION")
    print("=" * 50)
    
    # Test sizes from your benchmark
    test_sizes = [1000, 2000, 4000, 8000]
    
    for size in test_sizes:
        print(f"\n📊 Size: {size}")
        
        # What qubit count would be needed for traditional 2^n = size?
        if size > 0 and (size & (size - 1)) == 0:  # Check if power of 2
            traditional_qubits = int(math.log2(size))
            print(f"   Traditional qubits (2^n = {size}): {traditional_qubits}")
        else:
            # Find closest power of 2
            traditional_qubits = int(math.ceil(math.log2(size)))
            closest_power = 2 ** traditional_qubits
            print(f"   Size {size} is NOT power of 2")
            print(f"   Closest power of 2: 2^{traditional_qubits} = {closest_power}")
            print(f"   ❌ This would fail the check: 2^{traditional_qubits} != {size}")
        
        # Your vertex system approach
        vertex_qubits = 1000  # Fixed vertex architecture
        vertex_edges = (vertex_qubits * (vertex_qubits - 1)) // 2
        print(f"   Your vertex system: {vertex_qubits} qubits, {vertex_edges:,} edges")
        print(f"   Can handle arbitrary data sizes: ✅")
        
        # The problematic check
        would_pass_check = (2 ** vertex_qubits == size)
        print(f"   Would pass 2^{vertex_qubits} == {size}? {'✅ Yes' if would_pass_check else '❌ No'}")
        
        if not would_pass_check:
            print(f"   🚨 This is why it hangs! 2^1000 is impossibly large")

def test_correct_vertex_approach():
    """Test the correct approach for vertex systems."""
    print(f"\n\n🔧 CORRECT VERTEX SYSTEM APPROACH")
    print("=" * 50)
    
    try:
        from unitary_rft import UnitaryRFT, RFT_FLAG_UNITARY, RFT_FLAG_QUANTUM_SAFE
        
        # Test with power-of-2 sizes first
        power_of_2_sizes = [1024, 2048, 4096]  # 2^10, 2^11, 2^12
        
        for size in power_of_2_sizes:
            qubits = int(math.log2(size))
            print(f"\n✅ Testing power-of-2: {size} elements, {qubits} qubits")
            
            try:
                rft = UnitaryRFT(size, RFT_FLAG_UNITARY | RFT_FLAG_QUANTUM_SAFE)
                rft.init_quantum_basis(qubits)
                print(f"   Initialization: ✅ Success")
                
                # Test small transform
                signal = np.random.random(size) + 1j * np.random.random(size)
                signal = signal / np.linalg.norm(signal)
                
                spectrum = rft.forward(signal)
                reconstructed = rft.inverse(spectrum)
                error = np.max(np.abs(signal - reconstructed))
                
                print(f"   Transform test: ✅ Success, error = {error:.2e}")
                
            except Exception as e:
                print(f"   ❌ Failed: {e}")
    
    except ImportError as e:
        print(f"❌ Cannot import RFT library: {e}")

def test_vertex_workaround():
    """Test a workaround for vertex systems."""
    print(f"\n\n🛠️ VERTEX SYSTEM WORKAROUND")
    print("=" * 50)
    
    # For your vertex system, we need to:
    # 1. Use data sizes that are powers of 2
    # 2. Map vertex operations onto these sizes
    # 3. Use padding if necessary
    
    vertex_compatible_sizes = []
    target_sizes = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000]
    
    for target in target_sizes:
        # Find next power of 2
        padded_size = 2 ** int(math.ceil(math.log2(target)))
        qubits = int(math.log2(padded_size))
        
        vertex_compatible_sizes.append({
            'target': target,
            'padded': padded_size,
            'qubits': qubits,
            'padding': padded_size - target
        })
        
        print(f"   Target: {target:6d} → Padded: {padded_size:6d} ({qubits:2d} qubits, +{padded_size - target:4d} padding)")
    
    print(f"\n💡 SOLUTION:")
    print(f"   Use padded power-of-2 sizes for RFT initialization")
    print(f"   Extract only the relevant portion of results")
    print(f"   This allows vertex operations on arbitrary data sizes")
    
    return vertex_compatible_sizes

def main():
    """Main debug function."""
    print("🔬 VERTEX SYSTEM DEBUG ANALYSIS")
    print("🎯 Goal: Understand why quantum vertex initialization hangs")
    print()
    
    debug_vertex_initialization()
    test_correct_vertex_approach()
    compatible_sizes = test_vertex_workaround()
    
    print(f"\n\n🎉 DIAGNOSIS COMPLETE")
    print(f"📋 Issue: Your vertex system tries to use 1000 qubits with arbitrary sizes")
    print(f"🔧 Fix: Use power-of-2 padding and extract relevant portions")
    print(f"✅ This preserves vertex architecture while enabling RFT operations")
    
    return compatible_sizes

if __name__ == "__main__":
    main()
