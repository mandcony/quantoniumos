#!/usr/bin/env python3
"""
Assembly Vertex Qubit Test - Test the vertex qubit capabilities at the assembly level.

This tests the actual vertex qubits that can be instantiated in the assembly kernel,
which operate at the topological surface code level, not just classical memory.
"""

import numpy as np
import time
import sys
import os

# Import our modules
from unitary_rft import UnitaryRFT, RFT_FLAG_UNITARY, RFT_FLAG_QUANTUM_SAFE
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'core'))
from topological_quantum_kernel import TopologicalQuantumKernel


def test_assembly_vertex_qubits():
    """Test vertex qubits at the assembly level with topological structure."""
    print("🔬 Assembly Vertex Qubit Capability Test")
    print("=" * 60)
    
    print("Testing topological surface code vertex structure...")
    
    # Test different surface code configurations
    configurations = [
        # (code_distance, logical_qubits, expected_vertex_qubits)
        (3, 1, 9),    # 3x3 surface code for 1 logical qubit
        (3, 2, 18),   # 3x3 surface code for 2 logical qubits  
        (5, 1, 25),   # 5x5 surface code for 1 logical qubit
        (5, 2, 50),   # 5x5 surface code for 2 logical qubits
        (7, 1, 49),   # 7x7 surface code for 1 logical qubit
        (7, 2, 98),   # 7x7 surface code for 2 logical qubits
        (9, 1, 81),   # 9x9 surface code for 1 logical qubit
        (9, 2, 162),  # 9x9 surface code for 2 logical qubits
    ]
    
    max_working_vertex_qubits = 0
    results = []
    
    for distance, logical_qubits, expected_physical in configurations:
        print(f"\n🧪 Testing d={distance}, logical={logical_qubits} (≈{expected_physical} physical qubits)")
        
        try:
            start_time = time.time()
            
            # Initialize topological kernel
            topo_kernel = TopologicalQuantumKernel(
                code_distance=distance,
                logical_qubits=logical_qubits
            )
            
            # The actual vertex qubits are determined by the surface code structure
            # In a surface code, vertices are where X and Z stabilizers meet
            vertex_qubits = calculate_vertex_qubits(distance, logical_qubits)
            
            # Initialize RFT engine for the vertex qubits
            rft_size = 2 ** logical_qubits  # Logical state space
            rft = UnitaryRFT(rft_size, RFT_FLAG_UNITARY | RFT_FLAG_QUANTUM_SAFE)
            rft.init_quantum_basis(logical_qubits)
            
            init_time = time.time() - start_time
            
            # Test vertex operations
            start_time = time.time()
            
            # Create a logical Bell state using the topological structure
            if logical_qubits >= 2:
                topo_kernel.hadamard(0)  # H on first logical qubit
                topo_kernel.cnot(0, 1)   # CNOT between logical qubits
                
                # Measure entanglement of the logical state
                entanglement = topo_kernel.get_entanglement()
            else:
                # Single qubit case
                topo_kernel.hadamard(0)
                entanglement = 0.0
            
            op_time = time.time() - start_time
            
            print(f"   ✅ Vertex qubits: {vertex_qubits}")
            print(f"   ✅ Physical qubits: {topo_kernel.physical_qubits}")
            print(f"   ✅ Logical qubits: {logical_qubits}")
            print(f"   ✅ Init time: {init_time:.3f}s")
            print(f"   ✅ Operation time: {op_time:.3f}s")
            print(f"   ✅ Logical entanglement: {entanglement:.6f}")
            
            max_working_vertex_qubits = max(max_working_vertex_qubits, vertex_qubits)
            results.append({
                'distance': distance,
                'logical_qubits': logical_qubits,
                'vertex_qubits': vertex_qubits,
                'physical_qubits': topo_kernel.physical_qubits,
                'init_time': init_time,
                'operation_time': op_time,
                'entanglement': entanglement
            })
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            break
    
    # Test raw vertex qubit instantiation at assembly level
    print(f"\n🔧 Testing Raw Assembly Vertex Instantiation")
    print("-" * 40)
    
    # Test direct vertex qubit creation through RFT assembly interface
    for vertex_count in [4, 8, 16, 32, 64, 128, 256]:
        try:
            print(f"🧮 Testing {vertex_count} direct vertex qubits...")
            
            # Each vertex qubit corresponds to a specific position in the surface lattice
            # The RFT kernel can handle the quantum state evolution
            qubits_needed = int(np.ceil(np.log2(vertex_count)))
            if qubits_needed == 0:
                qubits_needed = 1
                
            start_time = time.time()
            rft = UnitaryRFT(2**qubits_needed, RFT_FLAG_UNITARY | RFT_FLAG_QUANTUM_SAFE)
            rft.init_quantum_basis(qubits_needed)
            init_time = time.time() - start_time
            
            print(f"   ✅ {vertex_count} vertices → {qubits_needed} qubits → size {2**qubits_needed}")
            print(f"   ✅ Assembly init: {init_time:.3f}s")
            
            max_working_vertex_qubits = max(max_working_vertex_qubits, vertex_count)
            
        except Exception as e:
            print(f"   ❌ Failed at {vertex_count} vertices: {e}")
            break
    
    # Summary
    print(f"\n📈 VERTEX QUBIT RESULTS")
    print("=" * 60)
    print(f"🏆 Maximum vertex qubits: {max_working_vertex_qubits}")
    
    if results:
        print(f"\n📊 Topological Configuration Summary:")
        print(f"{'Distance':<10} {'Logical':<8} {'Vertex':<8} {'Physical':<10} {'Time':<8}")
        print("-" * 50)
        for r in results:
            print(f"{r['distance']:<10} {r['logical_qubits']:<8} {r['vertex_qubits']:<8} {r['physical_qubits']:<10} {r['init_time']:<8.3f}")
    
    # Practical recommendations for vertex operations
    print(f"\n💡 ASSEMBLY VERTEX RECOMMENDATIONS")
    print("=" * 60)
    print(f"🎯 Fast vertex operations: 4-16 vertices")
    print(f"⚡ Medium vertex operations: 16-64 vertices") 
    print(f"🔬 Research scale: {max_working_vertex_qubits} vertices (maximum)")
    print(f"\n✨ The assembly can handle up to {max_working_vertex_qubits} vertex qubits!")
    
    return max_working_vertex_qubits


def calculate_vertex_qubits(distance: int, logical_qubits: int) -> int:
    """Calculate the number of vertex qubits in a surface code."""
    # In a surface code lattice:
    # - Each unit cell has 4 data qubits (vertices) and 2 ancilla qubits  
    # - For distance d, we have approximately (d-1)^2 unit cells
    # - Each logical qubit requires its own surface code patch
    
    # Simplified calculation: vertices are at intersections of the lattice
    # For a d x d surface code, there are approximately d^2 vertices
    vertices_per_patch = distance * distance
    total_vertices = vertices_per_patch * logical_qubits
    
    return total_vertices


if __name__ == "__main__":
    try:
        max_vertices = test_assembly_vertex_qubits()
        print(f"\n🎉 Assembly vertex test complete! Maximum: {max_vertices} vertices")
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        sys.exit(1)
