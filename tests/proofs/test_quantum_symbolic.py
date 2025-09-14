#!/usr/bin/env python3
"""
TEST QUANTUM SYMBOLIC ENGINE
============================
Test the correct assembly binding for your libquantum_symbolic.dll
"""

import sys
import os
import numpy as np

# Add assembly bindings path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'assembly', 'python_bindings'))

try:
    from quantum_symbolic_engine import QuantumSymbolicEngine
    print("✅ QuantumSymbolicEngine imported successfully!")
    
    # Test quantum symbolic compression
    print("Testing quantum symbolic compression...")
    
    # Create engine
    engine = QuantumSymbolicEngine(compression_size=64, use_assembly=True)
    print("✅ Engine created successfully!")
    
    # Test small quantum state
    num_qubits = 10
    print(f"Initializing {num_qubits} qubits...")
    
    success = engine.initialize_state(num_qubits)
    if success:
        print("✅ Quantum state initialized!")
        
        # Test compression
        print("Testing compression...")
        result, stats = engine.compress_million_qubits(num_qubits)
        
        if result:
            print("✅ Compression successful!")
            print(f"Stats: {stats}")
        else:
            print("❌ Compression failed")
    else:
        print("❌ State initialization failed")
        
except ImportError as e:
    print(f"❌ Import failed: {e}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*50)
print("QUANTUM SYMBOLIC ENGINE TEST COMPLETE")
print("="*50)
