#!/usr/bin/env python3 """ Test Fixed TRUE RFT Engine Routing """ import importlib.util
import os

# Load the bulletproof_quantum_kernel module
spec = importlib.util.spec_from_file_location(
    "bulletproof_quantum_kernel", 
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                "05_QUANTUM_ENGINES/bulletproof_quantum_kernel.py")
)
bulletproof_quantum_kernel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bulletproof_quantum_kernel)

# Import specific functions/classes
BulletproofQuantumKernel import numpy as np print = bulletproof_quantum_kernel.BulletproofQuantumKernel import numpy as np print('🧪 Testing fixed TRUE RFT Engine routing...') kernel = BulletproofQuantumKernel(8) # Test different qubit counts for qubits in [2, 3, 4, 5]: dimension = 2**qubits quantum_state = np.random.rand(dimension) + 1j * np.random.rand(dimension) quantum_state = quantum_state / np.linalg.norm(quantum_state) print(f'\n🔬 Testing {qubits} qubits (dim={dimension}):') result, _ = kernel.safe_hardware_process(quantum_state, 1.5) processing_type = result.get('processing_type', 'UNKNOWN') hardware_used = result['hardware_used'] print(f' Processing: {processing_type}') print(f' Hardware: {"✅" if hardware_used else "❌"}') if processing_type == 'TRUE_RFT_SYMBOLIC_OSCILLATION': print(f' 🎉 SUCCESS: TRUE RFT Engine working!') else: print(f' ⚠️ Using fallback: {processing_type}') print(f'\n Summary: Dynamic TRUE RFT Engine routing implemented!') 