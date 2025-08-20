#!/usr/bin/env python3
"""
Launch script for the 150 Qubit Quantum Processor Frontend
"""

import sys
import os

# Add the current directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Add the apps directory to the path
apps_dir = os.path.join(current_dir, "11_QUANTONIUMOS", "apps")
sys.path.insert(0, apps_dir)

def main():
    try:
        from quantum_processor_frontend import QuantumProcessorFrontend
        
        print("Launching Dynamic Quantum Processor Frontend...")
        print(f"Integrating with QuantoniumOS Quantum Kernel (up to 1000 qubits)")
        
        app = QuantumProcessorFrontend()
        app.create_window()
        
        if app.window:
            print(f"Frontend launched successfully! Capacity: {app.max_qubits} qubits")
            app.window.mainloop()
        else:
            print("Failed to create frontend window")
            
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure the quantum_processor_frontend.py is in the correct location")
    except Exception as e:
        print(f"Error launching frontend: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
