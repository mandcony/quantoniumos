#!/usr/bin/env python3
"""
MINIMAL SIMULATOR TEST - ISOLATE THE CRASH
"""

import sys, os, numpy as np
import pytest  # type: ignore[import]
from PyQt5.QtWidgets import QApplication, QMainWindow  # type: ignore[import]

# Add assembly bindings path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'assembly', 'python_bindings'))

unitary_rft = pytest.importorskip(
    "unitary_rft",
    reason="UnitaryRFT assembly bindings are unavailable; skipping minimal simulator test.",
)

from unitary_rft import UnitaryRFT, RFT_FLAG_QUANTUM_SAFE, RFT_FLAG_USE_RESONANCE  # type: ignore[import]

print("🔧 MINIMAL SIMULATOR TEST")
print("=" * 30)

class MinimalRFTTest(QMainWindow):
    def __init__(self):
        print("Creating QMainWindow...")
        super().__init__()
        self.setWindowTitle("Minimal RFT Test")
        self.resize(800, 600)
        
        print("Initializing RFT...")
        self.num_qubits = 5
        rft_size = 2 ** self.num_qubits  # 32
        
        try:
            print(f"Creating UnitaryRFT with size {rft_size}...")
            self.rft_engine = UnitaryRFT(rft_size, RFT_FLAG_QUANTUM_SAFE | RFT_FLAG_USE_RESONANCE)
            print(f"✓ RFT engine created: {self.rft_engine}")
            print(f"✓ RFT size: {self.rft_engine.size}")
            
            print("Creating quantum state...")
            self.quantum_state = np.zeros(rft_size, dtype=complex)
            self.quantum_state[0] = 1.0
            print("✓ Quantum state created")
            
            print("Calling forward transform...")
            self.resonance_state = self.rft_engine.forward(self.quantum_state)
            print(f"✓ Forward transform SUCCESS: {len(self.resonance_state)} elements")
            
        except Exception as e:
            print(f"✗ RFT initialization FAILED: {e}")
            import traceback
            traceback.print_exc()
        
        print("Showing window...")
        self.show()
        print("✓ MinimalRFTTest completed successfully")

if __name__ == "__main__":
    app = QApplication([])
    
    window = MinimalRFTTest()
    
    print("Running app.exec_()...")
    # Don't actually run the event loop, just create and show
    # app.exec_()
    print("TEST COMPLETE")
