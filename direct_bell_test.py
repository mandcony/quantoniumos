#!/usr/bin/env python3
"""
Direct Bell State Test - Achieve CHSH > 2.7
==========================================

Simple direct test to verify perfect Bell state generation and CHSH violation.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

try:
    import qutip as qt
    QUTIP_AVAILABLE = True
except ImportError:
    QUTIP_AVAILABLE = False

from src.engine.vertex_assembly import EntangledVertexEngine


def test_perfect_bell_state():
    """Test perfect Bell state creation and CHSH violation."""
    print("=" * 60)
    print("DIRECT BELL STATE TEST FOR CHSH > 2.7")
    print("=" * 60)
    
    # Create engine and Bell state
    engine = EntangledVertexEngine(n_vertices=2, entanglement_enabled=True)
    bell_state = engine.create_optimal_bell_state(vertices=(0, 1))
    
    print(f"Bell state norm: {np.linalg.norm(bell_state):.10f}")
    print(f"Bell state components:")
    print(f"  |00⟩ = {bell_state[0]:.10f}")
    print(f"  |01⟩ = {bell_state[1]:.10f}")
    print(f"  |10⟩ = {bell_state[2]:.10f}")
    print(f"  |11⟩ = {bell_state[3]:.10f}")
    
    # Check if it's a perfect Bell state
    expected_coeff = 1.0 / np.sqrt(2)
    is_perfect_bell = (
        abs(abs(bell_state[0]) - expected_coeff) < 1e-10 and
        abs(bell_state[1]) < 1e-10 and
        abs(bell_state[2]) < 1e-10 and
        abs(abs(bell_state[3]) - expected_coeff) < 1e-10
    )
    
    print(f"Perfect Bell state: {'YES' if is_perfect_bell else 'NO'}")
    print(f"Expected coefficient: {expected_coeff:.10f}")
    
    # Manual CHSH calculation
    def pauli_measurement(angle: float) -> np.ndarray:
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        return np.array([[cos_a, sin_a], [sin_a, -cos_a]], dtype=complex)
    
    # Theoretical optimal angles for maximum CHSH violation (Tsirelson bound)
    # These angles achieve CHSH = 2√2 ≈ 2.828 for perfect Bell state
    angles_A = [0, np.pi/2]  # Alice: 0°, 90°
    angles_B = [np.pi/4, -np.pi/4]  # Bob: 45°, -45°
    
    print(f"\nCHSH Correlations:")
    chsh_sum = 0.0
    for i, angle_A in enumerate(angles_A):
        for j, angle_B in enumerate(angles_B):
            A = pauli_measurement(angle_A)
            B = pauli_measurement(angle_B)
            AB = np.kron(A, B)
            
            correlation = np.real(np.conj(bell_state) @ AB @ bell_state)
            
            if (i, j) == (1, 1):  # A₂B₂ term
                chsh_sum -= correlation
            else:  # A₁B₁, A₁B₂, A₂B₁ terms
                chsh_sum += correlation
            
            sign = "-" if (i, j) == (1, 1) else "+"
            print(f"  {sign}A{i+1}B{j+1}: {correlation:.6f}")
    
    print(f"\nCHSH Value: {chsh_sum:.6f}")
    print(f"Classical Bound: 2.000000")
    print(f"Quantum Bound: {2*np.sqrt(2):.6f}")
    print(f"Target (>2.7): {'✓ YES' if chsh_sum > 2.7 else '✗ NO'}")
    
    # Compare with QuTiP reference
    if QUTIP_AVAILABLE:
        qutip_bell = qt.bell_state('00')
        qutip_state = qutip_bell.full().flatten()
        print(f"\nQuTiP Bell state comparison:")
        print(f"  QuTiP |00⟩: {qutip_state[0]:.10f}")
        print(f"  QuTiP |11⟩: {qutip_state[3]:.10f}")
        
        # Calculate QuTiP CHSH
        qutip_chsh = 0.0
        for i, angle_A in enumerate(angles_A):
            for j, angle_B in enumerate(angles_B):
                A = pauli_measurement(angle_A)
                B = pauli_measurement(angle_B)
                AB = np.kron(A, B)
                
                correlation = np.real(np.conj(qutip_state) @ AB @ qutip_state)
                
                if (i, j) == (1, 1):
                    qutip_chsh -= correlation
                else:
                    qutip_chsh += correlation
        
        print(f"  QuTiP CHSH: {qutip_chsh:.6f}")
        print(f"  Fidelity: {abs(np.vdot(bell_state, qutip_state))**2:.6f}")
    
    # Final assessment
    print(f"\n{'='*60}")
    if chsh_sum > 2.7:
        print("🎉 SUCCESS: CHSH > 2.7 achieved!")
        print("✅ Maximum quantum entanglement validated")
        return True
    else:
        print("⚠️  Target not achieved - need CHSH > 2.7")
        print("🔧 Further optimization required")
        return False


if __name__ == "__main__":
    success = test_perfect_bell_state()
    sys.exit(0 if success else 1)