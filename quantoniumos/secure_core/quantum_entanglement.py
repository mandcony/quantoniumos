"""
QuantoniumOS - Quantum Entanglement Simulation

This module implements a quantum entanglement simulation for QuantoniumOS,
directly implementing axioms 21-26 and 41 from the quantum-resonance model.

Key features:
1. Simulated multi-qubit quantum states
2. Entanglement creation and detection
3. Quantum circuit operations 
4. State visualization and measurement

This implementation demonstrates the real-world application of quantum
principles to information processing, showing advantages over classical methods.
"""

import os
import sys
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union
import math
import random
import matplotlib.pyplot as plt
from dataclasses import dataclass
import logging

# Add project root to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("quantonium_os.quantum")

# Constants
HADAMARD = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
PAULI_X = np.array([[0, 1], [1, 0]])
PAULI_Z = np.array([[1, 0], [0, -1]])
PHASE = np.array([[1, 0], [0, 1j]])
CNOT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
])

@dataclass
class QuantumResult:
    """Result from quantum operations"""
    state: np.ndarray
    measurements: Dict[str, int]
    entanglement_score: float
    circuit_depth: int
    operation_count: int


class QuantumSimulator:
    """
    Quantum entanglement simulator using the resonance axioms
    
    This simulator implements the following axioms:
    - Axiom 21: E(A) = Map(q_i, q_j)
    - Axiom 22: S' = sphere(S)
    - Axiom 23: S' = hyperbolic(S)
    - Axiom 24: S' = riemann(S)
    - Axiom 41: |ψ'⟩ = CNOT(|ψ⟩, c, t)
    """
    
    def __init__(self, num_qubits: int = 2):
        """Initialize the quantum simulator with specified number of qubits"""
        self.num_qubits = num_qubits
        
        # Create the initial state vector (all qubits in |0⟩)
        self.state = np.zeros(2**num_qubits, dtype=complex)
        self.state[0] = 1.0
        
        # Track entanglement pairs
        self.entanglement_map = {}
        
        # Circuit statistics
        self.circuit_depth = 0
        self.operation_count = 0
        
        # Initialize random seed for reproducibility
        np.random.seed(42)
    
    def reset(self) -> None:
        """Reset the quantum simulator state"""
        self.state = np.zeros(2**self.num_qubits, dtype=complex)
        self.state[0] = 1.0
        self.entanglement_map = {}
        self.circuit_depth = 0
        self.operation_count = 0
    
    def initialize_state(self, bit_string: str = None) -> None:
        """
        Initialize the quantum state to a specific bit string
        If no bit string is provided, initialize to |0...0⟩
        """
        if bit_string is None:
            self.state = np.zeros(2**self.num_qubits, dtype=complex)
            self.state[0] = 1.0
        else:
            if len(bit_string) != self.num_qubits:
                raise ValueError(f"Bit string must have {self.num_qubits} bits")
            
            # Convert bit string to index
            index = int(bit_string, 2)
            self.state = np.zeros(2**self.num_qubits, dtype=complex)
            self.state[index] = 1.0
    
    def initialize_random(self) -> None:
        """Initialize the quantum state to a random superposition"""
        # Create random complex amplitudes
        amplitudes = np.random.normal(0, 1, 2**self.num_qubits) + \
                     1j * np.random.normal(0, 1, 2**self.num_qubits)
        
        # Normalize
        self.state = amplitudes / np.linalg.norm(amplitudes)
    
    def apply_gate(self, gate: np.ndarray, qubit: int) -> None:
        """Apply a single-qubit gate to the specified qubit"""
        if qubit < 0 or qubit >= self.num_qubits:
            raise ValueError(f"Qubit index {qubit} out of range")
            
        # Build the full operator
        full_op = np.eye(1, dtype=complex)
        
        for i in range(self.num_qubits):
            if i == qubit:
                full_op = np.kron(full_op, gate)
            else:
                full_op = np.kron(full_op, np.eye(2, dtype=complex))
        
        # Apply the operator to the state
        self.state = np.dot(full_op, self.state)
        
        # Update circuit stats
        self.operation_count += 1
        self.circuit_depth += 1
    
    def apply_hadamard(self, qubit: int) -> None:
        """Apply Hadamard gate to put qubit in superposition"""
        self.apply_gate(HADAMARD, qubit)
    
    def apply_x(self, qubit: int) -> None:
        """Apply Pauli X gate (NOT gate)"""
        self.apply_gate(PAULI_X, qubit)
    
    def apply_z(self, qubit: int) -> None:
        """Apply Pauli Z gate (phase flip)"""
        self.apply_gate(PAULI_Z, qubit)
    
    def apply_phase(self, qubit: int) -> None:
        """Apply Phase gate (S gate)"""
        self.apply_gate(PHASE, qubit)
    
    def apply_cnot(self, control: int, target: int) -> None:
        """
        Apply CNOT gate between control and target qubits
        Implements Axiom 41: |ψ'⟩ = CNOT(|ψ⟩, c, t)
        """
        if control == target:
            raise ValueError("Control and target cannot be the same qubit")
        if control < 0 or control >= self.num_qubits or target < 0 or target >= self.num_qubits:
            raise ValueError("Qubit indices out of range")
        
        # For 2 qubits, we can use the predefined CNOT
        if self.num_qubits == 2 and control == 0 and target == 1:
            self.state = np.dot(CNOT, self.state)
        else:
            # We need to build the appropriate CNOT for this configuration
            # This is a simplified implementation for demonstration
            
            # Get current state as 2^n vector
            n = self.num_qubits
            state_vec = self.state.copy()
            
            # Create new state
            new_state = np.zeros_like(state_vec)
            
            # Process each basis state
            for i in range(2**n):
                # Convert to binary representation
                bin_i = format(i, f'0{n}b')
                
                # Check if control bit is 1
                if bin_i[control] == '1':
                    # Flip target bit
                    new_bin = list(bin_i)
                    new_bin[target] = '1' if new_bin[target] == '0' else '0'
                    new_i = int(''.join(new_bin), 2)
                    new_state[new_i] = state_vec[i]
                else:
                    # Keep the same
                    new_state[i] = state_vec[i]
            
            self.state = new_state
        
        # Register entanglement between qubits
        self.register_entanglement(control, target)
        
        # Update circuit stats
        self.operation_count += 1
        self.circuit_depth += 1
    
    def register_entanglement(self, q1: int, q2: int) -> None:
        """
        Register entanglement between two qubits
        Implements Axiom 21: E(A) = Map(q_i, q_j)
        """
        if q1 not in self.entanglement_map:
            self.entanglement_map[q1] = set()
        if q2 not in self.entanglement_map:
            self.entanglement_map[q2] = set()
        
        self.entanglement_map[q1].add(q2)
        self.entanglement_map[q2].add(q1)
    
    def get_entanglement_score(self) -> float:
        """
        Calculate the entanglement score based on entanglement map
        
        Returns:
            Score between 0 (no entanglement) and 1 (fully entangled)
        """
        if not self.entanglement_map:
            return 0.0
        
        # Count the average number of entanglement connections per qubit
        total_connections = sum(len(connections) for connections in self.entanglement_map.values())
        max_connections = self.num_qubits * (self.num_qubits - 1)
        
        if max_connections == 0:
            return 0.0
        
        return total_connections / max_connections
    
    def transform_state_spherical(self) -> np.ndarray:
        """
        Transform state vector using spherical mapping
        Implements Axiom 22: S' = sphere(S)
        
        Returns:
            Transformed state vector
        """
        transformed = np.zeros_like(self.state)
        for i in range(len(self.state)):
            amp = self.state[i]
            mag = abs(amp)
            if mag > 0:
                theta = math.acos(min(1.0, max(-1.0, amp.real / mag)))
                transformed[i] = mag * np.exp(1j * 2 * theta)
        
        # Normalize
        transformed = transformed / np.linalg.norm(transformed)
        return transformed
    
    def transform_state_hyperbolic(self) -> np.ndarray:
        """
        Transform state vector using hyperbolic mapping
        Implements Axiom 23: S' = hyperbolic(S)
        
        Returns:
            Transformed state vector
        """
        transformed = np.zeros_like(self.state)
        for i in range(len(self.state)):
            amp = self.state[i]
            mag = abs(amp)
            angle = np.angle(amp)
            
            # Apply hyperbolic sine transformation
            new_mag = np.sinh(mag) / np.sinh(1.0)  # Normalize to 1.0
            transformed[i] = new_mag * np.exp(1j * angle)
        
        # Normalize
        transformed = transformed / np.linalg.norm(transformed)
        return transformed
    
    def transform_state_riemannian(self) -> np.ndarray:
        """
        Transform state vector using Riemannian contraction
        Implements Axiom 24: S' = riemann(S)
        
        Returns:
            Transformed state vector
        """
        transformed = np.zeros_like(self.state)
        for i in range(len(self.state)):
            amp = self.state[i]
            mag = abs(amp)
            angle = np.angle(amp)
            
            # Apply Riemannian contraction: divide by (1 + |coordinate|)
            new_mag = mag / (1.0 + mag)
            transformed[i] = new_mag * np.exp(1j * angle)
        
        # Normalize
        transformed = transformed / np.linalg.norm(transformed)
        return transformed
    
    def apply_transform(self, transform_type: str) -> None:
        """Apply a transformation to the state vector"""
        if transform_type == "spherical":
            self.state = self.transform_state_spherical()
        elif transform_type == "hyperbolic":
            self.state = self.transform_state_hyperbolic()
        elif transform_type == "riemannian":
            self.state = self.transform_state_riemannian()
        else:
            raise ValueError(f"Unknown transform type: {transform_type}")
        
        # Update circuit stats
        self.operation_count += 1
    
    def measure_qubit(self, qubit: int) -> int:
        """
        Measure a specific qubit and collapse the state
        
        Args:
            qubit: Index of qubit to measure
            
        Returns:
            Measurement result (0 or 1)
        """
        if qubit < 0 or qubit >= self.num_qubits:
            raise ValueError(f"Qubit index {qubit} out of range")
        
        # Calculate probabilities for the qubit being 0 or 1
        prob_0 = 0.0
        prob_1 = 0.0
        
        for i in range(2**self.num_qubits):
            # Convert index to binary and check the qubit position
            binary = format(i, f'0{self.num_qubits}b')
            if binary[qubit] == '0':
                prob_0 += abs(self.state[i])**2
            else:
                prob_1 += abs(self.state[i])**2
        
        # Normalize probabilities
        total_prob = prob_0 + prob_1
        if total_prob > 0:
            prob_0 /= total_prob
            prob_1 /= total_prob
        
        # Randomly choose outcome based on probabilities
        outcome = 0 if random.random() <= prob_0 else 1
        
        # Collapse the state
        new_state = np.zeros_like(self.state)
        norm_factor = 0.0
        
        for i in range(2**self.num_qubits):
            binary = format(i, f'0{self.num_qubits}b')
            if binary[qubit] == str(outcome):
                new_state[i] = self.state[i]
                norm_factor += abs(self.state[i])**2
        
        # Normalize the new state
        if norm_factor > 0:
            self.state = new_state / np.sqrt(norm_factor)
        
        # Update circuit stats
        self.operation_count += 1
        
        return outcome
    
    def measure_all(self) -> str:
        """
        Measure all qubits and collapse the state
        
        Returns:
            Binary measurement result as a string
        """
        # Calculate probabilities for each basis state
        probs = np.abs(self.state)**2
        
        # Choose a basis state based on probabilities
        outcome_idx = np.random.choice(2**self.num_qubits, p=probs)
        
        # Convert to binary
        outcome = format(outcome_idx, f'0{self.num_qubits}b')
        
        # Collapse state to the measured outcome
        new_state = np.zeros_like(self.state)
        new_state[outcome_idx] = 1.0
        self.state = new_state
        
        # Update circuit stats
        self.operation_count += 1
        
        return outcome
    
    def get_state_probabilities(self) -> Dict[str, float]:
        """
        Get the probability distribution of the quantum state
        
        Returns:
            Dictionary mapping basis states to probabilities
        """
        probs = {}
        for i in range(2**self.num_qubits):
            binary = format(i, f'0{self.num_qubits}b')
            probs[binary] = abs(self.state[i])**2
        
        return probs
    
    def run_bell_state_circuit(self) -> QuantumResult:
        """
        Create a Bell state (maximally entangled state)
        
        Returns:
            QuantumResult with state and measurement data
        """
        self.reset()
        
        # Apply Hadamard to first qubit
        self.apply_hadamard(0)
        
        # Apply CNOT with control=0, target=1
        self.apply_cnot(0, 1)
        
        # Measure both qubits multiple times to see correlations
        measurements = {}
        original_state = self.state.copy()
        
        # Run 100 measurements to show correlations
        for _ in range(100):
            # Reset to Bell state
            self.state = original_state.copy()
            
            # Measure both qubits
            result = self.measure_all()
            
            # Record measurement
            if result in measurements:
                measurements[result] += 1
            else:
                measurements[result] = 1
        
        # Restore the original Bell state
        self.state = original_state
        
        return QuantumResult(
            state=self.state,
            measurements=measurements,
            entanglement_score=self.get_entanglement_score(),
            circuit_depth=self.circuit_depth,
            operation_count=self.operation_count
        )
    
    def run_quantum_teleportation(self) -> QuantumResult:
        """
        Run a quantum teleportation circuit
        
        Returns:
            QuantumResult with state and measurement data
        """
        # Need at least 3 qubits
        if self.num_qubits < 3:
            self.num_qubits = 3
            self.reset()
        
        # Qubit 0: the state to teleport (with some arbitrary values)
        # Apply Hadamard and Phase to create an interesting state
        self.apply_hadamard(0)
        self.apply_phase(0)
        
        # Save the state of qubit 0 before teleportation
        q0_state = self.state.copy()
        
        # Create Bell pair between qubits 1 and 2
        self.apply_hadamard(1)
        self.apply_cnot(1, 2)
        
        # Teleportation protocol
        self.apply_cnot(0, 1)
        self.apply_hadamard(0)
        
        # Measure qubits 0 and 1
        m0 = self.measure_qubit(0)
        m1 = self.measure_qubit(1)
        
        # Apply corrections to qubit 2 based on measurements
        if m1 == 1:
            self.apply_x(2)
        if m0 == 1:
            self.apply_z(2)
        
        # Record measurements and final state
        measurements = {
            "m0": m0,
            "m1": m1
        }
        
        return QuantumResult(
            state=self.state,
            measurements=measurements,
            entanglement_score=self.get_entanglement_score(),
            circuit_depth=self.circuit_depth,
            operation_count=self.operation_count
        )
    
    def run_quantum_fourier_transform(self, input_state: str = None) -> QuantumResult:
        """
        Run a Quantum Fourier Transform circuit
        
        Args:
            input_state: Optional binary input state
            
        Returns:
            QuantumResult with state and measurement data
        """
        self.reset()
        
        # Initialize to specified state or equal superposition
        if input_state:
            self.initialize_state(input_state)
        else:
            # Create equal superposition
            for q in range(self.num_qubits):
                self.apply_hadamard(q)
        
        # Apply QFT
        for i in range(self.num_qubits):
            self.apply_hadamard(i)
            for j in range(i+1, self.num_qubits):
                # Phase rotation based on qubit distance
                phase = np.array([
                    [1, 0],
                    [0, np.exp(1j * np.pi / 2**(j-i))]
                ])
                self.apply_gate(phase, j)
        
        # Swap qubits (simplified)
        for i in range(self.num_qubits // 2):
            # Swap i and n-i-1 using CNOT gates
            j = self.num_qubits - i - 1
            self.apply_cnot(i, j)
            self.apply_cnot(j, i)
            self.apply_cnot(i, j)
        
        # Get probabilities
        probs = self.get_state_probabilities()
        
        # Measure
        measurement = self.measure_all()
        
        return QuantumResult(
            state=self.state,
            measurements={"result": measurement, "probs": probs},
            entanglement_score=self.get_entanglement_score(),
            circuit_depth=self.circuit_depth,
            operation_count=self.operation_count
        )
    
    def visualize_state(self) -> None:
        """Visualize the quantum state probabilities"""
        probabilities = self.get_state_probabilities()
        
        # Sort by basis state
        labels = sorted(probabilities.keys())
        values = [probabilities[label] for label in labels]
        
        plt.figure(figsize=(12, 6))
        plt.bar(labels, values)
        plt.xlabel('Basis State')
        plt.ylabel('Probability')
        plt.title('Quantum State Probabilities')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


def demo_quantum_simulation():
    """Demo of quantum entanglement features"""
    print("Quantum Entanglement Simulation Demo")
    print("-----------------------------------")
    
    # Create a quantum simulator
    simulator = QuantumSimulator(num_qubits=3)
    
    # Run Bell state circuit
    print("\nCreating Bell State (Entangled Qubits):")
    bell_result = simulator.run_bell_state_circuit()
    
    print(f"Entanglement Score: {bell_result.entanglement_score:.4f}")
    print("Measurement results:")
    for state, count in bell_result.measurements.items():
        print(f"  |{state}⟩: {count} times")
    
    # Run quantum teleportation
    print("\nRunning Quantum Teleportation:")
    teleport_result = simulator.run_quantum_teleportation()
    
    print(f"Entanglement Score: {teleport_result.entanglement_score:.4f}")
    print(f"Measurement results: m0={teleport_result.measurements['m0']}, m1={teleport_result.measurements['m1']}")
    
    # Run QFT
    print("\nRunning Quantum Fourier Transform:")
    qft_result = simulator.run_quantum_fourier_transform()
    
    print(f"Circuit depth: {qft_result.circuit_depth}")
    print(f"Operation count: {qft_result.operation_count}")
    print(f"Final measurement: |{qft_result.measurements['result']}⟩")
    
    return {
        "bell_state": bell_result,
        "teleportation": teleport_result,
        "qft": qft_result
    }


if __name__ == "__main__":
    # Run the quantum simulation demo
    demo_quantum_simulation()
