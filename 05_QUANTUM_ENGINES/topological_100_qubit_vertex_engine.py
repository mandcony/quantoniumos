#!/usr/bin/env python3
"""
100-NODE NETWORK WITH COUPLED OSCILLATORS

Each node stores 2 complex numbers (alpha, beta) representing qubit amplitudes.
Each amplitude has an associated oscillator with frequency and phase evolution.
Network connections allow amplitude exchange between neighboring nodes.
"""

import tracemalloc
from typing import Any, Dict

import networkx as nx
import numpy as np
import psutil

PHI = (1 + np.sqrt(5)) / 2  # Golden ratio


class NetworkNode:
    """
    Network node storing two complex amplitudes with oscillator dynamics.

    Data stored:
    - alpha: complex number (amplitude for state 0)
    - beta: complex number (amplitude for state 1)
    - Two oscillators with frequencies and phases
    - 2D position coordinates
    """

    def __init__(self, node_id: int, initial_state: str = "0"):
        self.node_id = node_id

        # Two complex amplitudes (normalized: |alpha|^2 + |beta|^2 = 1)
        if initial_state == "0":
            self.alpha = complex(1.0, 0.0)  # State 0 amplitude
            self.beta = complex(0.0, 0.0)  # State 1 amplitude
        elif initial_state == "1":
            self.alpha = complex(0.0, 0.0)
            self.beta = complex(1.0, 0.0)
        else:
            # Random amplitudes
            theta = np.random.uniform(0, 2 * np.pi)
            self.alpha = np.cos(theta / 2)
            self.beta = np.sin(theta / 2) * np.exp(1j * np.random.uniform(0, 2 * np.pi))

            # Normalize to unit vector
            norm = np.sqrt(abs(self.alpha) ** 2 + abs(self.beta) ** 2)
            self.alpha /= norm
            self.beta /= norm

        # Each amplitude has an oscillator with frequency proportional to golden ratio
        base_freq = PHI ** (node_id / 100.0)
        self.oscillator_0 = HarmonicOscillator(base_freq, self.alpha)
        self.oscillator_1 = HarmonicOscillator(base_freq * PHI, self.beta)

        # 2D grid position
        self.position = self._compute_grid_position()

    def _compute_grid_position(self) -> np.ndarray:
        """Map node ID to 2D grid coordinates"""
        grid_size = int(np.ceil(np.sqrt(100)))  # 10x10 grid for 100 nodes
        x = self.node_id % grid_size
        y = self.node_id // grid_size
        return np.array([x / grid_size, y / grid_size])

    def get_state_vector(self) -> np.ndarray:
        """Get qubit state as [α, β] vector"""
        return np.array([self.alpha, self.beta])

    def apply_pauli_x(self):
        """Apply Pauli-X gate (bit flip)"""
        self.alpha, self.beta = self.beta, self.alpha
        self.oscillator_0.amplitude, self.oscillator_1.amplitude = (
            self.oscillator_1.amplitude,
            self.oscillator_0.amplitude,
        )

    def apply_pauli_y(self):
        """Apply Pauli-Y gate"""
        new_alpha = -1j * self.beta
        new_beta = 1j * self.alpha
        self.alpha, self.beta = new_alpha, new_beta
        self.oscillator_0.amplitude = new_alpha
        self.oscillator_1.amplitude = new_beta

    def apply_pauli_z(self):
        """Apply Pauli-Z gate (phase flip)"""
        self.beta = -self.beta
        self.oscillator_1.amplitude = -self.oscillator_1.amplitude

    def apply_hadamard(self):
        """Apply Hadamard gate (superposition)"""
        new_alpha = (self.alpha + self.beta) / np.sqrt(2)
        new_beta = (self.alpha - self.beta) / np.sqrt(2)
        self.alpha, self.beta = new_alpha, new_beta
        self.oscillator_0.amplitude = new_alpha
        self.oscillator_1.amplitude = new_beta

    def apply_phase_gate(self, phi: float):
        """Apply phase gate with angle phi"""
        self.beta *= np.exp(1j * phi)
        self.oscillator_1.amplitude *= np.exp(1j * phi)

    def apply_rft_resonance(self, parameter: float = 1.0):
        """Apply RFT-specific resonance operation"""
        # Resonance affects both oscillators
        self.oscillator_0.vibrational_mode("resonance_burst", parameter)
        self.oscillator_1.vibrational_mode("resonance_burst", parameter * PHI)

        # Update qubit amplitudes from oscillators
        self.alpha = self.oscillator_0.amplitude
        self.beta = self.oscillator_1.amplitude

        # Renormalize
        norm = np.sqrt(abs(self.alpha) ** 2 + abs(self.beta) ** 2)
        if norm > 1e-10:
            self.alpha /= norm
            self.beta /= norm

    def evolve_quantum_step(self, dt: float = 0.1):
        """Evolve qubit using quantum oscillators"""
        self.oscillator_0.quantum_step(dt)
        self.oscillator_1.quantum_step(dt)

        # Update qubit state from oscillators
        self.alpha = self.oscillator_0.amplitude
        self.beta = self.oscillator_1.amplitude

        # Maintain normalization
        norm = np.sqrt(abs(self.alpha) ** 2 + abs(self.beta) ** 2)
        if norm > 1e-10:
            self.alpha /= norm
            self.beta /= norm

    def measure_probability_0(self) -> float:
        """Probability of measuring |0⟩"""
        return abs(self.alpha) ** 2

    def measure_probability_1(self) -> float:
        """Probability of measuring |1⟩"""
        return abs(self.beta) ** 2

    def __repr__(self):
        p0 = self.measure_probability_0()
        p1 = self.measure_probability_1()
        return f"Qubit{self.node_id}(P(0)={p0:.3f}, P(1)={p1:.3f})"


class HarmonicOscillator:
    """
    Simple harmonic oscillator with frequency, amplitude, and phase.
    Evolution: amplitude *= exp(i * frequency * dt) * damping_factor
    """

    def __init__(self, frequency: float, amplitude: complex = 1.0):
        self.frequency = frequency  # Oscillation frequency (Hz)
        self.amplitude = amplitude  # Complex amplitude
        self.energy_level = 0  # Integer energy level
        self.phase = 0.0  # Current phase (radians)
        self.damping = 0.001  # Damping coefficient

    def time_step(self, dt: float = 0.1):
        """Evolve oscillator by time step dt"""
        self.phase += self.frequency * dt
        self.amplitude *= np.exp(1j * self.phase) * (1 - self.damping * dt)

    def vibrational_mode(self, mode: str, parameter: float = 1.0):
        """Modify oscillator frequency"""
        if mode == "resonance_burst":
            self.energy_level += 1
            self.frequency *= PHI**parameter
            self.amplitude *= PHI ** (-parameter / 2)  # Keep amplitude bounded

    def quantum_step(self, dt: float = 0.1):
        """Quantum evolution step"""
        self.time_step(dt)


class Network100Nodes:
    """
    Network of 100 nodes, each storing two complex amplitudes.
    Connections allow amplitude exchange between neighboring nodes.
    Each amplitude has an oscillator with frequency evolution.
    """

    def __init__(self):
        self.num_qubits = 100
        self.qubits = {}  # node_id -> NetworkNode
        self.graph = nx.Graph()
        self.entanglement_edges = {}  # connection_weights

        # Memory tracking
        tracemalloc.start()
        self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        print("🔺 100-node network initialized")
        print(f"📊 Memory at start: {self.start_memory:.2f} MB")

    def initialize_nodes(self, initial_pattern: str = "random"):
        """Initialize all 100 nodes with amplitude pairs"""
        print(f"\n🔺 Initializing 100 nodes (pattern: {initial_pattern})")

        for i in range(self.num_qubits):
            if initial_pattern == "all_zero":
                state = "0"
            elif initial_pattern == "all_one":
                state = "1"
            elif initial_pattern == "alternating":
                state = "0" if i % 2 == 0 else "1"
            else:  # random
                state = "random"

            node = NetworkNode(i, initial_state=state)
            self.qubits[i] = node
            self.graph.add_node(i, pos=node.position)

        print(f"✅ Created {len(self.qubits)} nodes on 2D grid")

    def create_network_connections(self, connectivity: str = "nearest_neighbor"):
        """Create connections between nodes for amplitude exchange"""
        print(f"\n🔗 Creating network connections ({connectivity})")

        if connectivity == "nearest_neighbor":
            # Connect nearest neighbors in 2D grid
            grid_size = int(np.ceil(np.sqrt(self.num_qubits)))

            for i in range(self.num_qubits):
                x, y = i % grid_size, i // grid_size

                # Connect to right neighbor
                if x + 1 < grid_size:
                    j = (x + 1) + y * grid_size
                    if j < self.num_qubits:
                        self.graph.add_edge(i, j)
                        self.entanglement_edges[(i, j)] = complex(1.0 / np.sqrt(2))

                # Connect to bottom neighbor
                if y + 1 < grid_size:
                    j = x + (y + 1) * grid_size
                    if j < self.num_qubits:
                        self.graph.add_edge(i, j)
                        self.entanglement_edges[(i, j)] = complex(1.0 / np.sqrt(2))

        elif connectivity == "golden_ratio":
            # Connect nodes based on golden ratio distances
            for i in range(self.num_qubits):
                for j in range(i + 1, self.num_qubits):
                    distance = abs(i - j)
                    if distance <= 5 or distance % int(PHI) == 0:
                        weight = np.exp(1j * PHI * distance) / np.sqrt(distance + 1)
                        self.graph.add_edge(i, j)
                        self.entanglement_edges[(i, j)] = weight

        print(f"✅ Created {len(self.entanglement_edges)} network connections")

    def apply_single_qubit_gate(self, qubit_id: int, gate: str, parameter: float = 0.0):
        """Apply single-qubit gate to specific qubit"""
        if qubit_id not in self.qubits:
            raise ValueError(f"Qubit {qubit_id} does not exist")

        qubit = self.qubits[qubit_id]

        if gate == "X":
            qubit.apply_pauli_x()
        elif gate == "Y":
            qubit.apply_pauli_y()
        elif gate == "Z":
            qubit.apply_pauli_z()
        elif gate == "H":
            qubit.apply_hadamard()
        elif gate == "Phase":
            qubit.apply_phase_gate(parameter)
        elif gate == "RFT_Resonance":
            qubit.apply_rft_resonance(parameter)
        else:
            raise ValueError(f"Unknown gate: {gate}")

    def apply_cnot_gate(self, control_id: int, target_id: int):
        """Apply CNOT gate between two qubits"""
        if control_id not in self.qubits or target_id not in self.qubits:
            raise ValueError("Control or target qubit does not exist")

        control = self.qubits[control_id]
        target = self.qubits[target_id]

        # CNOT: if control is |1⟩, flip target
        # This is a simplified implementation for demonstration
        if abs(control.beta) > abs(control.alpha):  # Control is more |1⟩ than |0⟩
            target.apply_pauli_x()

    def evolve_quantum_network(self, time_steps: int = 10, dt: float = 0.05):
        """Evolve entire 100-qubit network using quantum oscillators"""
        print(f"\n🌊 Evolving 100-qubit network ({time_steps} steps, dt={dt})")

        for step in range(time_steps):
            # Evolve each qubit's oscillators
            for qubit in self.qubits.values():
                qubit.evolve_quantum_step(dt)

            # Apply entanglement through edges
            for (i, j), weight in self.entanglement_edges.items():
                qubit_i = self.qubits[i]
                qubit_j = self.qubits[j]

                # Simple entanglement: couple the |1⟩ states
                coupling_strength = abs(weight) * 0.01
                if coupling_strength > 0:
                    # Exchange some amplitude between |1⟩ states
                    exchange = coupling_strength * (
                        qubit_i.beta * np.conj(qubit_j.beta)
                    )
                    qubit_i.beta += exchange * 0.1
                    qubit_j.beta += np.conj(exchange) * 0.1

                    # Renormalize both qubits
                    for qubit in [qubit_i, qubit_j]:
                        norm = np.sqrt(abs(qubit.alpha) ** 2 + abs(qubit.beta) ** 2)
                        if norm > 1e-10:
                            qubit.alpha /= norm
                            qubit.beta /= norm

            # Progress report
            if step % 3 == 0:
                total_prob_1 = sum(
                    q.measure_probability_1() for q in self.qubits.values()
                )
                avg_entanglement = np.mean(
                    [abs(w) for w in self.entanglement_edges.values()]
                )
                print(
                    f"   Step {step}: total_P(1)={total_prob_1:.3f}, avg_entanglement={avg_entanglement:.4f}"
                )

    def get_quantum_state_summary(self) -> Dict[str, Any]:
        """Get summary of 100-qubit quantum state"""
        # Memory usage
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_used = current_memory - self.start_memory

        # Quantum state statistics
        prob_0_list = [q.measure_probability_0() for q in self.qubits.values()]
        prob_1_list = [q.measure_probability_1() for q in self.qubits.values()]

        # Entanglement measure (simplified)
        total_entanglement = sum(abs(w) for w in self.entanglement_edges.values())

        return {
            "num_qubits": self.num_qubits,
            "num_entanglement_edges": len(self.entanglement_edges),
            "memory_used_mb": memory_used,
            "total_probability_0": sum(prob_0_list),
            "total_probability_1": sum(prob_1_list),
            "average_probability_0": np.mean(prob_0_list),
            "average_probability_1": np.mean(prob_1_list),
            "probability_variance": np.var(prob_1_list),
            "total_entanglement_strength": total_entanglement,
            "max_probability_1": max(prob_1_list),
            "min_probability_1": min(prob_1_list),
            "quantum_state_dimension": 2**self.num_qubits,  # Theoretical dimension
            "hilbert_space_log2": self.num_qubits,  # log2 of dimension
        }

    def run_quantum_algorithm_demo(self):
        """Demonstrate quantum algorithm on 100-qubit topology"""
        print("\n🧮 Running quantum algorithm demonstration:")

        # Step 1: Apply Hadamard to create superposition
        print("   Step 1: Creating superposition with Hadamard gates")
        for i in range(0, 10):  # First 10 qubits
            self.apply_single_qubit_gate(i, "H")

        # Step 2: Apply some CNOT gates for entanglement
        print("   Step 2: Creating entanglement with CNOT gates")
        for i in range(0, 8, 2):
            self.apply_cnot_gate(i, i + 1)

        # Step 3: Apply RFT resonance operations
        print("   Step 3: Applying RFT resonance operations")
        for i in range(10, 20):
            self.apply_single_qubit_gate(i, "RFT_Resonance", 0.5)

        # Step 4: Evolve the network
        print("   Step 4: Evolving quantum network")
        self.evolve_quantum_network(time_steps=5, dt=0.1)


def demonstrate_network():
    """Test the 100-node network with oscillator dynamics"""
    print("🔺 100-NODE NETWORK DEMONSTRATION")
    print("=" * 100)

    # Create network
    engine = Network100Nodes()

    # Initialize nodes
    engine.initialize_nodes(initial_pattern="random")

    # Create connections
    engine.create_network_connections(connectivity="nearest_neighbor")

    # Run quantum algorithm demo
    engine.run_quantum_algorithm_demo()

    # Get final state summary
    final_data = engine.get_quantum_state_summary()
    print("\n📊 Final network state:")
    for key, value in final_data.items():
        if isinstance(value, float):
            if key == "quantum_state_dimension":
                print(f"   {key}: 2^{final_data['num_qubits']} = {value:.2e}")
            else:
                print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")

    print("\n✅ Network simulation completed")
    print(f"📊 Memory: {final_data['memory_used_mb']:.2f} MB")
    print(f"🔗 Nodes: {final_data['num_qubits']}")
    print(f"🌐 Connections: {final_data['num_entanglement_edges']}")

    return engine


if __name__ == "__main__":
    result = demonstrate_network()
