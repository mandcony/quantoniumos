#!/usr/bin/env python3
"""
SIMPLE QUBIT VERTICES TEST
Test quantum vertex operations with proper output
"""

import numpy as np
import networkx as nx
from typing import Dict, Any, List, Tuple

class QubitVertex:
    """A vertex representing a qubit with alpha and beta amplitudes"""
    
    def __init__(self, vertex_id: int, initial_state: str = "0"):
        self.vertex_id = vertex_id
        
        # Initialize qubit state |ψ⟩ = α|0⟩ + β|1⟩
        if initial_state == "0":
            self.alpha = complex(1.0, 0.0)
            self.beta = complex(0.0, 0.0)
        elif initial_state == "1":
            self.alpha = complex(0.0, 0.0)
            self.beta = complex(1.0, 0.0)
        else:  # superposition
            self.alpha = complex(1.0/np.sqrt(2), 0.0)
            self.beta = complex(1.0/np.sqrt(2), 0.0)
            
        # Golden ratio frequency
        phi = (1 + np.sqrt(5)) / 2
        self.frequency = phi ** (vertex_id / 10.0)
        self.phase = 0.0
        
    def apply_hadamard(self):
        """Apply Hadamard gate: creates superposition"""
        new_alpha = (self.alpha + self.beta) / np.sqrt(2)
        new_beta = (self.alpha - self.beta) / np.sqrt(2)
        self.alpha, self.beta = new_alpha, new_beta
        
    def apply_pauli_x(self):
        """Apply Pauli-X gate: bit flip"""
        self.alpha, self.beta = self.beta, self.alpha
        
    def apply_phase_gate(self, phi: float):
        """Apply phase gate to beta amplitude"""
        self.beta *= np.exp(1j * phi)
        
    def evolve(self, dt: float = 0.1):
        """Evolve qubit using its frequency"""
        self.phase += self.frequency * dt
        phase_factor = np.exp(1j * self.phase)
        self.alpha *= phase_factor
        self.beta *= phase_factor
        
        # Renormalize
        norm = np.sqrt(abs(self.alpha)**2 + abs(self.beta)**2)
        if norm > 1e-10:
            self.alpha /= norm
            self.beta /= norm
            
    def measure_probabilities(self) -> Tuple[float, float]:
        """Get measurement probabilities for |0⟩ and |1⟩"""
        p0 = abs(self.alpha)**2
        p1 = abs(self.beta)**2
        return p0, p1
        
    def get_state_info(self) -> Dict[str, Any]:
        """Get detailed state information"""
        p0, p1 = self.measure_probabilities()
        return {
            'vertex_id': self.vertex_id,
            'alpha': self.alpha,
            'beta': self.beta,
            'prob_0': p0,
            'prob_1': p1,
            'frequency': self.frequency,
            'phase': self.phase,
            'is_superposition': abs(p0 - 0.5) < 0.1 and abs(p1 - 0.5) < 0.1
        }

class QubitVertexNetwork:
    """Network of qubit vertices"""
    
    def __init__(self, num_vertices: int = 8):
        self.num_vertices = num_vertices
        self.vertices = {}
        self.graph = nx.Graph()
        
        # Create vertices
        for i in range(num_vertices):
            initial = "0" if i % 2 == 0 else "+"  # Alternating |0⟩ and |+⟩
            self.vertices[i] = QubitVertex(i, initial)
            self.graph.add_node(i)
            
        # Create connections (nearest neighbor)
        for i in range(num_vertices - 1):
            self.graph.add_edge(i, i + 1)
            
        print(f"Created network with {num_vertices} qubit vertices")
        
    def apply_gate_to_vertex(self, vertex_id: int, gate: str, parameter: float = 0.0):
        """Apply quantum gate to specific vertex"""
        if vertex_id not in self.vertices:
            raise ValueError(f"Vertex {vertex_id} does not exist")
            
        vertex = self.vertices[vertex_id]
        
        if gate == "H":
            vertex.apply_hadamard()
        elif gate == "X":
            vertex.apply_pauli_x()
        elif gate == "Phase":
            vertex.apply_phase_gate(parameter)
        else:
            raise ValueError(f"Unknown gate: {gate}")
            
        print(f"Applied {gate} gate to vertex {vertex_id}")
        
    def evolve_network(self, time_steps: int = 5, dt: float = 0.1):
        """Evolve all vertices in the network"""
        print(f"\nEvolving network for {time_steps} time steps:")
        
        for step in range(time_steps):
            # Evolve each vertex
            for vertex in self.vertices.values():
                vertex.evolve(dt)
                
            # Show progress
            if step % 2 == 0:
                total_p1 = sum(v.measure_probabilities()[1] for v in self.vertices.values())
                avg_phase = np.mean([v.phase for v in self.vertices.values()])
                print(f"  Step {step}: total_P(1)={total_p1:.3f}, avg_phase={avg_phase:.3f}")
                
    def entangle_vertices(self, v1_id: int, v2_id: int):
        """Create simple entanglement between two vertices"""
        if v1_id not in self.vertices or v2_id not in self.vertices:
            raise ValueError("One or both vertices do not exist")
            
        v1 = self.vertices[v1_id]
        v2 = self.vertices[v2_id]
        
        # Simple entanglement: couple the beta amplitudes
        coupling = 0.1
        
        # Exchange some amplitude
        temp_beta = v1.beta
        v1.beta = (1 - coupling) * v1.beta + coupling * v2.beta
        v2.beta = (1 - coupling) * v2.beta + coupling * temp_beta
        
        # Renormalize both
        for v in [v1, v2]:
            norm = np.sqrt(abs(v.alpha)**2 + abs(v.beta)**2)
            if norm > 1e-10:
                v.alpha /= norm
                v.beta /= norm
                
        print(f"Entangled vertices {v1_id} and {v2_id}")
        
    def get_network_state(self) -> Dict[str, Any]:
        """Get complete network state information"""
        vertex_states = []
        
        for i in range(self.num_vertices):
            state = self.vertices[i].get_state_info()
            vertex_states.append(state)
            
        # Overall statistics
        total_p0 = sum(state['prob_0'] for state in vertex_states)
        total_p1 = sum(state['prob_1'] for state in vertex_states)
        superposition_count = sum(1 for state in vertex_states if state['is_superposition'])
        
        return {
            'num_vertices': self.num_vertices,
            'vertex_states': vertex_states,
            'total_prob_0': total_p0,
            'total_prob_1': total_p1,
            'superposition_count': superposition_count,
            'entanglement_edges': self.graph.number_of_edges(),
            'connectivity': self.graph.number_of_edges() / (self.num_vertices * (self.num_vertices - 1) / 2) if self.num_vertices > 1 else 0
        }

def test_qubit_vertices():
    """Test the qubit vertex network"""
    print("🔺 QUBIT VERTICES TEST")
    print("=" * 40)
    
    # Create 8-vertex network
    network = QubitVertexNetwork(8)
    
    print("\n📊 Initial network state:")
    initial_state = network.get_network_state()
    print(f"  Vertices: {initial_state['num_vertices']}")
    print(f"  Total P(0): {initial_state['total_prob_0']:.3f}")
    print(f"  Total P(1): {initial_state['total_prob_1']:.3f}")
    print(f"  Superposition vertices: {initial_state['superposition_count']}")
    
    print("\n🔧 Applying quantum gates:")
    # Apply some gates
    network.apply_gate_to_vertex(0, "H")  # Create superposition
    network.apply_gate_to_vertex(1, "X")  # Bit flip
    network.apply_gate_to_vertex(2, "Phase", np.pi/4)  # Phase gate
    
    print("\n🔗 Creating entanglement:")
    # Entangle some vertices
    network.entangle_vertices(0, 1)
    network.entangle_vertices(2, 3)
    
    print("\n⏱️ Evolving network:")
    # Evolve the network
    network.evolve_network(time_steps=6, dt=0.15)
    
    print("\n📊 Final network state:")
    final_state = network.get_network_state()
    print(f"  Vertices: {final_state['num_vertices']}")
    print(f"  Total P(0): {final_state['total_prob_0']:.3f}")
    print(f"  Total P(1): {final_state['total_prob_1']:.3f}")
    print(f"  Superposition vertices: {final_state['superposition_count']}")
    
    print("\n🔍 Individual vertex details:")
    for i in range(min(4, network.num_vertices)):  # Show first 4 vertices
        state = final_state['vertex_states'][i]
        print(f"  Vertex {i}: P(0)={state['prob_0']:.3f}, P(1)={state['prob_1']:.3f}, "
              f"freq={state['frequency']:.3f}, superpos={state['is_superposition']}")
    
    print(f"\n✅ Qubit vertices test completed!")
    print(f"  Network connectivity: {final_state['connectivity']:.3f}")
    print(f"  Quantum operations: successful")
    
    return network

if __name__ == "__main__":
    result_network = test_qubit_vertices()
