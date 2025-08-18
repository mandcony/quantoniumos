#!/usr/bin/env python3
"""
50-QUBIT VERTEX NETWORK TEST
Test larger scale qubit vertex operations with proper output
"""

import numpy as np
import networkx as nx
import psutil
from typing import Dict, Any, List, Tuple

class QubitVertex50:
    """A vertex representing a qubit in a 50-node network"""
    
    def __init__(self, vertex_id: int):
        self.vertex_id = vertex_id
        
        # Random initial state
        theta = np.random.uniform(0, 2*np.pi)
        self.alpha = np.cos(theta/2)
        self.beta = np.sin(theta/2) * np.exp(1j * np.random.uniform(0, 2*np.pi))
        
        # Normalize
        norm = np.sqrt(abs(self.alpha)**2 + abs(self.beta)**2)
        self.alpha /= norm
        self.beta /= norm
        
        # Golden ratio frequency
        phi = (1 + np.sqrt(5)) / 2
        self.frequency = phi ** (vertex_id / 50.0)
        self.phase = 0.0
        
        # Grid position
        grid_size = 8  # 8x8 grid for 50 nodes
        self.x = vertex_id % grid_size
        self.y = vertex_id // grid_size
        
    def evolve_step(self, dt: float = 0.1):
        """Evolve qubit by one time step"""
        self.phase += self.frequency * dt
        phase_factor = np.exp(1j * self.phase * 0.1)  # Small phase evolution
        
        self.alpha *= phase_factor
        self.beta *= phase_factor
        
        # Maintain normalization
        norm = np.sqrt(abs(self.alpha)**2 + abs(self.beta)**2)
        if norm > 1e-10:
            self.alpha /= norm
            self.beta /= norm
            
    def measure_probabilities(self) -> Tuple[float, float]:
        """Get |0⟩ and |1⟩ measurement probabilities"""
        return abs(self.alpha)**2, abs(self.beta)**2
        
    def apply_operation(self, operation: str):
        """Apply quantum operation"""
        if operation == "flip":
            self.alpha, self.beta = self.beta, self.alpha
        elif operation == "phase":
            self.beta *= np.exp(1j * np.pi/4)
        elif operation == "hadamard":
            new_alpha = (self.alpha + self.beta) / np.sqrt(2)
            new_beta = (self.alpha - self.beta) / np.sqrt(2)
            self.alpha, self.beta = new_alpha, new_beta

class Network50Qubits:
    """Network of 50 qubit vertices"""
    
    def __init__(self):
        self.num_vertices = 50
        self.vertices = {}
        self.graph = nx.Graph()
        self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Create vertices
        for i in range(self.num_vertices):
            self.vertices[i] = QubitVertex50(i)
            self.graph.add_node(i, pos=(self.vertices[i].x, self.vertices[i].y))
            
        # Create nearest neighbor connections
        self._create_connections()
        
        print(f"Created 50-qubit vertex network")
        print(f"Initial memory: {self.start_memory:.2f} MB")
        
    def _create_connections(self):
        """Create grid connections between vertices"""
        grid_size = 8
        edges_added = 0
        
        for i in range(self.num_vertices):
            vertex = self.vertices[i]
            x, y = vertex.x, vertex.y
            
            # Right neighbor
            if x + 1 < grid_size:
                j = (x + 1) + y * grid_size
                if j < self.num_vertices:
                    self.graph.add_edge(i, j)
                    edges_added += 1
                    
            # Bottom neighbor
            if y + 1 < grid_size:
                j = x + (y + 1) * grid_size
                if j < self.num_vertices:
                    self.graph.add_edge(i, j)
                    edges_added += 1
                    
        print(f"Created {edges_added} connections")
        
    def apply_operations(self):
        """Apply quantum operations to subsets of vertices"""
        print("\nApplying quantum operations:")
        
        # Hadamard on first 10 vertices
        for i in range(10):
            self.vertices[i].apply_operation("hadamard")
        print("  Applied Hadamard to vertices 0-9")
        
        # Bit flip on vertices 10-19
        for i in range(10, 20):
            self.vertices[i].apply_operation("flip")
        print("  Applied bit flip to vertices 10-19")
        
        # Phase gates on vertices 20-29
        for i in range(20, 30):
            self.vertices[i].apply_operation("phase")
        print("  Applied phase gates to vertices 20-29")
        
    def evolve_network(self, time_steps: int = 10, dt: float = 0.1):
        """Evolve the entire 50-vertex network"""
        print(f"\nEvolving 50-vertex network ({time_steps} steps):")
        
        for step in range(time_steps):
            # Evolve each vertex
            for vertex in self.vertices.values():
                vertex.evolve_step(dt)
                
            # Apply nearest neighbor coupling every few steps
            if step % 3 == 0:
                self._apply_neighbor_coupling()
                
            # Progress report
            if step % 3 == 0:
                total_p1 = sum(v.measure_probabilities()[1] for v in self.vertices.values())
                avg_freq = np.mean([v.frequency for v in self.vertices.values()])
                print(f"  Step {step}: total_P(1)={total_p1:.3f}, avg_freq={avg_freq:.3f}")
                
    def _apply_neighbor_coupling(self):
        """Apply weak coupling between neighboring vertices"""
        coupling_strength = 0.02
        
        for edge in list(self.graph.edges())[:20]:  # Only first 20 edges for efficiency
            i, j = edge
            v1, v2 = self.vertices[i], self.vertices[j]
            
            # Simple coupling: exchange small amount of beta amplitude
            exchange = coupling_strength * (v1.beta - v2.beta) * 0.5
            v1.beta -= exchange
            v2.beta += exchange
            
            # Renormalize
            for v in [v1, v2]:
                norm = np.sqrt(abs(v.alpha)**2 + abs(v.beta)**2)
                if norm > 1e-10:
                    v.alpha /= norm
                    v.beta /= norm
                    
    def get_network_statistics(self) -> Dict[str, Any]:
        """Get comprehensive network statistics"""
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_used = current_memory - self.start_memory
        
        # Probability statistics
        p0_values = [v.measure_probabilities()[0] for v in self.vertices.values()]
        p1_values = [v.measure_probabilities()[1] for v in self.vertices.values()]
        
        # Frequency statistics
        frequencies = [v.frequency for v in self.vertices.values()]
        
        # Amplitude statistics
        alpha_magnitudes = [abs(v.alpha) for v in self.vertices.values()]
        beta_magnitudes = [abs(v.beta) for v in self.vertices.values()]
        
        return {
            "num_vertices": self.num_vertices,
            "num_connections": self.graph.number_of_edges(),
            "memory_used_mb": memory_used,
            "total_prob_0": sum(p0_values),
            "total_prob_1": sum(p1_values),
            "avg_prob_0": np.mean(p0_values),
            "avg_prob_1": np.mean(p1_values),
            "prob_variance": np.var(p1_values),
            "avg_frequency": np.mean(frequencies),
            "max_frequency": max(frequencies),
            "min_frequency": min(frequencies),
            "avg_alpha_magnitude": np.mean(alpha_magnitudes),
            "avg_beta_magnitude": np.mean(beta_magnitudes),
            "hilbert_space_dimension": 2**self.num_vertices,
            "superposition_count": sum(1 for p1 in p1_values if 0.3 < p1 < 0.7)
        }

def test_50_qubit_network():
    """Test the 50-qubit vertex network"""
    print("🔺 50-QUBIT VERTEX NETWORK TEST")
    print("=" * 50)
    
    # Create network
    network = Network50Qubits()
    
    # Initial statistics
    print("\n📊 Initial network state:")
    initial_stats = network.get_network_statistics()
    
    key_stats = ["num_vertices", "num_connections", "memory_used_mb", 
                 "total_prob_1", "avg_frequency", "superposition_count"]
    
    for key in key_stats:
        value = initial_stats[key]
        if key == "hilbert_space_dimension":
            print(f"  {key}: 2^50 = {value:.2e}")
        elif isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
            
    # Apply operations
    network.apply_operations()
    
    # Evolve network
    network.evolve_network(time_steps=9, dt=0.12)
    
    # Final statistics
    print("\n📊 Final network state:")
    final_stats = network.get_network_statistics()
    
    for key in key_stats:
        initial_val = initial_stats[key]
        final_val = final_stats[key]
        
        if isinstance(final_val, float) and isinstance(initial_val, float):
            change = final_val - initial_val
            print(f"  {key}: {final_val:.4f} (Δ{change:+.4f})")
        else:
            print(f"  {key}: {final_val}")
            
    print(f"\n✅ 50-qubit network test completed!")
    print(f"  Hilbert space dimension: 2^50 = {final_stats['hilbert_space_dimension']:.2e}")
    print(f"  Memory usage: {final_stats['memory_used_mb']:.2f} MB")
    print(f"  Network connectivity: {final_stats['num_connections']/((50*49)/2):.4f}")
    
    return network

if __name__ == "__main__":
    result = test_50_qubit_network()
