# File: attached_assets/symbolic_qubit_resonance_test.py

import random
import uuid
from datetime import datetime
from attached_assets.symbolic_qubit_state import SymbolicQubitState
from attached_assets.symbolic_quantum_search import SymbolicQuantumSearch
from attached_assets.geometric_container import GeometricContainer

def create_symbolic_containers(count=10, owner_id=None):
    """Create resonant containers for demonstration."""
    containers = []
    
    for i in range(count):
        # Create a container with varying complexity
        complexity = random.randint(2, 6)
        dimensions = random.randint(2, 4)
        
        container = GeometricContainer(
            name=f"Symbolic Container {i+1}",
            owner_id=owner_id
        )
        
        # Generate the container
        container.generate_container(complexity=complexity, dimensions=dimensions)
        
        containers.append(container)
    
    return containers

def run_test_with_containers(test_waveform, containers):
    """
    Run a resonance test with the given containers.
    
    Args:
        test_waveform: A list of amplitude values to test
        containers: List of Container objects to check against
        
    Returns:
        Matched container or None
    """
    # Convert the test waveform to mean amplitude for demo
    if not test_waveform:
        return None
        
    test_value = sum(test_waveform) / len(test_waveform)
    
    # Create a quantum search instance
    search = SymbolicQuantumSearch()
    
    # Find container with closest matching resonance
    return search.search_database(containers, test_value)

def create_demo_container(name="Demo Container", owner_id=None):
    """Create a demo container with known resonance."""
    container = GeometricContainer(name=name, owner_id=owner_id)
    container.resonant_frequencies = [0.567]
    container.hash_value = "RH-A0.57-P0.43-C2A"
    container.id = str(uuid.uuid4())
    container.name = name
    container.owner_id = owner_id
    container.locked = True
    container.vertices = [
        [0.2, 0.3, 0.7],
        [0.7, 0.2, 0.3],
        [0.3, 0.7, 0.2],
        [0.5, 0.5, 0.5]
    ]
    return container

def create_fixed_demo_containers(owner_id=None):
    """Create a set of demo containers with specific resonance frequencies."""
    containers = []
    
    # Container 1: 0.333 resonance
    c1 = GeometricContainer(name="Low Resonance Container", owner_id=owner_id)
    c1.resonant_frequencies = [0.333]
    c1.hash_value = "RH-A0.33-P0.25-C2A"
    c1.id = str(uuid.uuid4())
    c1.locked = True
    c1.vertices = [
        [0.2, 0.3, 0.4],
        [0.3, 0.4, 0.5],
        [0.4, 0.5, 0.6]
    ]
    containers.append(c1)
    
    # Container 2: 0.567 resonance - the target container
    c2 = create_demo_container(name="Medium Resonance Container", owner_id=owner_id)
    containers.append(c2)
    
    # Container 3: 0.789 resonance
    c3 = GeometricContainer(name="High Resonance Container", owner_id=owner_id)
    c3.resonant_frequencies = [0.789]
    c3.hash_value = "RH-A0.79-P0.61-C2A"
    c3.id = str(uuid.uuid4())
    c3.locked = True
    c3.vertices = [
        [0.6, 0.7, 0.8],
        [0.7, 0.8, 0.9],
        [0.8, 0.9, 0.9]
    ]
    containers.append(c3)
    
    return containers

if __name__ == '__main__':
    # Create some test containers
    containers = create_symbolic_containers(8)
    containers.append(create_demo_container())
    
    # Generate a test waveform that should match the demo container
    test_waveform = [0.55, 0.58, 0.56, 0.57, 0.59, 0.54]
    
    # Run the test
    match = run_test_with_containers(test_waveform, containers)
    
    if match:
        print(f"✅ Found matching container: {match.name}, resonance = {match.resonant_frequencies[0]}")
    else:
        print("❌ No matching container found.")
