"""
QuantoniumOS - System Resonance Manager

This module implements the core system resonance management for process scheduling
using quantum-inspired resonance patterns and geometric containers.
"""

import time
import logging
import threading
import numpy as np
import math
import random
from typing import List, Dict, Any

try:
    from core.encryption.wave_primitives import WaveNumber
except ImportError:
    # Fallback WaveNumber implementation if the main one isn't available
    class WaveNumber:
        def __init__(self, amplitude=1.0, phase=0.0):
            self.amplitude = float(amplitude)
            self.phase = float(phase)
        
        def scale_amplitude(self, factor):
            self.amplitude *= factor

logger = logging.getLogger(__name__)

class GeometricContainer:
    """Base class for objects with geometric properties for resonance calculations"""
    def __init__(self, id, vertices=None, transformations=None, material_props=None):
        self.id = id
        self.vertices = vertices if vertices else []
        self.transformations = transformations if transformations else []
        self.material_props = material_props if material_props else {}
        self.resonant_frequencies = []
        
        # Calculate initial resonant frequencies if we have vertices
        if self.vertices:
            self._calculate_resonance()
    
    def _calculate_resonance(self):
        """Calculate resonant frequencies based on geometric properties"""
        if not self.vertices:
            return
        
        # Simple implementation: calculate resonance as sum of vertex distances
        if len(self.vertices) >= 2:
            distances = []
            for i in range(len(self.vertices)):
                v1 = self.vertices[i]
                v2 = self.vertices[(i + 1) % len(self.vertices)]
                # Calculate Euclidean distance
                distance = sum((a - b) ** 2 for a, b in zip(v1, v2)) ** 0.5
                distances.append(distance)
            
            # Calculate resonant frequency from distances
            avg_distance = sum(distances) / len(distances)
            self.resonant_frequencies = [1.0 / avg_distance]

class Process(GeometricContainer):
    """Process class with quantum-inspired scheduling properties"""
    def __init__(self, id: int, priority, amplitude=None, vertices=None):
        # Initialize geometric container properties
        super().__init__(id, vertices=vertices if vertices else [])
        
        # Initialize process-specific properties
        self.id = id
        if isinstance(priority, (int, float)):
            self.priority = WaveNumber(amplitude=float(priority), phase=0.0)
        else:
            self.priority = priority  # WaveNumber object
        
        self.amplitude = amplitude if amplitude is not None else complex(1.0, 0.0)
        self.resonance = 0.0  # Float
        self.state = "ready"  # ready, running, blocked, terminated
        self.start_time = None
        self.end_time = None
        self.run_time = 0
        self.wait_time = 0
        self.last_state_change = time.time()
        self.time = 0.0
        self.priority_phase = random.uniform(0, 2 * math.pi)
        self.amplitude_phase = random.uniform(0, 2 * math.pi)
        self.resonance_phase = random.uniform(0, 2 * math.pi)
    
    def __repr__(self):
        return (f"Process(id={self.id}, state={self.state}, "
                f"priority={self.priority.amplitude:.2f}, "
                f"amplitude={abs(self.amplitude):.2f}, "
                f"resonance={self.resonance:.2f})")
    
    def update_state(self, new_state: str):
        now = time.time()
        duration = now - self.last_state_change
        
        if self.state == "running":
            self.run_time += duration
        elif self.state == "ready" or self.state == "blocked":
            self.wait_time += duration
        
        self.state = new_state
        self.last_state_change = now
        
        if new_state == "running" and self.start_time is None:
            self.start_time = now
        elif new_state == "terminated" and self.end_time is None:
            self.end_time = now
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get process metrics for monitoring and analysis"""
        return {
            "id": self.id,
            "state": self.state,
            "priority": self.priority.amplitude if hasattr(self.priority, "amplitude") else self.priority,
            "amplitude": abs(self.amplitude),
            "resonance": self.resonance,
            "run_time": self.run_time,
            "wait_time": self.wait_time,
            "turnaround_time": self.end_time - self.start_time if self.end_time and self.start_time else None
        }

def monitor_resonance_states(processes, dt=0.1, max_samples=10):
    """
    Advanced monitor for process resonance states that simulates
    quantum-inspired fluctuations in priority, amplitude, and resonance.
    
    Args:
        processes: List of Process objects
        dt: Time step for simulation
        max_samples: Maximum number of samples to record
        
    Returns:
        List of state snapshots and process metrics over time
    """
    results = []
    freq_val = 1.0  # Default resonance frequency if config isn't available
    
    try:
        from core.config import Config
        cfg = Config()
        freq_val = cfg.data.get("resonance_frequency", 1.0)
    except ImportError:
        raise NotImplementedError("TODO: implement")
    
    system_freq = WaveNumber(freq_val, phase=0.0)
    logger.info(f"[Resonance Manager] freq={freq_val}, dt={dt}")
    
    for sample_idx in range(max_samples):
        # Create a snapshot of all process states
        snapshot = {
            "time": sample_idx * dt,
            "running_pid": None,
            "processes": {}
        }
        
        # Process quantum-inspired fluctuations and update states
        for p in processes:
            # Increment time for oscillation
            p.time += dt
            
            # Simulate fluctuating priority (CPU load simulation)
            if hasattr(p.priority, "scale_amplitude"):
                load_variation = math.sin(freq_val * p.time + p.priority_phase)
                damping_factor = 0.5 + 0.5 * load_variation  # Oscillates between 0 and 1
                p.priority.scale_amplitude(damping_factor)
                p.priority.amplitude = min(max(p.priority.amplitude, 0.1), 10)  # Clamp to reasonable range
            
            # Simulate fluctuating amplitude (Memory usage simulation)
            if isinstance(p.amplitude, complex):
                p.amplitude = complex(
                    5 + 5 * math.sin(freq_val * p.time * 0.8 + p.amplitude_phase), 
                    2 * math.cos(freq_val * p.time * 0.9)
                )
            else:
                # Fallback for non-complex amplitude
                p.amplitude = 5 + 5 * math.sin(freq_val * p.time * 0.8 + p.amplitude_phase)
            
            # Simulate fluctuating resonance (Disk/IO simulation)
            p.resonance = 5 + 5 * math.sin(freq_val * p.time * 0.6 + p.resonance_phase)
            p.resonance = min(max(p.resonance, 0), 10)  # Clamp to reasonable range
        
        # Quantum-inspired scheduling: select a process to run based on complex amplitudes
        # Use amplitude magnitude as the probability weight
        total_amplitude = sum(abs(p.amplitude) for p in processes)
        
        if total_amplitude > 0:
            # Create probability distribution based on amplitude magnitudes
            probabilities = [abs(p.amplitude) / total_amplitude for p in processes]
            
            # Interference effects: processes with similar resonant frequencies interfere constructively
            for i in range(len(processes)):
                for j in range(i+1, len(processes)):
                    if len(processes[i].resonant_frequencies) > 0 and len(processes[j].resonant_frequencies) > 0:
                        # Calculate resonance similarity (simplified)
                        res_i = processes[i].resonant_frequencies[0]
                        res_j = processes[j].resonant_frequencies[0]
                        similarity = 1 / (1 + abs(res_i - res_j))
                        
                        # Apply quantum interference effect to probabilities
                        interference = similarity * 0.1 * math.sin(processes[i].time - processes[j].time)
                        probabilities[i] += interference
                        probabilities[j] += interference
            
            # Normalize probabilities after interference
            total_prob = sum(probabilities)
            if total_prob > 0:
                probabilities = [p / total_prob for p in probabilities]
            
            # Select running process using probability distribution
            try:
                running_index = np.random.choice(len(processes), p=probabilities)
                snapshot["running_pid"] = processes[running_index].id
                
                # Update process states
                for j, process in enumerate(processes):
                    process.update_state("running" if j == running_index else "ready")
            except ValueError:
                # Fallback if probability distribution is invalid
                logger.warning("Invalid probability distribution in quantum scheduler")
                snapshot["running_pid"] = processes[0].id
        else:
            # Fallback scheduling
            snapshot["running_pid"] = processes[0].id
        
        # Record detailed process metrics in snapshot
        for p in processes:
            metrics = p.get_metrics()
            snapshot["processes"][str(p.id)] = {
                "state": p.state,
                "priority": p.priority.amplitude if hasattr(p.priority, "amplitude") else p.priority,
                "amplitude_real": p.amplitude.real if isinstance(p.amplitude, complex) else p.amplitude,
                "amplitude_imag": p.amplitude.imag if isinstance(p.amplitude, complex) else 0,
                "resonance": p.resonance
            }
        
        results.append(snapshot)
        time.sleep(dt)  # Simulate real-time delay
    
    return results

if __name__ == "__main__":
    # Simple test
    processes = [Process(i, priority=0.5 + i * 0.25) for i in range(4)]
    results = monitor_resonance_states(processes, interval=0.1, max_samples=5)
    print(results)
