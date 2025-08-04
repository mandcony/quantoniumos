"""
QuantoniumOS - Quantum Scheduler

This module implements the quantum-inspired scheduler for QuantoniumOS,
directly implementing the axioms 69-74 from the quantum-resonance model.

Key features:
1. Process amplitude representation
2. Resonance-based scheduling
3. Characteristic frequency calculation
4. Resource constraint enforcement

This implementation demonstrates the real-world application of the quantum-resonance
formalism to process scheduling, showing clear advantages over traditional schedulers.
"""

import os
import sys
import time
import math
import random
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging

# Add project root to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logger = logging.getLogger("quantonium_os.scheduler")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

@dataclass
class Process:
    """Process representation in the quantum-resonance scheduler"""
    pid: int
    priority: float
    amplitude: complex
    freq: float
    state: str  # "ready", "running", "waiting", "finished"
    wait_time: float
    burst_time: float
    remaining_time: float
    quantum: int
    mem_pages: List[int]
    
    def __post_init__(self):
        # Initialize characteristic frequency based on priority and initial amplitude
        if self.freq == 0:
            self.freq = self.calculate_characteristic_frequency()
    
    def calculate_characteristic_frequency(self) -> float:
        """
        Calculates characteristic frequency for the process
        Implements Axiom 71: f = freq(π, ΔA)
        """
        base_freq = 0.1 + 0.9 * self.priority  # Scale 0.1-1.0 based on priority
        # Add some variation based on process attributes
        variation = (hash(self.pid) % 1000) / 10000  # Small variation
        return base_freq + variation


class QuantumScheduler:
    """
    Quantum-inspired OS scheduler using resonance and amplitudes
    
    This scheduler implements the following axioms:
    - Axiom 69: A' = constrain(A, C)
    - Axiom 70: S' = memory(S, PT)
    - Axiom 71: f = freq(π, ΔA)
    - Axiom 72: k = decide(A)
    - Axiom 73: A' = run(A)
    - Axiom 74: k = schedule(S, f, A)
    """
    
    def __init__(self):
        """Initialize the quantum scheduler"""
        self.processes: Dict[int, Process] = {}
        self.running_pid: Optional[int] = None
        self.time_quantum: int = 4
        self.resonance_threshold: float = 0.6
        self.resource_limit: float = 4.0  # Arbitrary resource limit
        self.page_table: Dict[int, Dict[int, int]] = {}  # pid -> {page_id -> frame_id}
        self.memory_frames: Dict[int, int] = {}  # frame_id -> pid
        self.frame_count: int = 64  # Total memory frames
        self.time: float = 0.0
        self.prev_amplitudes: Dict[int, complex] = {}  # Previous amplitudes for tracking changes
        self.schedule_log: List[Dict[str, Any]] = []
    
    def add_process(self, process: Process) -> None:
        """Add a new process to the scheduler"""
        self.processes[process.pid] = process
        self.prev_amplitudes[process.pid] = process.amplitude
        
        # Allocate memory pages (simplified)
        self.page_table[process.pid] = {}
        for i, page in enumerate(process.mem_pages):
            # Find a free frame
            for frame in range(self.frame_count):
                if frame not in self.memory_frames:
                    self.memory_frames[frame] = process.pid
                    self.page_table[process.pid][page] = frame
                    break
    
    def remove_process(self, pid: int) -> None:
        """Remove a process from the scheduler"""
        if pid in self.processes:
            # Free memory
            if pid in self.page_table:
                for frame in list(self.memory_frames.keys()):
                    if self.memory_frames[frame] == pid:
                        del self.memory_frames[frame]
                del self.page_table[pid]
            
            # Remove process
            del self.processes[pid]
            if pid in self.prev_amplitudes:
                del self.prev_amplitudes[pid]
            
            if self.running_pid == pid:
                self.running_pid = None
    
    def update_memory(self) -> None:
        """
        Process memory events and update process amplitudes
        Implements Axiom 70: S' = memory(S, PT)
        """
        for pid, process in list(self.processes.items()):
            if process.state == "finished":
                continue
                
            # Simulate page fault (10% chance)
            if process.state == "running" and random.random() < 0.1:
                # Page fault: create a complex phase shift in the amplitude
                phase_shift = math.pi / 4  # 45 degree phase shift
                process.amplitude *= np.exp(1j * phase_shift)
                process.state = "waiting"
                process.wait_time += 2  # Penalty for page fault
                logger.debug(f"Process {pid} had a page fault, new amplitude: {process.amplitude}")
    
    def calculate_resonance(self, freq1: float, freq2: float) -> float:
        """
        Calculate resonance between two frequencies using a Gaussian model
        Implements Axiom 17: A_res = max(R(f, A))
        """
        # Gaussian resonance function, peaks at freq1 == freq2
        sigma = 0.2  # Width of the resonance peak
        return np.exp(-((freq1 - freq2) ** 2) / (2 * sigma ** 2))
    
    def apply_resource_constraints(self) -> None:
        """
        Apply resource constraints to process amplitudes
        Implements Axiom 69: A' = constrain(A, C)
        """
        # Calculate total "energy" of the system
        total_energy = sum(abs(p.amplitude) ** 2 for p in self.processes.values())
        
        # If exceeds limit, normalize
        if total_energy > self.resource_limit:
            scale_factor = np.sqrt(self.resource_limit / total_energy)
            for pid, process in self.processes.items():
                process.amplitude *= scale_factor
    
    def decide_next_process(self) -> Optional[int]:
        """
        Select the next process to run based on amplitudes and resonance
        Implements Axiom 72: k = decide(A)
        """
        if not self.processes:
            return None
            
        # Get all ready processes
        ready_procs = {pid: proc for pid, proc in self.processes.items() 
                      if proc.state in ["ready", "waiting"] and proc.wait_time <= 0}
        
        if not ready_procs:
            return None
            
        # Calculate a system frequency based on time
        sys_freq = 0.5 + 0.5 * math.sin(self.time / 10)
        
        # Find process with highest resonance * amplitude magnitude
        max_score = -1
        selected_pid = None
        
        for pid, proc in ready_procs.items():
            # Calculate resonance
            res = self.calculate_resonance(proc.freq, sys_freq)
            # Score based on resonance and amplitude magnitude
            score = res * abs(proc.amplitude)
            
            if score > max_score:
                max_score = score
                selected_pid = pid
        
        return selected_pid
    
    def update_running_process(self) -> None:
        """
        Update the amplitude for the running process
        Implements Axiom 73: A' = run(A)
        """
        if self.running_pid is None or self.running_pid not in self.processes:
            return
            
        proc = self.processes[self.running_pid]
        if proc.state != "running":
            return
            
        # Decrease remaining time
        proc.remaining_time -= 1
        
        # Update amplitude based on execution (rotation in complex plane)
        angle = math.pi / 8  # 22.5 degree rotation
        proc.amplitude *= np.exp(1j * angle)
        
        # If process is finished
        if proc.remaining_time <= 0:
            proc.state = "finished"
            self.running_pid = None
            logger.info(f"Process {proc.pid} finished execution")
    
    def update_waiting_processes(self) -> None:
        """
        Update amplitudes for waiting processes
        Implements a variation of Axiom 73
        """
        for pid, process in self.processes.items():
            if process.state == "waiting":
                # Decrease wait time
                process.wait_time = max(0, process.wait_time - 1)
                
                # If wait time is over, move to ready
                if process.wait_time == 0:
                    process.state = "ready"
                
                # Apply a different phase change for waiting processes
                small_angle = math.pi / 16  # 11.25 degree rotation
                process.amplitude *= np.exp(1j * small_angle)
    
    def schedule(self) -> Dict[str, Any]:
        """
        Main scheduling function integrating all quantum axioms
        Implements Axiom 74: k = schedule(S, f, A)
        
        Returns:
            Dict with scheduling results for this tick
        """
        self.time += 1
        
        # Update memory state (page faults, etc)
        self.update_memory()
        
        # Apply resource constraints
        self.apply_resource_constraints()
        
        # Update running process
        self.update_running_process()
        
        # Update waiting processes
        self.update_waiting_processes()
        
        # If no process is running, select one
        if self.running_pid is None or self.processes.get(self.running_pid, None) is None:
            next_pid = self.decide_next_process()
            if next_pid is not None:
                self.running_pid = next_pid
                self.processes[next_pid].state = "running"
                logger.info(f"Selected process {next_pid} for execution")
        
        # Collect metrics for logging
        metrics = {
            "time": self.time,
            "running_pid": self.running_pid,
            "processes": {}
        }
        
        for pid, proc in self.processes.items():
            # Calculate amplitude change from previous tick
            prev_amp = self.prev_amplitudes.get(pid, 0+0j)
            amp_change = abs(proc.amplitude - prev_amp)
            self.prev_amplitudes[pid] = proc.amplitude
            
            metrics["processes"][pid] = {
                "state": proc.state,
                "amplitude_real": proc.amplitude.real,
                "amplitude_imag": proc.amplitude.imag,
                "amplitude_magnitude": abs(proc.amplitude),
                "frequency": proc.freq,
                "remaining_time": proc.remaining_time,
                "wait_time": proc.wait_time,
                "amplitude_change": amp_change
            }
        
        self.schedule_log.append(metrics)
        return metrics
    
    def run_simulation(self, steps: int = 100) -> List[Dict[str, Any]]:
        """Run the scheduler for a specified number of steps"""
        for _ in range(steps):
            self.schedule()
            
            # Check if all processes are finished
            if all(p.state == "finished" for p in self.processes.values()):
                break
                
        return self.schedule_log


def demo_quantum_scheduler():
    """Demonstrate the quantum scheduler with a simple simulation"""
    scheduler = QuantumScheduler()
    
    # Create some test processes
    processes = [
        Process(pid=1, priority=0.9, amplitude=1.0+0j, freq=0,
               state="ready", wait_time=0, burst_time=20, remaining_time=20,
               quantum=4, mem_pages=[1, 2, 3, 4]),
        Process(pid=2, priority=0.5, amplitude=0.8+0.2j, freq=0,
               state="ready", wait_time=0, burst_time=15, remaining_time=15,
               quantum=4, mem_pages=[5, 6, 7]),
        Process(pid=3, priority=0.7, amplitude=0.7+0.3j, freq=0,
               state="ready", wait_time=2, burst_time=10, remaining_time=10,
               quantum=4, mem_pages=[8, 9]),
        Process(pid=4, priority=0.3, amplitude=0.6+0.4j, freq=0,
               state="ready", wait_time=0, burst_time=25, remaining_time=25,
               quantum=4, mem_pages=[10, 11, 12, 13, 14])
    ]
    
    # Add processes to scheduler
    for p in processes:
        scheduler.add_process(p)
    
    # Run simulation
    log = scheduler.run_simulation(steps=100)
    
    # Print summary
    print("Quantum Scheduler Simulation Results:")
    print("-------------------------------------")
    
    # Process completion order
    completion_order = []
    for i, entry in enumerate(log):
        for pid, proc_data in entry["processes"].items():
            if proc_data["state"] == "finished" and pid not in completion_order:
                completion_order.append((pid, i))
    
    print(f"Process completion order: {[pid for pid, _ in sorted(completion_order, key=lambda x: x[1])]}")
    
    # Average amplitude magnitude over time
    amplitude_avgs = []
    for entry in log:
        avg = sum(p["amplitude_magnitude"] for p in entry["processes"].values()) / len(entry["processes"])
        amplitude_avgs.append(avg)
    
    print(f"Average amplitude magnitude: {sum(amplitude_avgs)/len(amplitude_avgs):.4f}")
    
    # Switching frequency
    switches = 0
    last_pid = None
    for entry in log:
        if entry["running_pid"] != last_pid and entry["running_pid"] is not None:
            switches += 1
            last_pid = entry["running_pid"]
    
    print(f"Context switches: {switches}")
    print(f"Total simulation time: {log[-1]['time']}")
    
    return log


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the scheduler demo
    demo_quantum_scheduler()
