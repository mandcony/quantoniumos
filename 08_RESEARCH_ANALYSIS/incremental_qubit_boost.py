#!/usr/bin/env python3
"""
Incremental Qubit Boost Analysis

This module provides incremental scaling analysis for quantum systems,
focusing on gradual qubit count increases and their computational implications.
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Any
import psutil
import tracemalloc


class IncrementalQubitAnalyzer:
    """
    Analyzes incremental qubit scaling patterns and performance characteristics.
    """
    
    def __init__(self):
        self.scaling_data = {}
        self.performance_metrics = {}
        
    def analyze_incremental_scaling(self, start_qubits: int = 5, end_qubits: int = 25, step: int = 2):
        """
        Analyze incremental qubit scaling from start to end qubits.
        
        Args:
            start_qubits: Starting number of qubits
            end_qubits: Ending number of qubits  
            step: Increment step size
        """
        print(f"🔬 INCREMENTAL QUBIT SCALING ANALYSIS")
        print(f"Range: {start_qubits} to {end_qubits} qubits (step: {step})")
        print("=" * 60)
        
        results = []
        
        for n_qubits in range(start_qubits, end_qubits + 1, step):
            print(f"\n📊 Analyzing {n_qubits} qubits...")
            
            # Start timing and memory tracking
            start_time = time.time()
            tracemalloc.start()
            
            # Calculate theoretical dimensions
            hilbert_dim = 2**n_qubits
            memory_complex128 = hilbert_dim * 16  # bytes for complex128
            memory_mb = memory_complex128 / (1024**2)
            memory_gb = memory_mb / 1024
            
            # Simulate quantum state vector operations
            try:
                if n_qubits <= 20:  # Only create actual arrays for small systems
                    state_vector = np.zeros(hilbert_dim, dtype=complex)
                    state_vector[0] = 1.0  # |00...0⟩ state
                    
                    # Apply some operations
                    for i in range(min(10, n_qubits)):
                        # Simulate gate application
                        state_vector = self._apply_simulated_gate(state_vector, i)
                    
                    actual_memory = True
                else:
                    # For larger systems, just calculate theoretical values
                    state_vector = None
                    actual_memory = False
                
                end_time = time.time()
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                
                # Calculate performance metrics
                execution_time = end_time - start_time
                
                result = {
                    'qubits': n_qubits,
                    'hilbert_dimension': hilbert_dim,
                    'memory_mb': memory_mb,
                    'memory_gb': memory_gb,
                    'execution_time': execution_time,
                    'peak_memory_mb': peak / (1024**2),
                    'actual_memory_used': actual_memory,
                    'feasible': memory_gb < 32.0  # Feasible on 32GB system
                }
                
                results.append(result)
                
                # Print result
                feasible_str = "✅ Feasible" if result['feasible'] else "❌ Too Large"
                print(f"   Dimension: 2^{n_qubits} = {hilbert_dim:,}")
                print(f"   Memory: {memory_gb:.3f} GB")
                print(f"   Time: {execution_time:.4f}s")
                print(f"   Status: {feasible_str}")
                
            except MemoryError:
                print(f"   ❌ Memory Error: Cannot allocate {memory_gb:.1f} GB")
                result = {
                    'qubits': n_qubits,
                    'hilbert_dimension': hilbert_dim,
                    'memory_gb': memory_gb,
                    'error': 'MemoryError',
                    'feasible': False
                }
                results.append(result)
                break
        
        self.scaling_data['incremental'] = results
        return results
    
    def _apply_simulated_gate(self, state_vector: np.ndarray, qubit_idx: int) -> np.ndarray:
        """
        Apply a simulated quantum gate to demonstrate computation.
        
        Args:
            state_vector: Current quantum state
            qubit_idx: Index of qubit to apply gate to
            
        Returns:
            Modified state vector
        """
        n_qubits = int(np.log2(len(state_vector)))
        
        # Create a simple rotation gate effect
        # This is a simplified simulation for demonstration
        rotation_angle = 0.1 * (qubit_idx + 1)
        
        # Apply phase rotation to subset of amplitudes
        for i in range(len(state_vector)):
            if (i >> qubit_idx) & 1:  # If qubit_idx is |1⟩ in state |i⟩
                state_vector[i] *= np.exp(1j * rotation_angle)
        
        # Renormalize
        norm = np.linalg.norm(state_vector)
        if norm > 1e-12:
            state_vector /= norm
            
        return state_vector
    
    def generate_scaling_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive scaling report.
        
        Returns:
            Dictionary containing scaling analysis results
        """
        if 'incremental' not in self.scaling_data:
            return {'error': 'No scaling data available. Run analyze_incremental_scaling() first.'}
        
        results = self.scaling_data['incremental']
        
        # Find transition points
        last_feasible = None
        first_infeasible = None
        
        for result in results:
            if result.get('feasible', False):
                last_feasible = result
            elif first_infeasible is None:
                first_infeasible = result
        
        # Calculate scaling trends
        feasible_results = [r for r in results if r.get('feasible', False)]
        
        if len(feasible_results) >= 2:
            execution_times = [r['execution_time'] for r in feasible_results]
            qubit_counts = [r['qubits'] for r in feasible_results]
            
            # Fit exponential scaling
            log_times = np.log(execution_times)
            coeffs = np.polyfit(qubit_counts, log_times, 1)
            scaling_factor = np.exp(coeffs[0])
        else:
            scaling_factor = None
        
        report = {
            'total_systems_analyzed': len(results),
            'feasible_systems': len(feasible_results),
            'last_feasible_qubits': last_feasible['qubits'] if last_feasible else None,
            'scaling_factor_per_qubit': scaling_factor,
            'memory_limit_reached': first_infeasible is not None,
            'transition_point': {
                'last_feasible': last_feasible,
                'first_infeasible': first_infeasible
            },
            'detailed_results': results
        }
        
        return report
    
    def print_summary(self):
        """Print summary of incremental scaling analysis."""
        report = self.generate_scaling_report()
        
        if 'error' in report:
            print(f"❌ {report['error']}")
            return
        
        print(f"\n📈 INCREMENTAL SCALING SUMMARY")
        print("=" * 50)
        print(f"Systems analyzed: {report['total_systems_analyzed']}")
        print(f"Feasible systems: {report['feasible_systems']}")
        
        if report['last_feasible_qubits']:
            print(f"Largest feasible: {report['last_feasible_qubits']} qubits")
        
        if report['scaling_factor_per_qubit']:
            print(f"Time scaling: {report['scaling_factor_per_qubit']:.2f}x per qubit")
        
        if report['memory_limit_reached']:
            transition = report['transition_point']
            if transition['last_feasible'] and transition['first_infeasible']:
                last_mem = transition['last_feasible']['memory_gb']
                first_mem = transition['first_infeasible']['memory_gb'] 
                print(f"Memory transition: {last_mem:.1f} GB → {first_mem:.1f} GB")


def run_incremental_analysis():
    """Run comprehensive incremental qubit analysis."""
    analyzer = IncrementalQubitAnalyzer()
    
    # Run incremental scaling analysis
    results = analyzer.analyze_incremental_scaling(
        start_qubits=5,
        end_qubits=25,
        step=2
    )
    
    # Print summary
    analyzer.print_summary()
    
    return analyzer


if __name__ == "__main__":
    print("🚀 INCREMENTAL QUBIT BOOST ANALYSIS")
    print("=" * 60)
    
    analyzer = run_incremental_analysis()
