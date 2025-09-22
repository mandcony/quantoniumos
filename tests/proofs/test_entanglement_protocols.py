#!/usr/bin/env python3
"""
Entanglement Validation Protocols for QuantoniumOS
==================================================

Comprehensive test suite for validating entanglement in vertex-based
quantum systems. Implements rigorous quantum mechanical tests including:

- Bell inequality violations (CHSH test)
- Schmidt decomposition and rank analysis
- Entanglement entropy calculations
- Separability witnesses
- Fidelity comparisons with known entangled states
- Quantum correlation measures

This module ensures that the entanglement enhancements in QuantoniumOS
produce genuine quantum non-locality rather than classical correlations.
"""

import numpy as np
import pytest
from typing import List, Dict, Tuple, Optional, Union, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings

# Import QuTiP for reference calculations
try:
    import qutip as qt
    QUTIP_AVAILABLE = True
except ImportError:
    QUTIP_AVAILABLE = False
    warnings.warn("QuTiP not available. Some validation protocols will be limited.")

# Import QuantoniumOS components
try:
    from ..engine.vertex_assembly import EntangledVertexEngine, HyperEdge
    from ..engine.open_quantum_systems import OpenQuantumSystem, NoiseModel
    from ..core.canonical_true_rft import CanonicalTrueRFT
    ENGINE_AVAILABLE = True
except ImportError:
    ENGINE_AVAILABLE = False
    warnings.warn("QuantoniumOS engines not available for testing")


@dataclass
class EntanglementMeasure:
    """Container for entanglement measurement results."""
    name: str
    value: float
    threshold: float
    passed: bool
    details: Dict = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


class EntanglementProtocol(ABC):
    """Abstract base class for entanglement validation protocols."""
    
    def __init__(self, name: str):
        """Initialize entanglement protocol."""
        self.name = name
        self.tolerance = 1e-10
    
    @abstractmethod
    def validate(self, engine: 'EntangledVertexEngine', **kwargs) -> EntanglementMeasure:
        """Validate entanglement using this protocol."""
        pass
    
    def set_tolerance(self, tolerance: float) -> None:
        """Set numerical tolerance for validations."""
        self.tolerance = tolerance


class BellTestProtocol(EntanglementProtocol):
    """
    Bell test protocol using CHSH inequality.
    Tests for quantum non-locality via Bell inequality violations.
    """
    
    def __init__(self):
        """Initialize Bell test protocol."""
        super().__init__("Bell-CHSH Test")
        self.classical_bound = 2.0  # Classical physics bound
        self.quantum_bound = 2.0 * np.sqrt(2)  # Tsirelson's bound (~2.828)
    
    def validate(self, engine: 'EntangledVertexEngine', 
                vertices: Optional[List[int]] = None,
                num_measurements: int = 1000,
                **kwargs) -> EntanglementMeasure:
        """
        Perform CHSH Bell test on vertex pairs.
        
        Args:
            engine: Entangled vertex engine to test
            vertices: Vertex pair to test (default: [0, 1])
            num_measurements: Number of measurement samples
            
        Returns:
            EntanglementMeasure with CHSH violation results
        """
        if vertices is None:
            vertices = [0, 1]
        
        if len(vertices) != 2:
            raise ValueError("Bell test requires exactly 2 vertices")
        
        if engine.n_vertices < 2:
            return EntanglementMeasure(
                name=self.name,
                value=0.0,
                threshold=self.classical_bound,
                passed=False,
                details={'error': 'Insufficient vertices for Bell test'}
            )
        
        # Get quantum state
        state = engine.assemble_state()
        
        # Define measurement operators (Pauli matrices at different angles)
        def pauli_measurement(angle: float) -> np.ndarray:
            """Create Pauli measurement operator at given angle."""
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            return np.array([
                [cos_a, sin_a],
                [sin_a, -cos_a]
            ], dtype=complex)
        
        # CHSH measurement angles (optimal for maximum violation = Tsirelson bound)
        angles_A = [0, np.pi/2]  # Alice's measurement angles: 0° and 90°
        angles_B = [np.pi/4, -np.pi/4]  # Bob's measurement angles: 45° and -45°
        
        # Calculate CHSH correlator: ⟨A₁B₁⟩ + ⟨A₁B₂⟩ + ⟨A₂B₁⟩ - ⟨A₂B₂⟩
        chsh_sum = 0.0
        correlations = {}
        
        v1, v2 = vertices[0], vertices[1]
        
        for i, angle_A in enumerate(angles_A):
            for j, angle_B in enumerate(angles_B):
                # Create joint measurement operator
                A = pauli_measurement(angle_A)
                B = pauli_measurement(angle_B)
                
                # Calculate expectation value for two-qubit system
                correlation = self._calculate_two_qubit_correlation(state, A, B, v1, v2)
                
                # CHSH combination
                if (i, j) == (1, 1):  # A₂B₂ term
                    chsh_sum -= correlation
                else:  # A₁B₁, A₁B₂, A₂B₁ terms
                    chsh_sum += correlation
                
                correlations[f'A{i+1}B{j+1}'] = correlation
        
        # Determine if Bell inequality is violated
        bell_violation = chsh_sum > self.classical_bound
        violation_strength = chsh_sum - self.classical_bound
        
        return EntanglementMeasure(
            name=self.name,
            value=chsh_sum,
            threshold=self.classical_bound,
            passed=bell_violation,
            details={
                'correlations': correlations,
                'violation_strength': violation_strength,
                'quantum_bound': self.quantum_bound,
                'classical_bound': self.classical_bound,
                'vertices_tested': vertices
            }
        )
    
    def _calculate_two_qubit_correlation(self, state: np.ndarray, 
                                       A: np.ndarray, B: np.ndarray,
                                       v1: int, v2: int) -> float:
        """Calculate correlation ⟨ψ|A⊗B|ψ⟩ for two-qubit subsystem."""
        n_vertices = int(np.log2(len(state)))
        
        if n_vertices == 2:
            # Direct calculation for 2-qubit system
            AB = np.kron(A, B)
            return np.real(np.conj(state) @ AB @ state)
        else:
            # Extract two-qubit reduced density matrix using proper partial trace
            rho_full = np.outer(state, np.conj(state))
            rho_reduced = self._extract_two_qubit_reduced_state(rho_full, v1, v2, n_vertices)
            
            # Calculate expectation value
            AB = np.kron(A, B)
            return np.real(np.trace(rho_reduced @ AB))
    
    def _extract_two_qubit_reduced_state(self, rho: np.ndarray, v1: int, v2: int, n_vertices: int) -> np.ndarray:
        """Extract proper two-qubit reduced density matrix."""
        if QUTIP_AVAILABLE:
            try:
                # Use QuTiP for accurate partial trace
                import qutip as qt
                
                # Convert to QuTiP format
                qt_rho = qt.Qobj(rho, dims=[[2]*n_vertices, [2]*n_vertices])
                
                # Partial trace to get two-qubit subsystem
                keep_qubits = [v1, v2]
                rho_reduced_qt = qt_rho.ptrace(keep_qubits)
                
                return rho_reduced_qt.full()
            except Exception:
                pass
        
        # Fallback: manual partial trace (simplified for demonstration)
        return self._partial_trace_to_two_qubits(rho, v1, v2, n_vertices)
    
    def _partial_trace_to_two_qubits(self, rho: np.ndarray, v1: int, v2: int, n_vertices: int) -> np.ndarray:
        """Extract two-qubit reduced density matrix."""
        # Simplified implementation - full version would use tensor reshaping
        # For now, approximate by projecting onto computational basis
        rho_2q = np.zeros((4, 4), dtype=complex)
        
        for i in range(4):
            for j in range(4):
                # Map 2-qubit indices to full system indices
                bit1_i = (i >> 1) & 1
                bit0_i = i & 1
                bit1_j = (j >> 1) & 1
                bit0_j = j & 1
                
                # Sum over all configurations of other qubits
                for config in range(2**(n_vertices - 2)):
                    full_i = (config << 2) | (bit1_i << v1) | (bit0_i << v2)
                    full_j = (config << 2) | (bit1_j << v1) | (bit0_j << v2)
                    
                    if full_i < len(rho) and full_j < len(rho):
                        rho_2q[i, j] += rho[full_i, full_j]
        
        return rho_2q


class SchmidtDecompositionProtocol(EntanglementProtocol):
    """
    Schmidt decomposition protocol for quantifying entanglement.
    Validates entanglement via Schmidt rank and coefficients.
    """
    
    def __init__(self):
        """Initialize Schmidt decomposition protocol."""
        super().__init__("Schmidt Decomposition")
        self.separable_threshold = 1.01  # Allow small numerical errors
    
    def validate(self, engine: 'EntangledVertexEngine',
                bipartition: Optional[Tuple[List[int], List[int]]] = None,
                **kwargs) -> EntanglementMeasure:
        """
        Perform Schmidt decomposition validation.
        
        Args:
            engine: Entangled vertex engine to test
            bipartition: Bipartition of vertices (A, B). If None, use balanced split.
            
        Returns:
            EntanglementMeasure with Schmidt rank and entropy
        """
        if engine.n_vertices < 2:
            return EntanglementMeasure(
                name=self.name,
                value=1.0,
                threshold=self.separable_threshold,
                passed=False,
                details={'error': 'Need at least 2 vertices for bipartition'}
            )
        
        # Default bipartition: split vertices roughly in half
        if bipartition is None:
            mid = engine.n_vertices // 2
            subsystem_A = list(range(mid))
            subsystem_B = list(range(mid, engine.n_vertices))
            bipartition = (subsystem_A, subsystem_B)
        
        try:
            # Get Schmidt decomposition from engine
            schmidt_coeffs, U, Vh = engine.schmidt_decomposition(bipartition)
            
            # Calculate Schmidt rank (number of non-zero coefficients)
            significant_coeffs = schmidt_coeffs[schmidt_coeffs > self.tolerance]
            schmidt_rank = len(significant_coeffs)
            
            # Calculate entanglement entropy
            entropy = 0.0
            if len(significant_coeffs) > 0:
                normalized_coeffs = significant_coeffs**2
                normalized_coeffs = normalized_coeffs / np.sum(normalized_coeffs)
                entropy = -np.sum(normalized_coeffs * np.log2(normalized_coeffs + 1e-16))
            
            # Determine if state is entangled (Schmidt rank > 1)
            is_entangled = schmidt_rank > self.separable_threshold
            
            return EntanglementMeasure(
                name=self.name,
                value=schmidt_rank,
                threshold=self.separable_threshold,
                passed=is_entangled,
                details={
                    'schmidt_coefficients': schmidt_coeffs.tolist(),
                    'schmidt_rank': schmidt_rank,
                    'entanglement_entropy': entropy,
                    'bipartition': bipartition,
                    'max_entropy': np.log2(min(len(bipartition[0]), len(bipartition[1])))
                }
            )
            
        except Exception as e:
            return EntanglementMeasure(
                name=self.name,
                value=0.0,
                threshold=self.separable_threshold,
                passed=False,
                details={'error': f'Schmidt decomposition failed: {e}'}
            )


class SeparabilityWitnessProtocol(EntanglementProtocol):
    """
    Separability witness protocol using entanglement witnesses.
    Tests for entanglement using linear operators that separate
    entangled states from separable states.
    """
    
    def __init__(self):
        """Initialize separability witness protocol."""
        super().__init__("Separability Witness")
        self.witness_threshold = 0.0  # Positive values indicate entanglement
    
    def validate(self, engine: 'EntangledVertexEngine',
                witness_type: str = "concurrence",
                **kwargs) -> EntanglementMeasure:
        """
        Apply entanglement witness.
        
        Args:
            engine: Entangled vertex engine to test
            witness_type: Type of witness ("concurrence", "negativity", "ppt")
            
        Returns:
            EntanglementMeasure with witness results
        """
        state = engine.assemble_state()
        rho = np.outer(state, np.conj(state))
        
        if witness_type == "concurrence":
            witness_value = self._calculate_concurrence(rho)
            return EntanglementMeasure(
                name=f"{self.name} (Concurrence)",
                value=witness_value,
                threshold=self.witness_threshold,
                passed=witness_value > self.witness_threshold,
                details={'witness_type': 'concurrence'}
            )
        
        elif witness_type == "negativity":
            negativity = self._calculate_negativity(rho, engine.n_vertices)
            return EntanglementMeasure(
                name=f"{self.name} (Negativity)",
                value=negativity,
                threshold=self.witness_threshold,
                passed=negativity > self.witness_threshold,
                details={'witness_type': 'negativity'}
            )
        
        elif witness_type == "ppt":
            ppt_violation = self._peres_horodecki_test(rho, engine.n_vertices)
            return EntanglementMeasure(
                name=f"{self.name} (PPT)",
                value=ppt_violation,
                threshold=self.witness_threshold,
                passed=ppt_violation > self.witness_threshold,
                details={'witness_type': 'ppt'}
            )
        
        else:
            raise ValueError(f"Unknown witness type: {witness_type}")
    
    def _calculate_concurrence(self, rho: np.ndarray) -> float:
        """Calculate concurrence for two-qubit states."""
        if rho.shape != (4, 4):
            # Not a two-qubit system
            return 0.0
        
        # Pauli-Y matrix
        sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Y_Y = np.kron(sigma_y, sigma_y)
        
        # Spin-flipped density matrix
        rho_tilde = Y_Y @ np.conj(rho) @ Y_Y
        
        # Calculate R matrix and its eigenvalues
        R = rho @ rho_tilde
        eigenvals = np.linalg.eigvals(R)
        sqrt_eigenvals = np.sqrt(np.maximum(0, eigenvals.real))
        sqrt_eigenvals.sort()
        
        # Concurrence formula
        concurrence = max(0, sqrt_eigenvals[-1] - sqrt_eigenvals[-2] - sqrt_eigenvals[-3] - sqrt_eigenvals[-4])
        return concurrence
    
    def _calculate_negativity(self, rho: np.ndarray, n_vertices: int) -> float:
        """Calculate negativity (sum of negative eigenvalues of partial transpose)."""
        if n_vertices < 2:
            return 0.0
        
        # Simplified partial transpose for first qubit
        rho_pt = self._partial_transpose_first_qubit(rho, n_vertices)
        
        # Calculate eigenvalues of partial transpose
        eigenvals = np.linalg.eigvals(rho_pt)
        negative_eigenvals = eigenvals[eigenvals.real < -self.tolerance]
        
        if len(negative_eigenvals) > 0:
            return -np.sum(negative_eigenvals.real)
        else:
            return 0.0
    
    def _partial_transpose_first_qubit(self, rho: np.ndarray, n_vertices: int) -> np.ndarray:
        """Apply partial transpose to first qubit."""
        dim = 2**n_vertices
        rho_pt = np.zeros_like(rho)
        
        for i in range(dim):
            for j in range(dim):
                # Extract first qubit indices
                i_first = i & 1
                j_first = j & 1
                i_rest = i >> 1
                j_rest = j >> 1
                
                # Apply transpose to first qubit subspace
                i_new = (i_rest << 1) | j_first
                j_new = (j_rest << 1) | i_first
                
                rho_pt[i_new, j_new] = rho[i, j]
        
        return rho_pt
    
    def _peres_horodecki_test(self, rho: np.ndarray, n_vertices: int) -> float:
        """Peres-Horodecki criterion (PPT test)."""
        # Same as negativity but return maximum negative eigenvalue
        negativity = self._calculate_negativity(rho, n_vertices)
        return negativity


class FidelityComparisonProtocol(EntanglementProtocol):
    """
    Fidelity comparison protocol against known entangled states.
    Validates that vertex engine produces states with high fidelity
    to target entangled states (Bell, GHZ, W, etc.).
    """
    
    def __init__(self):
        """Initialize fidelity comparison protocol."""
        super().__init__("Fidelity Comparison")
        self.fidelity_threshold = 0.95  # High fidelity requirement
    
    def validate(self, engine: 'EntangledVertexEngine',
                target_state: str = "bell",
                entanglement_level: float = 0.9,
                **kwargs) -> EntanglementMeasure:
        """
        Compare fidelity with target entangled state.
        
        Args:
            engine: Entangled vertex engine to test
            target_state: Target state name ("bell", "ghz", "w")
            entanglement_level: Entanglement level for state generation
            
        Returns:
            EntanglementMeasure with fidelity results
        """
        if QUTIP_AVAILABLE:
            # Use engine's built-in QuTiP fidelity
            fidelity = engine.fidelity_with_qutip(target_state)
            
            return EntanglementMeasure(
                name=f"{self.name} ({target_state.upper()})",
                value=fidelity,
                threshold=self.fidelity_threshold,
                passed=fidelity >= self.fidelity_threshold,
                details={
                    'target_state': target_state,
                    'entanglement_level': entanglement_level,
                    'method': 'qutip'
                }
            )
        else:
            # Manual fidelity calculation
            engine_state = engine.assemble_entangled_state(entanglement_level)
            target = self._create_target_state(target_state, engine.n_vertices)
            
            if target is not None:
                fidelity = abs(np.vdot(target, engine_state))**2
                
                return EntanglementMeasure(
                    name=f"{self.name} ({target_state.upper()})",
                    value=fidelity,
                    threshold=self.fidelity_threshold,
                    passed=fidelity >= self.fidelity_threshold,
                    details={
                        'target_state': target_state,
                        'entanglement_level': entanglement_level,
                        'method': 'manual'
                    }
                )
            else:
                return EntanglementMeasure(
                    name=f"{self.name} ({target_state.upper()})",
                    value=0.0,
                    threshold=self.fidelity_threshold,
                    passed=False,
                    details={'error': f'Cannot create target state: {target_state}'}
                )
    
    def _create_target_state(self, state_name: str, n_vertices: int) -> Optional[np.ndarray]:
        """Create target entangled state."""
        dim = 2**n_vertices
        state = np.zeros(dim, dtype=complex)
        
        if state_name.lower() == "bell" and n_vertices >= 2:
            # Bell state: (|00⟩ + |11⟩)/√2
            state[0] = 1.0 / np.sqrt(2)  # |00...0⟩
            state[3] = 1.0 / np.sqrt(2)  # |11...1⟩ (for first two qubits)
            
        elif state_name.lower() == "ghz":
            # GHZ state: (|00...0⟩ + |11...1⟩)/√2
            state[0] = 1.0 / np.sqrt(2)
            state[-1] = 1.0 / np.sqrt(2)
            
        elif state_name.lower() == "w":
            # W state: (|100...0⟩ + |010...0⟩ + ... + |000...1⟩)/√N
            for i in range(n_vertices):
                basis_idx = 1 << i
                state[basis_idx] = 1.0 / np.sqrt(n_vertices)
        
        else:
            return None
        
        return state


class EntanglementValidationSuite:
    """
    Comprehensive entanglement validation suite.
    Runs all validation protocols and provides summary report.
    """
    
    def __init__(self):
        """Initialize validation suite."""
        self.protocols = [
            BellTestProtocol(),
            SchmidtDecompositionProtocol(),
            SeparabilityWitnessProtocol(),
            FidelityComparisonProtocol()
        ]
        self.results: List[EntanglementMeasure] = []
    
    def run_full_validation(self, engine: 'EntangledVertexEngine',
                          test_configurations: Optional[List[Dict]] = None) -> Dict:
        """
        Run complete validation suite.
        
        Args:
            engine: Entangled vertex engine to validate
            test_configurations: List of test configurations for each protocol
            
        Returns:
            Comprehensive validation report
        """
        if test_configurations is None:
            test_configurations = [
                {'entanglement_level': 0.1},  # Low entanglement
                {'entanglement_level': 0.5},  # Medium entanglement
                {'entanglement_level': 0.9}   # High entanglement
            ]
        
        self.results = []
        summary = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'protocols': {},
            'configurations': test_configurations
        }
        
        for config in test_configurations:
            entanglement_level = config.get('entanglement_level', 0.5)
            
            # Set up engine for this configuration
            if hasattr(engine, 'add_hyperedge') and entanglement_level > 0.3:
                # Add hyperedges for higher entanglement levels
                if engine.n_vertices >= 2:
                    engine.add_hyperedge({0, 1}, correlation_strength=entanglement_level)
                if engine.n_vertices >= 4:
                    engine.add_hyperedge({2, 3}, correlation_strength=entanglement_level * 0.8)
            
            config_results = []
            
            # Run each protocol
            for protocol in self.protocols:
                try:
                    if isinstance(protocol, BellTestProtocol):
                        result = protocol.validate(engine, vertices=[0, 1])
                    elif isinstance(protocol, FidelityComparisonProtocol):
                        result = protocol.validate(engine, target_state="bell", 
                                                 entanglement_level=entanglement_level)
                    else:
                        result = protocol.validate(engine)
                    
                    config_results.append(result)
                    self.results.append(result)
                    
                    summary['total_tests'] += 1
                    if result.passed:
                        summary['passed_tests'] += 1
                    else:
                        summary['failed_tests'] += 1
                        
                except Exception as e:
                    error_result = EntanglementMeasure(
                        name=f"{protocol.name} (Error)",
                        value=0.0,
                        threshold=0.0,
                        passed=False,
                        details={'error': str(e)}
                    )
                    config_results.append(error_result)
                    self.results.append(error_result)
                    summary['total_tests'] += 1
                    summary['failed_tests'] += 1
            
            # Store results for this configuration
            config_key = f"entanglement_{entanglement_level:.1f}"
            summary['protocols'][config_key] = config_results
        
        # Calculate overall success rate
        summary['success_rate'] = (summary['passed_tests'] / summary['total_tests'] 
                                 if summary['total_tests'] > 0 else 0.0)
        
        return summary
    
    def generate_report(self, validation_results: Dict) -> str:
        """Generate human-readable validation report."""
        report = []
        report.append("=" * 80)
        report.append("QUANTONIUMOS ENTANGLEMENT VALIDATION REPORT")
        report.append("=" * 80)
        report.append()
        
        # Summary statistics
        report.append(f"Total Tests: {validation_results['total_tests']}")
        report.append(f"Passed: {validation_results['passed_tests']}")
        report.append(f"Failed: {validation_results['failed_tests']}")
        report.append(f"Success Rate: {validation_results['success_rate']:.1%}")
        report.append()
        
        # Protocol results by configuration
        for config_name, protocol_results in validation_results['protocols'].items():
            report.append(f"Configuration: {config_name}")
            report.append("-" * 40)
            
            for result in protocol_results:
                status = "PASS" if result.passed else "FAIL"
                report.append(f"  {result.name}: {status}")
                report.append(f"    Value: {result.value:.4f}")
                report.append(f"    Threshold: {result.threshold:.4f}")
                
                if result.details and 'error' not in result.details:
                    # Show relevant details
                    if 'violation_strength' in result.details:
                        report.append(f"    Bell violation: {result.details['violation_strength']:.4f}")
                    if 'schmidt_rank' in result.details:
                        report.append(f"    Schmidt rank: {result.details['schmidt_rank']}")
                    if 'entanglement_entropy' in result.details:
                        report.append(f"    Entropy: {result.details['entanglement_entropy']:.4f}")
                
                if result.details and 'error' in result.details:
                    report.append(f"    Error: {result.details['error']}")
                
                report.append()
        
        # Overall assessment
        report.append("ASSESSMENT")
        report.append("-" * 40)
        if validation_results['success_rate'] >= 0.8:
            report.append("✓ EXCELLENT: QuantoniumOS demonstrates robust entanglement support")
        elif validation_results['success_rate'] >= 0.6:
            report.append("⚠ GOOD: QuantoniumOS shows entanglement with some limitations")
        elif validation_results['success_rate'] >= 0.4:
            report.append("⚠ PARTIAL: QuantoniumOS has limited entanglement capabilities")
        else:
            report.append("✗ INADEQUATE: QuantoniumOS entanglement support needs improvement")
        
        report.append()
        
        # Recommendations
        if validation_results['success_rate'] < 1.0:
            report.append("RECOMMENDATIONS")
            report.append("-" * 40)
            report.append("• Increase correlation strength in hyperedges")
            report.append("• Improve RFT phase relationships")
            report.append("• Optimize Schmidt decomposition implementation")
            report.append("• Add more sophisticated entanglement witnesses")
            report.append()
        
        return "\n".join(report)


# Test fixtures and convenience functions
def create_test_engine(n_vertices: int = 4, entanglement_enabled: bool = True) -> 'EntangledVertexEngine':
    """Create test vertex engine for validation."""
    if not ENGINE_AVAILABLE:
        raise RuntimeError("QuantoniumOS engines not available for testing")
    
    return EntangledVertexEngine(n_vertices, entanglement_enabled)


def run_basic_entanglement_validation(n_vertices: int = 4) -> Dict:
    """Run basic entanglement validation for given number of vertices."""
    engine = create_test_engine(n_vertices)
    suite = EntanglementValidationSuite()
    results = suite.run_full_validation(engine)
    return results


# Export main classes and functions
__all__ = [
    'EntanglementMeasure', 'EntanglementProtocol', 'EntanglementValidationSuite',
    'BellTestProtocol', 'SchmidtDecompositionProtocol', 'SeparabilityWitnessProtocol',
    'FidelityComparisonProtocol', 'create_test_engine', 'run_basic_entanglement_validation',
    'ENGINE_AVAILABLE', 'QUTIP_AVAILABLE'
]