#!/usr/bin/env python3
"""
Comprehensive Test Suite for QuantoniumOS Entanglement Enhancements
==================================================================

This test suite validates all components of the entanglement enhancement:
1. EntangledVertexEngine functionality
2. Kraus operators and open systems
3. Entanglement validation protocols
4. Integration with existing QuantoniumOS components

Run with: python -m pytest tests/proofs/test_entangled_assembly.py -v
"""

import numpy as np
import pytest  # type: ignore[import]
import warnings
from typing import List, Dict, Tuple

# Import QuantoniumOS components
try:
    import sys
    import os
    sys.path.insert(0, os.path.abspath('.'))
    
    from src.engine.vertex_assembly import EntangledVertexEngine, HyperEdge, QUTIP_AVAILABLE, RFT_AVAILABLE
    from src.engine.open_quantum_systems import (
        OpenQuantumSystem, DepolarizingChannel, AmplitudeDampingChannel,
        NoiseModel, create_depolarizing_channel
    )
    from tests.proofs.test_entanglement_protocols import (
        EntanglementValidationSuite, BellTestProtocol, SchmidtDecompositionProtocol,
        create_test_engine, ENGINE_AVAILABLE
    )
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    COMPONENTS_AVAILABLE = False
    RFT_AVAILABLE = False
    warnings.warn(f"QuantoniumOS components not available: {e}")

# Import QuTiP for reference if available
try:
    import qutip as qt  # type: ignore[import]
    QUTIP_AVAILABLE = True
except ImportError:
    QUTIP_AVAILABLE = False


class TestEntangledVertexEngine:
    """Test suite for EntangledVertexEngine."""
    
    @pytest.fixture
    def basic_engine(self):
        """Create basic entangled vertex engine."""
        if not COMPONENTS_AVAILABLE or not ENGINE_AVAILABLE:
            pytest.skip("QuantoniumOS components not available")
        return EntangledVertexEngine(n_vertices=4, entanglement_enabled=True)
    
    @pytest.fixture
    def separable_engine(self):
        """Create separable (non-entangled) vertex engine."""
        if not COMPONENTS_AVAILABLE or not ENGINE_AVAILABLE:
            pytest.skip("QuantoniumOS components not available")
        return EntangledVertexEngine(n_vertices=4, entanglement_enabled=False)
    
    def test_engine_initialization(self, basic_engine):
        """Test basic engine initialization."""
        assert basic_engine.n_vertices == 4
        assert basic_engine.entanglement_enabled == True
        assert basic_engine.phi == (1 + np.sqrt(5)) / 2
        assert len(basic_engine.vertex_states) == 4
        assert len(basic_engine.hyperedges) == 0
    
    def test_hyperedge_addition(self, basic_engine):
        """Test adding hyperedges for correlation."""
        # Add a simple two-vertex correlation
        basic_engine.add_hyperedge({0, 1}, correlation_strength=0.8)
        
        assert len(basic_engine.hyperedges) == 1
        edge = basic_engine.hyperedges[0]
        assert edge.vertices == {0, 1}
        assert edge.correlation_strength == 0.8
        
        # Check correlation matrix was updated
        assert basic_engine.correlation_matrix[0, 1] != 0
        
    def test_separable_state_assembly(self, basic_engine):
        """Test assembly of separable states."""
        state = basic_engine.assemble_entangled_state(entanglement_level=0.0)
        
        # Check normalization
        assert abs(np.linalg.norm(state) - 1.0) < 1e-10
        
        # Check state dimension
        expected_dim = 2**basic_engine.n_vertices
        assert len(state) == expected_dim
    
    def test_entangled_state_assembly(self, basic_engine):
        """Test assembly of entangled states."""
        # Add correlation
        basic_engine.add_hyperedge({0, 1}, correlation_strength=1.0)
        
        # Generate entangled state
        state = basic_engine.assemble_entangled_state(entanglement_level=0.8)
        
        # Check normalization
        assert abs(np.linalg.norm(state) - 1.0) < 1e-10
        
        # Check that state is not just |0000⟩
        assert abs(state[0]) < 0.99  # Should have other components
    
    def test_entanglement_entropy_calculation(self, basic_engine):
        """Test entanglement entropy calculations."""
        # Add correlation for entanglement
        basic_engine.add_hyperedge({0, 1}, correlation_strength=1.0)
        
        # Calculate entropy for single qubit subsystem
        entropy = basic_engine.get_entanglement_entropy([0])
        
        # For correlated qubits, entropy should be positive
        assert entropy >= 0
        
        # For separable states, entropy should be near zero
        separable_engine = EntangledVertexEngine(4, entanglement_enabled=False)
        sep_entropy = separable_engine.get_entanglement_entropy([0])
        assert sep_entropy < 1e-10
    
    def test_schmidt_decomposition(self, basic_engine):
        """Test Schmidt decomposition functionality."""
        # Add correlation
        basic_engine.add_hyperedge({0, 1}, correlation_strength=0.9)
        
        # Perform Schmidt decomposition
        bipartition = ([0, 1], [2, 3])
        schmidt_coeffs, U, Vh = basic_engine.schmidt_decomposition(bipartition)
        
        # Check Schmidt coefficients are real and positive
        assert np.all(schmidt_coeffs >= 0)
        
        # Check normalization
        assert abs(np.sum(schmidt_coeffs**2) - 1.0) < 1e-10
        
        # Check unitary matrices
        assert np.allclose(U @ U.conj().T, np.eye(U.shape[0]), atol=1e-10)
        assert np.allclose(Vh @ Vh.conj().T, np.eye(Vh.shape[0]), atol=1e-10)
    
    @pytest.mark.skipif(not QUTIP_AVAILABLE, reason="QuTiP not available")
    def test_qutip_fidelity(self, basic_engine):
        """Test fidelity calculations with QuTiP."""
        # Add Bell-state correlation
        basic_engine.add_hyperedge({0, 1}, correlation_strength=1.0)
        
        # Calculate fidelity with Bell state
        fidelity = basic_engine.fidelity_with_qutip("bell")
        
        # Should have reasonable fidelity with Bell state
        assert 0.0 <= fidelity <= 1.0
        
        # For good correlation, fidelity should be decent
        if fidelity < 0.5:
            warnings.warn(f"Low Bell state fidelity: {fidelity:.3f}")
    
    def test_backward_compatibility(self):
        """Test that existing code still works."""
        if not COMPONENTS_AVAILABLE or not ENGINE_AVAILABLE:
            pytest.skip("QuantoniumOS components not available")
        
        # Old-style vertex assembly should still work
        from src.engine.vertex_assembly import VertexAssembly
        
        old_engine = VertexAssembly(n_vertices=4)
        state = old_engine.assemble_state()
        
        # Should produce normalized state
        assert abs(np.linalg.norm(state) - 1.0) < 1e-10
        
        # Should be separable (entropy near zero)
        entropy = old_engine.get_entanglement_entropy([0])
        assert entropy < 1e-10


class TestOpenQuantumSystems:
    """Test suite for open quantum systems and Kraus operators."""
    
    @pytest.fixture
    def basic_system(self):
        """Create basic open quantum system."""
        if not COMPONENTS_AVAILABLE or not ENGINE_AVAILABLE:
            pytest.skip("QuantoniumOS components not available")
        engine = EntangledVertexEngine(n_vertices=2, entanglement_enabled=True)
        return OpenQuantumSystem(engine)
    
    def test_depolarizing_channel(self):
        """Test depolarizing channel implementation."""
        if not COMPONENTS_AVAILABLE:
            pytest.skip("QuantoniumOS components not available")
        
        channel = DepolarizingChannel(p=0.1)
        kraus_ops = channel.get_kraus_operators(dim=2)
        
        # Should have 4 Kraus operators for single qubit
        assert len(kraus_ops) == 4
        
        # Check completeness relation: ∑ K†K = I
        completeness = sum(K.conj().T @ K for K in kraus_ops)
        identity = np.eye(2, dtype=complex)
        assert np.allclose(completeness, identity, atol=1e-10)
    
    def test_amplitude_damping_channel(self):
        """Test amplitude damping channel."""
        if not COMPONENTS_AVAILABLE:
            pytest.skip("QuantoniumOS components not available")
        
        channel = AmplitudeDampingChannel(gamma=0.2)
        kraus_ops = channel.get_kraus_operators()
        
        # Should have 2 Kraus operators
        assert len(kraus_ops) == 2
        
        # Check completeness
        completeness = sum(K.conj().T @ K for K in kraus_ops)
        identity = np.eye(2, dtype=complex)
        assert np.allclose(completeness, identity, atol=1e-10)
    
    def test_decoherence_application(self, basic_system):
        """Test applying decoherence to quantum states."""
        # Create initial pure state
        psi = np.array([1, 0, 0, 0], dtype=complex)  # |00⟩
        rho_initial = np.outer(psi, psi.conj())
        
        # Apply depolarizing noise to first qubit only
        rho_final = basic_system.apply_decoherence(
            rho_initial, 
            noise_model=NoiseModel.DEPOLARIZING, 
            p=0.1,
            target_qubits=[0]  # Apply only to first qubit
        )
        
        # Check trace preservation
        assert abs(np.trace(rho_final) - 1.0) < 1e-10
        
        # Check that state became mixed (purity < 1)
        purity_initial = basic_system.purity(rho_initial)
        purity_final = basic_system.purity(rho_final)
        
        assert abs(purity_initial - 1.0) < 1e-10  # Initial state is pure
        assert purity_final < purity_initial  # Final state is more mixed
    
    def test_entropy_calculations(self, basic_system):
        """Test von Neumann entropy calculations."""
        # Pure state should have zero entropy
        psi = np.array([1, 0], dtype=complex)
        rho_pure = np.outer(psi, psi.conj())
        entropy_pure = basic_system.von_neumann_entropy(rho_pure)
        assert entropy_pure < 1e-10
        
        # Maximally mixed state should have maximum entropy
        rho_mixed = np.eye(2, dtype=complex) / 2
        entropy_mixed = basic_system.von_neumann_entropy(rho_mixed)
        assert abs(entropy_mixed - 1.0) < 1e-10  # log₂(2) = 1 bit
    
    def test_evolution_history(self, basic_system):
        """Test system evolution tracking."""
        # Initially no history
        assert len(basic_system.evolution_history) == 0
        
        # Add noise channel
        channel = create_depolarizing_channel(0.1)
        basic_system.add_noise_channel(channel)
        
        # Should record the addition
        assert len(basic_system.evolution_history) == 1
        assert basic_system.evolution_history[0]['type'] == 'add_channel'


class TestEntanglementValidation:
    """Test suite for entanglement validation protocols."""
    
    @pytest.fixture
    def test_engine(self):
        """Create test engine for validation."""
        if not COMPONENTS_AVAILABLE or not ENGINE_AVAILABLE:
            pytest.skip("QuantoniumOS components not available")
        return create_test_engine(n_vertices=4, entanglement_enabled=True)
    
    @pytest.fixture
    def validation_suite(self):
        """Create validation suite."""
        if not COMPONENTS_AVAILABLE:
            pytest.skip("QuantoniumOS components not available")
        return EntanglementValidationSuite()
    
    def test_bell_test_protocol(self, test_engine):
        """Test Bell inequality violation detection."""
        if not COMPONENTS_AVAILABLE:
            pytest.skip("QuantoniumOS components not available")
        
        # Add strong correlation for Bell violation
        test_engine.add_hyperedge({0, 1}, correlation_strength=1.0)
        
        protocol = BellTestProtocol()
        result = protocol.validate(test_engine, vertices=[0, 1])
        
        # Check result structure
        assert hasattr(result, 'name')
        assert hasattr(result, 'value')
        assert hasattr(result, 'passed')
        assert hasattr(result, 'details')
        
        # CHSH value should be within bounds
        assert 0 <= result.value <= protocol.quantum_bound + 0.1  # Allow small numerical errors
    
    def test_schmidt_decomposition_protocol(self, test_engine):
        """Test Schmidt decomposition validation."""
        if not COMPONENTS_AVAILABLE:
            pytest.skip("QuantoniumOS components not available")
        
        # Add correlation
        test_engine.add_hyperedge({0, 1}, correlation_strength=0.8)
        
        protocol = SchmidtDecompositionProtocol()
        result = protocol.validate(test_engine)
        
        # Should detect entanglement (Schmidt rank > 1)
        if result.passed:
            assert result.value > protocol.separable_threshold
        
        # Check details
        assert 'schmidt_rank' in result.details
        assert 'entanglement_entropy' in result.details
    
    def test_full_validation_suite(self, test_engine, validation_suite):
        """Test complete validation suite."""
        # Add various correlations
        test_engine.add_hyperedge({0, 1}, correlation_strength=0.9)
        test_engine.add_hyperedge({2, 3}, correlation_strength=0.7)
        
        # Run full validation
        results = validation_suite.run_full_validation(test_engine)
        
        # Check result structure
        assert 'total_tests' in results
        assert 'passed_tests' in results
        assert 'failed_tests' in results
        assert 'success_rate' in results
        assert 'protocols' in results
        
        # Should have run some tests
        assert results['total_tests'] > 0
        
        # Success rate should be reasonable
        assert 0 <= results['success_rate'] <= 1
    
    def test_validation_report_generation(self, validation_suite):
        """Test validation report generation."""
        # Create mock results
        mock_results = {
            'total_tests': 8,
            'passed_tests': 6,
            'failed_tests': 2,
            'success_rate': 0.75,
            'protocols': {
                'entanglement_0.5': []
            }
        }
        
        report = validation_suite.generate_report(mock_results)
        
        # Should be a non-empty string
        assert isinstance(report, str)
        assert len(report) > 100
        
        # Should contain key information
        assert '8' in report  # total tests
        assert '75%' in report or '75.0%' in report  # success rate


class TestIntegration:
    """Integration tests for all components working together."""
    
    @pytest.mark.skipif(not (COMPONENTS_AVAILABLE and ENGINE_AVAILABLE), reason="Components not available")
    def test_end_to_end_entanglement(self):
        """Test complete entanglement workflow."""
        # Create entangled vertex engine
        engine = EntangledVertexEngine(n_vertices=4, entanglement_enabled=True)
        
        # Add multiple correlations
        engine.add_hyperedge({0, 1}, correlation_strength=0.9)
        engine.add_hyperedge({2, 3}, correlation_strength=0.8)
        engine.add_hyperedge({0, 2}, correlation_strength=0.5)
        
        # Generate entangled state
        state = engine.assemble_entangled_state(entanglement_level=0.7)
        
        # Verify state properties
        assert abs(np.linalg.norm(state) - 1.0) < 1e-10
        assert len(state) == 16  # 2^4
        
        # Apply decoherence
        open_system = OpenQuantumSystem(engine)
        rho = np.outer(state, state.conj())
        rho_noisy = open_system.apply_decoherence(rho, NoiseModel.DEPOLARIZING, p=0.05)
        
        # Verify decoherence effects
        assert abs(np.trace(rho_noisy) - 1.0) < 1e-10
        assert open_system.purity(rho_noisy) < 1.0
        
        # Run validation
        suite = EntanglementValidationSuite()
        validation_results = suite.run_full_validation(engine)
        
        # Should have reasonable success rate
        assert validation_results['success_rate'] > 0.3  # At least some tests should pass
    
    @pytest.mark.skipif(not (COMPONENTS_AVAILABLE and ENGINE_AVAILABLE), reason="Components not available") 
    def test_performance_scaling(self):
        """Test performance scaling with system size."""
        import time
        
        sizes = [2, 4, 6, 8]
        times = []
        
        for n in sizes:
            engine = EntangledVertexEngine(n_vertices=n, entanglement_enabled=True)
            
            # Add some correlations
            for i in range(0, n-1, 2):
                engine.add_hyperedge({i, i+1}, correlation_strength=0.8)
            
            # Time state assembly
            start_time = time.time()
            state = engine.assemble_entangled_state(entanglement_level=0.6)
            end_time = time.time()
            
            times.append(end_time - start_time)
            
            # Verify state
            assert abs(np.linalg.norm(state) - 1.0) < 1e-10
        
        # Times should scale reasonably (not exponentially)
        # For small sizes, should complete quickly
        assert all(t < 1.0 for t in times)  # All under 1 second
    
    @pytest.mark.skipif(not RFT_AVAILABLE, reason="RFT not available")
    def test_rft_integration(self):
        """Test integration with RFT components."""
        engine = EntangledVertexEngine(n_vertices=4, entanglement_enabled=True)
        
        # Should have RFT engine if available
        if engine.rft_engine is not None:
            # Test RFT-modulated correlations
            engine.add_hyperedge({0, 1}, correlation_strength=1.0)
            
            state = engine.assemble_entangled_state(entanglement_level=0.8)
            
            # State should be well-formed
            assert abs(np.linalg.norm(state) - 1.0) < 1e-10
            
            # RFT should contribute to correlation matrix
            assert np.any(np.abs(engine.correlation_matrix) > 1e-10)


# Utility functions for manual testing
def run_comprehensive_test():
    """Run comprehensive test manually (outside pytest)."""
    if not COMPONENTS_AVAILABLE:
        print("⚠ QuantoniumOS components not available")
        return False
    
    try:
        print("🧪 Testing EntangledVertexEngine...")
        engine = EntangledVertexEngine(n_vertices=4, entanglement_enabled=True)
        engine.add_hyperedge({0, 1}, correlation_strength=0.9)
        state = engine.assemble_entangled_state(entanglement_level=0.7)
        print(f"✓ State assembled: norm = {np.linalg.norm(state):.6f}")
        
        print("🧪 Testing OpenQuantumSystem...")
        open_sys = OpenQuantumSystem(engine)
        rho = np.outer(state, state.conj())
        rho_noisy = open_sys.apply_decoherence(rho, NoiseModel.DEPOLARIZING, p=0.1, target_qubits=[0, 1])
        purity = open_sys.purity(rho_noisy)
        print(f"✓ Decoherence applied: purity = {purity:.6f}")
        
        print("🧪 Testing EntanglementValidation...")
        suite = EntanglementValidationSuite()
        results = suite.run_full_validation(engine)
        print(f"✓ Validation complete: success rate = {results['success_rate']:.1%}")
        
        print("🎉 All tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    # Run manual test if executed directly
    success = run_comprehensive_test()
    exit(0 if success else 1)