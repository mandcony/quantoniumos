#!/usr/bin/env python3
"""
Open Quantum Systems for QuantoniumOS
=====================================

Implementation of Kraus operators and open system dynamics for
vertex-based quantum computing with decoherence modeling.

This module provides:
- Kraus operator channel implementations
- Density matrix evolution with superoperators
- Common noise models (depolarizing, amplitude damping, phase damping)
- Mixed state support with von Neumann entropy calculations
- Integration with RFT-based phase evolution
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Callable
from abc import ABC, abstractmethod
import warnings
from enum import Enum

# Import QuTiP for benchmarking if available
try:
    import qutip as qt
    QUTIP_AVAILABLE = True
except ImportError:
    QUTIP_AVAILABLE = False

# Import RFT and vertex assembly
try:
    from .vertex_assembly import EntangledVertexEngine
    from ..core.canonical_true_rft import CanonicalTrueRFT
    RFT_AVAILABLE = True
except ImportError:
    RFT_AVAILABLE = False


class NoiseModel(Enum):
    """Enumeration of supported noise models."""
    DEPOLARIZING = "depolarizing"
    AMPLITUDE_DAMPING = "amplitude_damping"
    PHASE_DAMPING = "phase_damping"
    BIT_FLIP = "bit_flip"
    PHASE_FLIP = "phase_flip"
    COMPOSITE = "composite"


class KrausChannel(ABC):
    """Abstract base class for quantum channels via Kraus operators."""
    
    def __init__(self, name: str):
        """Initialize Kraus channel."""
        self.name = name
        self._kraus_operators: List[np.ndarray] = []
        self._validated = False
    
    @abstractmethod
    def _generate_kraus_operators(self, **kwargs) -> List[np.ndarray]:
        """Generate Kraus operators for this channel."""
        pass
    
    def get_kraus_operators(self, **kwargs) -> List[np.ndarray]:
        """Get Kraus operators, generating them if needed."""
        if not self._validated:
            self._kraus_operators = self._generate_kraus_operators(**kwargs)
            self._validate_kraus_operators()
            self._validated = True
        return self._kraus_operators.copy()
    
    def _validate_kraus_operators(self, tolerance: float = 1e-10) -> None:
        """Validate that Kraus operators satisfy completeness relation."""
        if not self._kraus_operators:
            raise ValueError("No Kraus operators to validate")
        
        # Check completeness: ∑_i K†_i K_i = I
        dim = self._kraus_operators[0].shape[0]
        completeness = np.zeros((dim, dim), dtype=complex)
        
        for K in self._kraus_operators:
            completeness += K.conj().T @ K
        
        identity = np.eye(dim, dtype=complex)
        error = np.linalg.norm(completeness - identity)
        
        if error > tolerance:
            raise ValueError(f"Kraus operators not complete: error = {error:.2e}")
    
    def apply_to_density_matrix(self, rho: np.ndarray) -> np.ndarray:
        """
        Apply channel to density matrix: ρ → ∑_i K_i ρ K†_i
        
        Args:
            rho: Input density matrix
            
        Returns:
            Output density matrix after channel application
        """
        if not self._validated:
            self.get_kraus_operators()
        
        output_rho = np.zeros_like(rho)
        for K in self._kraus_operators:
            output_rho += K @ rho @ K.conj().T
        
        return output_rho
    
    def apply_to_state_vector(self, psi: np.ndarray) -> np.ndarray:
        """
        Apply channel to pure state (convert to mixed state).
        
        Args:
            psi: Input state vector
            
        Returns:
            Output density matrix (generally mixed)
        """
        # Convert pure state to density matrix
        rho_input = np.outer(psi, psi.conj())
        return self.apply_to_density_matrix(rho_input)


class DepolarizingChannel(KrausChannel):
    """Depolarizing channel: ρ → (1-p)ρ + p·I/d"""
    
    def __init__(self, p: float):
        """
        Initialize depolarizing channel.
        
        Args:
            p: Depolarization probability (0 ≤ p ≤ 1)
        """
        super().__init__(f"Depolarizing(p={p:.3f})")
        self.p = p
        if not 0 <= p <= 1:
            raise ValueError("Depolarization probability must be in [0, 1]")
    
    def _generate_kraus_operators(self, dim: int = 2) -> List[np.ndarray]:
        """Generate Kraus operators for depolarizing channel."""
        if dim == 2:
            # Single qubit depolarizing channel
            I = np.eye(2, dtype=complex)
            X = np.array([[0, 1], [1, 0]], dtype=complex)
            Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
            Z = np.array([[1, 0], [0, -1]], dtype=complex)
            
            K0 = np.sqrt(1 - self.p) * I
            K1 = np.sqrt(self.p / 3) * X
            K2 = np.sqrt(self.p / 3) * Y
            K3 = np.sqrt(self.p / 3) * Z
            
            return [K0, K1, K2, K3]
        else:
            # Multi-qubit: use tensor product construction
            # This should only be called for single-qubit channels
            # Multi-qubit application is handled by _apply_single_qubit_channel
            warnings.warn(f"Multi-qubit depolarizing channel (dim={dim}) using simplified model")
            K0 = np.sqrt(1 - self.p) * np.eye(dim, dtype=complex)
            # Add weak uniform mixing for multi-qubit case
            uniform_mix = np.ones((dim, dim), dtype=complex) / dim
            K1 = np.sqrt(self.p) * uniform_mix
            return [K0, K1]


class AmplitudeDampingChannel(KrausChannel):
    """Amplitude damping channel modeling energy relaxation."""
    
    def __init__(self, gamma: float):
        """
        Initialize amplitude damping channel.
        
        Args:
            gamma: Damping parameter (0 ≤ γ ≤ 1)
        """
        super().__init__(f"AmplitudeDamping(γ={gamma:.3f})")
        self.gamma = gamma
        if not 0 <= gamma <= 1:
            raise ValueError("Damping parameter must be in [0, 1]")
    
    def _generate_kraus_operators(self, **kwargs) -> List[np.ndarray]:
        """Generate Kraus operators for amplitude damping."""
        # K0 = |0⟩⟨0| + √(1-γ)|1⟩⟨1|
        # K1 = √γ |0⟩⟨1|
        
        K0 = np.array([
            [1, 0],
            [0, np.sqrt(1 - self.gamma)]
        ], dtype=complex)
        
        K1 = np.array([
            [0, np.sqrt(self.gamma)],
            [0, 0]
        ], dtype=complex)
        
        return [K0, K1]


class PhaseDampingChannel(KrausChannel):
    """Phase damping channel modeling dephasing."""
    
    def __init__(self, gamma: float):
        """
        Initialize phase damping channel.
        
        Args:
            gamma: Dephasing parameter (0 ≤ γ ≤ 1)
        """
        super().__init__(f"PhaseDamping(γ={gamma:.3f})")
        self.gamma = gamma
        if not 0 <= gamma <= 1:
            raise ValueError("Dephasing parameter must be in [0, 1]")
    
    def _generate_kraus_operators(self, **kwargs) -> List[np.ndarray]:
        """Generate Kraus operators for phase damping."""
        # K0 = |0⟩⟨0| + √(1-γ)|1⟩⟨1|
        # K1 = √γ |1⟩⟨1|
        
        K0 = np.array([
            [1, 0],
            [0, np.sqrt(1 - self.gamma)]
        ], dtype=complex)
        
        K1 = np.array([
            [0, 0],
            [0, np.sqrt(self.gamma)]
        ], dtype=complex)
        
        return [K0, K1]


class BitFlipChannel(KrausChannel):
    """Bit flip channel: X error with probability p."""
    
    def __init__(self, p: float):
        """Initialize bit flip channel."""
        super().__init__(f"BitFlip(p={p:.3f})")
        self.p = p
        if not 0 <= p <= 1:
            raise ValueError("Error probability must be in [0, 1]")
    
    def _generate_kraus_operators(self, **kwargs) -> List[np.ndarray]:
        """Generate Kraus operators for bit flip channel."""
        I = np.eye(2, dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        
        K0 = np.sqrt(1 - self.p) * I
        K1 = np.sqrt(self.p) * X
        
        return [K0, K1]


class PhaseFlipChannel(KrausChannel):
    """Phase flip channel: Z error with probability p."""
    
    def __init__(self, p: float):
        """Initialize phase flip channel."""
        super().__init__(f"PhaseFlip(p={p:.3f})")
        self.p = p
        if not 0 <= p <= 1:
            raise ValueError("Error probability must be in [0, 1]")
    
    def _generate_kraus_operators(self, **kwargs) -> List[np.ndarray]:
        """Generate Kraus operators for phase flip channel."""
        I = np.eye(2, dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        
        K0 = np.sqrt(1 - self.p) * I
        K1 = np.sqrt(self.p) * Z
        
        return [K0, K1]


class CompositeChannel(KrausChannel):
    """Composite channel combining multiple noise sources."""
    
    def __init__(self, channels: List[KrausChannel]):
        """Initialize composite channel."""
        super().__init__("Composite")
        self.channels = channels
    
    def _generate_kraus_operators(self, **kwargs) -> List[np.ndarray]:
        """Generate composite Kraus operators."""
        if not self.channels:
            # Identity channel
            return [np.eye(2, dtype=complex)]
        
        # Start with first channel
        composite_ops = self.channels[0].get_kraus_operators(**kwargs)
        
        # Compose with remaining channels
        for channel in self.channels[1:]:
            channel_ops = channel.get_kraus_operators(**kwargs)
            new_composite = []
            
            for K_comp in composite_ops:
                for K_chan in channel_ops:
                    new_composite.append(K_chan @ K_comp)
            
            composite_ops = new_composite
        
        return composite_ops


class RFTNoiseChannel(KrausChannel):
    """RFT-enhanced noise channel with resonant phase modulation."""
    
    def __init__(self, base_channel: KrausChannel, rft_modulation: bool = True):
        """
        Initialize RFT-modulated noise channel.
        
        Args:
            base_channel: Base noise channel to modulate
            rft_modulation: Enable RFT phase modulation
        """
        super().__init__(f"RFT-{base_channel.name}")
        self.base_channel = base_channel
        self.rft_modulation = rft_modulation
        self.phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        
        # Initialize RFT engine if available
        self.rft_engine = None
        if RFT_AVAILABLE and rft_modulation:
            try:
                self.rft_engine = CanonicalTrueRFT(8)  # Small RFT for phase generation
            except Exception:
                warnings.warn("RFT engine not available for noise modulation")
    
    def _generate_kraus_operators(self, **kwargs) -> List[np.ndarray]:
        """Generate RFT-modulated Kraus operators."""
        base_ops = self.base_channel.get_kraus_operators(**kwargs)
        
        if not self.rft_modulation or not self.rft_engine:
            return base_ops
        
        # Apply RFT phase modulation
        modulated_ops = []
        for i, K in enumerate(base_ops):
            # Generate RFT-based phase
            rft_input = np.zeros(self.rft_engine.size, dtype=complex)
            rft_input[i % len(rft_input)] = 1.0
            
            rft_output = self.rft_engine.forward_transform(rft_input)
            phase_factor = rft_output[0] / abs(rft_output[0]) if abs(rft_output[0]) > 1e-12 else 1.0
            
            # Apply golden ratio phase modulation
            golden_phase = np.exp(1j * 2 * np.pi * i / self.phi)
            total_phase = phase_factor * golden_phase
            
            modulated_ops.append(total_phase * K)
        
        return modulated_ops


class OpenQuantumSystem:
    """
    Open quantum system evolution with Kraus operators.
    Integrates with EntangledVertexEngine for realistic NISQ simulation.
    """
    
    def __init__(self, vertex_engine: Optional['EntangledVertexEngine'] = None):
        """
        Initialize open quantum system.
        
        Args:
            vertex_engine: Optional vertex engine for state management
        """
        self.vertex_engine = vertex_engine
        self.channels: List[KrausChannel] = []
        self.evolution_history: List[Dict] = []
    
    def add_noise_channel(self, 
                         channel: KrausChannel, 
                         target_vertices: Optional[List[int]] = None) -> None:
        """
        Add noise channel to the system.
        
        Args:
            channel: Kraus channel to add
            target_vertices: Vertices affected by this channel (None for all)
        """
        channel_info = {
            'channel': channel,
            'target_vertices': target_vertices,
            'timestamp': len(self.evolution_history)
        }
        self.channels.append(channel)
        self.evolution_history.append({
            'type': 'add_channel',
            'channel': channel.name,
            'targets': target_vertices
        })
    
    def apply_decoherence(self, 
                         rho: np.ndarray, 
                         noise_model: Union[NoiseModel, str] = NoiseModel.DEPOLARIZING,
                         p: float = 0.01,
                         target_qubits: Optional[List[int]] = None,
                         **kwargs) -> np.ndarray:
        """
        Apply decoherence to density matrix.
        
        Args:
            rho: Input density matrix
            noise_model: Type of noise to apply
            p: Noise parameter (probability/strength)
            target_qubits: List of qubits to apply noise to (None = all qubits)
            **kwargs: Additional noise parameters
            
        Returns:
            Density matrix after decoherence
        """
        if isinstance(noise_model, str):
            noise_model = NoiseModel(noise_model)
        
        # Determine system size from density matrix
        n_qubits = int(np.log2(rho.shape[0]))
        if 2**n_qubits != rho.shape[0]:
            raise ValueError(f"Density matrix dimension {rho.shape[0]} is not a power of 2")
        
        # Default: apply noise to all qubits
        if target_qubits is None:
            target_qubits = list(range(n_qubits))
        
        # Validate target qubits
        for qubit in target_qubits:
            if qubit >= n_qubits:
                raise ValueError(f"Target qubit {qubit} exceeds system size {n_qubits}")
        
        # Create appropriate single-qubit channel
        if noise_model == NoiseModel.DEPOLARIZING:
            single_qubit_channel = DepolarizingChannel(p)
        elif noise_model == NoiseModel.AMPLITUDE_DAMPING:
            gamma = kwargs.get('gamma', p)
            single_qubit_channel = AmplitudeDampingChannel(gamma)
        elif noise_model == NoiseModel.PHASE_DAMPING:
            gamma = kwargs.get('gamma', p)
            single_qubit_channel = PhaseDampingChannel(gamma)
        elif noise_model == NoiseModel.BIT_FLIP:
            single_qubit_channel = BitFlipChannel(p)
        elif noise_model == NoiseModel.PHASE_FLIP:
            single_qubit_channel = PhaseFlipChannel(p)
        else:
            raise ValueError(f"Unsupported noise model: {noise_model}")
        
        # Apply RFT modulation if enabled
        use_rft = kwargs.get('use_rft_modulation', RFT_AVAILABLE)
        if use_rft:
            single_qubit_channel = RFTNoiseChannel(single_qubit_channel, rft_modulation=True)
        
        # Apply channel to each target qubit
        output_rho = rho.copy()
        
        for qubit in target_qubits:
            output_rho = self._apply_single_qubit_channel(
                output_rho, single_qubit_channel, qubit, n_qubits
            )
        
        # Record evolution
        self.evolution_history.append({
            'type': 'apply_decoherence',
            'noise_model': noise_model.value,
            'parameters': {'p': p, 'target_qubits': target_qubits, **kwargs},
            'entropy_before': self.von_neumann_entropy(rho),
            'entropy_after': self.von_neumann_entropy(output_rho)
        })
        
        return output_rho
    
    def _apply_single_qubit_channel(self, 
                                   rho: np.ndarray, 
                                   channel: KrausChannel, 
                                   target_qubit: int, 
                                   n_qubits: int) -> np.ndarray:
        """
        Apply single-qubit channel to specific qubit in multi-qubit system.
        
        Args:
            rho: Multi-qubit density matrix
            channel: Single-qubit Kraus channel
            target_qubit: Index of target qubit
            n_qubits: Total number of qubits
            
        Returns:
            Density matrix after channel application
        """
        # Get Kraus operators for single qubit
        kraus_ops_1q = channel.get_kraus_operators(dim=2)
        
        # Extend Kraus operators to full system using tensor products
        dim = 2**n_qubits
        kraus_ops_full = []
        
        for K in kraus_ops_1q:
            # Create identity operators for all other qubits
            K_full = np.eye(1, dtype=complex)
            
            for qubit in range(n_qubits):
                if qubit == target_qubit:
                    K_full = np.kron(K_full, K)
                else:
                    K_full = np.kron(K_full, np.eye(2, dtype=complex))
            
            kraus_ops_full.append(K_full)
        
        # Apply Kraus operators to density matrix
        output_rho = np.zeros_like(rho)
        for K_full in kraus_ops_full:
            output_rho += K_full @ rho @ K_full.conj().T
        
        return output_rho
    
    def evolve_vertex_state(self, 
                           entanglement_level: float = 0.5,
                           noise_model: NoiseModel = NoiseModel.DEPOLARIZING,
                           p: float = 0.01) -> np.ndarray:
        """
        Evolve vertex state with decoherence.
        
        Args:
            entanglement_level: Initial entanglement level
            noise_model: Noise model to apply
            p: Noise strength
            
        Returns:
            Final density matrix (mixed state)
        """
        if self.vertex_engine is None:
            raise ValueError("Vertex engine required for state evolution")
        
        # Get initial pure state
        psi = self.vertex_engine.assemble_entangled_state(entanglement_level)
        rho_initial = np.outer(psi, psi.conj())
        
        # Apply decoherence
        rho_final = self.apply_decoherence(rho_initial, noise_model, p)
        
        return rho_final
    
    @staticmethod
    def von_neumann_entropy(rho: np.ndarray, base: float = 2.0) -> float:
        """
        Calculate von Neumann entropy S = -Tr(ρ log ρ).
        
        Args:
            rho: Density matrix
            base: Logarithm base (2 for bits, e for nats)
            
        Returns:
            Von Neumann entropy
        """
        eigenvals = np.linalg.eigvals(rho)
        eigenvals = eigenvals[eigenvals > 1e-12]  # Remove numerical zeros
        
        if len(eigenvals) == 0:
            return 0.0
        
        if base == 2:
            return -np.sum(eigenvals * np.log2(eigenvals)).real
        else:
            return -np.sum(eigenvals * np.log(eigenvals)).real
    
    @staticmethod
    def purity(rho: np.ndarray) -> float:
        """
        Calculate purity P = Tr(ρ²).
        
        Args:
            rho: Density matrix
            
        Returns:
            Purity (1 for pure states, < 1 for mixed states)
        """
        return np.trace(rho @ rho).real
    
    def benchmark_with_qutip(self, 
                            test_states: List[str] = ["bell", "ghz"],
                            noise_strengths: List[float] = [0.01, 0.05, 0.1]) -> Dict:
        """
        Benchmark decoherence against QuTiP reference.
        
        Args:
            test_states: List of test state names
            noise_strengths: List of noise parameters to test
            
        Returns:
            Benchmark results dictionary
        """
        if not QUTIP_AVAILABLE:
            warnings.warn("QuTiP not available for benchmarking")
            return {}
        
        if self.vertex_engine is None:
            warnings.warn("Vertex engine required for benchmarking")
            return {}
        
        results = {}
        
        for state_name in test_states:
            results[state_name] = {}
            
            # Create reference state with QuTiP
            if state_name == "bell" and self.vertex_engine.n_vertices >= 2:
                qt_state = qt.bell_state('00')
                if self.vertex_engine.n_vertices > 2:
                    for _ in range(self.vertex_engine.n_vertices - 2):
                        qt_state = qt.tensor(qt_state, qt.basis(2, 0))
            else:
                continue  # Skip unsupported states
            
            qt_rho = qt_state * qt_state.dag()
            
            for p in noise_strengths:
                # Our implementation
                if state_name == "bell":
                    self.vertex_engine.add_hyperedge({0, 1}, correlation_strength=1.0)
                psi_engine = self.vertex_engine.assemble_entangled_state(0.9)
                rho_engine = np.outer(psi_engine, psi_engine.conj())
                rho_engine_noisy = self.apply_decoherence(rho_engine, NoiseModel.DEPOLARIZING, p)
                
                # QuTiP reference
                if QUTIP_AVAILABLE:
                    try:
                        depol_map = qt.depolarizing_channel(p, N=1)  # Single qubit
                        qt_rho_noisy = depol_map(qt_rho)
                        
                        # Convert to numpy for comparison
                        rho_qutip_noisy = qt_rho_noisy.full()
                        
                        # Calculate fidelity
                        if rho_qutip_noisy.shape == rho_engine_noisy.shape:
                            fidelity = np.abs(np.trace(rho_qutip_noisy.conj().T @ rho_engine_noisy))
                        else:
                            fidelity = 0.0
                        
                        results[state_name][p] = {
                            'entropy_engine': self.von_neumann_entropy(rho_engine_noisy),
                            'entropy_qutip': self.von_neumann_entropy(rho_qutip_noisy),
                            'purity_engine': self.purity(rho_engine_noisy),
                            'purity_qutip': self.purity(rho_qutip_noisy),
                            'fidelity': fidelity
                        }
                        
                    except Exception as e:
                        warnings.warn(f"QuTiP benchmark failed for {state_name}, p={p}: {e}")
        
        return results
    
    def get_evolution_summary(self) -> Dict:
        """Get summary of system evolution."""
        return {
            'total_steps': len(self.evolution_history),
            'channels_added': len(self.channels),
            'channel_types': [ch.name for ch in self.channels],
            'history': self.evolution_history.copy()
        }


# Convenience functions for common noise models
def create_depolarizing_channel(p: float) -> DepolarizingChannel:
    """Create depolarizing channel with given probability."""
    return DepolarizingChannel(p)


def create_amplitude_damping_channel(gamma: float) -> AmplitudeDampingChannel:
    """Create amplitude damping channel with given damping rate."""
    return AmplitudeDampingChannel(gamma)


def create_phase_damping_channel(gamma: float) -> PhaseDampingChannel:
    """Create phase damping channel with given dephasing rate."""
    return PhaseDampingChannel(gamma)


def create_pauli_channel(px: float, py: float, pz: float) -> CompositeChannel:
    """Create Pauli channel with X, Y, Z error probabilities."""
    channels = []
    if px > 0:
        channels.append(BitFlipChannel(px))
    if py > 0:
        # Y = iXZ, approximate with composite XZ
        channels.extend([BitFlipChannel(py/2), PhaseFlipChannel(py/2)])
    if pz > 0:
        channels.append(PhaseFlipChannel(pz))
    
    return CompositeChannel(channels)


# Export main classes and functions
__all__ = [
    'KrausChannel', 'NoiseModel', 'OpenQuantumSystem',
    'DepolarizingChannel', 'AmplitudeDampingChannel', 'PhaseDampingChannel',
    'BitFlipChannel', 'PhaseFlipChannel', 'CompositeChannel', 'RFTNoiseChannel',
    'create_depolarizing_channel', 'create_amplitude_damping_channel',
    'create_phase_damping_channel', 'create_pauli_channel',
    'QUTIP_AVAILABLE', 'RFT_AVAILABLE'
]