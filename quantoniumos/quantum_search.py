# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
# Patent Application: USPTO #19/169,399
# Listed in CLAIMS_PRACTICING_FILES.txt
"""
Quantonium OS - Quantum Search Module

Symbolic Grover-style resonance search engine using amplitude and phase variance matching.
Integrates with Φ-RFT transform outputs for golden-ratio resonance-based search.

This module provides:
  • SymbolicContainerMetadata: Container for resonance signatures
  • QuantumSearch: Grover-style oracle and diffusion operations
  • Integration with RFT amplitude/phase outputs from hardware

Hardware Integration:
  • fpga_top.sv: Provides real-time RFT amplitude computation
  • makerchip_rft_closed_form.tlv: TL-Verilog simulation reference
  • rft_middleware_engine.sv: High-throughput transform engine
"""

from typing import List, Optional, Tuple

import numpy as np


# -----------------------------------------------------------------------------
# Golden Ratio Constants (matching hardware Q8.8 values)
# -----------------------------------------------------------------------------
PHI = 1.618033988749895  # Golden ratio φ
PHI_INV = 0.6180339887498949  # 1/φ
SQRT5 = 2.23606797749979  # √5
NORM_8 = 0.3535533905932738  # 1/√8 (8-point normalization)


# -----------------------------------------------------------------------------
# Symbolic Container Metadata
# -----------------------------------------------------------------------------
class SymbolicContainerMetadata:
    """
    Container describing a symbolic object in the QuantoniumOS graph.
    
    This maps to the 8-point RFT output structure from hardware:
    - amplitude[0:7] from fpga_top.sv magnitude calculation
    - phase derived from rft_real/rft_imag arctangent
    
    Parameters
    ----------
    label : str
        Human-readable identifier for the container.
    resonance : List[float]
        Resonance signature for this container (e.g., RFT amplitude samples).
    phi_modulated : bool
        If True, resonance values are φ-modulated (golden ratio weighted).
    """
    
    def __init__(
        self, 
        label: str, 
        resonance: List[float],
        phi_modulated: bool = False
    ):
        self.label = label
        self.resonance = np.array(resonance, dtype=np.float64)
        self.phi_modulated = phi_modulated
        
        # Apply golden ratio modulation if specified
        if phi_modulated and len(self.resonance) == 8:
            phi_weights = np.array([
                PHI ** (-k * PHI_INV) for k in range(8)
            ])
            self.resonance = self.resonance * phi_weights

    def amplitude_mean(self) -> float:
        """Returns mean absolute amplitude (matches hardware Manhattan distance)."""
        return float(np.mean(np.abs(self.resonance)))
    
    def amplitude_rms(self) -> float:
        """Returns RMS amplitude for energy-based matching."""
        return float(np.sqrt(np.mean(self.resonance ** 2)))

    def phase_profile(self) -> np.ndarray:
        """
        Returns the wrapped phase angle of the resonance.
        Maps to hardware: atan2(rft_imag[k], rft_real[k])
        """
        return np.angle(self.resonance + 1e-12)
    
    def golden_coherence(self) -> float:
        """
        Measures how well the resonance aligns with golden ratio harmonics.
        Returns a value between 0 (no coherence) and 1 (perfect φ-alignment).
        """
        if len(self.resonance) < 2:
            return 0.0
        
        # Check ratios between consecutive components
        ratios = np.abs(self.resonance[:-1]) / (np.abs(self.resonance[1:]) + 1e-12)
        phi_deviation = np.abs(ratios - PHI) / PHI
        coherence = 1.0 - np.mean(np.clip(phi_deviation, 0, 1))
        return float(coherence)
    
    @classmethod
    def from_rft_output(
        cls, 
        label: str,
        rft_real: List[float],
        rft_imag: List[float]
    ) -> "SymbolicContainerMetadata":
        """
        Create container from RFT real/imaginary outputs.
        Matches hardware rft_real[0:7], rft_imag[0:7] arrays.
        """
        real = np.array(rft_real, dtype=np.float64)
        imag = np.array(rft_imag, dtype=np.float64)
        complex_resonance = real + 1j * imag
        return cls(label, complex_resonance.tolist())


# -----------------------------------------------------------------------------
# Grover-style Symbolic Resonance Matcher
# -----------------------------------------------------------------------------
class QuantumSearch:
    """
    QuantumSearch operates on a list of symbolic containers and allows
    Grover-style resonance search via amplitude/phase matching.
    
    This implements a classical simulation of quantum search that:
    1. Uses amplitude matching (like hardware threshold comparisons)
    2. Uses phase alignment (matching kernel_real/kernel_imag angles)
    3. Applies diffusion (amplitude inversion about mean)
    
    Integration with hardware:
    - Containers can be populated from fpga_top.sv LED outputs
    - Oracle thresholds match hardware amplitude[k][29:22] > threshold
    - Phase alignment uses hardware kernel coefficient angles
    
    Parameters
    ----------
    containers : List[SymbolicContainerMetadata]
        Collection of containers to be searched.
    """
    
    def __init__(self, containers: List[SymbolicContainerMetadata]):
        self.containers = containers
        self._iteration_count = 0
        self._search_history: List[Tuple[str, float]] = []

    @staticmethod
    def _amplitude_distance(target_amp: float, candidate_amp: float) -> float:
        """
        Normalized amplitude distance.
        Maps to hardware: comparing amplitude[k][29:22] values.
        """
        return abs(target_amp - candidate_amp) / (abs(target_amp) + 1e-9)

    @staticmethod
    def _phase_alignment_score(target_phase: float, candidate_phase: float) -> float:
        """
        Returns a score between 0 and 1 indicating how well the phases align.
        Maps to hardware: phase = atan2(kernel_imag, kernel_real)
        """
        delta = np.angle(np.exp(1j * (candidate_phase - target_phase)))
        return float(1.0 - abs(delta) / np.pi)
    
    @staticmethod
    def _golden_ratio_bonus(coherence: float) -> float:
        """
        Provides bonus score for φ-coherent containers.
        Aligns with hardware golden ratio modulation.
        """
        return coherence * PHI_INV  # Up to ~0.618 bonus

    def grover_oracle(
        self, 
        target_amplitude: float, 
        target_phase: float,
        threshold: float = 0.75,
        use_phi_bonus: bool = True
    ) -> List[Tuple[SymbolicContainerMetadata, float]]:
        """
        Emulates an oracle that "marks" containers whose resonance matches the
        target amplitude and phase.
        
        This corresponds to hardware LED output logic:
        - led_output[k] <= (amplitude[k][29:22] > threshold)
        
        Parameters
        ----------
        target_amplitude : float
            Target amplitude to search for.
        target_phase : float
            Target phase in radians.
        threshold : float
            Minimum score to be considered "marked" (default 0.75).
        use_phi_bonus : bool
            If True, apply golden ratio coherence bonus.

        Returns
        -------
        List[Tuple[SymbolicContainerMetadata, float]]
            List of (container, score) pairs where score is a resonance match score.
        """
        marked = []
        for container in self.containers:
            amp = container.amplitude_mean()
            phase_profile = container.phase_profile()
            if phase_profile.size == 0:
                continue

            # Use the average phase as a crude symbolic signature
            avg_phase = float(np.mean(phase_profile))

            amp_dist = self._amplitude_distance(target_amplitude, amp)
            phase_score = self._phase_alignment_score(target_phase, avg_phase)

            # Composite Grover-style marking: low amplitude distance and high phase score
            resonance_score = (1.0 - amp_dist) * phase_score
            
            # Apply golden ratio coherence bonus
            if use_phi_bonus:
                phi_bonus = self._golden_ratio_bonus(container.golden_coherence())
                resonance_score = resonance_score * (1.0 + phi_bonus)
                resonance_score = min(resonance_score, 1.0)  # Cap at 1.0

            if resonance_score > threshold:
                marked.append((container, resonance_score))

        return marked

    def grover_iteration(
        self, 
        target_amplitude: float, 
        target_phase: float
    ) -> Tuple[List[SymbolicContainerMetadata], List[float]]:
        """
        Performs one symbolic Grover-style iteration over the containers.
        
        This implements:
        1. Oracle phase (mark matching containers)
        2. Diffusion operator (reflect about mean amplitude)
        
        The diffusion step corresponds to hardware:
        - Mean calculation across all amplitude[k] values
        - Amplitude inversion: 2*mean - amplitude[k]

        Returns
        -------
        Tuple[List[SymbolicContainerMetadata], List[float]]
            The containers list and their updated amplitudes in Hilbert space.
        """
        self._iteration_count += 1
        oracle_results = self.grover_oracle(target_amplitude, target_phase)

        # Construct symbolic amplitudes: start uniform, then boost marked items
        n = len(self.containers)
        if n == 0:
            return [], []

        amplitudes = np.ones(n, dtype=np.float64) / np.sqrt(n)

        # Marking phase: boost amplitudes of matching containers
        label_to_index = {c.label: i for i, c in enumerate(self.containers)}
        for container, score in oracle_results:
            idx = label_to_index[container.label]
            amplitudes[idx] *= (1.0 + score)
            self._search_history.append((container.label, score))

        # Diffusion step: reflect amplitudes around the mean
        # This is the key Grover operation: 2|ψ_mean⟩⟨ψ_mean| - I
        mean_amp = float(np.mean(amplitudes))
        amplitudes = 2 * mean_amp - amplitudes

        return self.containers, amplitudes.tolist()

    def search(
        self, 
        target_amplitude: float, 
        target_phase: float, 
        iterations: int = 3,
        early_exit_threshold: float = 0.95
    ) -> Tuple[Optional[SymbolicContainerMetadata], float]:
        """
        High-level API: runs multiple Grover-style iterations and returns
        the best-matching container.
        
        Optimal iteration count for N items: ⌊(π/4)√N⌋
        For 8 containers: ~2 iterations optimal

        Parameters
        ----------
        target_amplitude : float
            Target amplitude (e.g., typical resonance magnitude) to search for.
        target_phase : float
            Target phase, in radians, to align with.
        iterations : int
            Number of Grover-style iterations to perform.
        early_exit_threshold : float
            If a match exceeds this score, exit early.

        Returns
        -------
        Tuple[SymbolicContainerMetadata, float]
            (Best matching container, match score).
            If no container matches above the threshold, returns (None, 0.0).
        """
        best_container: Optional[SymbolicContainerMetadata] = None
        best_score = 0.0
        self._search_history = []
        self._iteration_count = 0

        for iteration in range(iterations):
            oracle_results = self.grover_oracle(target_amplitude, target_phase)
            
            for container, score in oracle_results:
                if score > best_score:
                    best_score = score
                    best_container = container
                    
                    # Early exit if we found an excellent match
                    if score >= early_exit_threshold:
                        return best_container, best_score

            # Symbolic Grover iteration (amplitude amplification)
            self.grover_iteration(target_amplitude, target_phase)

        return best_container, best_score
    
    def search_by_coherence(
        self, 
        min_coherence: float = 0.5
    ) -> List[Tuple[SymbolicContainerMetadata, float]]:
        """
        Search for containers with high golden ratio coherence.
        
        This is specific to QuantoniumOS and finds containers whose
        resonance patterns align with φ harmonics.
        
        Parameters
        ----------
        min_coherence : float
            Minimum golden ratio coherence (0 to 1).
            
        Returns
        -------
        List of (container, coherence) tuples sorted by coherence.
        """
        results = []
        for container in self.containers:
            coherence = container.golden_coherence()
            if coherence >= min_coherence:
                results.append((container, coherence))
        
        return sorted(results, key=lambda x: x[1], reverse=True)
    
    @property
    def iteration_count(self) -> int:
        """Returns the number of Grover iterations performed."""
        return self._iteration_count
    
    @property 
    def search_history(self) -> List[Tuple[str, float]]:
        """Returns history of (label, score) matches from iterations."""
        return self._search_history.copy()


# -----------------------------------------------------------------------------
# Hardware Integration Utilities
# -----------------------------------------------------------------------------
def containers_from_hardware_output(
    led_pattern: int,
    amplitude_thresholds: List[int] = None
) -> SymbolicContainerMetadata:
    """
    Create a container from hardware LED output pattern.
    
    Maps fpga_top.sv LED output to resonance values:
    - LED on = amplitude above threshold
    - LED off = amplitude below threshold
    
    Parameters
    ----------
    led_pattern : int
        8-bit LED pattern from WF_LED output.
    amplitude_thresholds : List[int]
        Thresholds used in hardware (default from fpga_top.sv).
        
    Returns
    -------
    SymbolicContainerMetadata with reconstructed resonance.
    """
    if amplitude_thresholds is None:
        # Default thresholds from fpga_top.sv
        amplitude_thresholds = [128, 50, 30, 25, 25, 30, 50, 128]
    
    resonance = []
    for k in range(8):
        bit_set = (led_pattern >> k) & 1
        # Estimate amplitude based on threshold crossing
        if bit_set:
            resonance.append(float(amplitude_thresholds[k]) * 1.5)
        else:
            resonance.append(float(amplitude_thresholds[k]) * 0.5)
    
    label = f"HW_Pattern_0x{led_pattern:02X}"
    return SymbolicContainerMetadata(label, resonance, phi_modulated=True)


def optimal_grover_iterations(n_containers: int) -> int:
    """
    Calculate optimal Grover iteration count for N items.
    Formula: ⌊(π/4)√N⌋
    """
    return max(1, int(np.floor(np.pi / 4 * np.sqrt(n_containers))))


# -----------------------------------------------------------------------------
# Example Usage (for local testing)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("🔍 QuantoniumOS Quantum Search Module")
    print("=" * 60)
    print()
    
    print("Initializing symbolic container metadata set...")
    print()

    # Resonance values for testing - using φ-related values
    sample_containers = [
        SymbolicContainerMetadata(
            "Container_PHI", 
            [PHI, PHI**2, PHI**3, PHI**4, PHI**5, PHI**6, PHI**7, PHI**8],
            phi_modulated=True
        ),
        SymbolicContainerMetadata(
            "Container_SQRT5", 
            [SQRT5, SQRT5, SQRT5, SQRT5, SQRT5, SQRT5, SQRT5, SQRT5]
        ),
        SymbolicContainerMetadata(
            "Container_RAMP", 
            [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
        ),
        SymbolicContainerMetadata(
            "Container_GHZ",  # Simulating GHZ-like state
            [NORM_8 * np.sqrt(2), 0, 0, 0, 0, 0, 0, NORM_8 * np.sqrt(2)]
        ),
    ]

    search_engine = QuantumSearch(sample_containers)
    
    # Calculate optimal iterations
    opt_iter = optimal_grover_iterations(len(sample_containers))
    print(f"📊 Container count: {len(sample_containers)}")
    print(f"📊 Optimal iterations: {opt_iter}")
    print()

    # Search for golden ratio amplitude
    target_amp = PHI
    target_phi = 0.0

    print(f"🔭 Searching for amplitude={target_amp:.4f}, phase={target_phi:.4f}...")
    match, score = search_engine.search(target_amp, target_phi, iterations=opt_iter)

    if match:
        print(f"✅ Match Found: {match.label}")
        print(f"   Confidence: {score:.4f}")
        print(f"   Golden Coherence: {match.golden_coherence():.4f}")
        print(f"   Amplitude Mean: {match.amplitude_mean():.4f}")
    else:
        print("❌ No matching container found.")
    
    print()
    print("🔬 Searching by golden ratio coherence...")
    coherent = search_engine.search_by_coherence(min_coherence=0.3)
    for container, coherence in coherent:
        print(f"   {container.label}: φ-coherence = {coherence:.4f}")
    
    print()
    print(f"📈 Search iterations performed: {search_engine.iteration_count}")
    print("=" * 60)
