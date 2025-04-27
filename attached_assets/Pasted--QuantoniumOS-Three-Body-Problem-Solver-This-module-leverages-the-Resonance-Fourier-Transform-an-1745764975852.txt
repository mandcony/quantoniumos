"""
QuantoniumOS Three-Body Problem Solver

This module leverages the Resonance Fourier Transform and symbolic
computational capabilities of QuantoniumOS to approach the classical
three-body problem in celestial mechanics.

The approach uses wave-based mathematics to represent gravitational fields 
as interacting waveforms, potentially revealing resonance patterns that
simplify the otherwise chaotic behavior.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Add the root directory to the path to access QuantoniumOS modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import core QuantoniumOS modules
try:
    from encryption.wave_primitives import generate_waveform, analyze_waveform
    from encryption.resonance_fourier import resonance_fourier_transform, inverse_resonance_fourier_transform
    from orchestration.symbolic_container import create_container, validate_container
    print("Successfully imported QuantoniumOS core modules")
except ImportError as e:
    print(f"Error importing QuantoniumOS modules: {e}")
    print("Falling back to simplified implementations")
    
    # Simplified implementation if core modules are not available
    def generate_waveform(length=32, seed=None):
        """Generate a test waveform"""
        if seed is not None:
            np.random.seed(seed)
        return np.random.random(length)
    
    def analyze_waveform(waveform):
        """Simple waveform analysis"""
        return {
            "mean": np.mean(waveform),
            "std": np.std(waveform),
            "min": np.min(waveform),
            "max": np.max(waveform)
        }
    
    def resonance_fourier_transform(waveform):
        """Simplified RFT implementation"""
        freq_data = np.fft.fft(waveform)
        amplitude = np.abs(freq_data)
        phase = np.angle(freq_data)
        frequencies = np.fft.fftfreq(len(waveform))
        return {
            "frequencies": frequencies,
            "amplitudes": amplitude,
            "phases": phase,
            "resonance_mask": amplitude > np.mean(amplitude)
        }
    
    def inverse_resonance_fourier_transform(freq_data):
        """Simplified inverse RFT"""
        complex_data = freq_data["amplitudes"] * np.exp(1j * freq_data["phases"])
        return np.fft.ifft(complex_data).real


class ThreeBodySystem:
    """
    Represents a three-body gravitational system using QuantoniumOS
    resonance mathematics.
    
    Instead of traditional numerical integration of differential equations,
    this approach represents gravitational fields as interacting waveforms
    and uses resonance patterns to identify stable configurations.
    """
    
    def __init__(self, masses, initial_positions, initial_velocities, G=6.67430e-11):
        """
        Initialize the three-body system.
        
        Parameters:
        ----------
        masses : array-like
            Masses of the three bodies [m1, m2, m3]
        initial_positions : array-like
            Initial positions as [[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]]
        initial_velocities : array-like
            Initial velocities as [[vx1, vy1, vz1], [vx2, vy2, vz2], [vx3, vy3, vz3]]
        G : float
            Gravitational constant
        """
        self.masses = np.array(masses, dtype=np.float64)
        self.positions = np.array(initial_positions, dtype=np.float64)
        self.velocities = np.array(initial_velocities, dtype=np.float64)
        self.G = G
        self.t = 0
        self.history = [self.positions.copy()]
        
        # Generate waveforms representing gravitational fields
        self.field_waveforms = self._generate_field_waveforms()
        
        # Identify resonance patterns
        self.resonance_data = self._analyze_resonance_patterns()
        
        print(f"Three-body system initialized with masses: {masses}")
        print(f"Resonance patterns identified: {len(self.resonance_data['resonance_points'])}")
    
    def _generate_field_waveforms(self, waveform_length=64):
        """
        Generate waveforms representing the gravitational field of each body.
        
        In traditional approaches, gravitational fields are represented as
        force vectors. In our resonance approach, we represent them as
        waveforms whose interaction can be analyzed for resonance patterns.
        
        The improved waveform model incorporates:
        1. Gravitational field strength (varies with 1/r²)
        2. Orbital velocity and period information
        3. Relative phase positioning in orbital cycle
        4. Harmonic components based on orbital resonances
        """
        waveforms = []
        
        # Calculate system properties for context
        total_mass = np.sum(self.masses)
        center_of_mass = np.sum(self.positions * self.masses[:, np.newaxis], axis=0) / total_mass
        
        # Calculate orbital periods and angular velocities (if applicable)
        orbital_periods = []
        angular_velocities = []
        
        for i, mass in enumerate(self.masses):
            # Skip the central body (typically index 0)
            if i == 0:
                orbital_periods.append(None)
                angular_velocities.append(None)
                continue
                
            # Distance from central body
            r = np.linalg.norm(self.positions[i] - self.positions[0])
            
            # Approximate orbital period using Kepler's third law
            # T² = (4π²/G(M+m)) * r³
            period = np.sqrt((4 * np.pi**2) / (self.G * (self.masses[0] + mass)) * r**3)
            orbital_periods.append(period)
            
            # Angular velocity ω = 2π/T
            angular_velocity = 2 * np.pi / period if period else 0
            angular_velocities.append(angular_velocity)
        
        for i, mass in enumerate(self.masses):
            # Use mass and position as seed for reproducibility
            # Handle potential NaN values safely
            pos_sum = np.sum(self.positions[i])
            if np.isfinite(pos_sum) and np.isfinite(mass):
                seed = int((mass * 1000 + pos_sum) % 10000)
            else:
                seed = 42  # Fallback seed
            
            # Base waveform now incorporates orbital information
            if i == 0:  # Central body
                # Central body's field has primarily low-frequency components
                # with harmonics representing its influence on orbiting bodies
                base_frequencies = np.linspace(0.01, 0.1, 5)  # Main frequency components
                amplitudes = np.array([1.0, 0.5, 0.25, 0.125, 0.0625])  # Amplitude decay
                phases = np.zeros(5)  # Initial phases
                
                # Create composite waveform
                base_waveform = np.zeros(waveform_length)
                t = np.linspace(0, 2*np.pi, waveform_length)
                for f, a, p in zip(base_frequencies, amplitudes, phases):
                    base_waveform += a * np.sin(f * t + p)
                
                # Normalize and scale by mass
                base_waveform = base_waveform / np.max(np.abs(base_waveform)) * mass
                
            else:  # Orbiting body
                # Orbiting body's field has components related to its orbital frequency
                period = orbital_periods[i]
                angular_vel = angular_velocities[i]
                
                # Current orbital phase
                orbital_vector = self.positions[i] - self.positions[0]
                current_phase = np.arctan2(orbital_vector[1], orbital_vector[0])
                
                # Generate waveform with main orbital frequency and harmonics
                base_waveform = np.zeros(waveform_length)
                t = np.linspace(0, 2*np.pi, waveform_length)
                
                # Fundamental orbital frequency
                base_waveform += np.sin(t + current_phase)
                
                # First harmonic (twice orbital frequency)
                base_waveform += 0.5 * np.sin(2*t + 2*current_phase)
                
                # Second harmonic
                base_waveform += 0.25 * np.sin(3*t + 3*current_phase)
                
                # Add resonance-specific components based on mass ratio
                mass_ratio = mass / self.masses[0]
                resonance_factor = np.sin(mass_ratio * 10 * t)
                base_waveform += mass_ratio * resonance_factor
                
                # Normalize and scale by mass
                base_waveform = base_waveform / np.max(np.abs(base_waveform)) * mass
            
            # Gravitational field modulation - inverse square law
            # We represent this as amplitude modulation across the waveform
            distances = np.zeros(waveform_length)
            for j in range(waveform_length):
                # Phase position in the orbit
                phase = 2*np.pi * j/waveform_length
                if i > 0:  # For orbiting bodies
                    # Approximate position at this phase
                    pos_at_phase = np.array([
                        np.cos(phase + current_phase),
                        np.sin(phase + current_phase),
                        0
                    ]) * np.linalg.norm(self.positions[i] - self.positions[0])
                    pos_at_phase += self.positions[0]  # Center around central body
                    
                    # Distance from current actual position
                    distances[j] = np.linalg.norm(pos_at_phase - self.positions[i])
                else:  # For central body
                    # Use distance from center of mass
                    distances[j] = np.linalg.norm(center_of_mass) * (1 + 0.1 * np.sin(phase))
            
            # Avoid division by zero
            distances = np.maximum(distances, 1e-10)
            
            # Inverse square law modulation with smoothing
            field_strength = 1.0 / (1.0 + distances**2)
            
            # Apply gravitational field strength modulation
            modulated_waveform = base_waveform * field_strength
            
            # Add a small amount of noise to represent quantum fluctuations
            # and other minor gravitational perturbations
            noise = np.random.normal(0, 0.01, waveform_length)
            modulated_waveform += noise * mass * 1e-5
            
            waveforms.append({
                "body_index": i,
                "mass": mass,
                "waveform": modulated_waveform,
                "analysis": analyze_waveform(modulated_waveform),
                "orbital_period": orbital_periods[i] if i > 0 else None,
                "angular_velocity": angular_velocities[i] if i > 0 else None
            })
        
        return waveforms
    
    def _analyze_resonance_patterns(self):
        """
        Analyze the waveforms for resonance patterns that might indicate
        stable orbital configurations.
        
        This is where the QuantoniumOS resonance mathematics provides a
        novel approach to the three-body problem.
        
        Enhanced version includes:
        1. Multi-scale resonance detection across frequency bands
        2. Weighted stability metrics based on amplitude significance
        3. Orbital period relationship analysis
        4. Known resonance configuration detection (Lagrange points)
        """
        resonance_data = {
            "resonance_points": [],
            "stability_metrics": [],
            "transform_data": [],
            "lagrange_points": [],
            "orbital_resonances": []
        }
        
        # Transform each field waveform
        transform_data = []
        for field in self.field_waveforms:
            rft_data = resonance_fourier_transform(field["waveform"])
            transform_data.append(rft_data)
        
        # Get orbital periods for resonance ratio analysis
        orbital_periods = []
        for field in self.field_waveforms:
            orbital_periods.append(field.get("orbital_period"))
        
        # Multi-scale analysis: examine different frequency bands
        frequency_bands = [
            (0, len(transform_data[0]["frequencies"]) // 16),    # Very low frequencies
            (len(transform_data[0]["frequencies"]) // 16, len(transform_data[0]["frequencies"]) // 8),    # Low frequencies
            (len(transform_data[0]["frequencies"]) // 8, len(transform_data[0]["frequencies"]) // 4),     # Mid frequencies
            (len(transform_data[0]["frequencies"]) // 4, len(transform_data[0]["frequencies"]) // 2),     # High frequencies
            (len(transform_data[0]["frequencies"]) // 2, len(transform_data[0]["frequencies"]))           # Very high frequencies
        ]
        
        # Analyze each frequency band
        for band_name, (band_start, band_end) in zip(
            ["very_low", "low", "mid", "high", "very_high"], 
            frequency_bands
        ):
            # Look for resonance between the three bodies in this band
            for i in range(band_start, band_end):
                # Skip DC component
                if i == 0:
                    continue
                
                # Calculate resonance strength in each body (normalized amplitude)
                body_resonance = []
                for body_idx in range(3):
                    # Get normalized amplitude for this frequency
                    amplitude = transform_data[body_idx]["amplitudes"][i]
                    max_amp = np.max(transform_data[body_idx]["amplitudes"][band_start:band_end])
                    if max_amp > 0:
                        normalized_amp = amplitude / max_amp
                    else:
                        normalized_amp = 0
                    
                    body_resonance.append(normalized_amp)
                
                # Calculate weighted resonance (how strongly all bodies resonate at this frequency)
                # For true resonance, we want all bodies to have significant amplitude
                min_resonance = min(body_resonance)
                mean_resonance = sum(body_resonance) / 3
                
                # Set adaptive threshold based on frequency band (lower threshold for lower frequencies)
                threshold = 0.4
                if band_name == "very_low":
                    threshold = 0.2
                elif band_name == "low":
                    threshold = 0.3
                elif band_name == "very_high":
                    threshold = 0.5
                
                # Check if this frequency has significant resonance in all bodies
                is_resonant = min_resonance > threshold and mean_resonance > 0.5
                    
                if is_resonant:
                    # Calculate phase relationships
                    phase_diff_1_2 = abs(transform_data[0]["phases"][i] - transform_data[1]["phases"][i]) % (2*np.pi)
                    phase_diff_2_3 = abs(transform_data[1]["phases"][i] - transform_data[2]["phases"][i]) % (2*np.pi)
                    phase_diff_3_1 = abs(transform_data[2]["phases"][i] - transform_data[0]["phases"][i]) % (2*np.pi)
                    
                    # If phases form a balanced relationship, we may have a stable configuration
                    phase_sum = phase_diff_1_2 + phase_diff_2_3 + phase_diff_3_1
                    
                    # Normalized deviation from 2π (0 = perfectly stable)
                    stability = abs(phase_sum - 2*np.pi) / (2*np.pi)
                    
                    # Weight stability by the resonance strength
                    weighted_stability = stability * (1 - mean_resonance)
                    
                    # Special case: Detect potential Lagrange point configurations
                    # L4/L5 points have phase differences of approximately 60° or π/3
                    lagrange_point = None
                    
                    # Check for L4/L5 Lagrange point pattern (equilateral triangle configuration)
                    if (abs(phase_diff_1_2 - 2*np.pi/3) < 0.2 and 
                        abs(phase_diff_2_3 - 2*np.pi/3) < 0.2 and
                        abs(phase_diff_3_1 - 2*np.pi/3) < 0.2):
                        lagrange_point = "L4/L5"
                        # L4/L5 are very stable in certain mass ratios
                        weighted_stability *= 0.5  # Increase stability score
                    
                    # Check for L1/L2/L3 Lagrange point pattern (collinear configuration)
                    elif (min(phase_diff_1_2, 2*np.pi - phase_diff_1_2) < 0.2 or
                          min(phase_diff_2_3, 2*np.pi - phase_diff_2_3) < 0.2 or
                          min(phase_diff_3_1, 2*np.pi - phase_diff_3_1) < 0.2):
                        lagrange_point = "L1/L2/L3"
                        # L1/L2/L3 are less stable
                        weighted_stability *= 0.8  # Slightly increase stability score
                    
                    # Check for orbital resonance (ratio of periods)
                    orbital_resonance = None
                    if orbital_periods[1] is not None and orbital_periods[2] is not None:
                        period_ratio = orbital_periods[1] / orbital_periods[2]
                        
                        # Check common resonance ratios (e.g., 2:1, 3:2, etc.)
                        common_ratios = [(2, 1), (3, 2), (5, 3), (3, 1), (4, 3), (5, 2)]
                        for p, q in common_ratios:
                            if abs(period_ratio - p/q) < 0.1 or abs(period_ratio - q/p) < 0.1:
                                orbital_resonance = f"{p}:{q}"
                                # Orbital resonances are often stable
                                weighted_stability *= 0.7  # Increase stability score
                                break
                    
                    resonance_data["resonance_points"].append({
                        "frequency_index": i,
                        "frequency": transform_data[0]["frequencies"][i],
                        "frequency_band": band_name,
                        "phase_diffs": [phase_diff_1_2, phase_diff_2_3, phase_diff_3_1],
                        "body_resonance": body_resonance,
                        "mean_resonance": mean_resonance,
                        "stability": stability,
                        "weighted_stability": weighted_stability,
                        "lagrange_point": lagrange_point,
                        "orbital_resonance": orbital_resonance
                    })
                    
                    resonance_data["stability_metrics"].append(weighted_stability)
                    
                    # Record Lagrange point if detected
                    if lagrange_point:
                        resonance_data["lagrange_points"].append({
                            "type": lagrange_point,
                            "frequency": transform_data[0]["frequencies"][i],
                            "stability": weighted_stability
                        })
                    
                    # Record orbital resonance if detected
                    if orbital_resonance:
                        resonance_data["orbital_resonances"].append({
                            "ratio": orbital_resonance,
                            "frequency": transform_data[0]["frequencies"][i],
                            "stability": weighted_stability
                        })
        
        resonance_data["transform_data"] = transform_data
        
        # Sort resonance points by weighted stability
        resonance_data["resonance_points"].sort(key=lambda x: x["weighted_stability"])
        
        return resonance_data
    
    def _calculate_system_energy(self):
        """
        Calculate the total energy of the system (kinetic + potential).
        Used for energy conservation enforcement.
        """
        # Kinetic energy: 1/2 * m * v^2
        kinetic_energy = 0.0
        for i in range(3):
            v_squared = np.sum(self.velocities[i] ** 2)
            kinetic_energy += 0.5 * self.masses[i] * v_squared
        
        # Potential energy: -G * m1 * m2 / r
        potential_energy = 0.0
        for i in range(3):
            for j in range(i+1, 3):  # Avoid double counting
                r = np.linalg.norm(self.positions[i] - self.positions[j])
                if r > 1e-10:  # Avoid division by zero
                    potential_energy -= self.G * self.masses[i] * self.masses[j] / r
        
        return kinetic_energy + potential_energy
    
    def _calculate_lagrange_correction(self, lagrange_type):
        """
        Calculate correction forces to stabilize the system at Lagrange points.
        """
        correction = np.zeros_like(self.positions)
        
        # For L4/L5 points (equilateral triangle configuration)
        if lagrange_type == "L4/L5":
            # Calculate ideal positions for equilateral triangle
            # We assume body 0 is primary and body 1 is secondary
            r01 = np.linalg.norm(self.positions[1] - self.positions[0])
            unit_01 = (self.positions[1] - self.positions[0]) / r01
            
            # L4/L5 are 60° from the secondary body
            angle = np.pi / 3  # 60 degrees
            rotation_matrix = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]
            ])
            
            # Rotate the unit vector to get L4 point direction
            l4_direction = np.dot(rotation_matrix, unit_01)
            
            # L4 is at same distance as secondary
            ideal_l4 = self.positions[0] + l4_direction * r01
            
            # Calculate correction for third body to move toward L4
            correction_vector = ideal_l4 - self.positions[2]
            correction_magnitude = np.linalg.norm(correction_vector)
            
            if correction_magnitude > 0:
                correction[2] = correction_vector / correction_magnitude * 1e-5 * correction_magnitude
        
        # For L1/L2/L3 points (collinear configuration)
        elif lagrange_type == "L1/L2/L3":
            # Calculate line connecting primary bodies
            line_vector = self.positions[1] - self.positions[0]
            line_length = np.linalg.norm(line_vector)
            
            if line_length > 0:
                unit_line = line_vector / line_length
                
                # Project third body position onto the line
                projected_point = self.positions[0] + np.dot(
                    self.positions[2] - self.positions[0], unit_line
                ) * unit_line
                
                # Calculate correction to move third body toward the line
                correction_vector = projected_point - self.positions[2]
                correction_magnitude = np.linalg.norm(correction_vector)
                
                if correction_magnitude > 0:
                    correction[2] = correction_vector / correction_magnitude * 1e-6 * correction_magnitude
        
        return correction
    
    def _calculate_orbital_resonance_correction(self, resonance):
        """
        Calculate correction forces to maintain orbital resonance.
        """
        correction = np.zeros_like(self.positions)
        
        # Extract the resonance ratio (e.g. "2:1" -> (2, 1))
        if isinstance(resonance, dict) and "ratio" in resonance:
            ratio_str = resonance["ratio"]
            try:
                p, q = map(int, ratio_str.split(':'))
                target_ratio = p / q
            except (ValueError, ZeroDivisionError):
                return correction
        else:
            return correction
        
        # Get the current orbital periods
        if self.field_waveforms[1].get("orbital_period") and self.field_waveforms[2].get("orbital_period"):
            period1 = self.field_waveforms[1]["orbital_period"]
            period2 = self.field_waveforms[2]["orbital_period"]
            
            current_ratio = period1 / period2
            
            # Calculate error in ratio
            ratio_error = current_ratio - target_ratio
            
            # Apply a gentle correction to velocities based on error
            # Speed up or slow down the orbiting bodies to achieve the target ratio
            if abs(ratio_error) > 0.01:
                # Adjust body 1
                v1_magnitude = np.linalg.norm(self.velocities[1])
                if v1_magnitude > 0:
                    adjustment_factor1 = 1.0 + 0.01 * ratio_error
                    self.velocities[1] *= adjustment_factor1
                
                # Adjust body 2
                v2_magnitude = np.linalg.norm(self.velocities[2])
                if v2_magnitude > 0:
                    adjustment_factor2 = 1.0 - 0.01 * ratio_error
                    self.velocities[2] *= adjustment_factor2
        
        return correction
    
    def _calculate_resonance_correction(self, resonance_point):
        """
        Calculate correction to accelerations based on resonance analysis.
        This is where the QuantoniumOS approach differs from traditional methods.
        """
        # Initialize correction array
        correction = np.zeros_like(self.positions)
        
        # Ideal phase differences for a stable system would be 2π/3 between each pair
        ideal_phase_diff = 2 * np.pi / 3
        
        # Calculate correction factors based on how far phases are from ideal
        for i in range(3):
            j = (i + 1) % 3
            k = (i + 2) % 3
            
            # Current phase differences
            current_diff = resonance_point["phase_diffs"][i]
            
            # Calculate correction factor
            phase_error = current_diff - ideal_phase_diff
            
            # Direction should be toward the optimal orbital plane
            direction = np.cross(
                self.positions[j] - self.positions[i],
                self.positions[k] - self.positions[i]
            )
            
            # Normalize direction
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction)
            
            # Scale correction by error and apply
            correction[i] += direction * phase_error * 0.01
        
        return correction
    
    def step(self, dt=0.01):
        """
        Advance the simulation by one timestep using resonance-based approach.
        
        Instead of directly integrating forces, we:
        1. Update the field waveforms based on current positions
        2. Analyze resonance patterns to identify stable configurations
        3. Adjust velocities to move toward resonant configurations
        4. Apply standard velocity->position update
        
        Enhanced version includes:
        1. Adaptive resonance correction based on stability metrics
        2. Lagrange point detection and stabilization
        3. Orbital resonance preservation
        4. Energy conservation enforcement
        """
        # Update field waveforms based on new positions
        self.field_waveforms = self._generate_field_waveforms()
        
        # Re-analyze resonance to find stable configurations
        resonance_data = self._analyze_resonance_patterns()
        
        # Traditional force calculation as baseline
        accelerations = self._calculate_accelerations()
        
        # Apply resonance-based corrections if available
        if resonance_data["resonance_points"]:
            # Get the most stable resonance point
            best_resonance = resonance_data["resonance_points"][0]
            
            # Adaptive stability threshold - more stringent as simulation progresses
            base_threshold = 0.3
            time_factor = min(1.0, self.t / 86400.0 / 10)  # Ramp up over 10 days
            adaptive_threshold = base_threshold * (1.0 - 0.5 * time_factor)
            
            # Scale correction factor based on stability - better stability gets stronger correction
            stability = best_resonance.get("weighted_stability", best_resonance.get("stability", 1.0))
            correction_strength = 0.5  # Default blend factor
            
            if stability < adaptive_threshold:
                # Stronger correction for very stable configurations
                if stability < 0.1:
                    correction_strength = 0.7
                elif stability < 0.2:
                    correction_strength = 0.6
                
                # Apply resonance-based correction to accelerations
                resonance_correction = self._calculate_resonance_correction(best_resonance)
                accelerations = accelerations + resonance_correction * correction_strength
            
            # Special handling for Lagrange points if detected
            if resonance_data["lagrange_points"]:
                lagrange_point = resonance_data["lagrange_points"][0]
                if lagrange_point["type"] == "L4/L5":
                    # L4/L5 points need special stabilization in 3-body systems
                    # They form equilateral triangles with the primary bodies
                    l_correction = self._calculate_lagrange_correction("L4/L5")
                    accelerations = accelerations + l_correction * 0.3
                elif lagrange_point["type"] == "L1/L2/L3":
                    # L1/L2/L3 are inherently unstable and need stronger correction
                    l_correction = self._calculate_lagrange_correction("L1/L2/L3")
                    accelerations = accelerations + l_correction * 0.2
            
            # Preserve orbital resonances if detected
            if resonance_data["orbital_resonances"]:
                orbital_resonance = resonance_data["orbital_resonances"][0]
                # Apply gentle correction to maintain the resonance ratio
                o_correction = self._calculate_orbital_resonance_correction(orbital_resonance)
                accelerations = accelerations + o_correction * 0.1
        
        # Energy conservation check and correction
        initial_energy = self._calculate_system_energy()
        
        # Update velocities using calculated accelerations
        self.velocities += accelerations * dt
        
        # Update positions using velocities
        self.positions += self.velocities * dt
        
        # Energy conservation enforcement
        final_energy = self._calculate_system_energy()
        
        # Handle potential division by zero or NaN
        if initial_energy != 0 and np.isfinite(initial_energy) and np.isfinite(final_energy):
            energy_ratio = final_energy / initial_energy
            
            # If energy has changed significantly, apply a correction
            if np.isfinite(energy_ratio) and abs(energy_ratio - 1.0) > 0.01:
                # Scale velocities to conserve energy (avoid negative values)
                if energy_ratio > 0:
                    scale_factor = np.sqrt(1.0 / energy_ratio)
                    if np.isfinite(scale_factor):
                        self.velocities *= scale_factor
        
        # Update time and record history
        self.t += dt
        self.history.append(self.positions.copy())
    
    def _calculate_accelerations(self):
        """
        Calculate accelerations using Newton's law of gravitation.
        This provides the baseline physical model.
        """
        accelerations = np.zeros_like(self.positions)
        
        # Calculate pairwise accelerations
        for i in range(3):
            for j in range(3):
                if i != j:
                    r_vec = self.positions[j] - self.positions[i]
                    r = np.linalg.norm(r_vec)
                    # Avoid division by zero
                    if r < 1e-10:
                        continue
                    # Newton's law of gravitation
                    acc_mag = self.G * self.masses[j] / (r * r)
                    accelerations[i] += acc_mag * r_vec / r
        
        return accelerations
    
    def simulate(self, duration=10.0, dt=0.01):
        """
        Run the simulation for the specified duration.
        
        Parameters:
        ----------
        duration : float
            Total time to simulate
        dt : float
            Time step for simulation
        """
        steps = int(duration / dt)
        
        for _ in range(steps):
            self.step(dt)
        
        return np.array(self.history)
    
    def get_orbital_stability_metric(self):
        """
        Calculate a metric for the stability of the system based on
        the resonance patterns.
        
        Returns:
        -------
        float
            Stability metric (0-1 where 0 is unstable, 1 is stable)
        """
        # If we have resonance points, use the best one's stability measure
        if self.resonance_data["resonance_points"]:
            # Invert stability since in _analyze_resonance_patterns, 
            # 0 means stable and higher values mean less stable
            return 1.0 - self.resonance_data["resonance_points"][0]["stability"]
        
        # No resonance points found, system is likely chaotic
        return 0.0
    
    def visualize(self, output_file=None):
        """
        Visualize the three-body simulation.
        
        Parameters:
        ----------
        output_file : str, optional
            If provided, save the animation to this file
        """
        history = np.array(self.history)
        
        # Create figure and 3D axis
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Initial plot
        lines = []
        points = []
        
        colors = ['r', 'g', 'b']
        for i in range(3):
            # Trajectory line
            line, = ax.plot([], [], [], colors[i], label=f"Body {i+1} (m={self.masses[i]:.1e})")
            lines.append(line)
            
            # Current position
            point, = ax.plot([], [], [], f'{colors[i]}o', markersize=10)
            points.append(point)
        
        # Add legend
        ax.legend()
        
        # Set axis labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Title with stability metric
        stability = self.get_orbital_stability_metric()
        ax.set_title(f"Three-Body Simulation using QuantoniumOS Resonance Mathematics\nStability Metric: {stability:.3f}")
        
        # Calculate axis limits
        max_range = np.max(np.ptp(history, axis=0)) * 0.6
        mid_x = np.mean(history[:, :, 0])
        mid_y = np.mean(history[:, :, 1])
        mid_z = np.mean(history[:, :, 2])
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # Animation update function
        def update(frame):
            frame = min(frame, len(history) - 1)
            
            for i in range(3):
                # Update trajectory
                lines[i].set_data(history[:frame, i, 0], history[:frame, i, 1])
                lines[i].set_3d_properties(history[:frame, i, 2])
                
                # Update current position
                points[i].set_data([history[frame, i, 0]], [history[frame, i, 1]])
                points[i].set_3d_properties([history[frame, i, 2]])
            
            return lines + points
        
        # Create animation
        ani = animation.FuncAnimation(
            fig, update, frames=range(0, len(history), 10),
            interval=50, blit=True
        )
        
        # Save animation if output file is provided
        if output_file:
            ani.save(output_file, writer='pillow', fps=20)
            print(f"Animation saved to {output_file}")
        
        plt.tight_layout()
        plt.show()


def run_three_body_simulation():
    """
    Run a three-body simulation with realistic solar system values.
    """
    # Define masses in kg (Sun, Jupiter, Saturn)
    masses = [1.989e30, 1.898e27, 5.683e26]
    
    # Initial positions in meters - simplified circular orbit
    initial_positions = [
        [0, 0, 0],               # Sun at center
        [7.78e11, 0, 0],         # Jupiter at 5.2 AU
        [0, 1.429e12, 0]         # Saturn at 9.54 AU
    ]
    
    # Initial velocities in m/s - simplified circular orbit
    initial_velocities = [
        [0, 0, 0],               # Sun stationary
        [0, 13070, 0],           # Jupiter orbital velocity
        [-9690, 0, 0]            # Saturn orbital velocity
    ]
    
    # Create the system
    system = ThreeBodySystem(masses, initial_positions, initial_velocities)
    
    # Run the simulation with a reduced timeframe for testing
    print("Running simulation...")
    system.simulate(duration=86400.0*30, dt=86400.0)  # 30 days, 1 day steps
    
    # Generate summary instead of visualization for command-line output
    print("Generating analysis...")
    # Skip visualization which requires display: system.visualize()
    
    # Report stability
    stability = system.get_orbital_stability_metric()
    print(f"System stability metric: {stability:.3f} (0 = chaotic, 1 = stable)")
    
    # Report resonance patterns
    resonance_count = len(system.resonance_data["resonance_points"])
    print(f"Identified {resonance_count} resonance patterns")
    
    if resonance_count > 0:
        best_resonance = system.resonance_data["resonance_points"][0]
        print(f"Best resonance at frequency: {best_resonance['frequency']:.6f}")
        print(f"Phase differences: {[f'{p:.2f}' for p in best_resonance['phase_diffs']]}")
        print(f"Stability metric: {best_resonance['stability']:.6f}")
        
        # Report Lagrange points if detected
        if system.resonance_data["lagrange_points"]:
            lagrange_point = system.resonance_data["lagrange_points"][0]
            print(f"Detected Lagrange point: {lagrange_point['type']}")
            
        # Report orbital resonances if detected
        if system.resonance_data["orbital_resonances"]:
            orbital_resonance = system.resonance_data["orbital_resonances"][0]
            print(f"Detected orbital resonance: {orbital_resonance['ratio']}")
            

def run_earth_moon_system():
    """
    Run a simulation of the Earth-Moon system, which is a classic restricted
    three-body problem with known Lagrange points.
    """
    # Masses in kg (Earth, Moon, Small satellite)
    masses = [5.972e24, 7.342e22, 1000.0]
    
    # Earth-Moon distance: 384,400 km
    earth_moon_distance = 3.844e8
    
    # Initial positions in meters
    initial_positions = [
        [0, 0, 0],                            # Earth at center
        [earth_moon_distance, 0, 0],          # Moon
        [earth_moon_distance * 0.5, earth_moon_distance * 0.866, 0]  # L4 Lagrange point
    ]
    
    # Moon's orbital velocity: ~1022 m/s
    moon_velocity = 1022.0
    
    # Initial velocities in m/s
    initial_velocities = [
        [0, 0, 0],                # Earth stationary
        [0, moon_velocity, 0],    # Moon orbital velocity
        [0, moon_velocity, 0]     # Small satellite matching Moon's velocity
    ]
    
    # Create the system
    system = ThreeBodySystem(masses, initial_positions, initial_velocities)
    
    # Run the simulation
    print("Running Earth-Moon-Satellite simulation...")
    system.simulate(duration=86400.0*30, dt=86400.0)  # 30 days, 1 day steps
    
    # Report stability and Lagrange points
    stability = system.get_orbital_stability_metric()
    print(f"Earth-Moon system stability metric: {stability:.3f}")
    
    # Analyze Lagrange points
    if system.resonance_data["lagrange_points"]:
        for lagrange in system.resonance_data["lagrange_points"]:
            print(f"Detected Lagrange point: {lagrange['type']} with stability {lagrange['stability']:.6f}")


if __name__ == "__main__":
    print("QuantoniumOS Three-Body Problem Solver")
    print("=====================================")
    print("\n1. Solar System Simulation")
    run_three_body_simulation()
    
    print("\n2. Earth-Moon Lagrange Point Simulation")
    run_earth_moon_system()