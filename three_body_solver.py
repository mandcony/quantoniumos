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
        """
        waveforms = []
        
        for i, mass in enumerate(self.masses):
            # Use mass as seed for reproducibility
            seed = int(mass * 1000) % 10000
            
            # Generate base waveform
            base_waveform = generate_waveform(waveform_length, seed)
            
            # Scale by mass to represent gravitational strength
            scaled_waveform = base_waveform * mass
            
            # Modulate by position to represent spatial distribution
            position_modulation = np.sin(np.linspace(0, 2*np.pi, waveform_length) + 
                                        np.sum(self.positions[i]))
            
            modulated_waveform = scaled_waveform * position_modulation
            
            waveforms.append({
                "body_index": i,
                "mass": mass,
                "waveform": modulated_waveform,
                "analysis": analyze_waveform(modulated_waveform)
            })
        
        return waveforms
    
    def _analyze_resonance_patterns(self):
        """
        Analyze the waveforms for resonance patterns that might indicate
        stable orbital configurations.
        
        This is where the QuantoniumOS resonance mathematics provides a
        novel approach to the three-body problem.
        """
        resonance_data = {
            "resonance_points": [],
            "stability_metrics": [],
            "transform_data": []
        }
        
        # Transform each field waveform
        transform_data = []
        for field in self.field_waveforms:
            rft_data = resonance_fourier_transform(field["waveform"])
            transform_data.append(rft_data)
        
        # Look for resonance between the three bodies
        for i in range(len(transform_data[0]["frequencies"])):
            # Skip DC component
            if i == 0:
                continue
                
            # Check if this frequency is resonant in all three bodies
            if (transform_data[0]["resonance_mask"][i] and
                transform_data[1]["resonance_mask"][i] and
                transform_data[2]["resonance_mask"][i]):
                
                # Calculate phase relationships
                phase_diff_1_2 = abs(transform_data[0]["phases"][i] - transform_data[1]["phases"][i]) % (2*np.pi)
                phase_diff_2_3 = abs(transform_data[1]["phases"][i] - transform_data[2]["phases"][i]) % (2*np.pi)
                phase_diff_3_1 = abs(transform_data[2]["phases"][i] - transform_data[0]["phases"][i]) % (2*np.pi)
                
                # If phases form a balanced relationship, we may have a stable configuration
                phase_sum = phase_diff_1_2 + phase_diff_2_3 + phase_diff_3_1
                stability = abs(phase_sum - 2*np.pi) / (2*np.pi)  # 0 = perfectly stable
                
                resonance_data["resonance_points"].append({
                    "frequency_index": i,
                    "frequency": transform_data[0]["frequencies"][i],
                    "phase_diffs": [phase_diff_1_2, phase_diff_2_3, phase_diff_3_1],
                    "stability": stability
                })
                
                resonance_data["stability_metrics"].append(stability)
        
        resonance_data["transform_data"] = transform_data
        
        # Sort resonance points by stability
        resonance_data["resonance_points"].sort(key=lambda x: x["stability"])
        
        return resonance_data
    
    def step(self, dt=0.01):
        """
        Advance the simulation by one timestep using resonance-based approach.
        
        Instead of directly integrating forces, we:
        1. Update the field waveforms based on current positions
        2. Analyze resonance patterns to identify stable configurations
        3. Adjust velocities to move toward resonant configurations
        4. Apply standard velocity->position update
        """
        # Update field waveforms based on new positions
        self.field_waveforms = self._generate_field_waveforms()
        
        # Re-analyze resonance to find stable configurations
        resonance_data = self._analyze_resonance_patterns()
        
        # Traditional force calculation as baseline
        accelerations = self._calculate_accelerations()
        
        # Modify accelerations based on resonance data if available
        if resonance_data["resonance_points"]:
            # Get the most stable resonance point
            best_resonance = resonance_data["resonance_points"][0]
            
            # Only apply resonance correction if it's sufficiently stable
            if best_resonance["stability"] < 0.3:  # Arbitrary threshold
                # Apply resonance-based correction to accelerations
                resonance_correction = self._calculate_resonance_correction(best_resonance)
                accelerations = accelerations + resonance_correction * 0.5  # Blend with traditional approach
        
        # Update velocities using calculated accelerations
        self.velocities += accelerations * dt
        
        # Update positions using velocities
        self.positions += self.velocities * dt
        
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


if __name__ == "__main__":
    print("QuantoniumOS Three-Body Problem Solver")
    print("=====================================")
    run_three_body_simulation()