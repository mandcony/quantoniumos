#!/usr/bin/env python3
"""
QuantoniumOS - Static RFT (Resonance Fourier Transform) Visualizer

A non-interactive version that generates static visualizations of the RFT
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import logging
import os

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Launching QuantoniumOS Static RFT Visualizer...")

# Create output directory
output_dir = "rft_visualizations"
os.makedirs(output_dir, exist_ok=True)

class ResonanceWaveform:
    """Generator for various quantum-inspired waveforms"""
    
    @staticmethod
    def get_waveform(waveform_type, params, time_points):
        """Generate a specific waveform type with given parameters"""
        if waveform_type == "quantum_superposition":
            return ResonanceWaveform.quantum_superposition(params, time_points)
        elif waveform_type == "resonance_pattern":
            return ResonanceWaveform.resonance_pattern(params, time_points)
        elif waveform_type == "wave_packet":
            return ResonanceWaveform.wave_packet(params, time_points)
        elif waveform_type == "entangled_waves":
            return ResonanceWaveform.entangled_waves(params, time_points)
        elif waveform_type == "quantum_beat":
            return ResonanceWaveform.quantum_beat(params, time_points)
        else:
            # Default to sine wave
            return np.sin(time_points)
    
    @staticmethod
    def quantum_superposition(params, t):
        """Simulated quantum superposition of states"""
        num_states = params.get("num_states", 3)
        amplitudes = params.get("amplitudes", np.ones(num_states)/np.sqrt(num_states))
        frequencies = params.get("frequencies", np.arange(1, num_states+1))
        phases = params.get("phases", np.zeros(num_states))
        
        result = np.zeros_like(t)
        for i in range(num_states):
            result += amplitudes[i] * np.sin(frequencies[i] * t + phases[i])
        return result
    
    @staticmethod
    def resonance_pattern(params, t):
        """Resonance pattern with multiple harmonics"""
        base_freq = params.get("base_freq", 1.0)
        num_harmonics = params.get("num_harmonics", 5)
        decay = params.get("decay", 0.5)
        
        result = np.zeros_like(t)
        for i in range(1, num_harmonics+1):
            harmonic_amp = 1.0 / (i ** decay)
            result += harmonic_amp * np.sin(i * base_freq * t)
        return result / num_harmonics
    
    @staticmethod
    def wave_packet(params, t):
        """Gaussian wave packet"""
        amplitude = params.get("amplitude", 1.0)
        center = params.get("center", 0.0)
        width = params.get("width", 1.0)
        frequency = params.get("frequency", 5.0)
        
        gaussian = amplitude * np.exp(-(t - center)**2 / (2 * width**2))
        carrier = np.sin(frequency * t)
        return gaussian * carrier
    
    @staticmethod
    def entangled_waves(params, t):
        """Simulation of entangled quantum waves"""
        amplitude = params.get("amplitude", 1.0)
        phase_diff = params.get("phase_diff", np.pi/2)
        freq1 = params.get("freq1", 1.0)
        freq2 = params.get("freq2", 1.5)
        
        wave1 = amplitude * np.sin(freq1 * t)
        wave2 = amplitude * np.sin(freq2 * t + phase_diff)
        entanglement = np.cos(t) * wave1 + np.sin(t) * wave2
        return entanglement
    
    @staticmethod
    def quantum_beat(params, t):
        """Quantum beat pattern (interference between closely spaced frequencies)"""
        amplitude = params.get("amplitude", 1.0)
        center_freq = params.get("center_freq", 10.0)
        beat_freq = params.get("beat_freq", 0.5)
        
        return amplitude * np.sin(center_freq * t) * np.cos(beat_freq * t)


class RFTCalculator:
    """Performs Resonance Fourier Transform calculations"""
    
    @staticmethod
    def compute_rft(waveform, time_points):
        """
        Compute the Resonance Fourier Transform
        This is an enhanced version of FFT that retains phase information
        and handles resonance patterns specially
        """
        # Standard FFT computation
        fft_result = np.fft.fft(waveform)
        frequencies = np.fft.fftfreq(len(time_points), time_points[1] - time_points[0])
        
        # Extract amplitude and phase
        amplitudes = np.abs(fft_result)
        phases = np.angle(fft_result)
        
        # We'll keep only the positive frequencies
        positive_freq_idx = frequencies > 0
        frequencies = frequencies[positive_freq_idx]
        amplitudes = amplitudes[positive_freq_idx]
        phases = phases[positive_freq_idx]
        
        # Enhance resonant frequencies
        # In RFT, resonant frequencies get special treatment
        resonance_mask = RFTCalculator.detect_resonances(amplitudes)
        enhanced_amplitudes = amplitudes.copy()
        enhanced_amplitudes[resonance_mask] *= 1.2  # Enhance resonant frequencies
        
        return {
            'frequencies': frequencies,
            'amplitudes': amplitudes,
            'enhanced_amplitudes': enhanced_amplitudes,
            'phases': phases,
            'resonance_mask': resonance_mask
        }
    
    @staticmethod
    def detect_resonances(amplitudes, threshold_factor=1.5):
        """
        Detect resonance peaks in the amplitude spectrum
        These are frequencies that stand out from their neighbors
        """
        # Simple peak detection algorithm
        is_resonance = np.zeros_like(amplitudes, dtype=bool)
        
        # Define what counts as a "neighborhood"
        window_size = min(5, len(amplitudes)//10 if len(amplitudes) > 20 else 2)
        
        for i in range(window_size, len(amplitudes)-window_size):
            neighborhood = amplitudes[i-window_size:i+window_size+1]
            neighborhood_mean = np.mean(neighborhood)
            # If amplitude is significantly higher than neighborhood average, mark as resonance
            if amplitudes[i] > threshold_factor * neighborhood_mean:
                is_resonance[i] = True
                
        return is_resonance


def generate_rft_visualization(time_points, waveform_type, phase_index, num_phases=5):
    """Generate static visualization for a waveform with specified phase"""
    # Calculate phase shift based on index
    phase_shift = (phase_index / num_phases) * 2 * np.pi
    
    # Parameters with specific phase
    params = {
        "amplitude": 1.0,
        "num_states": 3,
        "num_harmonics": 5,
        "base_freq": 1.0,
        "frequency": 2.5,
        "freq1": 1.0,
        "freq2": 1.5,
        "center_freq": 5.0,
        "beat_freq": 0.5,
        "phase_diff": phase_shift,
        "width": 2.0,
    }
    
    # Generate waveform
    waveform = ResonanceWaveform.get_waveform(waveform_type, params, time_points)
    
    # Compute RFT
    rft_result = RFTCalculator.compute_rft(waveform, time_points)
    
    # Create figure with 2 subplots for time and frequency domains
    fig, (ax_time, ax_freq) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot time domain
    ax_time.plot(time_points, waveform, 'b-', linewidth=1.5)
    ax_time.set_title(f'Time Domain: {waveform_type.replace("_", " ").title()} (Phase: {phase_index+1}/{num_phases})', fontsize=14)
    ax_time.set_xlabel('Time', fontsize=12)
    ax_time.set_ylabel('Amplitude', fontsize=12)
    ax_time.grid(True, linestyle='--', alpha=0.7)
    
    # Plot frequency domain
    ax_freq.plot(rft_result['frequencies'], rft_result['amplitudes'], 'b-', linewidth=1.5, alpha=0.7, label='FFT')
    ax_freq.plot(rft_result['frequencies'], rft_result['enhanced_amplitudes'], 'r-', linewidth=1.5, alpha=0.5, label='RFT')
    if np.any(rft_result['resonance_mask']):
        ax_freq.plot(rft_result['frequencies'][rft_result['resonance_mask']], 
                     rft_result['amplitudes'][rft_result['resonance_mask']], 
                     'ro', markersize=6, label='Resonances')
    ax_freq.set_title('Frequency Domain (Resonance Fourier Transform)', fontsize=14)
    ax_freq.set_xlabel('Frequency', fontsize=12)
    ax_freq.set_ylabel('Amplitude', fontsize=12)
    ax_freq.legend(fontsize=10)
    ax_freq.grid(True, linestyle='--', alpha=0.7)
    
    # Add overall title
    plt.suptitle(f"QuantoniumOS RFT Visualization: {waveform_type.replace('_', ' ').title()}", fontsize=16)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save figure
    filename = f"{output_dir}/{waveform_type}_phase{phase_index+1}.png"
    plt.savefig(filename, dpi=150)
    logger.info(f"Saved visualization to {filename}")
    plt.close(fig)
    
    return filename


def main():
    """Main function to run the RFT visualization"""
    # Set up time domain
    time_points = np.linspace(0, 10, 1000)
    
    # Define waveform types
    waveform_types = [
        "quantum_superposition",
        "resonance_pattern",
        "wave_packet",
        "entangled_waves",
        "quantum_beat"
    ]
    
    # Number of phases to generate for each waveform
    num_phases = 5
    
    # Generate visualizations for each waveform type and phase
    for waveform_type in waveform_types:
        logger.info(f"Generating visualizations for {waveform_type}")
        
        for phase_index in range(num_phases):
            filename = generate_rft_visualization(time_points, waveform_type, phase_index, num_phases)
            print(f"Generated: {filename}")
    
    logger.info(f"All visualizations complete. Files saved to {output_dir}/")
    
    # Create an index HTML file to view all images
    create_html_index(waveform_types, num_phases)


def create_html_index(waveform_types, num_phases):
    """Create an HTML file to view all generated visualizations"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>QuantoniumOS RFT Visualizations</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #f0f0f0; }
            h1 { color: #333366; }
            h2 { color: #336699; }
            .waveform-section { margin-bottom: 30px; background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .phase-images { display: flex; flex-wrap: wrap; gap: 10px; }
            .phase-image { border: 1px solid #ddd; border-radius: 4px; padding: 5px; background-color: white; }
            .phase-image img { max-width: 100%; height: auto; display: block; }
            .phase-image p { margin: 5px 0; text-align: center; font-size: 0.9em; }
        </style>
    </head>
    <body>
        <h1>QuantoniumOS Resonance Fourier Transform Visualizations</h1>
    """
    
    for waveform_type in waveform_types:
        waveform_name = waveform_type.replace("_", " ").title()
        html_content += f"""
        <div class="waveform-section">
            <h2>{waveform_name}</h2>
            <div class="phase-images">
        """
        
        for phase_index in range(num_phases):
            image_path = f"{waveform_type}_phase{phase_index+1}.png"
            html_content += f"""
                <div class="phase-image">
                    <img src="{image_path}" alt="{waveform_name} Phase {phase_index+1}">
                    <p>Phase {phase_index+1}/{num_phases}</p>
                </div>
            """
        
        html_content += """
            </div>
        </div>
        """
    
    html_content += """
    </body>
    </html>
    """
    
    # Write the HTML file
    html_path = f"{output_dir}/index.html"
    with open(html_path, "w") as f:
        f.write(html_content)
    
    logger.info(f"Created HTML index at {html_path}")


if __name__ == "__main__":
    main()