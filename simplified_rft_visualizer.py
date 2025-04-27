#!/usr/bin/env python3
"""
QuantoniumOS - Simplified RFT (Resonance Fourier Transform) Visualizer

A simplified version that uses Matplotlib interactive mode for better compatibility
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import logging

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Launching QuantoniumOS Simplified RFT Visualizer...")

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


def animate_rft(i, time_points, waveform_type, ax_time, ax_freq):
    """Update function for animation"""
    phase_shift = (i % 100) / 100.0 * 2 * np.pi
    
    # Parameters with animated phase
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
    
    # Generate animated waveform
    waveform = ResonanceWaveform.get_waveform(waveform_type, params, time_points)
    
    # Compute RFT
    rft_result = RFTCalculator.compute_rft(waveform, time_points)
    
    # Update time domain plot
    ax_time.clear()
    ax_time.plot(time_points, waveform, 'b-', linewidth=1.5)
    ax_time.set_title(f'Time Domain: {waveform_type.replace("_", " ").title()}')
    ax_time.set_xlabel('Time')
    ax_time.set_ylabel('Amplitude')
    ax_time.grid(True, linestyle='--', alpha=0.7)
    
    # Update frequency domain plot
    ax_freq.clear()
    ax_freq.plot(rft_result['frequencies'], rft_result['amplitudes'], 'b-', linewidth=1.5, alpha=0.7, label='FFT')
    ax_freq.plot(rft_result['frequencies'], rft_result['enhanced_amplitudes'], 'r-', linewidth=1.5, alpha=0.5, label='RFT')
    ax_freq.plot(rft_result['frequencies'][rft_result['resonance_mask']], 
                 rft_result['amplitudes'][rft_result['resonance_mask']], 
                 'ro', markersize=6, label='Resonances')
    ax_freq.set_title('Frequency Domain (Resonance Fourier Transform)')
    ax_freq.set_xlabel('Frequency')
    ax_freq.set_ylabel('Amplitude')
    ax_freq.legend()
    ax_freq.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()


def main():
    """Main function to run the RFT visualization"""
    # Set up time domain
    time_points = np.linspace(0, 10, 1000)
    
    # Choose waveform type
    waveform_types = [
        "quantum_superposition",
        "resonance_pattern",
        "wave_packet",
        "entangled_waves",
        "quantum_beat"
    ]
    
    # Create figure and subplots
    fig, (ax_time, ax_freq) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Run animations for each waveform type
    for waveform_type in waveform_types:
        print(f"Visualizing: {waveform_type.replace('_', ' ').title()}")
        
        # Create animation
        ani = FuncAnimation(
            fig, 
            animate_rft, 
            frames=100,
            interval=50,
            fargs=(time_points, waveform_type, ax_time, ax_freq),
            repeat=False
        )
        
        # Display animation
        plt.suptitle(f"QuantoniumOS RFT Visualization: {waveform_type.replace('_', ' ').title()}")
        plt.tight_layout()
        plt.show()
        
        # Ask user if they want to continue to next waveform
        response = input("Press Enter to see next waveform or type 'exit' to quit: ")
        if response.lower() == "exit":
            break
    
    print("RFT Visualization complete.")


if __name__ == "__main__":
    plt.style.use('dark_background')  # Use dark theme for better visualization
    main()