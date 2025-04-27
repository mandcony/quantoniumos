#!/usr/bin/env python3
"""
QuantoniumOS - RFT (Resonance Fourier Transform) Visualizer

An enhanced visualization tool that demonstrates the Resonance Fourier Transform
in motion, showing both the time domain and frequency domain representations
with animated transitions between states.
"""

import sys
import os
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLabel, QComboBox, QGridLayout, QGroupBox, QSplitter
)
from PyQt5.QtCore import Qt, QTimer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.cm as cm
import logging

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Launching QuantoniumOS RFT Visualizer...")

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
        elif waveform_type == "user_defined":
            return params.get("custom_function", lambda t: np.zeros_like(t))(time_points)
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
    
    @staticmethod
    def inverse_rft(rft_result, num_points):
        """
        Compute the inverse RFT to reconstruct the original waveform
        """
        # Reconstruct the full spectrum (positive and negative frequencies)
        full_frequencies = np.fft.fftfreq(num_points, 1/num_points)
        full_amplitudes = np.zeros(num_points, dtype=complex)
        
        # Map the positive frequencies back
        positive_indices = full_frequencies > 0
        full_amplitudes[positive_indices] = rft_result['enhanced_amplitudes'] * np.exp(1j * rft_result['phases'])
        
        # Add negative frequencies (complex conjugates)
        negative_indices = full_frequencies < 0
        positive_indices_flipped = np.flip(positive_indices)
        full_amplitudes[negative_indices] = np.conj(np.flip(full_amplitudes[positive_indices]))
        
        # Inverse FFT to get back to time domain
        reconstructed = np.fft.ifft(full_amplitudes)
        
        return np.real(reconstructed)


class RFTVisualizer(QMainWindow):
    """Main visualization application for Resonance Fourier Transform"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QuantoniumOS - Resonance Fourier Transform Visualizer")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create the main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        self.setCentralWidget(main_widget)
        
        # Create a splitter for adjustable panels
        splitter = QSplitter(Qt.Vertical)
        main_layout.addWidget(splitter)
        
        # Create the visualization panel
        viz_widget = QWidget()
        viz_layout = QHBoxLayout(viz_widget)
        splitter.addWidget(viz_widget)
        
        # Create the control panel
        control_widget = QWidget()
        control_layout = QGridLayout(control_widget)
        splitter.addWidget(control_widget)
        splitter.setSizes([600, 200])  # Initial sizes
        
        # Set up the visualization figures
        # Time domain plot
        self.fig_time = Figure(figsize=(6, 4), dpi=100)
        self.ax_time = self.fig_time.add_subplot(111)
        self.canvas_time = FigureCanvas(self.fig_time)
        viz_layout.addWidget(self.canvas_time)
        
        # Frequency domain plot
        self.fig_freq = Figure(figsize=(6, 4), dpi=100)
        self.ax_freq = self.fig_freq.add_subplot(111)
        self.canvas_freq = FigureCanvas(self.fig_freq)
        viz_layout.addWidget(self.canvas_freq)
        
        # 3D phase space plot
        self.fig_3d = Figure(figsize=(6, 4), dpi=100)
        self.ax_3d = self.fig_3d.add_subplot(111, projection='3d')
        self.canvas_3d = FigureCanvas(self.fig_3d)
        viz_layout.addWidget(self.canvas_3d)
        
        # Set up the control widgets
        # Waveform selection
        waveform_group = QGroupBox("Waveform Selection")
        waveform_layout = QVBoxLayout()
        waveform_group.setLayout(waveform_layout)
        control_layout.addWidget(waveform_group, 0, 0)
        
        self.waveform_combo = QComboBox()
        self.waveform_combo.addItems([
            "quantum_superposition", 
            "resonance_pattern", 
            "wave_packet", 
            "entangled_waves", 
            "quantum_beat"
        ])
        self.waveform_combo.currentIndexChanged.connect(self.update_waveform)
        waveform_layout.addWidget(self.waveform_combo)
        
        # Animation controls
        animation_group = QGroupBox("Animation")
        animation_layout = QHBoxLayout()
        animation_group.setLayout(animation_layout)
        control_layout.addWidget(animation_group, 0, 1)
        
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_animation)
        animation_layout.addWidget(self.play_button)
        
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(10)
        self.speed_slider.setMaximum(200)
        self.speed_slider.setValue(100)
        self.speed_slider.setTickInterval(10)
        self.speed_slider.setTickPosition(QSlider.TicksBelow)
        animation_layout.addWidget(self.speed_slider)
        animation_layout.addWidget(QLabel("Animation Speed"))
        
        # Parameter controls
        param_group = QGroupBox("Waveform Parameters")
        param_layout = QGridLayout()
        param_group.setLayout(param_layout)
        control_layout.addWidget(param_group, 1, 0, 1, 2)
        
        # Parameter sliders
        self.param_sliders = {}
        
        # Frequency slider
        param_layout.addWidget(QLabel("Frequency:"), 0, 0)
        self.param_sliders["frequency"] = QSlider(Qt.Horizontal)
        self.param_sliders["frequency"].setMinimum(1)
        self.param_sliders["frequency"].setMaximum(20)
        self.param_sliders["frequency"].setValue(5)
        self.param_sliders["frequency"].valueChanged.connect(self.update_waveform)
        param_layout.addWidget(self.param_sliders["frequency"], 0, 1)
        
        # Amplitude slider
        param_layout.addWidget(QLabel("Amplitude:"), 1, 0)
        self.param_sliders["amplitude"] = QSlider(Qt.Horizontal)
        self.param_sliders["amplitude"].setMinimum(1)
        self.param_sliders["amplitude"].setMaximum(10)
        self.param_sliders["amplitude"].setValue(5)
        self.param_sliders["amplitude"].valueChanged.connect(self.update_waveform)
        param_layout.addWidget(self.param_sliders["amplitude"], 1, 1)
        
        # Complexity slider
        param_layout.addWidget(QLabel("Complexity:"), 2, 0)
        self.param_sliders["complexity"] = QSlider(Qt.Horizontal)
        self.param_sliders["complexity"].setMinimum(1)
        self.param_sliders["complexity"].setMaximum(10)
        self.param_sliders["complexity"].setValue(3)
        self.param_sliders["complexity"].valueChanged.connect(self.update_waveform)
        param_layout.addWidget(self.param_sliders["complexity"], 2, 1)
        
        # Phase slider
        param_layout.addWidget(QLabel("Phase:"), 3, 0)
        self.param_sliders["phase"] = QSlider(Qt.Horizontal)
        self.param_sliders["phase"].setMinimum(0)
        self.param_sliders["phase"].setMaximum(100)
        self.param_sliders["phase"].setValue(0)
        self.param_sliders["phase"].valueChanged.connect(self.update_waveform)
        param_layout.addWidget(self.param_sliders["phase"], 3, 1)
        
        # Initialize data
        self.time_points = np.linspace(0, 10, 1000)
        self.waveform = None
        self.rft_result = None
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.animate)
        self.animation_phase = 0.0
        self.animation_running = False
        
        # Initialize the plots
        self.update_waveform()
    
    def update_waveform(self):
        """Update the waveform based on current parameters"""
        waveform_type = self.waveform_combo.currentText()
        
        # Get parameters from sliders
        params = {
            "amplitude": self.param_sliders["amplitude"].value() / 5.0,
            "num_states": self.param_sliders["complexity"].value(),
            "num_harmonics": self.param_sliders["complexity"].value(),
            "base_freq": self.param_sliders["frequency"].value() / 5.0,
            "frequency": self.param_sliders["frequency"].value() / 2.0,
            "freq1": self.param_sliders["frequency"].value() / 5.0,
            "freq2": self.param_sliders["frequency"].value() / 3.0,
            "center_freq": self.param_sliders["frequency"].value(),
            "beat_freq": self.param_sliders["frequency"].value() / 10.0,
            "phase_diff": self.param_sliders["phase"].value() / 100.0 * 2 * np.pi,
            "width": 2.0,
        }
        
        # Generate waveform
        self.waveform = ResonanceWaveform.get_waveform(waveform_type, params, self.time_points)
        
        # Compute RFT
        self.rft_result = RFTCalculator.compute_rft(self.waveform, self.time_points)
        
        # Update plots
        self.update_plots()
    
    def update_plots(self):
        """Update all visualization plots"""
        if self.waveform is None or self.rft_result is None:
            return
        
        # Clear previous plots
        self.ax_time.clear()
        self.ax_freq.clear()
        self.ax_3d.clear()
        
        # Plot time domain
        self.ax_time.plot(self.time_points, self.waveform, 'b-', linewidth=1.5)
        self.ax_time.set_title('Time Domain Waveform')
        self.ax_time.set_xlabel('Time')
        self.ax_time.set_ylabel('Amplitude')
        self.ax_time.grid(True, linestyle='--', alpha=0.7)
        
        # Plot frequency domain
        frequencies = self.rft_result['frequencies']
        amplitudes = self.rft_result['amplitudes']
        enhanced_amplitudes = self.rft_result['enhanced_amplitudes']
        resonance_mask = self.rft_result['resonance_mask']
        
        # Plot regular amplitudes
        self.ax_freq.plot(frequencies, amplitudes, 'b-', linewidth=1.5, alpha=0.7, label='FFT')
        
        # Highlight resonances
        self.ax_freq.plot(frequencies, enhanced_amplitudes, 'r-', linewidth=1.5, alpha=0.5, label='RFT')
        self.ax_freq.plot(frequencies[resonance_mask], amplitudes[resonance_mask], 'ro', markersize=6, label='Resonances')
        
        self.ax_freq.set_title('Frequency Domain (Resonance Fourier Transform)')
        self.ax_freq.set_xlabel('Frequency')
        self.ax_freq.set_ylabel('Amplitude')
        self.ax_freq.legend()
        self.ax_freq.grid(True, linestyle='--', alpha=0.7)
        
        # 3D phase space visualization
        # Use a subset of points for clearer visualization
        step = 10
        t_subset = self.time_points[::step]
        w_subset = self.waveform[::step]
        
        # Compute derivative and second derivative for phase space
        dw = np.gradient(self.waveform)[::step]
        ddw = np.gradient(dw)[::step]
        
        # Plot 3D phase space
        self.ax_3d.plot3D(w_subset, dw, ddw, 'b-', linewidth=1.5)
        
        # Add a scattered point to represent current state
        point_idx = int((self.animation_phase % 1.0) * len(t_subset))
        self.ax_3d.scatter([w_subset[point_idx]], [dw[point_idx]], [ddw[point_idx]], 
                          color='red', s=100, label='Current State')
        
        self.ax_3d.set_title('Phase Space Representation')
        self.ax_3d.set_xlabel('Position')
        self.ax_3d.set_ylabel('Velocity')
        self.ax_3d.set_zlabel('Acceleration')
        
        # Refresh canvases
        self.canvas_time.draw()
        self.canvas_freq.draw()
        self.canvas_3d.draw()
    
    def toggle_animation(self):
        """Start or stop the animation"""
        if self.animation_running:
            self.animation_timer.stop()
            self.play_button.setText("Play")
            self.animation_running = False
        else:
            self.animation_timer.start(50)  # Update every 50ms
            self.play_button.setText("Pause")
            self.animation_running = True
    
    def animate(self):
        """Update animation frame"""
        # Update animation phase
        speed = self.speed_slider.value() / 1000.0
        self.animation_phase += speed
        
        # Apply phase shift to the time domain
        phase_shift = (self.animation_phase % 1.0) * 2 * np.pi
        waveform_type = self.waveform_combo.currentText()
        
        # Get parameters from sliders with animated phase
        params = {
            "amplitude": self.param_sliders["amplitude"].value() / 5.0,
            "num_states": self.param_sliders["complexity"].value(),
            "num_harmonics": self.param_sliders["complexity"].value(),
            "base_freq": self.param_sliders["frequency"].value() / 5.0,
            "frequency": self.param_sliders["frequency"].value() / 2.0,
            "freq1": self.param_sliders["frequency"].value() / 5.0,
            "freq2": self.param_sliders["frequency"].value() / 3.0,
            "center_freq": self.param_sliders["frequency"].value(),
            "beat_freq": self.param_sliders["frequency"].value() / 10.0,
            "phase_diff": (self.param_sliders["phase"].value() / 100.0 * 2 * np.pi) + phase_shift,
            "width": 2.0,
        }
        
        # Generate animated waveform
        self.waveform = ResonanceWaveform.get_waveform(waveform_type, params, self.time_points)
        
        # Compute RFT
        self.rft_result = RFTCalculator.compute_rft(self.waveform, self.time_points)
        
        # Update plots
        self.update_plots()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RFTVisualizer()
    window.show()
    sys.exit(app.exec_())