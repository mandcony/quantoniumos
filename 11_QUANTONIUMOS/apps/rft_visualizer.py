"""
QuantoniumOS - Resonance Fourier Transform (RFT) Engine
Real-time visualization of RFT wave processes and resonance analysis
Advanced quantum-resonance transformation engine
"""

import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.backends.backend_qt5agg import \
    FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

try:
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal
    from PyQt5.QtGui import QColor, QFont, QPainter, QPen
    from PyQt5.QtWidgets import (QCheckBox, QComboBox, QGridLayout, QGroupBox,
                                 QHBoxLayout, QLabel, QProgressBar,
                                 QPushButton, QSlider, QSpinBox, QTextEdit,
                                 QVBoxLayout, QWidget)

    PYQT5_AVAILABLE = True
except ImportError:
    PYQT5_AVAILABLE = False

# Set matplotlib style for dark theme
plt.style.use("dark_background")
sns.set_palette("bright")


class RFTVisualizer(QWidget if PYQT5_AVAILABLE else object):
    """Real-time RFT analysis and visualization"""

    def __init__(self):
        if not PYQT5_AVAILABLE:
            print("[WARNING] PyQt5 required for RFT Visualizer GUI")
            return

        super().__init__()
        # Initialize RFT wave processing parameters
        self.sample_rate = 44100  # Hz
        self.frequency_resolution = 64
        self.wave_harmonics = []
        self.resonance_peaks = []
        self.wave_amplitude = 1.0
        self.phase_shift = 0.0
        self.input_data = None
        self.transformation_history = []
        self.analysis_results = {}

        self.init_ui()
        self.initialize_rft_system()

        # Auto-update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_visualizations)
        self.update_timer.start(1000)  # Update every second

    def init_ui(self):
        """Initialize the RFT visualizer interface"""
        self.setWindowTitle("QuantoniumOS - RFT Visualizer")
        self.setGeometry(200, 200, 1400, 1000)

        # Apply cream design styling
        self.setStyleSheet(
            """
            QWidget {
                background-color: #f0ead6;
                color: #2d2d2d;
                font-family: "Segoe UI", Arial, sans-serif;
                font-size: 9pt;
            }
            QGroupBox {
                border: 1px solid #c0c0c0;
                border-radius: 4px;
                padding: 10px;
                margin-top: 10px;
                font-weight: normal;
                background-color: #f8f6f0;
            }
            QGroupBox::title {
                color: #2d2d2d;
                padding: 5px;
                font-weight: bold;
            }
            QPushButton {
                background-color: #0078d4;
                border: 1px solid #005a9e;
                border-radius: 3px;
                padding: 6px 12px;
                color: white;
                font-weight: normal;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #106ebe;
                border: 1px solid #005a9e;
            }
            QSlider::groove:horizontal {
                border: 1px solid #c0c0c0;
                height: 4px;
                background-color: #e8e8e8;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #3b82f6;
                border: 1px solid #2563eb;
                width: 16px;
                border-radius: 8px;
                margin: -5px 0;
            }
            QSpinBox, QComboBox {
                background: rgba(255, 255, 255, 0.8);
                border: 1px solid rgba(255, 255, 255, 0.3);
                border-radius: 6px;
                padding: 6px;
                color: #1a1a1a;
            }
            QTextEdit {
                background: rgba(255, 255, 255, 0.8);
                border: 1px solid rgba(255, 255, 255, 0.3);
                border-radius: 8px;
                color: #1a1a1a;
                padding: 8px;
            }
            QProgressBar {
                border: 1px solid rgba(255, 255, 255, 0.3);
                border-radius: 6px;
                background: rgba(255, 255, 255, 0.6);
                text-align: center;
                color: #1a1a1a;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #3b82f6, stop:1 #2563eb);
                border-radius: 4px;
            }
            QCheckBox {
                color: #1a1a1a;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 1px solid #d1d5db;
                border-radius: 4px;
                background: rgba(255, 255, 255, 0.8);
            }
            QCheckBox::indicator:checked {
                background: #3b82f6;
            }
            QLabel {
                color: #1a1a1a;
            }
        """
        )

        layout = QVBoxLayout(self)

        # Title
        title = QLabel("~ Resonance Fourier Transform (RFT) Engine")
        title.setFont(QFont("Consolas", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            "color: #1a1a1a; margin: 12px; background: rgba(255, 255, 255, 0.8); padding: 12px; border-radius: 12px; font-weight: 600; border: 1px solid rgba(255, 255, 255, 0.3);"
        )
        layout.addWidget(title)

        # Main content
        main_layout = QHBoxLayout()

        # Left side - Controls and analysis
        self.create_control_panel(main_layout)

        # Right side - Visualizations
        self.create_visualization_panel(main_layout)

        layout.addLayout(main_layout)

        # Bottom - Status and metrics
        self.create_status_panel(layout)

        # Status
        self.status_label = QLabel("[OK] RFT Visualizer Ready")
        self.status_label.setStyleSheet(
            "color: #dc2626; font-weight: 600; margin: 6px;"
        )
        layout.addWidget(self.status_label)

    def create_control_panel(self, parent_layout):
        """Create RFT controls panel"""
        controls_widget = QWidget()
        controls_widget.setMaximumWidth(450)
        controls_layout = QVBoxLayout(controls_widget)

        # RFT Wave Configuration
        config_group = QGroupBox("~ RFT Wave Configuration")
        config_layout = QGridLayout(config_group)

        config_layout.addWidget(QLabel("[?] Sample Rate (Hz):"), 0, 0)
        self.sample_rate_spin = QSpinBox()
        self.sample_rate_spin.setRange(8000, 192000)
        self.sample_rate_spin.setValue(self.sample_rate)
        self.sample_rate_spin.setSingleStep(1000)
        self.sample_rate_spin.valueChanged.connect(self.update_sample_rate)
        config_layout.addWidget(self.sample_rate_spin, 0, 1)

        config_layout.addWidget(QLabel("[?] Frequency Resolution:"), 1, 0)
        self.freq_res_spin = QSpinBox()
        self.freq_res_spin.setRange(16, 1024)
        self.freq_res_spin.setValue(self.frequency_resolution)
        self.freq_res_spin.valueChanged.connect(self.update_frequency_resolution)
        config_layout.addWidget(self.freq_res_spin, 1, 1)

        config_layout.addWidget(QLabel("[?] Wave Type:"), 2, 0)
        self.wave_type = QComboBox()
        self.wave_type.addItems(
            ["Sine", "Cosine", "Square", "Sawtooth", "Triangle", "White Noise"]
        )
        self.wave_type.currentTextChanged.connect(self.generate_wave_data)
        config_layout.addWidget(self.wave_type, 2, 1)

        self.generate_btn = QPushButton("[?] Generate Wave Data")
        self.generate_btn.clicked.connect(self.generate_wave_data)
        config_layout.addWidget(self.generate_btn, 3, 0, 1, 2)

        controls_layout.addWidget(config_group)

        # RFT Wave Transformation Controls
        transform_group = QGroupBox("~ RFT Transformation Control")
        transform_layout = QVBoxLayout(transform_group)

        self.start_btn = QPushButton("?? Start RFT Processing")
        self.start_btn.clicked.connect(self.start_rft_transformation)
        transform_layout.addWidget(self.start_btn)

        self.step_btn = QPushButton("?? Step Forward")
        self.step_btn.clicked.connect(self.step_rft_transformation)
        transform_layout.addWidget(self.step_btn)

        self.reset_btn = QPushButton("Reset")
        self.reset_btn.clicked.connect(self.reset_rft_system)
        transform_layout.addWidget(self.reset_btn)

        # Progress indicator
        transform_layout.addWidget(QLabel("[STATS] Progress:"))
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(5)  # 5 RFT processing stages
        transform_layout.addWidget(self.progress_bar)

        controls_layout.addWidget(transform_group)

        # Analysis Options
        analysis_group = QGroupBox("[STATS] Analysis Options")
        analysis_layout = QVBoxLayout(analysis_group)

        self.entropy_check = QCheckBox(">> Entropy Analysis")
        self.entropy_check.setChecked(True)
        analysis_layout.addWidget(self.entropy_check)

        self.diffusion_check = QCheckBox("~ Diffusion Analysis")
        self.diffusion_check.setChecked(True)
        analysis_layout.addWidget(self.diffusion_check)

        self.avalanche_check = QCheckBox("?? Avalanche Effect")
        self.avalanche_check.setChecked(True)
        analysis_layout.addWidget(self.avalanche_check)

        self.frequency_check = QCheckBox("[STATS] Frequency Analysis")
        self.frequency_check.setChecked(False)
        analysis_layout.addWidget(self.frequency_check)

        controls_layout.addWidget(analysis_group)

        # Round Keys Display
        keys_group = QGroupBox("? Round Keys")
        keys_layout = QVBoxLayout(keys_group)

        self.keys_text = QTextEdit()
        self.keys_text.setMaximumHeight(120)
        self.keys_text.setReadOnly(True)
        keys_layout.addWidget(self.keys_text)

        controls_layout.addWidget(keys_group)

        # Metrics Display
        metrics_group = QGroupBox("[STATS] Real-time Metrics")
        metrics_layout = QVBoxLayout(metrics_group)

        self.metrics_text = QTextEdit()
        self.metrics_text.setMaximumHeight(150)
        self.metrics_text.setReadOnly(True)
        metrics_layout.addWidget(self.metrics_text)

        controls_layout.addWidget(metrics_group)

        parent_layout.addWidget(controls_widget)

    def create_visualization_panel(self, parent_layout):
        """Create visualization panel with matplotlib"""
        viz_group = QGroupBox("[STATS] RFT Visualizations")
        viz_layout = QVBoxLayout(viz_group)

        # Create matplotlib figure
        self.figure = Figure(figsize=(12, 8), facecolor="#0a0e1a")
        self.canvas = FigureCanvas(self.figure)
        viz_layout.addWidget(self.canvas)

        parent_layout.addWidget(viz_group)

    def create_status_panel(self, parent_layout):
        """Create status and information panel"""
        status_group = QGroupBox("? System Status")
        status_layout = QHBoxLayout(status_group)

        # Current round info
        self.stage_label = QLabel("~ Stage: Initialization")
        self.stage_label.setStyleSheet("color: #1a1a1a; font-weight: 600;")
        status_layout.addWidget(self.stage_label)

        # Transformation time
        self.time_label = QLabel("?? Time: 0.00s")
        self.time_label.setStyleSheet("color: #1a1a1a; font-weight: 600;")
        status_layout.addWidget(self.time_label)

        # Security level
        self.security_label = QLabel("?? Security: High")
        self.security_label.setStyleSheet("color: #059669; font-weight: 600;")
        status_layout.addWidget(self.security_label)

        parent_layout.addWidget(status_group)

    def initialize_rft_system(self):
        """Initialize the RFT wave processing system"""
        self.generate_wave_data()
        self.update_visualizations()
        self.status_label.setText(
            "~ RFT System initialized - Ready for wave processing"
        )

    def generate_wave_data(self):
        """Generate wave data based on selected type for RFT processing"""
        wave_type = self.wave_type.currentText()

        # Generate time array
        duration = 1.0  # 1 second
        t = np.linspace(0, duration, int(self.sample_rate * duration))

        # Generate base frequency
        base_freq = 440.0  # A4 note

        if wave_type == "Sine":
            self.input_data = np.sin(2 * np.pi * base_freq * t) * self.wave_amplitude
        elif wave_type == "Cosine":
            self.input_data = np.cos(2 * np.pi * base_freq * t) * self.wave_amplitude
        elif wave_type == "Square":
            self.input_data = (
                np.sign(np.sin(2 * np.pi * base_freq * t)) * self.wave_amplitude
            )
        elif wave_type == "Sawtooth":
            self.input_data = (
                2
                * (t * base_freq - np.floor(t * base_freq + 0.5))
                * self.wave_amplitude
            )
        elif wave_type == "Triangle":
            self.input_data = (
                2
                * np.arcsin(np.sin(2 * np.pi * base_freq * t))
                / np.pi
                * self.wave_amplitude
            )
        elif wave_type == "White Noise":
            self.input_data = np.random.normal(0, self.wave_amplitude, len(t))

        # Add harmonics for richness
        for i in range(2, 5):  # Add 2nd, 3rd, 4th harmonics
            harmonic_amplitude = self.wave_amplitude / i
            harmonic = np.sin(2 * np.pi * base_freq * i * t) * harmonic_amplitude
            self.input_data += harmonic

        self.transformation_history = [self.input_data.copy()]
        self.status_label.setText(
            f"~ Generated {wave_type.lower()} wave data ({len(self.input_data)} samples)"
        )

    def update_sample_rate(self, new_rate):
        """Update sample rate for RFT processing"""
        self.sample_rate = new_rate
        self.generate_wave_data()
        self.status_label.setText(f"[?] Sample rate updated to {new_rate} Hz")

    def update_frequency_resolution(self, new_resolution):
        """Update frequency resolution for RFT analysis"""
        self.frequency_resolution = new_resolution
        self.status_label.setText(
            f"[?] Frequency resolution updated to {new_resolution}"
        )

    def start_rft_transformation(self):
        """Start RFT wave transformation processing"""
        if self.input_data is None:
            self.generate_wave_data()

        self.transformation_history = [self.input_data.copy()]
        current_data = self.input_data.copy()

        # Perform RFT analysis stages
        stages = [
            "Windowing",
            "FFT",
            "Resonance Detection",
            "Phase Analysis",
            "Reconstruction",
        ]

        for stage_num, stage_name in enumerate(stages):
            # Apply RFT stage processing
            current_data = self.apply_rft_stage(current_data, stage_name)
            self.transformation_history.append(current_data.copy())
            self.progress_bar.setValue(stage_num + 1)
            self.status_label.setText(f"~ Processing: {stage_name}")

        self.status_label.setText("[OK] RFT transformation complete!")
        self.update_visualizations()

    def apply_rft_stage(self, data, stage_name):
        """Apply specific RFT processing stage"""
        if stage_name == "Windowing":
            # Apply Hanning window
            window = np.hanning(len(data))
            return data * window
        elif stage_name == "FFT":
            # Perform FFT for frequency domain analysis
            fft_result = np.fft.fft(data)
            return np.abs(fft_result)
        elif stage_name == "Resonance Detection":
            # Detect resonance peaks
            peaks = []
            for i in range(1, len(data) - 1):
                if (
                    data[i] > data[i - 1]
                    and data[i] > data[i + 1]
                    and data[i] > np.mean(data)
                ):
                    peaks.append(i)
            self.resonance_peaks = peaks
            return data
        elif stage_name == "Phase Analysis":
            # Analyze phase relationships
            return np.angle(np.fft.fft(data))
        elif stage_name == "Reconstruction":
            # Reconstruct signal with resonance enhancement
            enhanced = data.copy()
            for peak in self.resonance_peaks:
                if peak < len(enhanced):
                    enhanced[peak] *= 1.5  # Enhance resonance peaks
            return enhanced
        else:
            return data

        self.update_visualizations()
        self.status_label.setText("[OK] RFT transformation completed")

    def step_rft_transformation(self):
        """Step through RFT transformation one stage at a time"""
        current_stage = len(self.transformation_history) - 1

        if current_stage >= 5:  # 5 RFT stages
            self.status_label.setText("[OK] RFT processing already complete")
            return

        current_data = self.transformation_history[-1].copy()
        stages = [
            "Windowing",
            "FFT",
            "Resonance Detection",
            "Phase Analysis",
            "Reconstruction",
        ]

        if current_stage < len(stages):
            new_data = self.apply_rft_stage(current_data, stages[current_stage])
            self.transformation_history.append(new_data)

            self.progress_bar.setValue(current_stage + 1)
            self.update_visualizations()
            self.status_label.setText(f"?? Completed stage: {stages[current_stage]}")

    def reset_rft_system(self):
        """Reset the RFT system"""
        if self.input_data is not None:
            self.transformation_history = [self.input_data.copy()]
        else:
            self.generate_wave_data()
        self.progress_bar.setValue(0)
        self.update_visualizations()
        self.status_label.setText("RFT system reset")

    def update_visualizations(self):
        """Update all visualizations"""
        self.figure.clear()

        if len(self.transformation_history) < 2:
            return

        # Create subplots
        gs = self.figure.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        # 1. Data evolution heatmap
        ax1 = self.figure.add_subplot(gs[0, 0])
        if len(self.transformation_history) > 1:
            data_matrix = np.array(self.transformation_history).T
            im1 = ax1.imshow(data_matrix, cmap="plasma", aspect="auto")
            ax1.set_title("> Data Evolution", color="#ff6b35", fontweight="bold")
            ax1.set_xlabel("Round")
            ax1.set_ylabel("Byte Position")
            self.figure.colorbar(im1, ax=ax1, shrink=0.8)

        # 2. Entropy analysis
        ax2 = self.figure.add_subplot(gs[0, 1])
        if self.entropy_check.isChecked():
            entropies = []
            for data in self.transformation_history:
                # Calculate Shannon entropy
                _, counts = np.unique(data, return_counts=True)
                probabilities = counts / len(data)
                entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
                entropies.append(entropy)

            ax2.plot(entropies, "o-", color="#ff6b35", linewidth=2, markersize=6)
            ax2.set_title(">> Entropy Evolution", color="#ff6b35", fontweight="bold")
            ax2.set_xlabel("Round")
            ax2.set_ylabel("Shannon Entropy")
            ax2.grid(True, alpha=0.3)

        # 3. Diffusion analysis
        ax3 = self.figure.add_subplot(gs[0, 2])
        if self.diffusion_check.isChecked() and len(self.transformation_history) > 1:
            # Calculate bit differences between rounds
            differences = []
            for i in range(1, len(self.transformation_history)):
                prev_data = self.transformation_history[i - 1]
                curr_data = self.transformation_history[i]
                diff = np.sum(prev_data != curr_data)
                differences.append(diff)

            ax3.bar(range(len(differences)), differences, color="#ff6b35", alpha=0.7)
            ax3.set_title("~ Diffusion Pattern", color="#ff6b35", fontweight="bold")
            ax3.set_xlabel("Round")
            ax3.set_ylabel("Changed Bytes")

        # 4. Frequency analysis
        ax4 = self.figure.add_subplot(gs[1, :])
        if self.frequency_check.isChecked():
            # Show byte frequency distribution for current state
            current_data = self.transformation_history[-1]
            values, counts = np.unique(current_data, return_counts=True)
            ax4.bar(values, counts, color="#ff6b35", alpha=0.7, width=1.0)
            ax4.set_title(
                "[STATS] Byte Frequency Distribution",
                color="#ff6b35",
                fontweight="bold",
            )
            ax4.set_xlabel("Byte Value")
            ax4.set_ylabel("Frequency")
            ax4.set_xlim(0, 255)
        else:
            # Show wave data analysis
            ax4.axis("off")
            if len(self.transformation_history) >= 2:
                for i, data in enumerate(self.transformation_history[-4:]):
                    # Format wave data for display
                    if len(data) > 0:
                        wave_info = f"Stage {i}: Samples={len(data)}, Max={np.max(data):.3f}, Min={np.min(data):.3f}, RMS={np.sqrt(np.mean(data**2)):.3f}"
                    else:
                        wave_info = f"Stage {i}: No data"

                    stage_names = [
                        "Original",
                        "Windowed",
                        "FFT",
                        "Resonance",
                        "Phase",
                        "Enhanced",
                    ]
                    stage_name = stage_names[min(i, len(stage_names) - 1)]

                    ax4.text(
                        0.05,
                        0.8 - i * 0.15,
                        f"{stage_name}: {wave_info}",
                        transform=ax4.transAxes,
                        fontfamily="monospace",
                        color="#00ffcc",
                        fontsize=9,
                    )
                ax4.set_title(
                    "~ Wave Analysis Data", color="#00ffcc", fontweight="bold"
                )

        self.canvas.draw()

        # Update metrics
        self.update_metrics()

    def update_metrics(self):
        """Update real-time RFT wave processing metrics"""
        if len(self.transformation_history) < 2:
            return

        current_data = self.transformation_history[-1]
        initial_data = self.transformation_history[0]

        # Calculate RFT wave metrics
        current_stage = len(self.transformation_history) - 1

        # Signal-to-Noise Ratio (SNR)
        signal_power = np.mean(current_data**2)
        noise_estimate = np.var(current_data - np.mean(current_data))
        snr = 10 * np.log10(signal_power / (noise_estimate + 1e-10))

        # Frequency Domain Energy
        fft_data = np.fft.fft(current_data)
        freq_energy = np.sum(np.abs(fft_data) ** 2)

        # Resonance Detection Count
        resonance_count = (
            len(self.resonance_peaks) if hasattr(self, "resonance_peaks") else 0
        )

        # Phase Coherence
        phases = np.angle(fft_data)
        phase_coherence = np.abs(np.mean(np.exp(1j * phases)))

        # RFT Processing Quality Assessment
        if snr > 20 and phase_coherence > 0.7:
            quality = "? Excellent"
            quality_color = "#00ff88"
        elif snr > 10 and phase_coherence > 0.5:
            quality = "[OK] Good"
            quality_color = "#ffaa00"
        else:
            quality = "[WARNING] Needs Tuning"
            quality_color = "#ff4444"

        stage_names = ["Original", "Windowed", "FFT", "Resonance", "Phase", "Enhanced"]
        current_stage_name = stage_names[min(current_stage, len(stage_names) - 1)]

        metrics_str = f"~ RFT Processing Metrics:\n\n"
        metrics_str += f"Stage: {current_stage_name} ({current_stage}/5)\n"
        metrics_str += f"Sample Rate: {self.sample_rate} Hz\n"
        metrics_str += f"SNR: {snr:.2f} dB\n"
        metrics_str += f"Freq Energy: {freq_energy:.2e}\n"
        metrics_str += f"Phase Coherence: {phase_coherence:.3f}\n"
        metrics_str += f"Resonance Peaks: {resonance_count}\n"
        metrics_str += f"Data Length: {len(current_data)} samples\n"
        metrics_str += f"Processing: {'Complete' if current_stage >= 5 else 'Active'}"

        self.metrics_text.clear()
        self.metrics_text.append(metrics_str)

        # Update status labels
        self.security_label.setText(f"~ Quality: {quality.split()[-1]}")
        self.security_label.setStyleSheet(f"color: {quality_color}; font-weight: 600;")

        import time

        self.time_label.setText(f"?? Time: {time.time() % 100:.2f}s")


def main():
    """Main entry point"""
    if not PYQT5_AVAILABLE:
        print("[ERROR] PyQt5 required for RFT Visualizer")
        return

    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = RFTVisualizer()
    window.show()

    return app.exec_()


if __name__ == "__main__":
    main()
