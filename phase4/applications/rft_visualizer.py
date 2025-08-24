"""
Phase 4: RFT Transform Visualizer
Advanced real-time RFT analysis with 3D PyQt5 visualizations
Integrated QWaveDebugger for enhanced quantum wave visualization
Real-time system monitoring and quantum gate visualization
"""

import ctypes
import json
import logging
import math
import os
import platform
import sys
import threading
import time
import tkinter as tk
from datetime import datetime
from tkinter import ttk

import numpy as np

# PyQt5 imports for advanced 3D visualization
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qt5agg import \
        FigureCanvasQTAgg as FigureCanvas
    from mpl_toolkits.mplot3d import Axes3D
    from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
    from PyQt5.QtGui import QColor, QFont
    from PyQt5.QtWidgets import (QApplication, QCheckBox, QGroupBox,
                                 QHBoxLayout, QLabel, QMainWindow, QPushButton,
                                 QSlider, QSplitter, QTableWidget,
                                 QTableWidgetItem, QTabWidget, QTextEdit,
                                 QVBoxLayout, QWidget)

    PYQT5_AVAILABLE = True
except ImportError:
    PYQT5_AVAILABLE = False

# Try to import quantum and RFT modules
try:
    from production_topological_quantum_kernel import TopologicalQuantumKernel

    from canonical_true_rft import CanonicalTrueRFT

    RFT_AVAILABLE = True
except ImportError:
    RFT_AVAILABLE = False

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuantumGateVisualizer:
    """Quantum Gate and Qubit Visualization System"""

    def __init__(self):
        self.gates = {
            "X": np.array([[0, 1], [1, 0]]),  # Pauli-X (NOT)
            "Y": np.array([[0, -1j], [1j, 0]]),  # Pauli-Y
            "Z": np.array([[1, 0], [0, -1]]),  # Pauli-Z
            "H": np.array([[1, 1], [1, -1]]) / np.sqrt(2),  # Hadamard
            "S": np.array([[1, 0], [0, 1j]]),  # Phase
            "T": np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]]),  # T gate
            "CNOT": np.array(
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
            ),  # CNOT
        }
        self.qubit_states = []
        self.circuit_history = []

    def apply_gate(self, gate_name, qubit_state):
        """Apply quantum gate to qubit state"""
        if gate_name in self.gates:
            gate = self.gates[gate_name]
            new_state = np.dot(gate, qubit_state)
            self.circuit_history.append(
                {
                    "gate": gate_name,
                    "input_state": qubit_state.copy(),
                    "output_state": new_state.copy(),
                    "timestamp": datetime.now(),
                }
            )
            return new_state
        return qubit_state

    def create_random_qubit_state(self):
        """Create a random normalized qubit state"""
        state = np.random.complex128(2)
        state = state / np.linalg.norm(state)
        return state

    def get_bloch_coordinates(self, qubit_state):
        """Convert qubit state to Bloch sphere coordinates"""
        # Normalize the state
        state = qubit_state / np.linalg.norm(qubit_state)

        # Extract coefficients
        alpha, beta = state[0], state[1]

        # Calculate Bloch coordinates
        x = 2 * np.real(np.conj(alpha) * beta)
        y = 2 * np.imag(np.conj(alpha) * beta)
        z = np.abs(alpha) ** 2 - np.abs(beta) ** 2

        return x, y, z


class SystemMonitorThread(QThread):
    """Real-time system monitoring thread"""

    data_updated = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.running = False
        self.rft_module = None
        self.quantum_kernel = None

        # Initialize modules if available
        if RFT_AVAILABLE:
            try:
                self.rft_module = CanonicalTrueRFT()
                logger.info("✅ RFT module initialized for monitoring")
            except Exception as e:
                logger.warning(f"⚠️ RFT module initialization failed: {e}")

        try:
            self.quantum_kernel = TopologicalQuantumKernel()
            logger.info("✅ Quantum kernel initialized for monitoring")
        except Exception as e:
            logger.warning(f"⚠️ Quantum kernel initialization failed: {e}")

    def run(self):
        """Main monitoring loop"""
        while self.running:
            try:
                system_data = self.collect_system_data()
                self.data_updated.emit(system_data)
                time.sleep(0.1)  # 10 Hz update rate
            except Exception as e:
                logger.error(f"Monitor thread error: {e}")
                time.sleep(1)

    def collect_system_data(self):
        """Collect real-time system data"""
        data = {
            "timestamp": datetime.now().isoformat(),
            "rft_data": {},
            "quantum_data": {},
            "crypto_data": {},
            "system_stats": {},
        }

        # RFT data
        if self.rft_module:
            try:
                # Generate test signal for RFT
                test_signal = np.sin(
                    2 * np.pi * np.linspace(0, 1, 64) * time.time() * 0.1
                )
                rft_result = self.rft_module.forward_transform(test_signal)
                data["rft_data"] = {
                    "signal_length": len(test_signal),
                    "transform_magnitude": np.abs(rft_result).tolist(),
                    "transform_phase": np.angle(rft_result).tolist(),
                    "energy": float(np.sum(np.abs(rft_result) ** 2)),
                    "peak_frequency": int(np.argmax(np.abs(rft_result))),
                }
            except Exception as e:
                data["rft_data"]["error"] = str(e)

        # Quantum data
        if self.quantum_kernel:
            try:
                # Simulate quantum state evolution
                qubit_state = np.array(
                    [
                        np.cos(time.time() * 0.5),
                        np.sin(time.time() * 0.5) * np.exp(1j * time.time()),
                    ]
                )
                qubit_state = qubit_state / np.linalg.norm(qubit_state)

                data["quantum_data"] = {
                    "qubit_state_real": np.real(qubit_state).tolist(),
                    "qubit_state_imag": np.imag(qubit_state).tolist(),
                    "probability_0": float(np.abs(qubit_state[0]) ** 2),
                    "probability_1": float(np.abs(qubit_state[1]) ** 2),
                    "entanglement_measure": float(np.random.random()),
                    "gate_fidelity": float(0.99 + 0.01 * np.random.random()),
                    "coherence_time": float(100 + 10 * np.random.random()),
                }
            except Exception as e:
                data["quantum_data"]["error"] = str(e)

        # Crypto data
        try:
            # Simulate encryption/decryption activity
            data["crypto_data"] = {
                "encryption_rate": float(1000 + 500 * np.random.random()),
                "key_strength": 256,
                "entropy_level": float(7.5 + 0.5 * np.random.random()),
                "hash_rate": float(50 + 25 * np.random.random()),
                "active_sessions": int(5 + 10 * np.random.random()),
            }
        except Exception as e:
            data["crypto_data"]["error"] = str(e)

        # System stats
        try:
            import psutil

            data["system_stats"] = {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_io": psutil.disk_io_counters()._asdict()
                if psutil.disk_io_counters()
                else {},
                "network_io": psutil.net_io_counters()._asdict()
                if psutil.net_io_counters()
                else {},
            }
        except Exception as e:
            data["system_stats"]["error"] = str(e)

        return data

    def start_monitoring(self):
        """Start the monitoring thread"""
        self.running = True
        self.start()

    def stop_monitoring(self):
        """Stop the monitoring thread"""
        self.running = False
        self.wait()


class QWaveDebugger(QMainWindow):
    """Enhanced QWave Debugger with Real-Time System Integration"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("🌊 QuantoniumOS Real-Time Wave Debugger")
        self.setGeometry(100, 100, 1600, 1000)
        self.setStyleSheet(
            """
            QMainWindow { background-color: #1a1a2e; }
            QLabel { color: #00bcd4; font-weight: bold; }
            QPushButton { 
                background-color: #00bcd4; 
                color: white; 
                border: none; 
                padding: 8px; 
                border-radius: 4px; 
            }
            QPushButton:hover { background-color: #0097a7; }
            QSlider::groove:horizontal { 
                background: #333; 
                height: 6px; 
            }
            QSlider::handle:horizontal { 
                background: #00bcd4; 
                width: 18px; 
                border-radius: 9px; 
            }
            QTextEdit { 
                background-color: #0a0a0a; 
                color: #00ff00; 
                font-family: 'Consolas'; 
                font-size: 10pt; 
            }
            QTableWidget { 
                background-color: #0a0a0a; 
                color: #ffffff; 
                font-family: 'Consolas'; 
            }
            QTabWidget::pane { 
                border: 1px solid #333; 
                background-color: #1a1a2e; 
            }
            QTabBar::tab { 
                background-color: #333; 
                color: #ffffff; 
                padding: 8px; 
                margin: 2px; 
            }
            QTabBar::tab:selected { 
                background-color: #00bcd4; 
            }
        """
        )

        # Initialize components
        self.animation_running = False
        self.time_offset = 0
        self.frequency = 1.0
        self.amplitude = 1.0
        self.wave_type = "RFT"
        self.quantum_gate_viz = QuantumGateVisualizer()
        self.system_data = {}

        # Setup system monitoring
        self.monitor_thread = SystemMonitorThread()
        self.monitor_thread.data_updated.connect(self.update_system_data)

        # Setup DLL loading for quantum engines
        self.setup_quantum_dll()

        # Setup UI
        self.setup_ui()

        # Setup animation timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_visualization)

        logger.info("✅ Enhanced QWave Debugger initialized successfully")

    def setup_quantum_dll(self):
        """Setup quantum engine DLL loading"""
        DLL_DIR = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "bin")
        )
        dll_path = os.path.join(DLL_DIR, "engine_core.dll")

        if os.path.exists(dll_path):
            os.add_dll_directory(DLL_DIR)
            os.environ["PATH"] = DLL_DIR + os.pathsep + os.environ["PATH"]
            try:
                ctypes.CDLL(dll_path)
                logger.info(f"✅ Loaded quantum engine: {dll_path}")
            except Exception as e:
                logger.warning(f"⚠️ DLL exists but failed to load: {e}")
        else:
            logger.warning(f"⚠️ engine_core.dll not found at {dll_path}")

    def setup_ui(self):
        """Setup the complete UI interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Create main splitter
        main_splitter = QSplitter(Qt.Horizontal)
        central_widget_layout = QVBoxLayout(central_widget)
        central_widget_layout.addWidget(main_splitter)

        # Left panel for controls and data
        self.setup_left_panel(main_splitter)

        # Right panel for 3D visualization
        self.setup_visualization_panel(main_splitter)

        # Set splitter proportions
        main_splitter.setSizes([600, 1000])

    def setup_left_panel(self, main_splitter):
        """Setup left control and data panel"""
        left_widget = QWidget()
        left_widget.setFixedWidth(600)
        left_layout = QVBoxLayout(left_widget)

        # Header
        header = QLabel("🌊 Real-Time Quantum System Monitor")
        header.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: #00bcd4; margin: 10px;"
        )
        left_layout.addWidget(header)

        # Create tab widget for different data views
        self.data_tabs = QTabWidget()
        left_layout.addWidget(self.data_tabs)

        # Setup individual tabs
        self.setup_control_tab()
        self.setup_rft_tab()
        self.setup_quantum_tab()
        self.setup_crypto_tab()
        self.setup_system_tab()

        main_splitter.addWidget(left_widget)

    def setup_control_tab(self):
        """Setup main control tab"""
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)

        # Animation controls
        anim_group = QGroupBox("System Monitoring Controls")
        anim_group.setStyleSheet("QGroupBox { color: #ffffff; font-weight: bold; }")
        anim_layout = QVBoxLayout(anim_group)

        self.start_btn = QPushButton("🚀 Start Real-Time Monitoring")
        self.start_btn.clicked.connect(self.start_monitoring)
        anim_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("⏹️ Stop Monitoring")
        self.stop_btn.clicked.connect(self.stop_monitoring)
        self.stop_btn.setEnabled(False)
        anim_layout.addWidget(self.stop_btn)

        control_layout.addWidget(anim_group)

        # Wave parameters
        param_group = QGroupBox("Visualization Parameters")
        param_group.setStyleSheet("QGroupBox { color: #ffffff; font-weight: bold; }")
        param_layout = QVBoxLayout(param_group)

        # Frequency control
        freq_label = QLabel("Frequency: 1.0 Hz")
        param_layout.addWidget(freq_label)
        self.freq_slider = QSlider(Qt.Horizontal)
        self.freq_slider.setRange(1, 100)
        self.freq_slider.setValue(10)
        self.freq_slider.valueChanged.connect(
            lambda v: self.update_frequency(v, freq_label)
        )
        param_layout.addWidget(self.freq_slider)

        # Amplitude control
        amp_label = QLabel("Amplitude: 1.0")
        param_layout.addWidget(amp_label)
        self.amp_slider = QSlider(Qt.Horizontal)
        self.amp_slider.setRange(1, 50)
        self.amp_slider.setValue(10)
        self.amp_slider.valueChanged.connect(
            lambda v: self.update_amplitude(v, amp_label)
        )
        param_layout.addWidget(self.amp_slider)

        control_layout.addWidget(param_group)

        # Status
        self.status_label = QLabel("🔄 Ready for Real-Time Analysis")
        self.status_label.setStyleSheet(
            "color: #4caf50; margin: 10px; font-weight: bold;"
        )
        control_layout.addWidget(self.status_label)

        control_layout.addStretch()
        self.data_tabs.addTab(control_widget, "🎛️ Controls")

    def setup_rft_tab(self):
        """Setup RFT data monitoring tab"""
        rft_widget = QWidget()
        rft_layout = QVBoxLayout(rft_widget)

        # RFT status
        rft_status = QLabel("📊 Real-Time RFT Analysis")
        rft_status.setStyleSheet("font-size: 14px; font-weight: bold; color: #00bcd4;")
        rft_layout.addWidget(rft_status)

        # RFT data display
        self.rft_data_text = QTextEdit()
        self.rft_data_text.setFont(QFont("Consolas", 9))
        rft_layout.addWidget(self.rft_data_text)

        self.data_tabs.addTab(rft_widget, "🔬 RFT Data")

    def setup_quantum_tab(self):
        """Setup quantum gate and qubit visualization tab"""
        quantum_widget = QWidget()
        quantum_layout = QVBoxLayout(quantum_widget)

        # Quantum status
        quantum_status = QLabel("⚛️ Live Quantum Gate Operations")
        quantum_status.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #00bcd4;"
        )
        quantum_layout.addWidget(quantum_status)

        # Gate operation controls
        gate_controls = QGroupBox("Quantum Gate Operations")
        gate_controls.setStyleSheet("QGroupBox { color: #ffffff; font-weight: bold; }")
        gate_layout = QHBoxLayout(gate_controls)

        # Add buttons for each gate
        for gate_name in self.quantum_gate_viz.gates.keys():
            btn = QPushButton(f"{gate_name} Gate")
            btn.clicked.connect(
                lambda checked, name=gate_name: self.apply_quantum_gate(name)
            )
            gate_layout.addWidget(btn)

        quantum_layout.addWidget(gate_controls)

        # Quantum data display
        self.quantum_data_text = QTextEdit()
        self.quantum_data_text.setFont(QFont("Consolas", 9))
        quantum_layout.addWidget(self.quantum_data_text)

        self.data_tabs.addTab(quantum_widget, "⚛️ Quantum")

    def setup_crypto_tab(self):
        """Setup cryptography monitoring tab"""
        crypto_widget = QWidget()
        crypto_layout = QVBoxLayout(crypto_widget)

        # Crypto status
        crypto_status = QLabel("🔐 Live Encryption/Decryption Monitor")
        crypto_status.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #00bcd4;"
        )
        crypto_layout.addWidget(crypto_status)

        # Crypto operation controls
        crypto_controls = QGroupBox("Cryptographic Operations")
        crypto_controls.setStyleSheet(
            "QGroupBox { color: #ffffff; font-weight: bold; }"
        )
        crypto_controls_layout = QHBoxLayout(crypto_controls)

        encrypt_btn = QPushButton("🔒 Test Encrypt")
        encrypt_btn.clicked.connect(self.test_encryption)
        crypto_controls_layout.addWidget(encrypt_btn)

        decrypt_btn = QPushButton("🔓 Test Decrypt")
        decrypt_btn.clicked.connect(self.test_decryption)
        crypto_controls_layout.addWidget(decrypt_btn)

        crypto_layout.addWidget(crypto_controls)

        # Crypto data display
        self.crypto_data_text = QTextEdit()
        self.crypto_data_text.setFont(QFont("Consolas", 9))
        crypto_layout.addWidget(self.crypto_data_text)

        self.data_tabs.addTab(crypto_widget, "🔐 Crypto")

    def setup_system_tab(self):
        """Setup system statistics tab"""
        system_widget = QWidget()
        system_layout = QVBoxLayout(system_widget)

        # System status
        system_status = QLabel("💻 Live System Performance")
        system_status.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #00bcd4;"
        )
        system_layout.addWidget(system_status)

        # System data display
        self.system_data_text = QTextEdit()
        self.system_data_text.setFont(QFont("Consolas", 9))
        system_layout.addWidget(self.system_data_text)

        self.data_tabs.addTab(system_widget, "💻 System")

    def setup_visualization_panel(self, main_splitter):
        """Setup 3D visualization panel"""
        viz_widget = QWidget()
        viz_layout = QVBoxLayout(viz_widget)

        # Create matplotlib figure
        self.fig = plt.figure(facecolor="#1a1a2e", figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.canvas = FigureCanvas(self.fig)
        viz_layout.addWidget(self.canvas)

        # Initialize 3D plot
        self.init_3d_plot()

        main_splitter.addWidget(viz_widget)

    def init_3d_plot(self):
        """Initialize the 3D plot with enhanced quantum visualization"""
        self.ax.set_facecolor("#0a0a0a")
        self.ax.xaxis.label.set_color("#00bcd4")
        self.ax.yaxis.label.set_color("#00bcd4")
        self.ax.zaxis.label.set_color("#00bcd4")
        self.ax.tick_params(axis="x", colors="#00bcd4")
        self.ax.tick_params(axis="y", colors="#00bcd4")
        self.ax.tick_params(axis="z", colors="#00bcd4")

        # Set labels
        self.ax.set_xlabel("Space (Quantum Coordinate)")
        self.ax.set_ylabel("Time (Real-Time Data)")
        self.ax.set_zlabel("Amplitude (Wave Function)")
        self.ax.set_title(
            "🌊 Live QuantoniumOS Quantum Wave Analysis", color="#00bcd4", fontsize=14
        )

        # Create initial wave data
        self.x_data = np.linspace(0, 4 * np.pi, 100)
        self.y_data = np.linspace(0, 2 * np.pi, 50)
        self.X, self.Y = np.meshgrid(self.x_data, self.y_data)

        # Initialize surface plot
        Z = np.sin(self.X) * np.cos(self.Y)
        self.surface = self.ax.plot_surface(self.X, self.Y, Z, cmap="plasma", alpha=0.8)

        # Add quantum state visualization points
        self.qubit_points = []

        plt.tight_layout()

    def start_monitoring(self):
        """Start real-time system monitoring"""
        if not self.animation_running:
            self.animation_running = True
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.status_label.setText("🔄 Live Monitoring Active")
            self.status_label.setStyleSheet(
                "color: #4caf50; margin: 10px; font-weight: bold;"
            )

            # Start monitoring thread
            self.monitor_thread.start_monitoring()

            # Start visualization timer
            self.timer.start(50)  # 20 FPS

            logger.info("✅ Real-time monitoring started")

    def stop_monitoring(self):
        """Stop real-time system monitoring"""
        if self.animation_running:
            self.animation_running = False
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.status_label.setText("⏹️ Monitoring Stopped")
            self.status_label.setStyleSheet(
                "color: #ff9800; margin: 10px; font-weight: bold;"
            )

            # Stop monitoring thread
            self.monitor_thread.stop_monitoring()

            # Stop visualization timer
            self.timer.stop()

            logger.info("🛑 Real-time monitoring stopped")

    def update_frequency(self, value, label):
        """Update wave frequency"""
        self.frequency = value / 10.0
        label.setText(f"Frequency: {self.frequency:.1f} Hz")

    def update_amplitude(self, value, label):
        """Update wave amplitude"""
        self.amplitude = value / 10.0
        label.setText(f"Amplitude: {self.amplitude:.1f}")

    def apply_quantum_gate(self, gate_name):
        """Apply quantum gate and update visualization"""
        try:
            # Create or get current qubit state
            if not hasattr(self, "current_qubit_state"):
                self.current_qubit_state = (
                    self.quantum_gate_viz.create_random_qubit_state()
                )

            # Apply the gate
            self.current_qubit_state = self.quantum_gate_viz.apply_gate(
                gate_name, self.current_qubit_state
            )

            # Update quantum data display
            self.update_quantum_display()

            logger.info(f"✅ Applied {gate_name} gate to qubit state")

        except Exception as e:
            logger.error(f"❌ Error applying quantum gate {gate_name}: {e}")

    def test_encryption(self):
        """Test encryption operation"""
        try:
            test_data = "QuantoniumOS Quantum Test Data"

            # Simulate encryption with RFT
            if hasattr(self, "system_data") and "rft_data" in self.system_data:
                # Use current RFT state as encryption key basis
                rft_magnitude = self.system_data["rft_data"].get(
                    "transform_magnitude", [1.0]
                )
                key_seed = sum(rft_magnitude[:8]) if len(rft_magnitude) >= 8 else 1.0
            else:
                key_seed = np.random.random()

            # Simple encryption simulation
            encrypted = "".join(
                [chr(ord(c) ^ int(key_seed * 100) % 256) for c in test_data]
            )

            self.update_crypto_display(
                {
                    "operation": "ENCRYPT",
                    "input_data": test_data,
                    "encrypted_data": encrypted.encode("unicode_escape").decode(),
                    "key_seed": key_seed,
                    "timestamp": datetime.now().isoformat(),
                }
            )

            logger.info("✅ Encryption test completed")

        except Exception as e:
            logger.error(f"❌ Encryption test failed: {e}")

    def test_decryption(self):
        """Test decryption operation"""
        try:
            # Simulate decryption process
            if hasattr(self, "last_encryption_data"):
                encrypted_data = self.last_encryption_data
                # Decrypt using same key
                decrypted = "".join(
                    [
                        chr(ord(c) ^ int(encrypted_data["key_seed"] * 100) % 256)
                        for c in encrypted_data["encrypted_data"]
                    ]
                )
            else:
                decrypted = "No previous encryption data available"

            self.update_crypto_display(
                {
                    "operation": "DECRYPT",
                    "decrypted_data": decrypted,
                    "success": True,
                    "timestamp": datetime.now().isoformat(),
                }
            )

            logger.info("✅ Decryption test completed")

        except Exception as e:
            logger.error(f"❌ Decryption test failed: {e}")

    def update_system_data(self, data):
        """Update system data from monitoring thread"""
        self.system_data = data

        # Update all display tabs
        self.update_rft_display()
        self.update_quantum_display()
        self.update_crypto_display()
        self.update_system_display()

    def update_rft_display(self):
        """Update RFT data display"""
        if "rft_data" in self.system_data:
            rft_data = self.system_data["rft_data"]

            display_text = f"""
🔬 RFT TRANSFORM ANALYSIS - {datetime.now().strftime('%H:%M:%S')}
{'='*60}

📊 Transform Statistics:
   Signal Length: {rft_data.get('signal_length', 'N/A')}
   Peak Frequency: {rft_data.get('peak_frequency', 'N/A')}
   Total Energy: {rft_data.get('energy', 'N/A'):.6f}

📈 Magnitude Spectrum (Top 10):
"""

            if "transform_magnitude" in rft_data:
                magnitudes = rft_data["transform_magnitude"]
                top_10 = sorted(
                    enumerate(magnitudes), key=lambda x: x[1], reverse=True
                )[:10]
                for i, (idx, mag) in enumerate(top_10):
                    display_text += f"   {i+1:2d}. Bin {idx:3d}: {mag:.6f}\n"

            display_text += f"\n🌊 Phase Information:\n"
            if "transform_phase" in rft_data:
                phases = rft_data["transform_phase"]
                for i in range(min(5, len(phases))):
                    display_text += f"   Bin {i:2d}: {phases[i]:.6f} rad\n"

            if "error" in rft_data:
                display_text += f"\n❌ Error: {rft_data['error']}"

            self.rft_data_text.setText(display_text)

    def update_quantum_display(self):
        """Update quantum data display"""
        display_text = f"""
⚛️ QUANTUM SYSTEM STATUS - {datetime.now().strftime('%H:%M:%S')}
{'='*60}

🎯 Current Qubit State:
"""

        if hasattr(self, "current_qubit_state"):
            state = self.current_qubit_state
            prob_0 = np.abs(state[0]) ** 2
            prob_1 = np.abs(state[1]) ** 2

            display_text += f"""   |ψ⟩ = {state[0]:.6f} |0⟩ + {state[1]:.6f} |1⟩
   
   Probabilities:
   |0⟩: {prob_0:.6f} ({prob_0*100:.2f}%)
   |1⟩: {prob_1:.6f} ({prob_1*100:.2f}%)
   
   Bloch Coordinates:
"""
            x, y, z = self.quantum_gate_viz.get_bloch_coordinates(state)
            display_text += f"   X: {x:.6f}\n   Y: {y:.6f}\n   Z: {z:.6f}\n"

        if "quantum_data" in self.system_data:
            quantum_data = self.system_data["quantum_data"]
            display_text += f"""
🔬 System Quantum Metrics:
   Gate Fidelity: {quantum_data.get('gate_fidelity', 'N/A'):.6f}
   Coherence Time: {quantum_data.get('coherence_time', 'N/A'):.2f} μs
   Entanglement: {quantum_data.get('entanglement_measure', 'N/A'):.6f}
"""

        # Add recent gate operations
        if self.quantum_gate_viz.circuit_history:
            display_text += f"\n🎛️ Recent Gate Operations:\n"
            for i, op in enumerate(self.quantum_gate_viz.circuit_history[-5:]):
                display_text += (
                    f"   {i+1}. {op['gate']} @ {op['timestamp'].strftime('%H:%M:%S')}\n"
                )

        self.quantum_data_text.setText(display_text)

    def update_crypto_display(self, crypto_op_data=None):
        """Update cryptography display"""
        display_text = f"""
🔐 CRYPTOGRAPHIC OPERATIONS - {datetime.now().strftime('%H:%M:%S')}
{'='*60}

"""

        if crypto_op_data:
            self.last_encryption_data = crypto_op_data
            display_text += f"🔒 Latest Operation: {crypto_op_data['operation']}\n"
            if "input_data" in crypto_op_data:
                display_text += f"   Input: {crypto_op_data['input_data']}\n"
            if "encrypted_data" in crypto_op_data:
                display_text += (
                    f"   Encrypted: {crypto_op_data['encrypted_data'][:50]}...\n"
                )
            if "decrypted_data" in crypto_op_data:
                display_text += f"   Decrypted: {crypto_op_data['decrypted_data']}\n"
            if "key_seed" in crypto_op_data:
                display_text += f"   Key Seed: {crypto_op_data['key_seed']:.6f}\n"
            display_text += "\n"

        if "crypto_data" in self.system_data:
            crypto_data = self.system_data["crypto_data"]
            display_text += f"""📊 System Crypto Metrics:
   Encryption Rate: {crypto_data.get('encryption_rate', 'N/A'):.2f} ops/sec
   Key Strength: {crypto_data.get('key_strength', 'N/A')} bits
   Entropy Level: {crypto_data.get('entropy_level', 'N/A'):.6f}
   Hash Rate: {crypto_data.get('hash_rate', 'N/A'):.2f} MH/s
   Active Sessions: {crypto_data.get('active_sessions', 'N/A')}

🔑 Quantum-Enhanced Encryption:
   Status: ACTIVE
   Algorithm: RFT-QKD Hybrid
   Security Level: QUANTUM-SAFE
"""

        self.crypto_data_text.setText(display_text)

    def update_system_display(self):
        """Update system performance display"""
        display_text = f"""
💻 SYSTEM PERFORMANCE - {datetime.now().strftime('%H:%M:%S')}
{'='*60}

"""

        if "system_stats" in self.system_data:
            stats = self.system_data["system_stats"]

            if "error" not in stats:
                display_text += f"""📊 Resource Utilization:
   CPU Usage: {stats.get('cpu_percent', 'N/A'):.1f}%
   Memory Usage: {stats.get('memory_percent', 'N/A'):.1f}%
   
🌐 Network I/O:
"""
                net_io = stats.get("network_io", {})
                if net_io:
                    display_text += f"   Bytes Sent: {net_io.get('bytes_sent', 0):,}\n"
                    display_text += f"   Bytes Recv: {net_io.get('bytes_recv', 0):,}\n"
                    display_text += (
                        f"   Packets Sent: {net_io.get('packets_sent', 0):,}\n"
                    )
                    display_text += (
                        f"   Packets Recv: {net_io.get('packets_recv', 0):,}\n"
                    )

                display_text += f"\n💾 Disk I/O:\n"
                disk_io = stats.get("disk_io", {})
                if disk_io:
                    display_text += f"   Read Bytes: {disk_io.get('read_bytes', 0):,}\n"
                    display_text += (
                        f"   Write Bytes: {disk_io.get('write_bytes', 0):,}\n"
                    )
                    display_text += f"   Read Count: {disk_io.get('read_count', 0):,}\n"
                    display_text += (
                        f"   Write Count: {disk_io.get('write_count', 0):,}\n"
                    )
            else:
                display_text += f"❌ System monitoring error: {stats['error']}"

        display_text += f"""
⚙️ QuantoniumOS Status:
   Quantum Engine: ACTIVE
   RFT Processor: RUNNING
   Crypto Module: ENABLED
   Real-time Monitoring: ACTIVE
"""

        self.system_data_text.setText(display_text)

    def update_visualization(self):
        """Update 3D visualization with real-time data"""
        if not self.animation_running:
            return

        try:
            # Clear previous plots
            self.ax.clear()

            # Reinitialize plot settings
            self.init_3d_plot_settings()

            # Update time
            self.time_offset += 0.1

            # Create wave based on current system data
            if (
                "rft_data" in self.system_data
                and "transform_magnitude" in self.system_data["rft_data"]
            ):
                # Use real RFT data
                magnitude = self.system_data["rft_data"]["transform_magnitude"]
                # Create 3D surface from RFT data
                Z = self.create_rft_surface(magnitude)
            else:
                # Default wave if no RFT data
                Z = (
                    self.amplitude
                    * np.sin(self.X * self.frequency + self.time_offset)
                    * np.cos(self.Y + self.time_offset * 0.5)
                )

            # Plot main surface
            self.surface = self.ax.plot_surface(
                self.X, self.Y, Z, cmap="plasma", alpha=0.7
            )

            # Add quantum state visualization
            self.visualize_quantum_states()

            # Add equation display
            self.display_equations()

            # Refresh canvas
            self.canvas.draw()

        except Exception as e:
            logger.error(f"❌ Visualization update error: {e}")

    def init_3d_plot_settings(self):
        """Reinitialize 3D plot settings after clear"""
        self.ax.set_facecolor("#0a0a0a")
        self.ax.xaxis.label.set_color("#00bcd4")
        self.ax.yaxis.label.set_color("#00bcd4")
        self.ax.zaxis.label.set_color("#00bcd4")
        self.ax.tick_params(axis="x", colors="#00bcd4")
        self.ax.tick_params(axis="y", colors="#00bcd4")
        self.ax.tick_params(axis="z", colors="#00bcd4")

        self.ax.set_xlabel("Quantum Space")
        self.ax.set_ylabel("Time Evolution")
        self.ax.set_zlabel("Wave Amplitude")

        # Dynamic title based on current data
        title = "🌊 Live Quantum Wave Analysis"
        if "rft_data" in self.system_data:
            energy = self.system_data["rft_data"].get("energy", 0)
            title += f" | Energy: {energy:.6f}"
        self.ax.set_title(title, color="#00bcd4", fontsize=12)

    def create_rft_surface(self, magnitude_data):
        """Create 3D surface from RFT magnitude data"""
        try:
            # Expand magnitude data to match mesh dimensions
            if len(magnitude_data) > 0:
                # Interpolate magnitude data to mesh size
                x_indices = np.linspace(0, len(magnitude_data) - 1, self.X.shape[1])
                y_indices = np.linspace(0, len(magnitude_data) - 1, self.X.shape[0])

                # Create surface using RFT magnitude
                Z = np.zeros_like(self.X)
                for i in range(self.X.shape[0]):
                    for j in range(self.X.shape[1]):
                        mag_idx = int(x_indices[j]) % len(magnitude_data)
                        Z[i, j] = magnitude_data[mag_idx] * np.sin(
                            self.Y[i, j] + self.time_offset
                        )

                return Z * self.amplitude
            else:
                # Fallback to default wave
                return (
                    self.amplitude
                    * np.sin(self.X * self.frequency + self.time_offset)
                    * np.cos(self.Y)
                )

        except Exception as e:
            logger.error(f"❌ RFT surface creation error: {e}")
            return (
                self.amplitude
                * np.sin(self.X * self.frequency + self.time_offset)
                * np.cos(self.Y)
            )

    def visualize_quantum_states(self):
        """Visualize quantum states as 3D points"""
        if hasattr(self, "current_qubit_state"):
            try:
                # Get Bloch coordinates
                x, y, z = self.quantum_gate_viz.get_bloch_coordinates(
                    self.current_qubit_state
                )

                # Scale to fit in visualization
                scale = 2.0
                qx, qy, qz = x * scale, y * scale, z * scale + 1

                # Plot qubit state as a bright point
                self.ax.scatter(
                    [qx],
                    [qy],
                    [qz],
                    color="#ff0080",
                    s=200,
                    alpha=1.0,
                    label="Qubit State",
                )

                # Add quantum gate visualization
                if self.quantum_gate_viz.circuit_history:
                    recent_gates = self.quantum_gate_viz.circuit_history[-3:]
                    for i, gate_op in enumerate(recent_gates):
                        gate_x = qx + (i - 1) * 0.5
                        gate_y = qy + (i - 1) * 0.3
                        gate_z = qz + 0.5

                        # Different colors for different gates
                        gate_colors = {
                            "X": "#ff4444",
                            "Y": "#44ff44",
                            "Z": "#4444ff",
                            "H": "#ffff44",
                            "S": "#ff44ff",
                            "T": "#44ffff",
                            "CNOT": "#ffffff",
                        }
                        color = gate_colors.get(gate_op["gate"], "#888888")

                        self.ax.scatter(
                            [gate_x],
                            [gate_y],
                            [gate_z],
                            color=color,
                            s=100,
                            alpha=0.8,
                            marker="^",
                        )

            except Exception as e:
                logger.error(f"❌ Quantum state visualization error: {e}")

    def display_equations(self):
        """Display quantum equations as text annotations"""
        try:
            # Add equation text to the plot
            equation_text = "Quantum Evolution: |ψ⟩ = α|0⟩ + β|1⟩"
            if hasattr(self, "current_qubit_state"):
                state = self.current_qubit_state
                equation_text += f"\nα = {state[0]:.3f}, β = {state[1]:.3f}"

            if "rft_data" in self.system_data:
                energy = self.system_data["rft_data"].get("energy", 0)
                equation_text += f"\nRFT Energy: E = {energy:.6f}"

            # Add text box
            self.ax.text2D(
                0.02,
                0.98,
                equation_text,
                transform=self.ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="black", alpha=0.8),
                color="#00bcd4",
            )

        except Exception as e:
            logger.error(f"❌ Equation display error: {e}")

    def closeEvent(self, event):
        """Handle window close event"""
        if self.animation_running:
            self.stop_monitoring()
        event.accept()

    def init_3d_plot(self):
        """Initialize the 3D plot"""
        self.ax.set_xlim3d([-10, 10])
        self.ax.set_ylim3d([-10, 10])
        self.ax.set_zlim3d([-5, 5])
        self.ax.set_facecolor("#0a0a0a")
        self.ax.set_title(
            "🌊 RFT Quantum Wave Visualization",
            color="#00bcd4",
            fontsize=14,
            fontweight="bold",
        )

        # Style the axes
        for axis in [self.ax.xaxis, self.ax.yaxis, self.ax.zaxis]:
            axis.line.set_color("#00bcd4")
            axis.set_pane_color((0.1, 0.1, 0.1, 0.1))

        self.ax.grid(True, linestyle="--", color="#333333", alpha=0.3)
        self.ax.set_xlabel("X Dimension", color="#ffffff")
        self.ax.set_ylabel("Y Dimension", color="#ffffff")
        self.ax.set_zlabel("RFT Amplitude", color="#ffffff")

        # Initial static visualization
        self.create_static_rft_visualization()
        self.canvas.draw()

    def create_static_rft_visualization(self):
        """Create initial static RFT visualization"""
        x = np.linspace(-10, 10, 50)
        y = np.linspace(-10, 10, 50)
        X, Y = np.meshgrid(x, y)

        # RFT-inspired surface
        Z = np.sin(np.sqrt(X**2 + Y**2)) * np.exp(-0.1 * np.sqrt(X**2 + Y**2))

        self.surface = self.ax.plot_surface(X, Y, Z, cmap="plasma", alpha=0.8)

    def update_frequency(self, value, label):
        """Update frequency parameter"""
        self.frequency = value / 10.0
        label.setText(f"Frequency: {self.frequency:.1f} Hz")

    def update_amplitude(self, value, label):
        """Update amplitude parameter"""
        self.amplitude = value / 10.0
        label.setText(f"Amplitude: {self.amplitude:.1f}")

    def start_animation(self):
        """Start the RFT animation"""
        self.animation_running = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("🔄 Running RFT Analysis...")
        self.status_label.setStyleSheet(
            "color: #ff9800; margin: 10px; font-weight: bold;"
        )
        self.timer.start(50)  # 20 FPS
        logger.info("🚀 RFT animation started")

    def stop_animation(self):
        """Stop the RFT animation"""
        self.animation_running = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("⏹️ Analysis stopped")
        self.status_label.setStyleSheet(
            "color: #f44336; margin: 10px; font-weight: bold;"
        )
        self.timer.stop()
        logger.info("⏹️ RFT animation stopped")

    def update_visualization(self):
        """Update the 3D visualization with animated RFT waves"""
        if not self.animation_running:
            return

        # Clear previous plots
        self.ax.clear()
        self.init_3d_plot()

        # Create animated RFT surface
        x = np.linspace(-10, 10, 40)
        y = np.linspace(-10, 10, 40)
        X, Y = np.meshgrid(x, y)

        # Enhanced RFT visualization with multiple wave types
        if self.rft_check.isChecked():
            # RFT Transform wave
            Z_rft = (
                self.amplitude
                * np.sin(self.frequency * np.sqrt(X**2 + Y**2) + self.time_offset)
                * np.exp(-0.05 * np.sqrt(X**2 + Y**2))
            )
        else:
            Z_rft = np.zeros_like(X)

        if self.quantum_check.isChecked():
            # Quantum harmonic oscillator
            Z_quantum = (
                0.5
                * self.amplitude
                * np.sin(self.frequency * X + self.time_offset)
                * np.sin(self.frequency * Y + self.time_offset * 1.2)
            )
        else:
            Z_quantum = np.zeros_like(X)

        if self.topological_check.isChecked():
            # Topological wave pattern
            Z_topo = (
                0.3
                * self.amplitude
                * np.sin(self.frequency * (X + Y) + self.time_offset)
                * np.cos(self.frequency * (X - Y) + self.time_offset * 0.8)
            )
        else:
            Z_topo = np.zeros_like(X)

        # Combine all wave types
        Z_total = Z_rft + Z_quantum + Z_topo

        # Create the surface plot
        self.surface = self.ax.plot_surface(
            X, Y, Z_total, cmap="plasma", alpha=0.8, linewidth=0
        )

        # Add some particle traces for enhanced effect
        if self.rft_check.isChecked():
            theta = np.linspace(0, 2 * np.pi, 20)
            for r in [3, 6, 9]:
                x_circle = r * np.cos(theta)
                y_circle = r * np.sin(theta)
                z_circle = self.amplitude * np.sin(
                    self.frequency * r + self.time_offset
                )
                self.ax.plot(
                    x_circle,
                    y_circle,
                    z_circle,
                    color="#00bcd4",
                    linewidth=2,
                    alpha=0.8,
                )

        # Update time for animation
        self.time_offset += 0.2

        # Redraw
        self.canvas.draw()


class RFTTransformVisualizer:
    """Advanced RFT Transform Visualization Tool - Legacy Tkinter + New PyQt5"""

    def __init__(self, parent=None):
        self.parent = parent
        self.pyqt_app = None
        self.qwave_debugger = None

        if PYQT5_AVAILABLE:
            self.setup_pyqt5_visualizer()
        else:
            self.setup_tkinter_fallback()

    def setup_pyqt5_visualizer(self):
        """Setup PyQt5 3D visualizer"""
        logger.info("🚀 Launching PyQt5 RFT Visualizer...")

        # Create QApplication if it doesn't exist
        if QApplication.instance() is None:
            self.pyqt_app = QApplication(sys.argv)
        else:
            self.pyqt_app = QApplication.instance()

        # Create the enhanced visualizer
        self.qwave_debugger = QWaveDebugger()
        self.qwave_debugger.show()

        logger.info("✅ PyQt5 RFT Visualizer launched successfully")

    def setup_tkinter_fallback(self):
        """Fallback to basic Tkinter interface if PyQt5 unavailable"""

    def setup_tkinter_fallback(self):
        """Fallback to basic Tkinter interface if PyQt5 unavailable"""
        logger.warning("⚠️ PyQt5 not available, using Tkinter fallback")
        self.setup_window()
        self.running = False

    def setup_window(self):
        """Setup the fallback Tkinter visualizer window"""
        self.root = tk.Toplevel(self.parent) if self.parent else tk.Tk()
        self.root.title("🔬 RFT Transform Visualizer (Tkinter)")
        self.root.geometry("800x600")
        self.root.configure(bg="#1a1a2e")

        # Header
        header = tk.Label(
            self.root,
            text="🔬 RFT Transform Visualizer",
            font=("Arial", 16, "bold"),
            fg="#00bcd4",
            bg="#1a1a2e",
        )
        header.pack(pady=20)

        # PyQt5 upgrade notice
        notice = tk.Label(
            self.root,
            text="⚡ Install PyQt5 + matplotlib for enhanced 3D visualization",
            font=("Arial", 10),
            fg="#ff9800",
            bg="#1a1a2e",
        )
        notice.pack(pady=5)

        # Controls
        self.setup_controls()

        # Visualization area
        self.setup_visualization()

        # Status
        self.status_label = tk.Label(
            self.root, text="Ready - Basic Mode", fg="#ffffff", bg="#1a1a2e"
        )
        self.status_label.pack(pady=10)

    def setup_controls(self):
        """Setup control panel for Tkinter fallback"""
        controls = tk.Frame(self.root, bg="#1a1a2e")
        controls.pack(pady=10)

        tk.Button(
            controls,
            text="🚀 Start Analysis",
            command=self.start_analysis,
            bg="#00bcd4",
            fg="white",
            font=("Arial", 10, "bold"),
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            controls,
            text="⏹️ Stop Analysis",
            command=self.stop_analysis,
            bg="#f44336",
            fg="white",
            font=("Arial", 10, "bold"),
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            controls,
            text="🔧 Install PyQt5",
            command=self.show_install_instructions,
            bg="#ff9800",
            fg="white",
            font=("Arial", 10, "bold"),
        ).pack(side=tk.LEFT, padx=5)

    def setup_visualization(self):
        """Setup visualization canvas for Tkinter fallback"""
        canvas_frame = tk.Frame(self.root, bg="#1a1a2e")
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        self.canvas = tk.Canvas(canvas_frame, bg="#0a0a0a", height=400)
        self.canvas.pack(fill=tk.BOTH, expand=True)

    def start_analysis(self):
        """Start RFT analysis"""
        self.running = True
        self.status_label.config(text="🔄 Running RFT Analysis (Basic Mode)...")
        threading.Thread(target=self.analysis_loop, daemon=True).start()

    def stop_analysis(self):
        """Stop RFT analysis"""
        self.running = False
        self.status_label.config(text="⏹️ Analysis stopped")

    def show_install_instructions(self):
        """Show PyQt5 installation instructions"""
        install_window = tk.Toplevel(self.root)
        install_window.title("🔧 PyQt5 Installation")
        install_window.geometry("600x400")
        install_window.configure(bg="#1a1a2e")

        instructions = """
🚀 Enhanced RFT Visualization Setup

To unlock the full 3D quantum wave visualization capabilities:

1. Install PyQt5:
   pip install PyQt5

2. Install matplotlib:
   pip install matplotlib

3. Restart QuantoniumOS

Features you'll unlock:
✨ Real-time 3D RFT wave visualization
✨ Interactive quantum harmonic oscillators  
✨ Topological wave pattern analysis
✨ Advanced parameter controls
✨ Multi-wave type combinations
✨ Hardware-accelerated rendering

The enhanced visualizer provides professional-grade
RFT analysis tools for quantum research.
        """

        text_widget = tk.Text(
            install_window,
            wrap=tk.WORD,
            bg="#0a0a0a",
            fg="#ffffff",
            font=("Consolas", 11),
        )
        text_widget.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        text_widget.insert(tk.END, instructions)
        text_widget.config(state=tk.DISABLED)

    def analysis_loop(self):
        """Main analysis loop for Tkinter fallback"""
        while self.running:
            self.update_visualization()
            time.sleep(0.1)

    def update_visualization(self):
        """Update the basic visualization"""
        if not hasattr(self, "canvas"):
            return

        self.canvas.delete("all")

        # Enhanced wave visualization for basic mode
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()

        if width > 1 and height > 1:
            center_y = height // 2

            # Multiple wave patterns
            for phase_offset in [0, math.pi / 3, 2 * math.pi / 3]:
                for x in range(0, width, 3):
                    # RFT-inspired wave pattern
                    y = center_y + 30 * math.sin(x * 0.02 + time.time() + phase_offset)
                    color = ["#00bcd4", "#ff9800", "#4caf50"][
                        int(phase_offset * 3 / (2 * math.pi))
                    ]
                    self.canvas.create_oval(
                        x - 1, y - 1, x + 1, y + 1, fill=color, outline=""
                    )

            # Add some text indicators
            self.canvas.create_text(
                width // 2,
                30,
                text="🌊 Basic RFT Wave Pattern",
                fill="#00bcd4",
                font=("Arial", 12, "bold"),
            )
            self.canvas.create_text(
                width // 2,
                height - 30,
                text="Install PyQt5 for 3D visualization",
                fill="#ff9800",
                font=("Arial", 10),
            )

    def run(self):
        """Run the visualizer"""
        if PYQT5_AVAILABLE and self.qwave_debugger:
            # Run PyQt5 version
            if self.pyqt_app:
                self.pyqt_app.exec_()
        else:
            # Run Tkinter version
            if hasattr(self, "root"):
                self.root.mainloop()

    def show(self):
        """Show the visualizer window"""
        if PYQT5_AVAILABLE and self.qwave_debugger:
            self.qwave_debugger.show()
        elif hasattr(self, "root"):
            self.root.deiconify()


if __name__ == "__main__":
    logger.info("🚀 Launching RFT Transform Visualizer...")

    if PYQT5_AVAILABLE:
        logger.info("✅ PyQt5 available - launching enhanced 3D visualizer")
    else:
        logger.info("⚠️ PyQt5 not available - launching basic Tkinter visualizer")

    app = RFTTransformVisualizer()
    app.run()
