#!/usr/bin/env python3
"""
Quantum Simulator - Scientific Interface
=======================================
Advanced quantum state simulation and visualization
"""

import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                            QWidget, QLabel, QPushButton, QTextEdit, QTabWidget,
                            QSlider, QComboBox, QSpinBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QColor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class QuantumSimulator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Quantum Simulator")
        self.setGeometry(100, 100, 1400, 900)
        self.setup_scientific_ui()
        self.init_quantum_state()
        
    def setup_scientific_ui(self):
        """Setup minimal scientific interface"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #fafafa;
                color: #2c3e50;
                font-family: "SF Pro Display", "Segoe UI";
            }
            QTabWidget::pane {
                border: 1px solid #e0e0e0;
                background-color: #ffffff;
            }
            QTabBar::tab {
                background-color: #f8f9fa;
                color: #2c3e50;
                padding: 12px 20px;
                border: none;
                margin-right: 2px;
                font-weight: 500;
            }
            QTabBar::tab:selected {
                background-color: #8e44ad;
                color: white;
            }
            QPushButton {
                background-color: #8e44ad;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 4px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #7d3c98;
            }
            QSlider::groove:horizontal {
                background: #e0e0e0;
                height: 6px;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #8e44ad;
                width: 18px;
                border-radius: 9px;
                margin: -6px 0;
            }
        """)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Header
        header = QLabel("QUANTUM SIMULATOR")
        header.setStyleSheet("font-size: 20px; font-weight: 600; color: #2c3e50; padding: 20px;")
        layout.addWidget(header)
        
        # Tabs
        tabs = QTabWidget()
        layout.addWidget(tabs)
        
        # Quantum State Tab
        state_tab = self.create_state_tab()
        tabs.addTab(state_tab, "Quantum State")
        
        # Gates Tab
        gates_tab = self.create_gates_tab()
        tabs.addTab(gates_tab, "Quantum Gates")
        
        # Measurements Tab
        measure_tab = self.create_measurement_tab()
        tabs.addTab(measure_tab, "Measurements")
        
    def create_state_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Controls
        controls = QHBoxLayout()
        
        # Qubit count
        qubit_label = QLabel("Qubits:")
        self.qubit_spin = QSpinBox()
        self.qubit_spin.setRange(1, 4)
        self.qubit_spin.setValue(2)
        self.qubit_spin.valueChanged.connect(self.init_quantum_state)
        
        # Initialize button
        init_btn = QPushButton("Initialize State")
        init_btn.clicked.connect(self.init_quantum_state)
        
        # Reset button
        reset_btn = QPushButton("Reset to |0⟩")
        reset_btn.clicked.connect(self.reset_state)
        
        controls.addWidget(qubit_label)
        controls.addWidget(self.qubit_spin)
        controls.addWidget(init_btn)
        controls.addWidget(reset_btn)
        controls.addStretch()
        
        layout.addLayout(controls)
        
        # State display
        self.state_display = QTextEdit()
        self.state_display.setMaximumHeight(200)
        layout.addWidget(self.state_display)
        
        # Plot area
        self.state_figure = Figure(figsize=(12, 6))
        self.state_canvas = FigureCanvas(self.state_figure)
        layout.addWidget(self.state_canvas)
        
        return widget
        
    def create_gates_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Controls
        controls = QHBoxLayout()
        
        # Gate selector
        gate_label = QLabel("Gate:")
        self.gate_combo = QComboBox()
        self.gate_combo.addItems(["Hadamard", "Pauli-X", "Pauli-Y", "Pauli-Z", "Phase", "T-Gate"])
        
        # Target qubit
        target_label = QLabel("Target Qubit:")
        self.target_spin = QSpinBox()
        self.target_spin.setRange(0, 1)
        
        # Apply gate button
        apply_btn = QPushButton("Apply Gate")
        apply_btn.clicked.connect(self.apply_gate)
        
        controls.addWidget(gate_label)
        controls.addWidget(self.gate_combo)
        controls.addWidget(target_label)
        controls.addWidget(self.target_spin)
        controls.addWidget(apply_btn)
        controls.addStretch()
        
        layout.addLayout(controls)
        
        # Circuit display
        self.circuit_display = QTextEdit()
        layout.addWidget(self.circuit_display)
        
        return widget
        
    def create_measurement_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Controls
        controls = QHBoxLayout()
        
        measure_btn = QPushButton("Measure State")
        measure_btn.clicked.connect(self.measure_state)
        
        histogram_btn = QPushButton("Show Histogram")
        histogram_btn.clicked.connect(self.show_histogram)
        
        controls.addWidget(measure_btn)
        controls.addWidget(histogram_btn)
        controls.addStretch()
        
        layout.addLayout(controls)
        
        # Results display
        self.results_display = QTextEdit()
        self.results_display.setMaximumHeight(200)
        layout.addWidget(self.results_display)
        
        # Plot area
        self.measure_figure = Figure(figsize=(12, 6))
        self.measure_canvas = FigureCanvas(self.measure_figure)
        layout.addWidget(self.measure_canvas)
        
        return widget
        
    def init_quantum_state(self):
        """Initialize quantum state"""
        num_qubits = self.qubit_spin.value()
        self.num_qubits = num_qubits
        self.state_vector = np.zeros(2**num_qubits, dtype=complex)
        self.state_vector[0] = 1.0  # |0...0⟩ state
        
        # Update target qubit range
        self.target_spin.setRange(0, num_qubits - 1)
        
        self.update_state_display()
        self.plot_state_amplitudes()
        
        # Initialize circuit
        self.circuit_history = []
        self.circuit_display.setPlainText("Circuit initialized. Apply gates to build quantum circuit.")
        
    def reset_state(self):
        """Reset to |0⟩ state"""
        self.state_vector = np.zeros(2**self.num_qubits, dtype=complex)
        self.state_vector[0] = 1.0
        self.update_state_display()
        self.plot_state_amplitudes()
        
    def update_state_display(self):
        """Update quantum state text display"""
        text = f"Quantum State ({self.num_qubits} qubits):\n"
        text += "=" * 40 + "\n\n"
        
        for i, amplitude in enumerate(self.state_vector):
            if abs(amplitude) > 1e-10:  # Only show non-zero amplitudes
                binary = format(i, f'0{self.num_qubits}b')
                prob = abs(amplitude)**2
                text += f"|{binary}⟩: {amplitude:.4f} (P = {prob:.4f})\n"
        
        self.state_display.setPlainText(text)
        
    def plot_state_amplitudes(self):
        """Plot quantum state amplitudes"""
        self.state_figure.clear()
        ax = self.state_figure.add_subplot(111)
        
        states = [format(i, f'0{self.num_qubits}b') for i in range(len(self.state_vector))]
        amplitudes = np.abs(self.state_vector)
        probabilities = amplitudes**2
        
        x = np.arange(len(states))
        bars = ax.bar(x, probabilities, color='purple', alpha=0.7)
        
        ax.set_xlabel('Quantum States')
        ax.set_ylabel('Probability')
        ax.set_title('Quantum State Probabilities')
        ax.set_xticks(x)
        ax.set_xticklabels([f'|{s}⟩' for s in states], rotation=45)
        ax.grid(True, alpha=0.3)
        
        self.state_canvas.draw()
        
    def apply_gate(self):
        """Apply selected quantum gate"""
        gate_name = self.gate_combo.currentText()
        target = self.target_spin.value()
        
        # Define quantum gates
        gates = {
            "Hadamard": np.array([[1, 1], [1, -1]]) / np.sqrt(2),
            "Pauli-X": np.array([[0, 1], [1, 0]]),
            "Pauli-Y": np.array([[0, -1j], [1j, 0]]),
            "Pauli-Z": np.array([[1, 0], [0, -1]]),
            "Phase": np.array([[1, 0], [0, 1j]]),
            "T-Gate": np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])
        }
        
        gate_matrix = gates[gate_name]
        
        # Apply gate to target qubit (simplified implementation)
        new_state = self.apply_single_qubit_gate(gate_matrix, target)
        self.state_vector = new_state
        
        # Update displays
        self.update_state_display()
        self.plot_state_amplitudes()
        
        # Update circuit history
        self.circuit_history.append(f"{gate_name} on qubit {target}")
        circuit_text = "Quantum Circuit:\n" + "=" * 30 + "\n\n"
        for i, operation in enumerate(self.circuit_history):
            circuit_text += f"Step {i+1}: {operation}\n"
        self.circuit_display.setPlainText(circuit_text)
        
    def apply_single_qubit_gate(self, gate, target_qubit):
        """Apply single qubit gate to state vector"""
        # Simplified implementation for demonstration
        new_state = self.state_vector.copy()
        
        # For each computational basis state
        for i in range(len(self.state_vector)):
            # Extract bit value at target position
            bit_val = (i >> target_qubit) & 1
            
            # Apply gate transformation
            if bit_val == 0:
                # Find corresponding |1⟩ state
                j = i | (1 << target_qubit)
                new_amp_0 = gate[0, 0] * self.state_vector[i] + gate[0, 1] * self.state_vector[j]
                new_amp_1 = gate[1, 0] * self.state_vector[i] + gate[1, 1] * self.state_vector[j]
                new_state[i] = new_amp_0
                new_state[j] = new_amp_1
        
        return new_state
        
    def measure_state(self):
        """Perform quantum measurement"""
        probabilities = np.abs(self.state_vector)**2
        
        # Sample from probability distribution
        outcome = np.random.choice(len(probabilities), p=probabilities)
        
        # Collapse state
        self.state_vector = np.zeros_like(self.state_vector)
        self.state_vector[outcome] = 1.0
        
        # Display result
        binary = format(outcome, f'0{self.num_qubits}b')
        result_text = f"Measurement Result:\n"
        result_text += f"Outcome: |{binary}⟩\n"
        result_text += f"Probability: {probabilities[outcome]:.4f}\n\n"
        result_text += "State collapsed to measured outcome."
        
        self.results_display.setPlainText(result_text)
        self.update_state_display()
        self.plot_state_amplitudes()
        
    def show_histogram(self):
        """Show measurement histogram"""
        # Simulate multiple measurements
        probabilities = np.abs(self.state_vector)**2
        num_shots = 1000
        
        outcomes = np.random.choice(len(probabilities), size=num_shots, p=probabilities)
        
        self.measure_figure.clear()
        ax = self.measure_figure.add_subplot(111)
        
        states = [format(i, f'0{self.num_qubits}b') for i in range(len(self.state_vector))]
        counts = np.bincount(outcomes, minlength=len(states))
        
        x = np.arange(len(states))
        bars = ax.bar(x, counts, color='purple', alpha=0.7)
        
        ax.set_xlabel('Measurement Outcomes')
        ax.set_ylabel('Counts')
        ax.set_title(f'Measurement Histogram ({num_shots} shots)')
        ax.set_xticks(x)
        ax.set_xticklabels([f'|{s}⟩' for s in states], rotation=45)
        ax.grid(True, alpha=0.3)
        
        self.measure_canvas.draw()

def main():
    app = QApplication(sys.argv)
    window = QuantumSimulator()
    window.show()
    return app.exec_()

if __name__ == "__main__":
    sys.exit(main())
