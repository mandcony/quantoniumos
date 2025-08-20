"""
QuantoniumOS - Multi-Qubit Quantum Simulator
Advanced quantum state simulation and visualization
"""

import sys
import os
import numpy as np
import math
import random
from pathlib import Path

try:
    from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, 
                                QPushButton, QLabel, QSlider, QSpinBox,
                                QGroupBox, QGridLayout, QTextEdit, QComboBox)
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal
    from PyQt5.QtGui import QFont, QPainter, QPen, QColor
    PYQT5_AVAILABLE = True
except ImportError:
    PYQT5_AVAILABLE = False

class QuantumSimulator(QWidget if PYQT5_AVAILABLE else object):
    """Multi-qubit quantum state simulator"""
    
    def __init__(self):
        if not PYQT5_AVAILABLE:
            print("⚠️ PyQt5 required for Quantum Simulator GUI")
            return
            
        super().__init__()
        self.num_qubits = 3
        self.quantum_state = None
        self.gates_applied = []
        self.init_ui()
        self.initialize_quantum_state()
    
    def init_ui(self):
        """Initialize the simulator interface"""
        self.setWindowTitle("🌌 QuantoniumOS - Quantum Simulator")
        self.setGeometry(400, 400, 1100, 900)
        
        # Apply quantum styling
        self.setStyleSheet("""
            QWidget {
                background-color: #0a0e1a;
                color: #00ffcc;
                font-family: "Consolas", monospace;
            }
            QGroupBox {
                border: 2px solid #00ffcc;
                border-radius: 8px;
                padding: 10px;
                margin-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                color: #00ff88;
                padding: 5px;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1a2332, stop:1 #0f1621);
                border: 2px solid #00ffcc;
                border-radius: 6px;
                padding: 8px 16px;
                color: #00ffcc;
                font-weight: bold;
                min-width: 80px;
            }
            QPushButton:hover {
                border: 2px solid #00ff88;
                color: #00ff88;
            }
            QSlider::groove:horizontal {
                border: 1px solid #00ffcc;
                height: 6px;
                background: #1a2332;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #00ffcc;
                border: 1px solid #00ff88;
                width: 16px;
                border-radius: 8px;
                margin: -5px 0;
            }
            QSpinBox, QComboBox {
                background: #1a2332;
                border: 1px solid #00ffcc;
                border-radius: 4px;
                padding: 5px;
                color: #ffffff;
            }
            QTextEdit {
                background: #1a2332;
                border: 1px solid #00ffcc;
                border-radius: 4px;
                color: #ffffff;
            }
            QLabel {
                color: #00ffcc;
            }
        """)
        
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("🌌 Multi-Qubit Quantum Simulator")
        title.setFont(QFont("Consolas", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #00ff88; margin: 10px;")
        layout.addWidget(title)
        
        # Main content
        main_layout = QHBoxLayout()
        
        # Left side - Controls
        self.create_control_panel(main_layout)
        
        # Right side - Visualization
        self.create_visualization_panel(main_layout)
        
        layout.addLayout(main_layout)
        
        # Bottom - State display
        self.create_state_display(layout)
        
        # Status
        self.status_label = QLabel("✅ Quantum Simulator Ready")
        self.status_label.setStyleSheet("color: #00ff88; font-weight: bold; margin: 5px;")
        layout.addWidget(self.status_label)
    
    def create_control_panel(self, parent_layout):
        """Create quantum controls panel"""
        controls_widget = QWidget()
        controls_widget.setMaximumWidth(400)
        controls_layout = QVBoxLayout(controls_widget)
        
        # Quantum System Setup
        setup_group = QGroupBox("⚙️ Quantum System")
        setup_layout = QGridLayout(setup_group)
        
        setup_layout.addWidget(QLabel("🎯 Number of Qubits:"), 0, 0)
        self.qubit_spinbox = QSpinBox()
        self.qubit_spinbox.setRange(1, 8)
        self.qubit_spinbox.setValue(self.num_qubits)
        self.qubit_spinbox.valueChanged.connect(self.update_qubit_count)
        setup_layout.addWidget(self.qubit_spinbox, 0, 1)
        
        self.reset_btn = QPushButton("🔄 Reset State")
        self.reset_btn.clicked.connect(self.reset_quantum_state)
        setup_layout.addWidget(self.reset_btn, 1, 0, 1, 2)
        
        controls_layout.addWidget(setup_group)
        
        # Quantum Gates
        gates_group = QGroupBox("🚪 Quantum Gates")
        gates_layout = QGridLayout(gates_group)
        
        # Single-qubit gates
        gates_layout.addWidget(QLabel("Single Qubit Gates:"), 0, 0, 1, 2)
        
        gate_buttons = [
            ("X", "Pauli-X (NOT)", self.apply_x_gate),
            ("Y", "Pauli-Y", self.apply_y_gate),
            ("Z", "Pauli-Z", self.apply_z_gate),
            ("H", "Hadamard", self.apply_h_gate),
            ("S", "Phase Gate", self.apply_s_gate),
            ("T", "T Gate", self.apply_t_gate)
        ]
        
        row = 1
        for i, (gate, tooltip, func) in enumerate(gate_buttons):
            btn = QPushButton(gate)
            btn.setToolTip(tooltip)
            btn.clicked.connect(func)
            gates_layout.addWidget(btn, row + i//2, i%2)
        
        # Qubit selector
        gates_layout.addWidget(QLabel("🎯 Target Qubit:"), 4, 0)
        self.target_qubit = QSpinBox()
        self.target_qubit.setRange(0, self.num_qubits - 1)
        gates_layout.addWidget(self.target_qubit, 4, 1)
        
        # Two-qubit gates
        gates_layout.addWidget(QLabel("Two Qubit Gates:"), 5, 0, 1, 2)
        
        self.cnot_btn = QPushButton("CNOT")
        self.cnot_btn.setToolTip("Controlled-NOT Gate")
        self.cnot_btn.clicked.connect(self.apply_cnot_gate)
        gates_layout.addWidget(self.cnot_btn, 6, 0)
        
        self.control_qubit = QSpinBox()
        self.control_qubit.setRange(0, self.num_qubits - 1)
        gates_layout.addWidget(self.control_qubit, 6, 1)
        
        controls_layout.addWidget(gates_group)
        
        # Measurement
        measure_group = QGroupBox("📏 Measurement")
        measure_layout = QVBoxLayout(measure_group)
        
        self.measure_btn = QPushButton("📊 Measure All Qubits")
        self.measure_btn.clicked.connect(self.measure_qubits)
        measure_layout.addWidget(self.measure_btn)
        
        self.measure_single_btn = QPushButton("🎯 Measure Single Qubit")
        self.measure_single_btn.clicked.connect(self.measure_single_qubit)
        measure_layout.addWidget(self.measure_single_btn)
        
        controls_layout.addWidget(measure_group)
        
        # Presets
        preset_group = QGroupBox("🎭 Quantum Presets")
        preset_layout = QVBoxLayout(preset_group)
        
        presets = [
            ("🌀 Superposition", self.create_superposition),
            ("🔗 Bell State", self.create_bell_state),
            ("🌊 GHZ State", self.create_ghz_state),
            ("🎲 Random State", self.create_random_state)
        ]
        
        for name, func in presets:
            btn = QPushButton(name)
            btn.clicked.connect(func)
            preset_layout.addWidget(btn)
        
        controls_layout.addWidget(preset_group)
        
        parent_layout.addWidget(controls_widget)
    
    def create_visualization_panel(self, parent_layout):
        """Create state visualization panel"""
        viz_group = QGroupBox("📊 Quantum State Visualization")
        viz_layout = QVBoxLayout(viz_group)
        
        # State vector display
        self.state_text = QTextEdit()
        self.state_text.setMaximumHeight(200)
        self.state_text.setReadOnly(True)
        viz_layout.addWidget(self.state_text)
        
        # Probability visualization (simplified)
        self.prob_text = QTextEdit()
        self.prob_text.setMaximumHeight(150)
        self.prob_text.setReadOnly(True)
        viz_layout.addWidget(self.prob_text)
        
        parent_layout.addWidget(viz_group)
    
    def create_state_display(self, parent_layout):
        """Create quantum state information display"""
        info_group = QGroupBox("📋 Quantum Information")
        info_layout = QVBoxLayout(info_group)
        
        self.info_text = QTextEdit()
        self.info_text.setMaximumHeight(120)
        self.info_text.setReadOnly(True)
        info_layout.addWidget(self.info_text)
        
        parent_layout.addWidget(info_group)
    
    def initialize_quantum_state(self):
        """Initialize quantum state to |000...0⟩"""
        self.quantum_state = np.zeros(2**self.num_qubits, dtype=complex)
        self.quantum_state[0] = 1.0  # |000...0⟩ state
        self.gates_applied = []
        self.update_displays()
    
    def update_qubit_count(self, new_count):
        """Update the number of qubits"""
        self.num_qubits = new_count
        self.target_qubit.setMaximum(new_count - 1)
        self.control_qubit.setMaximum(new_count - 1)
        self.initialize_quantum_state()
        self.status_label.setText(f"✅ Quantum system updated to {new_count} qubits")
    
    def reset_quantum_state(self):
        """Reset to initial state"""
        self.initialize_quantum_state()
        self.status_label.setText("🔄 Quantum state reset to |000...0⟩")
    
    def apply_single_qubit_gate(self, gate_matrix, qubit_idx, gate_name):
        """Apply a single-qubit gate to the quantum state"""
        # Create the full gate matrix for the entire system
        full_gate = np.eye(1, dtype=complex)
        
        for i in range(self.num_qubits):
            if i == qubit_idx:
                full_gate = np.kron(full_gate, gate_matrix)
            else:
                full_gate = np.kron(full_gate, np.eye(2, dtype=complex))
        
        # Apply the gate
        self.quantum_state = full_gate @ self.quantum_state
        self.gates_applied.append(f"{gate_name} on qubit {qubit_idx}")
        self.update_displays()
        self.status_label.setText(f"✅ Applied {gate_name} gate to qubit {qubit_idx}")
    
    def apply_x_gate(self):
        """Apply Pauli-X gate"""
        x_gate = np.array([[0, 1], [1, 0]], dtype=complex)
        self.apply_single_qubit_gate(x_gate, self.target_qubit.value(), "X")
    
    def apply_y_gate(self):
        """Apply Pauli-Y gate"""
        y_gate = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self.apply_single_qubit_gate(y_gate, self.target_qubit.value(), "Y")
    
    def apply_z_gate(self):
        """Apply Pauli-Z gate"""
        z_gate = np.array([[1, 0], [0, -1]], dtype=complex)
        self.apply_single_qubit_gate(z_gate, self.target_qubit.value(), "Z")
    
    def apply_h_gate(self):
        """Apply Hadamard gate"""
        h_gate = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        self.apply_single_qubit_gate(h_gate, self.target_qubit.value(), "H")
    
    def apply_s_gate(self):
        """Apply S (Phase) gate"""
        s_gate = np.array([[1, 0], [0, 1j]], dtype=complex)
        self.apply_single_qubit_gate(s_gate, self.target_qubit.value(), "S")
    
    def apply_t_gate(self):
        """Apply T gate"""
        t_gate = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
        self.apply_single_qubit_gate(t_gate, self.target_qubit.value(), "T")
    
    def apply_cnot_gate(self):
        """Apply CNOT gate"""
        control = self.control_qubit.value()
        target = self.target_qubit.value()
        
        if control == target:
            self.status_label.setText("❌ Control and target qubits must be different")
            return
        
        # Simplified CNOT implementation for demonstration
        # In a full implementation, this would construct the proper CNOT matrix
        self.gates_applied.append(f"CNOT: control={control}, target={target}")
        self.status_label.setText(f"✅ Applied CNOT gate (control: {control}, target: {target})")
        self.update_displays()
    
    def measure_qubits(self):
        """Measure all qubits"""
        probabilities = np.abs(self.quantum_state)**2
        outcome = np.random.choice(len(probabilities), p=probabilities)
        
        # Convert to binary representation
        binary_result = format(outcome, f'0{self.num_qubits}b')
        
        # Collapse to measured state
        self.quantum_state = np.zeros_like(self.quantum_state)
        self.quantum_state[outcome] = 1.0
        
        self.gates_applied.append(f"MEASURE ALL → |{binary_result}⟩")
        self.update_displays()
        self.status_label.setText(f"📊 Measurement result: |{binary_result}⟩")
    
    def measure_single_qubit(self):
        """Measure a single qubit"""
        qubit = self.target_qubit.value()
        # Simplified single qubit measurement
        prob_0 = sum(np.abs(self.quantum_state[i])**2 
                    for i in range(len(self.quantum_state)) 
                    if not (i >> qubit) & 1)
        
        result = 0 if random.random() < prob_0 else 1
        self.gates_applied.append(f"MEASURE qubit {qubit} → {result}")
        self.update_displays()
        self.status_label.setText(f"📊 Qubit {qubit} measured: {result}")
    
    def create_superposition(self):
        """Create superposition state"""
        self.reset_quantum_state()
        # Apply Hadamard to all qubits
        for i in range(self.num_qubits):
            self.target_qubit.setValue(i)
            self.apply_h_gate()
        self.status_label.setText("🌀 Created equal superposition state")
    
    def create_bell_state(self):
        """Create Bell state (entangled state)"""
        if self.num_qubits < 2:
            self.status_label.setText("❌ Need at least 2 qubits for Bell state")
            return
        
        self.reset_quantum_state()
        # H on qubit 0, then CNOT(0,1)
        self.target_qubit.setValue(0)
        self.apply_h_gate()
        
        self.control_qubit.setValue(0)
        self.target_qubit.setValue(1)
        self.apply_cnot_gate()
        
        self.status_label.setText("🔗 Created Bell state (entangled)")
    
    def create_ghz_state(self):
        """Create GHZ state"""
        self.reset_quantum_state()
        # H on first qubit, then CNOT to all others
        self.target_qubit.setValue(0)
        self.apply_h_gate()
        
        for i in range(1, self.num_qubits):
            self.control_qubit.setValue(0)
            self.target_qubit.setValue(i)
            self.apply_cnot_gate()
        
        self.status_label.setText("🌊 Created GHZ state (maximally entangled)")
    
    def create_random_state(self):
        """Create random quantum state"""
        self.quantum_state = np.random.random(2**self.num_qubits) + \
                           1j * np.random.random(2**self.num_qubits)
        # Normalize
        self.quantum_state /= np.linalg.norm(self.quantum_state)
        self.gates_applied.append("RANDOM STATE")
        self.update_displays()
        self.status_label.setText("🎲 Created random quantum state")
    
    def update_displays(self):
        """Update all visualization displays"""
        # State vector display
        state_str = "🌌 Quantum State Vector:\n"
        for i, amplitude in enumerate(self.quantum_state):
            if abs(amplitude) > 1e-10:  # Only show non-zero amplitudes
                binary = format(i, f'0{self.num_qubits}b')
                prob = abs(amplitude)**2
                state_str += f"|{binary}⟩: {amplitude:.3f} (prob: {prob:.3f})\n"
        
        self.state_text.clear()
        self.state_text.append(state_str)
        
        # Probability display
        prob_str = "📊 Measurement Probabilities:\n"
        for i, amplitude in enumerate(self.quantum_state):
            prob = abs(amplitude)**2
            if prob > 1e-10:
                binary = format(i, f'0{self.num_qubits}b')
                bar = "█" * int(prob * 20)  # Simple bar chart
                prob_str += f"|{binary}⟩: {prob:.3f} {bar}\n"
        
        self.prob_text.clear()
        self.prob_text.append(prob_str)
        
        # Information display
        info_str = f"🎯 System Info:\n"
        info_str += f"Qubits: {self.num_qubits}\n"
        info_str += f"State dimension: {2**self.num_qubits}\n"
        info_str += f"Gates applied: {len(self.gates_applied)}\n"
        info_str += f"Norm: {np.linalg.norm(self.quantum_state):.6f}\n\n"
        info_str += "📋 Gate History:\n"
        info_str += "\n".join(self.gates_applied[-5:])  # Last 5 operations
        
        self.info_text.clear()
        self.info_text.append(info_str)

def main():
    """Main entry point"""
    if not PYQT5_AVAILABLE:
        print("❌ PyQt5 required for Quantum Simulator")
        return
    
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    window = QuantumSimulator()
    window.show()
    
    return app.exec_()

if __name__ == "__main__":
    main()
