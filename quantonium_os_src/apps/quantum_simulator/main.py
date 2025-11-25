#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
# This file is listed in CLAIMS_PRACTICING_FILES.txt and is licensed
# under LICENSE-CLAIMS-NC.md (research/education only). Commercial
# rights require a separate patent license from the author.
"""Quantum Circuit Simulator - Visual quantum computing interface"""

from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QTextEdit, QComboBox, QSpinBox,
                             QGroupBox, QScrollArea, QSplitter)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
import numpy as np

class MainWindow(QMainWindow):
    """Quantum Circuit Simulator Application"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Quantum Simulator - QuantoniumOS")
        self.setGeometry(100, 100, 1200, 800)
        
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        # Title
        title = QLabel("🔬 Quantum Circuit Simulator")
        title.setFont(QFont("Sans Serif", 18, QFont.Bold))
        title.setStyleSheet("color: #00aaff; padding: 10px;")
        layout.addWidget(title)
        
        # Splitter for circuit builder and results
        splitter = QSplitter(Qt.Horizontal)
        
        # Left: Circuit Builder
        builder_group = QGroupBox("Circuit Builder")
        builder_layout = QVBoxLayout()
        
        # Qubit count
        qubit_layout = QHBoxLayout()
        qubit_layout.addWidget(QLabel("Qubits:"))
        self.qubit_spin = QSpinBox()
        self.qubit_spin.setRange(1, 10)
        self.qubit_spin.setValue(3)
        qubit_layout.addWidget(self.qubit_spin)
        qubit_layout.addStretch()
        builder_layout.addLayout(qubit_layout)
        
        # Gate selector
        gate_layout = QHBoxLayout()
        gate_layout.addWidget(QLabel("Gate:"))
        self.gate_combo = QComboBox()
        self.gate_combo.addItems(['H', 'X', 'Y', 'Z', 'CNOT', 'T', 'S', 'RFT'])
        gate_layout.addWidget(self.gate_combo)
        builder_layout.addLayout(gate_layout)
        
        # Circuit display
        self.circuit_text = QTextEdit()
        self.circuit_text.setPlaceholderText("Circuit will appear here...")
        self.circuit_text.setMaximumHeight(200)
        builder_layout.addWidget(QLabel("Current Circuit:"))
        builder_layout.addWidget(self.circuit_text)
        
        # Buttons
        btn_layout = QHBoxLayout()
        self.add_gate_btn = QPushButton("➕ Add Gate")
        self.simulate_btn = QPushButton("▶️ Simulate")
        self.clear_btn = QPushButton("🗑️ Clear")
        btn_layout.addWidget(self.add_gate_btn)
        btn_layout.addWidget(self.simulate_btn)
        btn_layout.addWidget(self.clear_btn)
        builder_layout.addLayout(btn_layout)
        
        builder_group.setLayout(builder_layout)
        splitter.addWidget(builder_group)
        
        # Right: Results
        results_group = QGroupBox("Simulation Results")
        results_layout = QVBoxLayout()
        
        self.results_text = QTextEdit()
        self.results_text.setPlaceholderText("Results will appear here after simulation...")
        self.results_text.setReadOnly(True)
        results_layout.addWidget(self.results_text)
        
        # State vector display
        state_label = QLabel("Quantum State Vector:")
        state_label.setFont(QFont("Sans Serif", 10, QFont.Bold))
        results_layout.addWidget(state_label)
        
        self.state_text = QTextEdit()
        self.state_text.setReadOnly(True)
        self.state_text.setMaximumHeight(150)
        results_layout.addWidget(self.state_text)
        
        results_group.setLayout(results_layout)
        splitter.addWidget(results_group)
        
        layout.addWidget(splitter)
        
        # Status bar
        self.statusBar().showMessage("Ready - Select gates and build your circuit")
        
        # Connect signals
        self.add_gate_btn.clicked.connect(self.add_gate)
        self.simulate_btn.clicked.connect(self.simulate)
        self.clear_btn.clicked.connect(self.clear_circuit)
        
        # Internal state
        self.circuit = []
        
        # Apply dark theme
        self.set_dark_theme()
        
    def set_dark_theme(self):
        """Apply quantum dark theme"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1a1a1a;
            }
            QWidget {
                background-color: #1a1a1a;
                color: #ffffff;
            }
            QGroupBox {
                border: 2px solid #00aaff;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
                color: #00aaff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                background-color: #00aaff;
                color: #ffffff;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #00ffaa;
            }
            QTextEdit, QSpinBox, QComboBox {
                background-color: #2a2a2a;
                color: #ffffff;
                border: 1px solid #00aaff;
                border-radius: 3px;
                padding: 5px;
            }
        """)
    
    def add_gate(self):
        """Add gate to circuit"""
        gate = self.gate_combo.currentText()
        qubits = self.qubit_spin.value()
        self.circuit.append({'gate': gate, 'qubits': qubits})
        
        # Update display
        circuit_str = " -> ".join([g['gate'] for g in self.circuit])
        self.circuit_text.setText(circuit_str)
        self.statusBar().showMessage(f"Added {gate} gate")
    
    def simulate(self):
        """Run quantum simulation"""
        if not self.circuit:
            self.results_text.setText("⚠️ No gates in circuit. Add gates first!")
            return
        
        num_qubits = self.qubit_spin.value()
        
        # Initialize state |0...0>
        state = np.zeros(2**num_qubits, dtype=complex)
        state[0] = 1.0
        
        # Simulate circuit (simplified)
        for gate_op in self.circuit:
            gate = gate_op['gate']
            if gate == 'H':
                # Hadamard simulation (simplified for qubit 0)
                state = self.apply_hadamard(state, num_qubits)
            elif gate == 'X':
                # Pauli-X (simplified)
                state = self.apply_x(state, num_qubits)
        
        # Display results
        probabilities = np.abs(state)**2
        
        results = "📊 Measurement Probabilities:\n\n"
        for i, prob in enumerate(probabilities):
            if prob > 1e-6:  # Only show non-negligible probabilities
                basis_state = format(i, f'0{num_qubits}b')
                results += f"|{basis_state}⟩: {prob:.4f} ({prob*100:.2f}%)\n"
        
        self.results_text.setText(results)
        
        # Display state vector
        state_str = "State Vector:\n"
        for i, amp in enumerate(state):
            if abs(amp) > 1e-6:
                basis_state = format(i, f'0{num_qubits}b')
                state_str += f"|{basis_state}⟩: {amp.real:.4f} + {amp.imag:.4f}i\n"
        
        self.state_text.setText(state_str)
        self.statusBar().showMessage("Simulation complete!")
    
    def apply_hadamard(self, state, num_qubits):
        """Apply Hadamard gate (simplified)"""
        # Full implementation would require proper tensor product
        # This is a demonstration
        return state / np.sqrt(2)
    
    def apply_x(self, state, num_qubits):
        """Apply Pauli-X gate (simplified)"""
        # Swap |0> and |1> amplitudes
        new_state = state.copy()
        for i in range(0, len(state), 2):
            new_state[i], new_state[i+1] = state[i+1], state[i]
        return new_state
    
    def clear_circuit(self):
        """Clear the circuit"""
        self.circuit = []
        self.circuit_text.clear()
        self.results_text.clear()
        self.state_text.clear()
        self.statusBar().showMessage("Circuit cleared")
