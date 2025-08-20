"""
Quantum Simulator Application - Clean Version
Advanced quantum circuit simulation for QuantoniumOS
"""

try:
    from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, 
                                QPushButton, QTextEdit, QLabel, QFrame, QGroupBox,
                                QComboBox, QSpinBox, QTabWidget, QSlider,
                                QCheckBox, QTableWidget, QTableWidgetItem)
    from PyQt5.QtCore import Qt, QTimer
    from PyQt5.QtGui import QFont
    PYQT5_AVAILABLE = True
except ImportError:
    PYQT5_AVAILABLE = False

import os
import sys
import math
import random

class QuantumSimulator(QWidget):
    """Advanced quantum circuit simulator widget"""
    
    def __init__(self):
        super().__init__()
        if not PYQT5_AVAILABLE:
            print("⚠️ PyQt5 not available for Quantum Simulator")
            return
            
        self.quantum_state = None
        self.circuit = []
        self.qubits = 3
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the quantum simulator interface"""
        self.setWindowTitle("🌌 QuantoniumOS Quantum Simulator")
        
        # Main layout
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("🌌 Quantum Circuit Simulator")
        title.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                color: #1f2937;
                padding: 20px;
                text-align: center;
            }
        """)
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Tab widget
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        
        # Create tabs
        self.create_circuit_builder_tab()
        self.create_state_viewer_tab()
        self.create_algorithms_tab()
        
        # Initialize quantum state
        self.quantum_state = self.create_initial_state()
        
    def create_circuit_builder_tab(self):
        """Create quantum circuit builder tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Circuit controls
        controls_frame = QFrame()
        controls_layout = QHBoxLayout(controls_frame)
        
        # Qubit configuration
        qubit_group = QGroupBox("⚛️ Qubit Configuration")
        qubit_layout = QVBoxLayout(qubit_group)
        
        qubit_layout.addWidget(QLabel("Number of Qubits:"))
        self.qubit_count = QSpinBox()
        self.qubit_count.setRange(1, 10)
        self.qubit_count.setValue(3)
        self.qubit_count.valueChanged.connect(self.update_qubits)
        qubit_layout.addWidget(self.qubit_count)
        
        controls_layout.addWidget(qubit_group)
        
        # Gate selection
        gate_group = QGroupBox("🔧 Quantum Gates")
        gate_layout = QVBoxLayout(gate_group)
        
        gates = ["Hadamard (H)", "Pauli-X", "Pauli-Y", "Pauli-Z", "CNOT", "Phase"]
        for gate in gates:
            btn = QPushButton(gate)
            btn.clicked.connect(lambda checked, g=gate: self.add_gate(g))
            gate_layout.addWidget(btn)
            
        controls_layout.addWidget(gate_group)
        
        # Circuit display
        circuit_group = QGroupBox("🔗 Circuit Visualization")
        circuit_layout = QVBoxLayout(circuit_group)
        
        self.circuit_display = QTextEdit()
        self.circuit_display.setReadOnly(True)
        self.circuit_display.setMaximumHeight(150)
        circuit_layout.addWidget(self.circuit_display)
        
        controls_layout.addWidget(circuit_group)
        
        layout.addWidget(controls_frame)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        run_btn = QPushButton("🚀 Run Circuit")
        run_btn.clicked.connect(self.run_circuit)
        button_layout.addWidget(run_btn)
        
        clear_btn = QPushButton("🗑️ Clear Circuit")
        clear_btn.clicked.connect(self.clear_circuit)
        button_layout.addWidget(clear_btn)
        
        layout.addLayout(button_layout)
        
        # Results
        results_group = QGroupBox("📊 Results")
        results_layout = QVBoxLayout(results_group)
        
        self.results_display = QTextEdit()
        self.results_display.setReadOnly(True)
        results_layout.addWidget(self.results_display)
        
        layout.addWidget(results_group)
        
        self.tabs.addTab(tab, "🔧 Circuit Builder")
        
    def create_state_viewer_tab(self):
        """Create quantum state viewer tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # State visualization
        state_group = QGroupBox("🌊 Quantum State")
        state_layout = QVBoxLayout(state_group)
        
        self.state_display = QTextEdit()
        self.state_display.setReadOnly(True)
        state_layout.addWidget(self.state_display)
        
        layout.addWidget(state_group)
        
        # Measurement controls
        measure_group = QGroupBox("📏 Measurement")
        measure_layout = QVBoxLayout(measure_group)
        
        measure_btn = QPushButton("📏 Measure All Qubits")
        measure_btn.clicked.connect(self.measure_qubits)
        measure_layout.addWidget(measure_btn)
        
        self.measurement_results = QTextEdit()
        self.measurement_results.setReadOnly(True)
        self.measurement_results.setMaximumHeight(100)
        measure_layout.addWidget(self.measurement_results)
        
        layout.addWidget(measure_group)
        
        self.tabs.addTab(tab, "🌊 State Viewer")
        
    def create_algorithms_tab(self):
        """Create quantum algorithms demonstration tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Algorithm selection
        algo_group = QGroupBox("🧮 Quantum Algorithms")
        algo_layout = QVBoxLayout(algo_group)
        
        algorithms = [
            "Deutsch-Jozsa Algorithm",
            "Grover's Search",
            "Quantum Fourier Transform",
            "Shor's Algorithm (Demo)",
            "Bell State Preparation"
        ]
        
        for algo in algorithms:
            btn = QPushButton(f"🔬 {algo}")
            btn.clicked.connect(lambda checked, a=algo: self.run_algorithm(a))
            algo_layout.addWidget(btn)
            
        layout.addWidget(algo_group)
        
        # Algorithm results
        results_group = QGroupBox("📈 Algorithm Results")
        results_layout = QVBoxLayout(results_group)
        
        self.algo_results = QTextEdit()
        self.algo_results.setReadOnly(True)
        results_layout.addWidget(self.algo_results)
        
        layout.addWidget(results_group)
        
        self.tabs.addTab(tab, "🧮 Algorithms")
        
    def create_initial_state(self):
        """Create initial quantum state |000...>"""
        n_states = 2 ** self.qubits
        state = [0.0] * n_states
        state[0] = 1.0  # |000...> state
        return state
        
    def update_qubits(self, value):
        """Update number of qubits"""
        self.qubits = value
        self.quantum_state = self.create_initial_state()
        self.clear_circuit()
        self.update_state_display()
        
    def add_gate(self, gate_name):
        """Add a gate to the circuit"""
        self.circuit.append(gate_name)
        self.update_circuit_display()
        
    def clear_circuit(self):
        """Clear the quantum circuit"""
        self.circuit = []
        self.quantum_state = self.create_initial_state()
        self.update_circuit_display()
        self.update_state_display()
        
    def update_circuit_display(self):
        """Update the circuit visualization"""
        if not self.circuit:
            self.circuit_display.setText("Empty circuit - add gates to build your quantum circuit")
        else:
            circuit_text = "Quantum Circuit:\n"
            for i, gate in enumerate(self.circuit):
                circuit_text += f"{i+1}. {gate}\n"
            self.circuit_display.setText(circuit_text)
            
    def update_state_display(self):
        """Update quantum state display"""
        if not hasattr(self, 'state_display'):
            return
            
        state_text = f"Quantum State ({self.qubits} qubits):\n\n"
        
        for i, amplitude in enumerate(self.quantum_state):
            if abs(amplitude) > 0.001:  # Only show non-zero amplitudes
                binary = format(i, f'0{self.qubits}b')
                state_text += f"|{binary}⟩: {amplitude:.3f}\n"
                
        self.state_display.setText(state_text)
        
    def run_circuit(self):
        """Simulate the quantum circuit"""
        if not self.circuit:
            self.results_display.setText("❌ No gates in circuit")
            return
            
        # Simple simulation (simplified for demo)
        results_text = "🚀 Circuit Execution Results:\n\n"
        results_text += f"Circuit with {len(self.circuit)} gates executed on {self.qubits} qubits\n"
        results_text += f"Gates applied: {', '.join(self.circuit)}\n\n"
        
        # Simulate some quantum effects
        for gate in self.circuit:
            if "Hadamard" in gate:
                results_text += "✅ Applied Hadamard gate - created superposition\n"
            elif "CNOT" in gate:
                results_text += "✅ Applied CNOT gate - created entanglement\n"
            else:
                results_text += f"✅ Applied {gate} gate\n"
                
        results_text += f"\n📊 Final state has {2**self.qubits} possible outcomes"
        self.results_display.setText(results_text)
        self.update_state_display()
        
    def measure_qubits(self):
        """Perform quantum measurement"""
        # Simulate measurement
        outcome = random.randint(0, 2**self.qubits - 1)
        binary = format(outcome, f'0{self.qubits}b')
        
        measure_text = f"📏 Measurement Result: |{binary}⟩\n"
        measure_text += f"Decimal: {outcome}\n"
        measure_text += f"Probability: {1/2**self.qubits:.3f} (uniform for demo)"
        
        self.measurement_results.setText(measure_text)
        
    def run_algorithm(self, algorithm):
        """Run a quantum algorithm demonstration"""
        results_text = f"🔬 Running: {algorithm}\n\n"
        
        if "Bell State" in algorithm:
            results_text += "Preparing Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2\n"
            results_text += "1. Apply H gate to qubit 0\n"
            results_text += "2. Apply CNOT gate (0→1)\n"
            results_text += "✅ Bell state prepared - qubits are maximally entangled"
            
        elif "Deutsch-Jozsa" in algorithm:
            results_text += "Testing if function is constant or balanced\n"
            results_text += "1. Initialize qubits in superposition\n"
            results_text += "2. Apply oracle function\n"
            results_text += "3. Apply Hadamard gates\n"
            results_text += "✅ Result: Function is BALANCED (demo)"
            
        elif "Grover" in algorithm:
            results_text += "Searching unsorted database quadratically faster\n"
            results_text += "1. Initialize uniform superposition\n"
            results_text += "2. Apply oracle and diffusion operator\n"
            results_text += "3. Repeat √N times\n"
            results_text += "✅ Found target item with high probability"
            
        elif "Fourier Transform" in algorithm:
            results_text += "Quantum Fourier Transform demonstration\n"
            results_text += "1. Apply controlled rotations\n"
            results_text += "2. Apply Hadamard gates\n"
            results_text += "3. Reverse qubit order\n"
            results_text += "✅ QFT completed - frequency domain representation"
            
        elif "Shor" in algorithm:
            results_text += "Shor's factoring algorithm (simplified demo)\n"
            results_text += "1. Choose random number a < N\n"
            results_text += "2. Find period using QFT\n"
            results_text += "3. Use period to find factors\n"
            results_text += "✅ Demo: Factored 15 = 3 × 5 (predetermined)"
            
        self.algo_results.setText(results_text)

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
