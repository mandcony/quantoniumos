#!/usr/bin/env python3
"""
QuantoniumOS Quantum Simulator
==============================
Interactive quantum circuit simulator
"""

import os
import sys
import math
import random
from typing import List, Dict, Tuple, Optional

# Import the base launcher
try:
    from launcher_base import AppLauncherBase, AppWindow, AppTerminal, HAS_PYQT
except ImportError:
    # Try to find the launcher_base module
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    try:
        from launcher_base import AppLauncherBase, AppWindow, AppTerminal, HAS_PYQT
    except ImportError:
        print("Error: launcher_base.py not found")
        sys.exit(1)

# Try to import PyQt5 for the GUI
if HAS_PYQT:
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                              QHBoxLayout, QPushButton, QLabel, QComboBox,
                              QSlider, QLineEdit, QTabWidget, QFrame, QGridLayout)
    from PyQt5.QtGui import QIcon, QPixmap, QFont, QColor, QPainter, QPen, QBrush
    from PyQt5.QtCore import Qt, QSize, QPoint, QRect, QTimer

# Try to import numpy for calculations
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("Numpy not found. Please install it with: pip install numpy")

class QuantumSimulator(AppWindow):
    """Quantum Simulator window"""
    
    def __init__(self, app_name: str, app_icon: str):
        """Initialize the Quantum Simulator window"""
        super().__init__(app_name, app_icon)
        
        # Check if numpy is available
        if not HAS_NUMPY:
            error_label = QLabel("Numpy not available. Please install it with: pip install numpy")
            error_label.setStyleSheet("color: red; font-size: 16px;")
            error_label.setAlignment(Qt.AlignCenter)
            self.layout.addWidget(error_label)
            return
        
        # Load the quantum kernel if available
        self.quantum_kernel = None
        try:
            # Try to find and import the quantum kernel
            kernel_paths = [
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bulletproof_quantum_kernel.py"),
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "working_quantum_kernel.py")
            ]
            
            kernel_found = False
            for path in kernel_paths:
                if os.path.exists(path):
                    # Add the directory to the path
                    sys.path.append(os.path.dirname(path))
                    
                    # Import the module
                    module_name = os.path.basename(path)[:-3]  # Remove .py
                    
                    try:
                        # Try direct import
                        module = __import__(module_name)
                        self.quantum_kernel = module
                        kernel_found = True
                        print(f"Quantum kernel loaded from {path}")
                        break
                    except ImportError:
                        # Try to load as a file
                        import importlib.util
                        spec = importlib.util.spec_from_file_location(module_name, path)
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        self.quantum_kernel = module
                        kernel_found = True
                        print(f"Quantum kernel loaded from {path}")
                        break
            
            if not kernel_found:
                print("Quantum kernel not found")
                # Create a simple quantum simulator
                self.quantum_kernel = type('', (), {})()
                self.quantum_kernel.simulate = self.simple_quantum_simulation
        except Exception as e:
            print(f"Failed to load quantum kernel: {e}")
            # Create a simple quantum simulator
            self.quantum_kernel = type('', (), {})()
            self.quantum_kernel.simulate = self.simple_quantum_simulation
        
        # Create the UI
        self.create_ui()
    
    def simple_quantum_simulation(self, circuit, shots=1000):
        """Simple quantum simulation for demonstration"""
        # Create a simple quantum state
        state = np.zeros(2**len(circuit["qubits"]))
        state[0] = 1.0  # |0...0⟩ state
        
        # Apply gates
        for gate in circuit["gates"]:
            gate_type = gate["type"]
            target = gate["target"]
            
            if gate_type == "H":  # Hadamard gate
                # Apply Hadamard to target qubit
                for i in range(len(state)):
                    if (i >> target) & 1:  # If target qubit is 1
                        state[i], state[i ^ (1 << target)] = (state[i] - state[i ^ (1 << target)]) / np.sqrt(2), (state[i] + state[i ^ (1 << target)]) / np.sqrt(2)
            
            elif gate_type == "X":  # Pauli-X (NOT) gate
                # Apply X to target qubit
                for i in range(len(state)):
                    if (i >> target) & 1:  # If target qubit is 1
                        state[i], state[i ^ (1 << target)] = state[i ^ (1 << target)], state[i]
            
            elif gate_type == "Z":  # Pauli-Z gate
                # Apply Z to target qubit
                for i in range(len(state)):
                    if (i >> target) & 1:  # If target qubit is 1
                        state[i] = -state[i]
        
        # Measure the state
        probabilities = np.abs(state)**2
        
        # Simulate shots
        results = {}
        for _ in range(shots):
            outcome = np.random.choice(len(state), p=probabilities)
            outcome_bits = format(outcome, f'0{len(circuit["qubits"])}b')
            results[outcome_bits] = results.get(outcome_bits, 0) + 1
        
        return {"results": results, "state": state}
    
    def create_ui(self):
        """Create the user interface"""
        # Clear the layout
        for i in reversed(range(self.layout.count())): 
            self.layout.itemAt(i).widget().setParent(None)
        
        # Add tabs for different simulations
        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)
        
        # Add the circuit tab
        self.create_circuit_tab()
        
        # Add the visualization tab
        self.create_visualization_tab()
    
    def create_circuit_tab(self):
        """Create the circuit editor tab"""
        # Create the tab widget
        circuit_tab = QWidget()
        circuit_layout = QVBoxLayout(circuit_tab)
        
        # Add the circuit editor
        editor_frame = QFrame()
        editor_frame.setFrameShape(QFrame.StyledPanel)
        editor_frame.setStyleSheet("QFrame { background-color: rgba(40, 40, 60, 150); border-radius: 5px; }")
        editor_layout = QVBoxLayout(editor_frame)
        
        # Add the qubit count selector
        qubit_layout = QHBoxLayout()
        qubit_label = QLabel("Qubits:")
        qubit_label.setStyleSheet("color: white;")
        qubit_layout.addWidget(qubit_label)
        
        self.qubit_slider = QSlider(Qt.Horizontal)
        self.qubit_slider.setMinimum(1)
        self.qubit_slider.setMaximum(5)
        self.qubit_slider.setValue(2)
        self.qubit_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                background-color: rgba(60, 60, 80, 200);
                height: 8px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background-color: rgba(100, 100, 200, 200);
                border: 1px solid rgba(150, 150, 255, 200);
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
        """)
        self.qubit_slider.valueChanged.connect(self.update_circuit)
        qubit_layout.addWidget(self.qubit_slider)
        
        self.qubit_value_label = QLabel("2")
        self.qubit_value_label.setStyleSheet("color: white;")
        qubit_layout.addWidget(self.qubit_value_label)
        
        editor_layout.addLayout(qubit_layout)
        
        # Add the circuit grid
        self.circuit_frame = QFrame()
        self.circuit_frame.setFrameShape(QFrame.StyledPanel)
        self.circuit_frame.setStyleSheet("QFrame { background-color: rgba(30, 30, 40, 150); border-radius: 5px; }")
        self.circuit_layout = QGridLayout(self.circuit_frame)
        
        editor_layout.addWidget(self.circuit_frame)
        
        # Add the gate buttons
        gate_layout = QHBoxLayout()
        
        gate_buttons = [
            ("H", "Hadamard"),
            ("X", "Pauli-X"),
            ("Y", "Pauli-Y"),
            ("Z", "Pauli-Z"),
            ("CNOT", "Controlled-NOT")
        ]
        
        for gate, tooltip in gate_buttons:
            button = QPushButton(gate)
            button.setToolTip(tooltip)
            button.setFixedSize(40, 40)
            button.setStyleSheet("""
                QPushButton {
                    background-color: rgba(60, 60, 80, 200);
                    color: white;
                    border: 1px solid rgba(100, 100, 200, 200);
                    border-radius: 5px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: rgba(80, 80, 120, 200);
                    border: 1px solid rgba(150, 150, 255, 200);
                }
                QPushButton:pressed {
                    background-color: rgba(40, 40, 60, 200);
                    border: 1px solid rgba(100, 100, 200, 200);
                }
            """)
            button.clicked.connect(lambda checked, g=gate: self.add_gate(g))
            gate_layout.addWidget(button)
        
        editor_layout.addLayout(gate_layout)
        
        # Add the run button
        run_layout = QHBoxLayout()
        
        run_button = QPushButton("Run Simulation")
        run_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(60, 80, 60, 200);
                color: white;
                border: 1px solid rgba(100, 200, 100, 200);
                border-radius: 5px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgba(80, 120, 80, 200);
                border: 1px solid rgba(150, 255, 150, 200);
            }
            QPushButton:pressed {
                background-color: rgba(40, 60, 40, 200);
                border: 1px solid rgba(100, 200, 100, 200);
            }
        """)
        run_button.clicked.connect(self.run_simulation)
        run_layout.addWidget(run_button)
        
        clear_button = QPushButton("Clear Circuit")
        clear_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(80, 60, 60, 200);
                color: white;
                border: 1px solid rgba(200, 100, 100, 200);
                border-radius: 5px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgba(120, 80, 80, 200);
                border: 1px solid rgba(255, 150, 150, 200);
            }
            QPushButton:pressed {
                background-color: rgba(60, 40, 40, 200);
                border: 1px solid rgba(200, 100, 100, 200);
            }
        """)
        clear_button.clicked.connect(self.clear_circuit)
        run_layout.addWidget(clear_button)
        
        editor_layout.addLayout(run_layout)
        
        circuit_layout.addWidget(editor_frame)
        
        # Add the results display
        results_frame = QFrame()
        results_frame.setFrameShape(QFrame.StyledPanel)
        results_frame.setStyleSheet("QFrame { background-color: rgba(40, 40, 60, 150); border-radius: 5px; }")
        results_layout = QVBoxLayout(results_frame)
        
        results_label = QLabel("Simulation Results")
        results_label.setStyleSheet("color: white; font-weight: bold;")
        results_label.setAlignment(Qt.AlignCenter)
        results_layout.addWidget(results_label)
        
        self.results_text = QLabel("Run a simulation to see results")
        self.results_text.setStyleSheet("color: white;")
        self.results_text.setAlignment(Qt.AlignCenter)
        results_layout.addWidget(self.results_text)
        
        circuit_layout.addWidget(results_frame)
        
        # Add the tab
        self.tabs.addTab(circuit_tab, "Circuit Editor")
        
        # Initialize the circuit
        self.circuit = {"qubits": 2, "gates": []}
        self.update_circuit()
    
    def create_visualization_tab(self):
        """Create the visualization tab"""
        # Create the tab widget
        viz_tab = QWidget()
        viz_layout = QVBoxLayout(viz_tab)
        
        # Add a label
        viz_label = QLabel("Quantum State Visualization")
        viz_label.setStyleSheet("color: white; font-size: 18px; font-weight: bold;")
        viz_label.setAlignment(Qt.AlignCenter)
        viz_layout.addWidget(viz_label)
        
        # Add the visualization frame
        viz_frame = QFrame()
        viz_frame.setFrameShape(QFrame.StyledPanel)
        viz_frame.setStyleSheet("QFrame { background-color: rgba(40, 40, 60, 150); border-radius: 5px; }")
        viz_frame_layout = QVBoxLayout(viz_frame)
        
        # Add the state vector display
        self.state_vector_label = QLabel("Run a simulation to see the state vector")
        self.state_vector_label.setStyleSheet("color: white;")
        self.state_vector_label.setAlignment(Qt.AlignCenter)
        viz_frame_layout.addWidget(self.state_vector_label)
        
        viz_layout.addWidget(viz_frame)
        
        # Add the tab
        self.tabs.addTab(viz_tab, "Visualization")
    
    def update_circuit(self):
        """Update the circuit display"""
        # Update the qubit count
        qubit_count = self.qubit_slider.value()
        self.qubit_value_label.setText(str(qubit_count))
        
        # Update the circuit
        self.circuit["qubits"] = qubit_count
        
        # Clear the circuit grid
        for i in reversed(range(self.circuit_layout.count())):
            self.circuit_layout.itemAt(i).widget().setParent(None)
        
        # Add the qubit labels
        for i in range(qubit_count):
            qubit_label = QLabel(f"Q{i}")
            qubit_label.setStyleSheet("color: white; font-weight: bold;")
            qubit_label.setAlignment(Qt.AlignCenter)
            self.circuit_layout.addWidget(qubit_label, i, 0)
        
        # Add the gate labels
        for i, gate in enumerate(self.circuit["gates"]):
            gate_label = QLabel(gate["type"])
            gate_label.setStyleSheet("""
                color: white;
                background-color: rgba(60, 60, 80, 200);
                border: 1px solid rgba(100, 100, 200, 200);
                border-radius: 5px;
                padding: 5px;
                font-weight: bold;
            """)
            gate_label.setAlignment(Qt.AlignCenter)
            gate_label.setFixedSize(30, 30)
            
            # Add a remove button
            remove_button = QPushButton("×")
            remove_button.setStyleSheet("""
                QPushButton {
                    color: white;
                    background-color: rgba(80, 60, 60, 200);
                    border: 1px solid rgba(200, 100, 100, 200);
                    border-radius: 5px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: rgba(120, 80, 80, 200);
                    border: 1px solid rgba(255, 150, 150, 200);
                }
            """)
            remove_button.setFixedSize(15, 15)
            remove_button.clicked.connect(lambda checked, idx=i: self.remove_gate(idx))
            
            # Add to the grid
            cell_widget = QWidget()
            cell_layout = QVBoxLayout(cell_widget)
            cell_layout.addWidget(gate_label, alignment=Qt.AlignCenter)
            cell_layout.addWidget(remove_button, alignment=Qt.AlignCenter)
            
            self.circuit_layout.addWidget(cell_widget, gate["target"], i + 1)
    
    def add_gate(self, gate_type):
        """Add a gate to the circuit"""
        # Get the target qubit (just use the first qubit for now)
        target = 0
        
        # Add the gate
        self.circuit["gates"].append({
            "type": gate_type,
            "target": target
        })
        
        # Update the display
        self.update_circuit()
    
    def remove_gate(self, index):
        """Remove a gate from the circuit"""
        if 0 <= index < len(self.circuit["gates"]):
            del self.circuit["gates"][index]
            self.update_circuit()
    
    def clear_circuit(self):
        """Clear the circuit"""
        self.circuit["gates"] = []
        self.update_circuit()
        self.results_text.setText("Run a simulation to see results")
        self.state_vector_label.setText("Run a simulation to see the state vector")
    
    def run_simulation(self):
        """Run the quantum simulation"""
        try:
            # Run the simulation
            result = self.quantum_kernel.simulate(self.circuit)
            
            # Format the results
            results_text = "Results:\n"
            for outcome, count in result["results"].items():
                results_text += f"|{outcome}⟩: {count} ({count/sum(result['results'].values())*100:.1f}%)\n"
            
            self.results_text.setText(results_text)
            
            # Format the state vector
            state_text = "State Vector:\n"
            for i, amplitude in enumerate(result["state"]):
                if abs(amplitude) > 1e-10:  # Only show non-zero amplitudes
                    bits = format(i, f'0{self.circuit["qubits"]}b')
                    state_text += f"{amplitude:.4f} |{bits}⟩\n"
            
            self.state_vector_label.setText(state_text)
        
        except Exception as e:
            self.results_text.setText(f"Simulation error: {e}")
            self.state_vector_label.setText(f"Simulation error: {e}")

class QuantumSimulatorTerminal(AppTerminal):
    """Quantum Simulator terminal"""
    
    def __init__(self, app_name: str):
        """Initialize the Quantum Simulator terminal"""
        super().__init__(app_name)
        
        # Try to import numpy for calculations
        try:
            import numpy as np
            self.np = np
            self.has_numpy = True
        except ImportError:
            self.has_numpy = False
            print("Numpy not found. Please install it with: pip install numpy")
        
        # Initialize the circuit
        self.circuit = {"qubits": 2, "gates": []}
    
    def start(self):
        """Start the terminal app"""
        print("\n" + "=" * 60)
        print(f"{self.app_name} Terminal Interface")
        print("=" * 60 + "\n")
        
        print("Available commands:")
        print("  help         - Show this help message")
        print("  qubits [n]   - Set the number of qubits")
        print("  add [gate]   - Add a gate (H, X, Y, Z)")
        print("  list         - List the current circuit")
        print("  clear        - Clear the circuit")
        print("  run          - Run the simulation")
        print("  exit         - Exit the application\n")
        
        # Main loop
        while self.running:
            command = input(f"{self.app_name}> ").strip()
            self.process_command(command)
    
    def process_command(self, command: str):
        """Process a terminal command"""
        parts = command.split()
        
        if not parts:
            return
        
        cmd = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []
        
        if cmd == "help":
            print("\nAvailable commands:")
            print("  help         - Show this help message")
            print("  qubits [n]   - Set the number of qubits")
            print("  add [gate]   - Add a gate (H, X, Y, Z)")
            print("  list         - List the current circuit")
            print("  clear        - Clear the circuit")
            print("  run          - Run the simulation")
            print("  exit         - Exit the application\n")
        
        elif cmd == "qubits":
            if not args:
                print(f"Current qubits: {self.circuit['qubits']}")
                return
            
            try:
                n = int(args[0])
                if n < 1 or n > 5:
                    print("Error: Number of qubits must be between 1 and 5")
                    return
                
                self.circuit["qubits"] = n
                print(f"Set qubits to {n}")
            except ValueError:
                print("Error: Invalid number")
        
        elif cmd == "add":
            if not self.has_numpy:
                print("Error: Numpy not available")
                return
            
            if not args:
                print("Error: Missing gate type")
                print("Usage: add [H, X, Y, Z]")
                return
            
            gate_type = args[0].upper()
            if gate_type not in ["H", "X", "Y", "Z"]:
                print(f"Error: Unknown gate type: {gate_type}")
                print("Available gates: H, X, Y, Z")
                return
            
            target = 0
            if len(args) > 1:
                try:
                    target = int(args[1])
                    if target < 0 or target >= self.circuit["qubits"]:
                        print(f"Error: Target qubit must be between 0 and {self.circuit['qubits']-1}")
                        return
                except ValueError:
                    print("Error: Invalid target qubit")
                    return
            
            self.circuit["gates"].append({
                "type": gate_type,
                "target": target
            })
            
            print(f"Added {gate_type} gate to qubit {target}")
        
        elif cmd == "list":
            print("\nCurrent circuit:")
            print(f"Qubits: {self.circuit['qubits']}")
            print("Gates:")
            for i, gate in enumerate(self.circuit["gates"]):
                print(f"  {i}: {gate['type']} on qubit {gate['target']}")
            print("")
        
        elif cmd == "clear":
            self.circuit["gates"] = []
            print("Circuit cleared")
        
        elif cmd == "run":
            if not self.has_numpy:
                print("Error: Numpy not available")
                return
            
            print("\nRunning simulation...")
            
            try:
                # Simple quantum simulation
                state = self.np.zeros(2**self.circuit["qubits"])
                state[0] = 1.0  # |0...0⟩ state
                
                # Apply gates
                for gate in self.circuit["gates"]:
                    gate_type = gate["type"]
                    target = gate["target"]
                    
                    if gate_type == "H":  # Hadamard gate
                        # Apply Hadamard to target qubit
                        for i in range(len(state)):
                            if (i >> target) & 1:  # If target qubit is 1
                                state[i], state[i ^ (1 << target)] = (state[i] - state[i ^ (1 << target)]) / self.np.sqrt(2), (state[i] + state[i ^ (1 << target)]) / self.np.sqrt(2)
                    
                    elif gate_type == "X":  # Pauli-X (NOT) gate
                        # Apply X to target qubit
                        for i in range(len(state)):
                            if (i >> target) & 1:  # If target qubit is 1
                                state[i], state[i ^ (1 << target)] = state[i ^ (1 << target)], state[i]
                    
                    elif gate_type == "Z":  # Pauli-Z gate
                        # Apply Z to target qubit
                        for i in range(len(state)):
                            if (i >> target) & 1:  # If target qubit is 1
                                state[i] = -state[i]
                
                # Print the state vector
                print("State vector:")
                for i, amplitude in enumerate(state):
                    if abs(amplitude) > 1e-10:  # Only show non-zero amplitudes
                        bits = format(i, f'0{self.circuit["qubits"]}b')
                        print(f"  {amplitude:.4f} |{bits}⟩")
                
                # Measure the state
                probabilities = self.np.abs(state)**2
                
                # Simulate shots
                shots = 1000
                results = {}
                for _ in range(shots):
                    outcome = self.np.random.choice(len(state), p=probabilities)
                    outcome_bits = format(outcome, f'0{self.circuit["qubits"]}b')
                    results[outcome_bits] = results.get(outcome_bits, 0) + 1
                
                # Print the results
                print("\nMeasurement results (1000 shots):")
                for outcome, count in sorted(results.items()):
                    print(f"  |{outcome}⟩: {count} ({count/shots*100:.1f}%)")
                
                print("")
            
            except Exception as e:
                print(f"Simulation error: {e}")
        
        elif cmd == "exit":
            print(f"Exiting {self.app_name}...")
            self.running = False
        
        else:
            print(f"Unknown command: {cmd}")
            print("Type 'help' for a list of commands")

def main():
    """Main function"""
    # Create the app launcher
    launcher = AppLauncherBase("Quantum Simulator", "fa5s.atom")
    
    # Check if the GUI should be disabled
    if "--no-gui" in sys.argv:
        launcher.launch_terminal(QuantumSimulatorTerminal)
    else:
        launcher.launch_gui(QuantumSimulator)

if __name__ == "__main__":
    main()
