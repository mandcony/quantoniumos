import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QListWidget, QLineEdit
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from multi_qubit_state import MultiQubitState
from geometric_container import GeometricContainer
from quantum_search import QuantumSearch

class QuantumGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Quantonium OS - Quantum System")
        self.setGeometry(100, 100, 800, 600)

        self.container_list = []
        self.q_state = MultiQubitState(3)
        self.quantum_search = QuantumSearch()

        self.initUI()

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()

        self.label = QLabel("Quantum State Visualization")
        layout.addWidget(self.label)

        self.canvas = FigureCanvas(plt.figure(figsize=(5, 3)))
        layout.addWidget(self.canvas)

        self.update_graph()

        self.add_container_button = QPushButton("Add Geometric Container")
        self.add_container_button.clicked.connect(self.add_container)
        layout.addWidget(self.add_container_button)

        self.container_list_widget = QListWidget()
        layout.addWidget(self.container_list_widget)

        self.resonance_input = QLineEdit()
        self.resonance_input.setPlaceholderText("Enter frequency for quantum search")
        layout.addWidget(self.resonance_input)

        self.search_button = QPushButton("Run Resonance-Based Search")
        self.search_button.clicked.connect(self.run_search)
        layout.addWidget(self.search_button)

        self.result_label = QLabel("Search Result: ")
        layout.addWidget(self.result_label)

        central_widget.setLayout(layout)

    def update_graph(self):
        """Updates the quantum state visualization."""
        self.canvas.figure.clear()
        ax = self.canvas.figure.add_subplot(111)

        amplitudes = np.abs(self.q_state.state_vector) ** 2
        ax.bar(range(len(amplitudes)), amplitudes, color='blue')

        ax.set_xlabel("Quantum State Index")
        ax.set_ylabel("Probability Amplitude")
        ax.set_title("Quantum State Probabilities")
        self.canvas.draw()

    def add_container(self):
        """Adds a new geometric container."""
        new_container = GeometricContainer(f"Container_{len(self.container_list)}", [[0, 0, 0], [1, 1, 1]])
        new_container.resonant_frequencies = [0.1 * len(self.container_list)]
        self.container_list.append(new_container)
        self.container_list_widget.addItem(f"{new_container.id} - Resonance: {new_container.resonant_frequencies}")

    def run_search(self):
        """Runs quantum resonance search based on user input."""
        try:
            target_frequency = float(self.resonance_input.text())
        except ValueError:
            self.result_label.setText("Invalid input: Enter a valid number")
            return

        best_match = self.quantum_search.search_database(self.container_list, target_frequency)

        if best_match:
            self.result_label.setText(f"Found: {best_match.id} (Resonance: {best_match.resonant_frequencies})")
        else:
            self.result_label.setText("No matching container found.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = QuantumGUI()
    window.show()
    sys.exit(app.exec_())
