import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QSlider, QPushButton
from PyQt5.QtCore import Qt

# Add the root directory to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

from oscillator import Oscillator, validate_oscillator  # Assuming these are defined elsewhere
from config import Config

class QWaveComposer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Q-Wave Composer - Quantum Wave Generator")
        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)
        self.mainLayout = QVBoxLayout(self.centralWidget)
        self.mainLayout.addWidget(QLabel("Wave Frequency:"))
        cfg = Config()
        self.freqSlider = QSlider(Qt.Horizontal)
        self.freqSlider.setMinimum(1)
        self.freqSlider.setMaximum(50)
        self.freqSlider.setValue(int(cfg.data.get("quantum_frequency", 1.0) * 10))
        self.mainLayout.addWidget(self.freqSlider)
        self.composeBtn = QPushButton("Compose")
        self.composeBtn.clicked.connect(self.composeWave)
        self.mainLayout.addWidget(self.composeBtn)

        # Load stylesheet from root directory
        style_path = os.path.join(ROOT_DIR, "styles.qss")
        if os.path.exists(style_path):
            with open(style_path, "r") as f:
                self.setStyleSheet(f.read())
        else:
            print(f"Stylesheet not found at {style_path}")

    def composeWave(self):
        freq = self.freqSlider.value() / 10.0
        osc = Oscillator(freq, complex(1.0, 0.0), 0.0)
        waveform = validate_oscillator(osc, duration=20)
        print(f"[Q-Wave Composer] Generated waveform with frequency {freq}: {waveform[:5]}...")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    composer = QWaveComposer()
    composer.show()
    sys.exit(app.exec_())