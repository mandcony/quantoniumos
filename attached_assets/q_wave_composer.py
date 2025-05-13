import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QSlider, QPushButton
from PyQt5.QtCore import Qt

# Add the root directory to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

try:
    from oscillator import Oscillator, validate_oscillator  # Ensures oscillator module is present
except ImportError:
    print("⚠️ Warning: 'oscillator' module not found. Some features may be disabled.")
    Oscillator, validate_oscillator = None, None

try:
    from apps.config import Config
except ImportError:
    print("⚠️ Warning: 'config' module not found. Default values will be used.")
    Config = None

class QWaveComposer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Q-Wave Composer - Quantum Wave Generator")
        self.setGeometry(100, 100, 500, 300)  # Window size
        
        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)
        self.mainLayout = QVBoxLayout(self.centralWidget)

        # Frequency Label
        self.freqLabel = QLabel("Wave Frequency: 1.0 Hz", alignment=Qt.AlignCenter)
        self.mainLayout.addWidget(self.freqLabel)

        # Frequency Slider
        self.freqSlider = QSlider(Qt.Horizontal)
        self.freqSlider.setMinimum(1)  # 0.1 Hz (1 / 10)
        self.freqSlider.setMaximum(50)  # 5 Hz (50 / 10)
        self.freqSlider.setValue(10)  # Default: 1.0 Hz
        self.freqSlider.valueChanged.connect(self.updateLabel)
        self.mainLayout.addWidget(self.freqSlider)

        # Compose Button
        self.composeBtn = QPushButton("Compose")
        self.composeBtn.clicked.connect(self.composeWave)
        self.mainLayout.addWidget(self.composeBtn)

        # Load stylesheet
        self.load_stylesheet()

    def load_stylesheet(self):
        """
        Loads the stylesheet from file if available, otherwise logs an error.
        """
        style_path = os.path.join(ROOT_DIR, "styles.qss")
        try:
            if os.path.exists(style_path):
                with open(style_path, "r", encoding="utf-8") as f:
                    self.setStyleSheet(f.read())
                print(f"✅ Stylesheet loaded from {style_path}")
            else:
                print(f"⚠️ Stylesheet not found at {style_path}, using default Qt styles.")
        except Exception as e:
            print(f"❌ Error loading stylesheet: {e}")

    def updateLabel(self):
        """
        Updates the frequency label when slider is moved.
        """
        freq = self.freqSlider.value() / 10.0  # Convert to Hz
        self.freqLabel.setText(f"Wave Frequency: {freq:.1f} Hz")

    def composeWave(self):
        """
        Generates a waveform using the selected frequency.
        """
        if not Oscillator or not validate_oscillator:
            print("❌ Error: Oscillator module is missing.")
            return

        freq = self.freqSlider.value() / 10.0
        osc = Oscillator(freq, complex(1.0, 0.0), 0.0)
        waveform = validate_oscillator(osc, duration=20)

        print(f"🎼 [Q-Wave Composer] Generated waveform with frequency {freq:.1f} Hz: {waveform[:5]}...")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    composer = QWaveComposer()
    composer.show()
    sys.exit(app.exec_())
