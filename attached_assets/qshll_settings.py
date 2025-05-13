import sys
import os
import math
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QPushButton, QMessageBox
)
from PyQt5.QtCore import Qt, QPointF, QRectF, pyqtSignal
from PyQt5.QtGui import QPainter, QPen, QColor, QLinearGradient

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

from apps.config import Config
from system_resonance_manager import Process, monitor_resonance_states

def validate_settings(freq_val, sigma_val, dt):
    # Define a sample vertices list (e.g., a square in 3D space)
    # Replace this with actual vertex data relevant to your application
    vertices = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
    # Map freq_val to priority and sigma_val to amplitude for Process
    return [Process(i, priority=freq_val, amplitude=complex(sigma_val, 0), vertices=vertices) for i in range(3)]

def load_stylesheet(qss_path):
    """Load the stylesheet from the given path, with fallback if not found."""
    if os.path.exists(qss_path):
        try:
            with open(qss_path, "r", encoding="utf-8") as f:
                print(f"✅ Stylesheet loaded from {qss_path}")
                return f.read()
        except UnicodeDecodeError as e:
            print(f"⚠️ Error decoding stylesheet from {qss_path}: {e}")
            print(f"Position: {e.start}, Character: {e.object[e.start]}")
            return ""
    print(f"⚠️ Stylesheet not found: {qss_path}")
    return ""

class Knob(QWidget):
    valueChanged = pyqtSignal(float)

    def __init__(self, min_val, max_val, default_val, parent=None):
        super().__init__(parent)
        self.min_val = min_val
        self.max_val = max_val
        self.value = default_val
        self.angle = self.value_to_angle(default_val)
        self.setFixedSize(80, 80)
        self.setObjectName("Knob")
        self.colors = {
            "gradient-start": "#FFFFFF",
            "gradient-end": "#E0E0E0",
            "outer-ring": "#4E6E74",
            "inner-ring": "#B0B3B8",
            "indicator": "#A8D3C1",
            "center-dot": "#A8D3C1"
        }
        self.update_colors_from_stylesheet()

    def update_colors_from_stylesheet(self):
        """Parse the stylesheet to extract custom color variables."""
        style = self.styleSheet()
        if not style:
            return
        for line in style.split(';'):
            line = line.strip()
            if line.startswith('--'):
                try:
                    key, value = line.split(':')
                    key = key.strip()[2:]
                    value = value.strip()
                    if key in self.colors:
                        self.colors[key] = value
                except ValueError:
                    continue
        print(f"Updated Knob Colors: {self.colors}")

    def value_to_angle(self, value):
        range_val = self.max_val - self.min_val
        angle_range = 270
        return -135 + (value - self.min_val) * angle_range / range_val

    def angle_to_value(self, angle):
        range_val = self.max_val - self.min_val
        angle_range = 270
        normalized = (angle + 135) / angle_range
        return self.min_val + normalized * range_val

    def setValue(self, value):
        if self.min_val <= value <= self.max_val:
            self.value = value
            self.angle = self.value_to_angle(value)
            self.update()
            self.valueChanged.emit(self.value)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.update_angle(event.pos())

    def mouseMoveEvent(self, event):
        self.update_angle(event.pos())

    def update_angle(self, pos):
        center = QPointF(self.width() / 2, self.height() / 2)
        delta = pos - center
        angle = math.degrees(math.atan2(delta.y(), delta.x())) - 90
        if angle < -135:
            angle = -135
        elif angle > 135:
            angle = 135
        self.angle = angle
        self.value = self.angle_to_value(angle)
        self.update()
        self.valueChanged.emit(self.value)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        gradient_start = QColor(self.colors["gradient-start"])
        gradient_end = QColor(self.colors["gradient-end"])
        outer_ring_color = QColor(self.colors["outer-ring"])
        inner_ring_color = QColor(self.colors["inner-ring"])
        indicator_color = QColor(self.colors["indicator"])
        center_dot_color = QColor(self.colors["center-dot"])

        rect = QRectF(10, 10, self.width() - 20, self.height() - 20)
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0, gradient_start)
        gradient.setColorAt(1, gradient_end)
        painter.setBrush(gradient)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(rect)

        painter.setPen(QPen(outer_ring_color, 3))
        painter.drawEllipse(rect)

        inner_rect = QRectF(12, 12, self.width() - 24, self.height() - 24)
        painter.setPen(QPen(inner_ring_color, 2))
        painter.drawEllipse(inner_rect)

        center = QPointF(self.width() / 2, self.height() / 2)
        radius = (self.width() - 20) / 2
        indicator_length = radius * 0.7
        indicator_angle = math.radians(self.angle)
        indicator_end = center + QPointF(
            math.cos(indicator_angle) * indicator_length,
            math.sin(indicator_angle) * indicator_length
        )
        painter.setPen(QPen(indicator_color, 5))
        painter.drawLine(center, indicator_end)

        painter.setBrush(center_dot_color)
        painter.drawEllipse(center, 5, 5)

class QSHLLSettings(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setObjectName("QSHLLSettings")
        self.setWindowTitle("Quantum Settings Hub")

        self.style_path = os.path.join(ROOT_DIR, "styles.qss")
        self.stylesheet = load_stylesheet(self.style_path)
        if not self.stylesheet:
            raise ValueError("Stylesheet could not be loaded; cannot proceed without styles.")
        self.setStyleSheet(self.stylesheet)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.title_label = QLabel("Quantum Settings Hub")
        self.title_label.setObjectName("TitleLabel")
        self.layout.addWidget(self.title_label)

        instructions = """
        **Optimize Your Quantum System:**

        - **Resonance Frequency (0.1–5.0)**: Sets oscillation speed.
        - **Resonance Spread (0.01–1.0)**: Adjusts precision vs. stability.
        """
        self.instructions_label = QLabel(instructions)
        self.instructions_label.setObjectName("InstructionsLabel")
        self.layout.addWidget(self.instructions_label)

        self.layout.addWidget(QLabel("Resonance Frequency (Oscillation Speed)"))
        cfg = Config()
        self.freq_label = QLabel(f"Value: {cfg.data.get('resonance_frequency', 1.0)}")
        self.freq_label.setObjectName("KnobLabel")
        self.layout.addWidget(self.freq_label)
        self.freq_knob = Knob(min_val=0.1, max_val=5.0, default_val=cfg.data.get('resonance_frequency', 1.0))
        self.freq_knob.setObjectName("FreqKnob")
        self.freq_knob.valueChanged.connect(self.update_freq_label)
        self.layout.addWidget(self.freq_knob)

        self.layout.addWidget(QLabel("Resonance Spread (Precision vs. Stability)"))
        self.sigma_label = QLabel(f"Value: {cfg.data.get('resonance_spread', 0.1)}")
        self.sigma_label.setObjectName("KnobLabel")
        self.layout.addWidget(self.sigma_label)
        self.sigma_knob = Knob(min_val=0.01, max_val=1.0, default_val=cfg.data.get('resonance_spread', 0.1))
        self.sigma_knob.setObjectName("SigmaKnob")
        self.sigma_knob.valueChanged.connect(self.update_sigma_label)
        self.layout.addWidget(self.sigma_knob)

        self.apply_btn = QPushButton("Save Settings")
        self.apply_btn.setObjectName("ApplyButton")
        self.apply_btn.clicked.connect(self.apply_settings)
        self.layout.addWidget(self.apply_btn)

    def update_freq_label(self, value):
        self.freq_label.setText(f"Value: {value:.1f}")

    def update_sigma_label(self, value):
        self.sigma_label.setText(f"Value: {value:.2f}")

    def apply_settings(self):
        freq_val = self.freq_knob.value
        sigma_val = self.sigma_knob.value
        try:
            processes = validate_settings(freq_val, sigma_val, dt=0.1)
            monitor_resonance_states(processes, 0.1)
            cfg = Config()
            cfg.save({"resonance_frequency": freq_val, "resonance_spread": sigma_val})
            QMessageBox.information(self, "Settings Saved", f"Frequency: {freq_val:.1f}\nSpread: {sigma_val:.2f}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply settings: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    style_path = os.path.join(ROOT_DIR, "styles.qss")
    stylesheet = load_stylesheet(style_path)
    if stylesheet:
        app.setStyleSheet(stylesheet)
    else:
        print("⚠️ No stylesheet applied; using default styles.")
    settings = QSHLLSettings()
    settings.show()
    sys.exit(app.exec_())