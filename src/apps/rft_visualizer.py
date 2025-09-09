#!/usr/bin/env python3
"""
RFT Visualizer - 2D Wave Pattern Engine
Simple 2D visualization with real-time wave patterns
"""

import sys
import math
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QSlider, QPushButton,
    QTextEdit, QHBoxLayout, QVBoxLayout, QStatusBar
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPainter, QColor, QPen


class WaveCanvas(QWidget):
    """Simple 2D wave visualization canvas"""
    def __init__(self):
        super().__init__()
        self.time_step = 0.0
        self.recursive_depth = 5
        self.frequency = 1.0
        self.amplitude = 100
        self.wave_speed = 0.5
        self.quantum_coupling = 0.618
        self.dark_mode = False
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Set background
        bg_color = QColor(15, 18, 22) if self.dark_mode else QColor(250, 250, 250)
        painter.fillRect(self.rect(), bg_color)
        
        # Draw wave patterns
        width = self.width()
        height = self.height()
        center_y = height // 2
        
        # Generate multiple wave layers
        for depth in range(self.recursive_depth):
            # Calculate wave properties for this depth
            freq_scale = self.frequency * (1 + depth * self.quantum_coupling)
            amp_scale = self.amplitude / (1 + depth * 0.3)
            phase_offset = depth * math.pi / 4
            
            # Set pen color with alpha based on depth
            if self.dark_mode:
                color = QColor(125, 196, 255, 255 - depth * 30)
            else:
                color = QColor(25, 118, 210, 255 - depth * 30)
            
            pen = QPen(color, 2)
            painter.setPen(pen)
            
            # Draw wave
            points = []
            for x in range(0, width, 2):
                # Normalized x position
                norm_x = (x / width) * 6 - 3  # Map to -3 to 3
                
                # Wave calculation with quantum coupling
                wave1 = math.sin(freq_scale * norm_x - self.wave_speed * self.time_step + phase_offset)
                quantum_mod = math.sin(self.quantum_coupling * norm_x + self.time_step) * 0.2
                
                y_offset = amp_scale * (wave1 + quantum_mod)
                y = center_y - y_offset
                
                if 0 <= y <= height:
                    points.append((x, int(y)))
            
            # Draw the wave line
            for i in range(len(points) - 1):
                painter.drawLine(points[i][0], points[i][1], points[i+1][0], points[i+1][1])
        
        # Draw center line
        pen = QPen(QColor(100, 100, 100, 100), 1)
        painter.setPen(pen)
        painter.drawLine(0, center_y, width, center_y)


class RFTVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RFT Visualizer")
        self.setGeometry(100, 100, 1200, 700)

        # State
        self.dark_mode = False
        
        # Timer for animation
        self.timer = QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.update_visualization)

        # UI components
        self.canvas = None
        self.metrics_text = None
        
        self.init_ui()
        self.apply_theme()
        
        # Start animation
        self.timer.start()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        root = QHBoxLayout(central)
        root.setSpacing(0)
        root.setContentsMargins(0, 0, 0, 0)

        # Left panel
        left = QWidget()
        left.setFixedWidth(280)
        left.setObjectName("LeftPanel")
        lyt = QVBoxLayout(left)
        lyt.setContentsMargins(20, 20, 20, 20)
        lyt.setSpacing(15)

        title = QLabel("RFT Visualizer")
        title.setObjectName("Title")
        lyt.addWidget(title)

        subtitle = QLabel("2D Wave Pattern Engine")
        subtitle.setObjectName("SubTitle")
        lyt.addWidget(subtitle)

        # Depth control
        lyt.addWidget(QLabel("Recursive Depth:"))
        self.depth_slider = QSlider(Qt.Horizontal)
        self.depth_slider.setRange(1, 10)
        self.depth_slider.setValue(5)
        self.depth_slider.valueChanged.connect(self.update_depth)
        lyt.addWidget(self.depth_slider)
        self.depth_label = QLabel("Depth: 5")
        lyt.addWidget(self.depth_label)

        # Frequency control
        lyt.addWidget(QLabel("Frequency:"))
        self.freq_slider = QSlider(Qt.Horizontal)
        self.freq_slider.setRange(1, 50)
        self.freq_slider.setValue(10)
        self.freq_slider.valueChanged.connect(self.update_frequency)
        lyt.addWidget(self.freq_slider)
        self.freq_label = QLabel("Freq: 1.0 Hz")
        lyt.addWidget(self.freq_label)

        # Control buttons
        self.pause_btn = QPushButton("⏸ Pause")
        self.pause_btn.clicked.connect(self.toggle_animation)
        lyt.addWidget(self.pause_btn)

        self.theme_btn = QPushButton("🌙 Dark Mode")
        self.theme_btn.clicked.connect(self.toggle_theme)
        lyt.addWidget(self.theme_btn)

        # Metrics section
        metrics_label = QLabel("Wave Metrics")
        metrics_label.setStyleSheet("font-weight: bold; margin-top: 20px; font-size: 14px;")
        lyt.addWidget(metrics_label)

        self.metrics_text = QTextEdit()
        self.metrics_text.setMaximumHeight(250)
        self.metrics_text.setReadOnly(True)
        self.metrics_text.setStyleSheet("""
            QTextEdit {
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 10px;
                line-height: 1.2;
                padding: 8px;
                border-radius: 6px;
            }
        """)
        lyt.addWidget(self.metrics_text)

        lyt.addStretch()
        root.addWidget(left)

        # Right panel - Wave canvas
        self.canvas = WaveCanvas()
        root.addWidget(self.canvas, 1)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("RFT Engine Active - 2D Mode")

    def apply_theme(self):
        if self.dark_mode:
            qss = """
            QMainWindow, QWidget { background:#0f1216; color:#dfe7ef; font-family:'Segoe UI'; }
            #Title { font-size:20px; font-weight:300; color:#dfe7ef; }
            #SubTitle { font-size:11px; color:#8aa0b3; }
            #LeftPanel { background:#12161b; border-right:1px solid #1f2a36; }
            QSlider::groove:horizontal { border:1px solid #1f2a36; height:8px; background:#12161b; border-radius:4px; }
            QSlider::handle:horizontal { background:#7dc4ff; width:18px; border-radius:9px; margin:-2px 0; }
            QPushButton { background:#12161b; border:1px solid #2a3847; border-radius:6px; padding:8px 14px; color:#c8d3de; }
            QPushButton:hover { background:#1d2b3a; }
            QTextEdit { background:#0a0a0a; border:1px solid #1f2a36; color:#dfe7ef; }
            QStatusBar { background:#12161b; border-top:1px solid #1f2a36; color:#8aa0b3; }
            """
            self.theme_btn.setText("☀ Light Mode")
        else:
            qss = """
            QMainWindow, QWidget { background:#fafafa; color:#243342; font-family:'Segoe UI'; }
            #Title { font-size:20px; font-weight:300; color:#2c3e50; }
            #SubTitle { font-size:11px; color:#8aa0b3; }
            #LeftPanel { background:#f8f9fa; border-right:1px solid #dee2e6; }
            QSlider::groove:horizontal { border:1px solid #dee2e6; height:8px; background:#e9ecef; border-radius:4px; }
            QSlider::handle:horizontal { background:#1976d2; width:18px; border-radius:9px; margin:-2px 0; }
            QPushButton { background:#f8f9fa; border:1px solid #dee2e6; border-radius:6px; padding:8px 14px; color:#495057; }
            QPushButton:hover { background:#e9ecef; }
            QTextEdit { background:#ffffff; border:1px solid #dee2e6; color:#243342; }
            QStatusBar { background:#f8f9fa; border-top:1px solid #dee2e6; color:#6c757d; }
            """
            self.theme_btn.setText("🌙 Dark Mode")

        self.setStyleSheet(qss)
        
        # Update canvas theme
        if self.canvas:
            self.canvas.dark_mode = self.dark_mode
            self.canvas.update()

    def toggle_theme(self):
        self.dark_mode = not self.dark_mode
        self.apply_theme()

    def update_depth(self, value):
        if self.canvas:
            self.canvas.recursive_depth = value
        self.depth_label.setText(f"Depth: {value}")

    def update_frequency(self, value):
        freq = value / 10.0
        if self.canvas:
            self.canvas.frequency = freq
        self.freq_label.setText(f"Freq: {freq:.1f} Hz")

    def toggle_animation(self):
        if self.timer.isActive():
            self.timer.stop()
            self.pause_btn.setText("▶ Resume")
        else:
            self.timer.start()
            self.pause_btn.setText("⏸ Pause")

    def update_visualization(self):
        if self.canvas:
            self.canvas.time_step += 0.1
            self.canvas.update()
            self.update_metrics()

    def update_metrics(self):
        if not self.canvas:
            return
            
        # Calculate some basic wave metrics
        time_step = self.canvas.time_step
        depth = self.canvas.recursive_depth
        freq = self.canvas.frequency
        coupling = self.canvas.quantum_coupling
        
        # Simulate wave analysis
        complexity = min(100, int(depth * 15))
        coherence = max(0, min(100, int(100 - (freq * 10))))
        interference = min(100, int(abs(math.sin(time_step)) * 100))
        coupling_effect = min(100, int(coupling * 100))
        phase_sync = max(0, min(100, int(75 + 25 * math.cos(time_step))))

        metrics_text = f"""
🌊 WAVE FIELD ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 WAVE PROPERTIES:
   • Recursive Layers: {depth}
   • Base Frequency: {freq:.1f} Hz
   • Field Complexity: {complexity}%
   • Wave Amplitude: {self.canvas.amplitude}px

🔄 FIELD DYNAMICS:
   • Coherence Level: {coherence}%
   • Interference: {interference}%
   • Phase Sync: {phase_sync}%
   • Coupling Effect: {coupling_effect}%

⚛️ QUANTUM PARAMETERS:
   • Quantum Coupling: {coupling:.3f}
   • Time Evolution: {time_step:.1f}s
   • Wave Speed: {self.canvas.wave_speed}
   • Pattern Mode: 2D Recursive

📈 FIELD EXPLANATION:
   • Multiple wave layers create depth
   • Quantum coupling adds complexity
   • Phase relationships show stability
   • Real-time frequency modulation

🔬 OBSERVED EFFECTS:
   • Layered wave interference
   • Recursive pattern generation
   • Dynamic phase evolution
   • Quantum-inspired coupling
        """
        
        self.metrics_text.setText(metrics_text)


def main():
    app = QApplication.instance()
    created = False
    if app is None:
        created = True
        app = QApplication(sys.argv)
        app.setApplicationName("RFT Visualizer")

    win = RFTVisualizer()
    win.show()
    print("🌊 RFT Visualizer launched!")

    if created:
        sys.exit(app.exec_())
    return app, win


if __name__ == "__main__":
    main()
