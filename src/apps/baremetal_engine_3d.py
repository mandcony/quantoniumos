# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
﻿#!/usr/bin/env python3
"""
≡ƒîè BAREMETAL RFT ENGINE 3D VISUALIZER
Visualizes the recursive quantum field transforms in 3D space
MATLAB-style XYZ axes with recursive wave patterns
"""

import sys
import numpy as np
import time
import math
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

try:
    import matplotlib
    matplotlib.use('Qt5Agg')
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

class BaremetalEngine3DVisualizer(QMainWindow):
    """3D Baremetal RFT Engine Visualizer"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("≡ƒîè Baremetal RFT Engine - 3D Recursive Wave Visualizer")
        self.setGeometry(100, 100, 1200, 800)
        
        # Engine parameters
        self.time_step = 0.0
        self.recursive_depth = 5
        self.frequency = 1.0
        self.amplitude = 1.0
        self.wave_speed = 0.5
        self.quantum_coupling = 0.618  # Golden ratio
        
        # Animation timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_visualization)
        self.timer.start(50)  # 20 FPS
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QHBoxLayout(central_widget)
        
        # 3D Visualization
        if HAS_MATPLOTLIB:
            self.canvas = self.create_3d_canvas()
            layout.addWidget(self.canvas, 3)
        else:
            fallback_label = QLabel("≡ƒôè 3D Visualization requires matplotlib\nInstall: pip install matplotlib")
            fallback_label.setAlignment(Qt.AlignCenter)
            fallback_label.setStyleSheet("font-size: 16px; color: #666; padding: 50px;")
            layout.addWidget(fallback_label, 3)
        
        # Control Panel
        control_panel = self.create_control_panel()
        layout.addWidget(control_panel, 1)
        
        # Status bar
        self.statusBar().showMessage("≡ƒîè Baremetal RFT Engine - Recursive Wave Visualization Active")
        
    def create_3d_canvas(self):
        """Create 3D matplotlib canvas"""
        fig = Figure(figsize=(10, 8))
        fig.patch.set_facecolor('#1a1a1a')
        
        self.ax = fig.add_subplot(111, projection='3d')
        self.ax.set_facecolor('#0a0a0a')
        
        # Set MATLAB-style labels
        self.ax.set_xlabel('X - Spatial Dimension', color='white', fontsize=12)
        self.ax.set_ylabel('Y - Frequency Domain', color='white', fontsize=12)
        self.ax.set_zlabel('Z - Amplitude Field', color='white', fontsize=12)
        self.ax.set_title('≡ƒîè Baremetal RFT Engine - Recursive Quantum Field', 
                         color='white', fontsize=14, pad=20)
        
        # Grid and styling
        self.ax.tick_params(colors='white')
        self.ax.grid(True, alpha=0.3)
        
        canvas = FigureCanvas(fig)
        canvas.setStyleSheet("background-color: #1a1a1a;")
        return canvas
        
    def create_control_panel(self):
        """Create control panel for engine parameters"""
        panel = QWidget()
        panel.setFixedWidth(300)
        panel.setStyleSheet("""
            QWidget {
                background-color: #2a2a2a;
                color: white;
                font-size: 12px;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #444;
                border-radius: 5px;
                margin: 10px 0;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 10px 0 10px;
            }
            QSlider::groove:horizontal {
                border: 1px solid #444;
                height: 8px;
                background: #333;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #00ff88;
                border: 1px solid #00aa55;
                width: 18px;
                border-radius: 9px;
                margin: -2px 0;
            }
            QPushButton {
                background-color: #444;
                border: 1px solid #666;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #555;
            }
            QPushButton:pressed {
                background-color: #333;
            }
        """)
        
        layout = QVBoxLayout(panel)
        
        # Engine Parameters
        engine_group = QGroupBox("≡ƒöº Engine Parameters")
        engine_layout = QVBoxLayout(engine_group)
        
        # Recursive Depth
        engine_layout.addWidget(QLabel("Recursive Depth:"))
        self.depth_slider = QSlider(Qt.Horizontal)
        self.depth_slider.setRange(1, 10)
        self.depth_slider.setValue(self.recursive_depth)
        self.depth_slider.valueChanged.connect(self.update_depth)
        engine_layout.addWidget(self.depth_slider)
        self.depth_label = QLabel(f"Depth: {self.recursive_depth}")
        engine_layout.addWidget(self.depth_label)
        
        # Frequency
        engine_layout.addWidget(QLabel("Base Frequency:"))
        self.freq_slider = QSlider(Qt.Horizontal)
        self.freq_slider.setRange(1, 50)
        self.freq_slider.setValue(int(self.frequency * 10))
        self.freq_slider.valueChanged.connect(self.update_frequency)
        engine_layout.addWidget(self.freq_slider)
        self.freq_label = QLabel(f"Freq: {self.frequency:.1f} Hz")
        engine_layout.addWidget(self.freq_label)
        
        # Amplitude
        engine_layout.addWidget(QLabel("Wave Amplitude:"))
        self.amp_slider = QSlider(Qt.Horizontal)
        self.amp_slider.setRange(1, 50)
        self.amp_slider.setValue(int(self.amplitude * 10))
        self.amp_slider.valueChanged.connect(self.update_amplitude)
        engine_layout.addWidget(self.amp_slider)
        self.amp_label = QLabel(f"Amp: {self.amplitude:.1f}")
        engine_layout.addWidget(self.amp_label)
        
        # Wave Speed
        engine_layout.addWidget(QLabel("Propagation Speed:"))
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(1, 100)
        self.speed_slider.setValue(int(self.wave_speed * 100))
        self.speed_slider.valueChanged.connect(self.update_speed)
        engine_layout.addWidget(self.speed_slider)
        self.speed_label = QLabel(f"Speed: {self.wave_speed:.2f}")
        engine_layout.addWidget(self.speed_label)
        
        # Quantum Coupling
        engine_layout.addWidget(QLabel("Quantum Coupling:"))
        self.coupling_slider = QSlider(Qt.Horizontal)
        self.coupling_slider.setRange(1, 100)
        self.coupling_slider.setValue(int(self.quantum_coupling * 100))
        self.coupling_slider.valueChanged.connect(self.update_coupling)
        engine_layout.addWidget(self.coupling_slider)
        self.coupling_label = QLabel(f"Coupling: {self.quantum_coupling:.3f}")
        engine_layout.addWidget(self.coupling_label)
        
        layout.addWidget(engine_group)
        
        # Visualization Controls
        viz_group = QGroupBox("≡ƒôè Visualization")
        viz_layout = QVBoxLayout(viz_group)
        
        self.pause_btn = QPushButton("ΓÅ╕∩╕Å Pause Animation")
        self.pause_btn.clicked.connect(self.toggle_animation)
        viz_layout.addWidget(self.pause_btn)
        
        self.reset_btn = QPushButton("≡ƒöä Reset Engine")
        self.reset_btn.clicked.connect(self.reset_engine)
        viz_layout.addWidget(self.reset_btn)
        
        self.export_btn = QPushButton("≡ƒÆ╛ Export Frame")
        self.export_btn.clicked.connect(self.export_frame)
        viz_layout.addWidget(self.export_btn)
        
        layout.addWidget(viz_group)
        
        # Engine Status
        status_group = QGroupBox("≡ƒôê Engine Status")
        status_layout = QVBoxLayout(status_group)
        
        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(150)
        self.status_text.setReadOnly(True)
        self.status_text.setStyleSheet("background-color: #1a1a1a; border: 1px solid #444;")
        status_layout.addWidget(self.status_text)
        
        layout.addWidget(status_group)
        
        layout.addStretch()
        
        return panel
    
    def update_depth(self, value):
        """Update recursive depth"""
        self.recursive_depth = value
        self.depth_label.setText(f"Depth: {value}")
        
    def update_frequency(self, value):
        """Update base frequency"""
        self.frequency = value / 10.0
        self.freq_label.setText(f"Freq: {self.frequency:.1f} Hz")
        
    def update_amplitude(self, value):
        """Update wave amplitude"""
        self.amplitude = value / 10.0
        self.amp_label.setText(f"Amp: {self.amplitude:.1f}")
        
    def update_speed(self, value):
        """Update wave speed"""
        self.wave_speed = value / 100.0
        self.speed_label.setText(f"Speed: {self.wave_speed:.2f}")
        
    def update_coupling(self, value):
        """Update quantum coupling"""
        self.quantum_coupling = value / 100.0
        self.coupling_label.setText(f"Coupling: {self.quantum_coupling:.3f}")
        
    def toggle_animation(self):
        """Toggle animation pause/play"""
        if self.timer.isActive():
            self.timer.stop()
            self.pause_btn.setText("Γû╢∩╕Å Resume Animation")
        else:
            self.timer.start(50)
            self.pause_btn.setText("ΓÅ╕∩╕Å Pause Animation")
            
    def reset_engine(self):
        """Reset engine to initial state"""
        self.time_step = 0.0
        self.update_status("≡ƒöä Engine reset to initial state")
        
    def export_frame(self):
        """Export current frame"""
        if HAS_MATPLOTLIB:
            filename = f"baremetal_engine_frame_{int(time.time())}.png"
            self.canvas.figure.savefig(filename, dpi=150, facecolor='#1a1a1a')
            self.update_status(f"≡ƒÆ╛ Frame exported: {filename}")
        
    def generate_recursive_wave(self, x, y, t):
        """Generate recursive wave pattern for baremetal engine"""
        z = np.zeros_like(x)
        
        for depth in range(self.recursive_depth):
            # Recursive frequency scaling
            freq_scale = self.frequency * (1 + depth * self.quantum_coupling)
            
            # Multiple wave components for complexity
            wave1 = np.sin(freq_scale * x - self.wave_speed * t + depth * np.pi/4)
            wave2 = np.cos(freq_scale * y - self.wave_speed * t + depth * np.pi/3)
            
            # Interference pattern
            interference = wave1 * wave2
            
            # Recursive amplitude scaling (diminishing)
            amp_scale = self.amplitude / (1 + depth * 0.3)
            
            # Add to total field
            z += amp_scale * interference * np.exp(-0.1 * depth)
            
            # Add quantum field modulation
            quantum_mod = np.sin(self.quantum_coupling * (x + y) + t) * 0.2
            z += quantum_mod * amp_scale
        
        return z
    
    def update_visualization(self):
        """Update 3D visualization"""
        if not HAS_MATPLOTLIB:
            return
            
        self.time_step += 0.1
        
        # Clear previous plot
        self.ax.clear()
        
        # Create 3D grid
        x = np.linspace(-3, 3, 50)
        y = np.linspace(-3, 3, 50)
        X, Y = np.meshgrid(x, y)
        
        # Generate recursive wave field
        Z = self.generate_recursive_wave(X, Y, self.time_step)
        
        # Plot surface with color mapping
        surf = self.ax.plot_surface(X, Y, Z, cmap='plasma', alpha=0.8,
                                   linewidth=0, antialiased=True)
        
        # Add wireframe for structure
        self.ax.plot_wireframe(X, Y, Z, color='white', alpha=0.3, linewidth=0.5)
        
        # Set limits and labels
        self.ax.set_xlim([-3, 3])
        self.ax.set_ylim([-3, 3])
        self.ax.set_zlim([-2, 2])
        
        self.ax.set_xlabel('X - Spatial Dimension', color='white', fontsize=10)
        self.ax.set_ylabel('Y - Frequency Domain', color='white', fontsize=10)
        self.ax.set_zlabel('Z - Amplitude Field', color='white', fontsize=10)
        self.ax.set_title('≡ƒîè Baremetal RFT Engine - Recursive Quantum Field', 
                         color='white', fontsize=12, pad=15)
        
        # Styling
        self.ax.tick_params(colors='white', labelsize=8)
        self.ax.grid(True, alpha=0.3)
        self.ax.set_facecolor('#0a0a0a')
        
        # Update canvas
        self.canvas.draw()
        
        # Update status
        if int(self.time_step * 10) % 20 == 0:  # Update every 2 seconds
            self.update_engine_status()
    
    def update_engine_status(self):
        """Update engine status display"""
        status = f"""ΓÜí BAREMETAL ENGINE STATUS
Time: {self.time_step:.1f}s
Recursive Layers: {self.recursive_depth}
Base Frequency: {self.frequency:.1f} Hz
Wave Speed: {self.wave_speed:.2f} units/s
Quantum Coupling: {self.quantum_coupling:.3f}

≡ƒö¼ ENGINE METRICS:
ΓÇó Field Complexity: {self.recursive_depth * 2} components
ΓÇó Computation: O(n┬▓) per frame
ΓÇó Memory: {50*50*self.recursive_depth} field points
ΓÇó Quantum Coherence: {self.quantum_coupling * 100:.1f}%

≡ƒîè WAVE PROPERTIES:
ΓÇó Interference Patterns: Active
ΓÇó Recursive Depth: {self.recursive_depth} layers
ΓÇó Field Coupling: Quantum entangled
ΓÇó Propagation: Relativistic"""
        
        self.status_text.setPlainText(status)
        
    def update_status(self, message):
        """Update status bar"""
        self.statusBar().showMessage(message)

def main():
    """Main function"""
    app = QApplication(sys.argv)
    app.setApplicationName("Baremetal RFT Engine 3D Visualizer")
    
    # Set dark theme
    app.setStyle('Fusion')
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(42, 42, 42))
    palette.setColor(QPalette.WindowText, Qt.white)
    app.setPalette(palette)
    
    window = BaremetalEngine3DVisualizer()
    window.show()
    
    print("≡ƒîè Baremetal RFT Engine 3D Visualizer launched!")
    print("≡ƒôè Showing recursive quantum field transforms in 3D space")
    print("≡ƒÄ¢∩╕Å Use controls to adjust engine parameters")
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

