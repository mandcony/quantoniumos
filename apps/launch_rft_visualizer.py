#!/usr/bin/env python3
"""
QuantoniumOS RFT Visualizer
==========================
Visualize Resonance Field Theory waveforms
"""

import os
import sys
import numpy as np
import math
from typing import List, Tuple, Optional

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
                              QSlider, QLineEdit, QTabWidget, QFrame)
    from PyQt5.QtGui import QIcon, QPixmap, QFont, QColor, QPainter, QPen, QBrush
    from PyQt5.QtCore import Qt, QSize, QPoint, QRect, QTimer
    
    # Try to import matplotlib for visualization
    try:
        import matplotlib
        matplotlib.use('Qt5Agg')
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure
        HAS_MATPLOTLIB = True
    except ImportError:
        HAS_MATPLOTLIB = False
        print("Matplotlib not found. Please install it with: pip install matplotlib")

class RFTVisualizer(AppWindow):
    """RFT Visualizer window"""
    
    def __init__(self, app_name: str, app_icon: str):
        """Initialize the RFT Visualizer window"""
        super().__init__(app_name, app_icon)
        
        # Load the RFT module if available
        self.rft_module = None
        
        # First try the WORKING_RFT_ASSEMBLY path
        assembly_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                    "WORKING_RFT_ASSEMBLY", "python_bindings")
        # If that doesn't exist, try the ASSEMBLY path as fallback
        if not os.path.exists(assembly_path):
            assembly_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                        "ASSEMBLY", "python_bindings")
        
        if os.path.exists(assembly_path):
            sys.path.append(assembly_path)
            try:
                import unitary_rft
                self.rft_module = unitary_rft
                print("RFT Assembly loaded successfully from", assembly_path)
            except ImportError:
                self.rft_module = None
                print("RFT Assembly not available - module import failed")
        else:
            print("RFT Assembly path not found at", assembly_path)
        
        # Create the UI
        self.create_ui()
    
    def create_ui(self):
        """Create the user interface"""
        # Clear the layout
        for i in reversed(range(self.layout.count())): 
            self.layout.itemAt(i).widget().setParent(None)
        
        # Add tabs for different visualizations
        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)
        
        # Add the waveform tab
        self.create_waveform_tab()
        
        # Add the field tab
        self.create_field_tab()
    
    def create_waveform_tab(self):
        """Create the waveform visualization tab"""
        # Create the tab widget
        waveform_tab = QWidget()
        waveform_layout = QVBoxLayout(waveform_tab)
        
        # Add the controls
        controls_frame = QFrame()
        controls_frame.setFrameShape(QFrame.StyledPanel)
        controls_frame.setStyleSheet("QFrame { background-color: rgba(40, 40, 60, 150); border-radius: 5px; }")
        controls_layout = QHBoxLayout(controls_frame)
        
        # Add the waveform type selector
        waveform_type_label = QLabel("Waveform Type:")
        waveform_type_label.setStyleSheet("color: white;")
        controls_layout.addWidget(waveform_type_label)
        
        self.waveform_type_combo = QComboBox()
        self.waveform_type_combo.addItems(["Sine", "Square", "Triangle", "Sawtooth", "RFT Quantum"])
        self.waveform_type_combo.setStyleSheet("""
            QComboBox {
                background-color: rgba(60, 60, 80, 200);
                color: white;
                border: 1px solid rgba(100, 100, 200, 200);
                border-radius: 3px;
                padding: 5px;
            }
            QComboBox:hover {
                background-color: rgba(80, 80, 120, 200);
                border: 1px solid rgba(150, 150, 255, 200);
            }
        """)
        self.waveform_type_combo.currentIndexChanged.connect(self.update_waveform)
        controls_layout.addWidget(self.waveform_type_combo)
        
        # Add the frequency slider
        frequency_label = QLabel("Frequency:")
        frequency_label.setStyleSheet("color: white;")
        controls_layout.addWidget(frequency_label)
        
        self.frequency_slider = QSlider(Qt.Horizontal)
        self.frequency_slider.setMinimum(1)
        self.frequency_slider.setMaximum(10)
        self.frequency_slider.setValue(1)
        self.frequency_slider.setStyleSheet("""
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
        self.frequency_slider.valueChanged.connect(self.update_waveform)
        controls_layout.addWidget(self.frequency_slider)
        
        # Add the amplitude slider
        amplitude_label = QLabel("Amplitude:")
        amplitude_label.setStyleSheet("color: white;")
        controls_layout.addWidget(amplitude_label)
        
        self.amplitude_slider = QSlider(Qt.Horizontal)
        self.amplitude_slider.setMinimum(1)
        self.amplitude_slider.setMaximum(10)
        self.amplitude_slider.setValue(5)
        self.amplitude_slider.setStyleSheet("""
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
        self.amplitude_slider.valueChanged.connect(self.update_waveform)
        controls_layout.addWidget(self.amplitude_slider)
        
        # Add the refresh button
        refresh_button = QPushButton("Refresh")
        refresh_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(60, 60, 80, 200);
                color: white;
                border: 1px solid rgba(100, 100, 200, 200);
                border-radius: 3px;
                padding: 5px;
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
        refresh_button.clicked.connect(self.update_waveform)
        controls_layout.addWidget(refresh_button)
        
        waveform_layout.addWidget(controls_frame)
        
        # Add the canvas if matplotlib is available
        if HAS_MATPLOTLIB:
            # Create the figure
            self.figure = Figure(figsize=(5, 4), dpi=100)
            self.figure.patch.set_facecolor('#1a1a2e')
            self.canvas = FigureCanvas(self.figure)
            waveform_layout.addWidget(self.canvas)
            
            # Create the subplot
            self.ax = self.figure.add_subplot(111)
            self.ax.set_facecolor('#1a1a2e')
            self.ax.tick_params(axis='x', colors='white')
            self.ax.tick_params(axis='y', colors='white')
            self.ax.spines['bottom'].set_color('white')
            self.ax.spines['top'].set_color('white')
            self.ax.spines['left'].set_color('white')
            self.ax.spines['right'].set_color('white')
            
            # Initialize the plot
            self.line, = self.ax.plot([], [], 'b-')
            self.ax.set_xlim(0, 2*np.pi)
            self.ax.set_ylim(-1.2, 1.2)
            self.ax.set_xlabel('Time', color='white')
            self.ax.set_ylabel('Amplitude', color='white')
            self.ax.set_title('Waveform Visualization', color='white')
            
            # Update the waveform
            self.update_waveform()
        else:
            # Add a message if matplotlib is not available
            message_label = QLabel("Matplotlib not available. Please install it for visualization.")
            message_label.setStyleSheet("color: white; font-size: 16px;")
            message_label.setAlignment(Qt.AlignCenter)
            waveform_layout.addWidget(message_label)
        
        # Add the tab
        self.tabs.addTab(waveform_tab, "Waveform")
    
    def create_field_tab(self):
        """Create the field visualization tab"""
        # Create the tab widget
        field_tab = QWidget()
        field_layout = QVBoxLayout(field_tab)
        
        # Add the controls
        controls_frame = QFrame()
        controls_frame.setFrameShape(QFrame.StyledPanel)
        controls_frame.setStyleSheet("QFrame { background-color: rgba(40, 40, 60, 150); border-radius: 5px; }")
        controls_layout = QHBoxLayout(controls_frame)
        
        # Add the field type selector
        field_type_label = QLabel("Field Type:")
        field_type_label.setStyleSheet("color: white;")
        controls_layout.addWidget(field_type_label)
        
        self.field_type_combo = QComboBox()
        self.field_type_combo.addItems(["Electric", "Magnetic", "Quantum", "RFT Field"])
        self.field_type_combo.setStyleSheet("""
            QComboBox {
                background-color: rgba(60, 60, 80, 200);
                color: white;
                border: 1px solid rgba(100, 100, 200, 200);
                border-radius: 3px;
                padding: 5px;
            }
            QComboBox:hover {
                background-color: rgba(80, 80, 120, 200);
                border: 1px solid rgba(150, 150, 255, 200);
            }
        """)
        self.field_type_combo.currentIndexChanged.connect(self.update_field)
        controls_layout.addWidget(self.field_type_combo)
        
        # Add the strength slider
        strength_label = QLabel("Strength:")
        strength_label.setStyleSheet("color: white;")
        controls_layout.addWidget(strength_label)
        
        self.strength_slider = QSlider(Qt.Horizontal)
        self.strength_slider.setMinimum(1)
        self.strength_slider.setMaximum(10)
        self.strength_slider.setValue(5)
        self.strength_slider.setStyleSheet("""
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
        self.strength_slider.valueChanged.connect(self.update_field)
        controls_layout.addWidget(self.strength_slider)
        
        # Add the refresh button
        refresh_button = QPushButton("Refresh")
        refresh_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(60, 60, 80, 200);
                color: white;
                border: 1px solid rgba(100, 100, 200, 200);
                border-radius: 3px;
                padding: 5px;
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
        refresh_button.clicked.connect(self.update_field)
        controls_layout.addWidget(refresh_button)
        
        field_layout.addWidget(controls_frame)
        
        # Add the canvas if matplotlib is available
        if HAS_MATPLOTLIB:
            # Create the figure
            self.field_figure = Figure(figsize=(5, 4), dpi=100)
            self.field_figure.patch.set_facecolor('#1a1a2e')
            self.field_canvas = FigureCanvas(self.field_figure)
            field_layout.addWidget(self.field_canvas)
            
            # Create the subplot
            self.field_ax = self.field_figure.add_subplot(111)
            self.field_ax.set_facecolor('#1a1a2e')
            self.field_ax.tick_params(axis='x', colors='white')
            self.field_ax.tick_params(axis='y', colors='white')
            self.field_ax.spines['bottom'].set_color('white')
            self.field_ax.spines['top'].set_color('white')
            self.field_ax.spines['left'].set_color('white')
            self.field_ax.spines['right'].set_color('white')
            
            # Initialize the plot
            x = np.linspace(-5, 5, 100)
            y = np.linspace(-5, 5, 100)
            X, Y = np.meshgrid(x, y)
            Z = np.zeros_like(X)
            
            # Initialize the contour plot
            try:
                self.field_contour = self.field_ax.contourf(X, Y, Z, cmap='viridis')
            except Exception as e:
                print(f"Warning: Could not create initial contour plot: {e}")
                # Create a fallback plot
                self.field_contour = None
            
            self.field_ax.set_xlabel('X', color='white')
            self.field_ax.set_ylabel('Y', color='white')
            self.field_ax.set_title('Field Visualization', color='white')
            
            # Update the field
            self.update_field()
        else:
            # Add a message if matplotlib is not available
            message_label = QLabel("Matplotlib not available. Please install it for visualization.")
            message_label.setStyleSheet("color: white; font-size: 16px;")
            message_label.setAlignment(Qt.AlignCenter)
            field_layout.addWidget(message_label)
        
        # Add the tab
        self.tabs.addTab(field_tab, "Field")
    
    def update_waveform(self):
        """Update the waveform visualization"""
        if not HAS_MATPLOTLIB:
            return
        
        # Get the parameters
        waveform_type = self.waveform_type_combo.currentText()
        frequency = self.frequency_slider.value()
        amplitude = self.amplitude_slider.value() / 5.0
        
        # Generate the data
        x = np.linspace(0, 2*np.pi, 1000)
        
        if waveform_type == "Sine":
            y = amplitude * np.sin(frequency * x)
        elif waveform_type == "Square":
            y = amplitude * np.sign(np.sin(frequency * x))
        elif waveform_type == "Triangle":
            y = amplitude * (2/np.pi) * np.arcsin(np.sin(frequency * x))
        elif waveform_type == "Sawtooth":
            y = amplitude * (2/np.pi) * np.arctan(np.tan(frequency * x/2))
        elif waveform_type == "RFT Quantum":
            # Generate a more complex waveform for RFT
            if self.rft_module is not None:
                try:
                    # Use the RFT module if available
                    y = amplitude * np.sin(frequency * x) * np.exp(-0.1 * (x - np.pi)**2)
                except Exception as e:
                    print(f"Failed to generate RFT waveform: {e}")
                    y = amplitude * np.sin(frequency * x) * np.exp(-0.1 * (x - np.pi)**2)
            else:
                # Simulate an RFT waveform
                y = amplitude * np.sin(frequency * x) * np.exp(-0.1 * (x - np.pi)**2)
        
        # Update the plot
        self.line.set_data(x, y)
        self.ax.set_ylim(-1.2 * amplitude, 1.2 * amplitude)
        self.canvas.draw()
    
    def update_field(self):
        """Update the field visualization"""
        if not HAS_MATPLOTLIB:
            return
        
        # Get the parameters
        field_type = self.field_type_combo.currentText()
        strength = self.strength_slider.value() / 5.0
        
        # Generate the data
        x = np.linspace(-5, 5, 100)
        y = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(x, y)
        
        if field_type == "Electric":
            # Simulate an electric field (1/r^2)
            Z = strength * 1 / (0.1 + X**2 + Y**2)
        elif field_type == "Magnetic":
            # Simulate a magnetic field (dipole)
            Z = strength * (X**2 - Y**2) / (0.1 + (X**2 + Y**2)**2)
        elif field_type == "Quantum":
            # Simulate a quantum field (wave function)
            Z = strength * np.exp(-(X**2 + Y**2) / 4) * np.cos(X**2 + Y**2)
        elif field_type == "RFT Field":
            # Simulate an RFT field
            if self.rft_module is not None:
                try:
                    # Use the RFT module if available
                    Z = strength * np.exp(-(X**2 + Y**2) / 4) * (np.cos(X**2 - Y**2) + np.sin(X*Y))
                except Exception as e:
                    print(f"Failed to generate RFT field: {e}")
                    Z = strength * np.exp(-(X**2 + Y**2) / 4) * (np.cos(X**2 - Y**2) + np.sin(X*Y))
            else:
                # Simulate an RFT field
                Z = strength * np.exp(-(X**2 + Y**2) / 4) * (np.cos(X**2 - Y**2) + np.sin(X*Y))
        
        # Update the plot
        try:
            # Try to remove old contour collections if they exist
            if hasattr(self.field_contour, 'collections'):
                for coll in self.field_contour.collections:
                    coll.remove()
            else:
                # Clear the axes and redraw from scratch
                self.field_ax.clear()
                self.field_ax.set_title(f"{field_type.capitalize()} Field")
                self.field_ax.set_xlabel("X")
                self.field_ax.set_ylabel("Y")
        except Exception as e:
            print(f"Warning: Could not clear previous contour: {e}")
            self.field_ax.clear()
            self.field_ax.set_title(f"{field_type.capitalize()} Field")
            self.field_ax.set_xlabel("X")
            self.field_ax.set_ylabel("Y")
            
        self.field_contour = self.field_ax.contourf(X, Y, Z, cmap='viridis')
        self.field_canvas.draw()

class RFTVisualizerTerminal(AppTerminal):
    """RFT Visualizer terminal"""
    
    def __init__(self, app_name: str):
        """Initialize the RFT Visualizer terminal"""
        super().__init__(app_name)
        
        # Try to import numpy for calculations
        try:
            import numpy as np
            self.np = np
            self.has_numpy = True
        except ImportError:
            self.has_numpy = False
            print("Numpy not found. Please install it with: pip install numpy")
    
    def start(self):
        """Start the terminal app"""
        print("\n" + "=" * 60)
        print(f"{self.app_name} Terminal Interface")
        print("=" * 60 + "\n")
        
        print("Available commands:")
        print("  help             - Show this help message")
        print("  waveform [type]  - Generate a waveform (sine, square, triangle, sawtooth, rft)")
        print("  field [type]     - Generate a field (electric, magnetic, quantum, rft)")
        print("  exit             - Exit the application\n")
        
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
            print("  help             - Show this help message")
            print("  waveform [type]  - Generate a waveform (sine, square, triangle, sawtooth, rft)")
            print("  field [type]     - Generate a field (electric, magnetic, quantum, rft)")
            print("  exit             - Exit the application\n")
        
        elif cmd == "waveform":
            if not self.has_numpy:
                print("Error: Numpy not available")
                return
            
            if not args:
                print("Error: Missing waveform type")
                print("Usage: waveform [sine, square, triangle, sawtooth, rft]")
                return
            
            waveform_type = args[0].lower()
            if waveform_type not in ["sine", "square", "triangle", "sawtooth", "rft"]:
                print(f"Error: Unknown waveform type: {waveform_type}")
                print("Available types: sine, square, triangle, sawtooth, rft")
                return
            
            # Generate a simple text representation of the waveform
            self.generate_text_waveform(waveform_type)
        
        elif cmd == "field":
            if not self.has_numpy:
                print("Error: Numpy not available")
                return
            
            if not args:
                print("Error: Missing field type")
                print("Usage: field [electric, magnetic, quantum, rft]")
                return
            
            field_type = args[0].lower()
            if field_type not in ["electric", "magnetic", "quantum", "rft"]:
                print(f"Error: Unknown field type: {field_type}")
                print("Available types: electric, magnetic, quantum, rft")
                return
            
            # Generate a simple text representation of the field
            self.generate_text_field(field_type)
        
        elif cmd == "exit":
            print(f"Exiting {self.app_name}...")
            self.running = False
        
        else:
            print(f"Unknown command: {cmd}")
            print("Type 'help' for a list of commands")
    
    def generate_text_waveform(self, waveform_type: str):
        """Generate a text representation of a waveform"""
        # Generate the data
        x = self.np.linspace(0, 2*self.np.pi, 40)
        
        if waveform_type == "sine":
            y = self.np.sin(x)
        elif waveform_type == "square":
            y = self.np.sign(self.np.sin(x))
        elif waveform_type == "triangle":
            y = (2/self.np.pi) * self.np.arcsin(self.np.sin(x))
        elif waveform_type == "sawtooth":
            y = (2/self.np.pi) * self.np.arctan(self.np.tan(x/2))
        elif waveform_type == "rft":
            # Generate a more complex waveform for RFT
            y = self.np.sin(x) * self.np.exp(-0.1 * (x - self.np.pi)**2)
        
        # Scale the data to fit in the terminal
        y = (y + 1) * 10
        
        # Generate the text representation
        print(f"\n{waveform_type.capitalize()} Waveform:")
        for i in range(20, -1, -1):
            line = ""
            for j in range(len(x)):
                if abs(y[j] - i) < 0.5:
                    line += "*"
                else:
                    line += " "
            print(line)
        print("-" * len(x))
        print("\n")
    
    def generate_text_field(self, field_type: str):
        """Generate a text representation of a field"""
        # Generate the data
        x = self.np.linspace(-5, 5, 20)
        y = self.np.linspace(-5, 5, 20)
        X, Y = self.np.meshgrid(x, y)
        
        if field_type == "electric":
            # Simulate an electric field (1/r^2)
            Z = 1 / (0.1 + X**2 + Y**2)
        elif field_type == "magnetic":
            # Simulate a magnetic field (dipole)
            Z = (X**2 - Y**2) / (0.1 + (X**2 + Y**2)**2)
        elif field_type == "quantum":
            # Simulate a quantum field (wave function)
            Z = self.np.exp(-(X**2 + Y**2) / 4) * self.np.cos(X**2 + Y**2)
        elif field_type == "rft":
            # Simulate an RFT field
            Z = self.np.exp(-(X**2 + Y**2) / 4) * (self.np.cos(X**2 - Y**2) + self.np.sin(X*Y))
        
        # Scale the data to fit in the terminal
        Z = (Z - Z.min()) / (Z.max() - Z.min())
        
        # Generate the text representation
        print(f"\n{field_type.capitalize()} Field:")
        
        # Define the characters to use for different values
        chars = " .-+*#@"
        
        for i in range(len(x)):
            line = ""
            for j in range(len(y)):
                # Map the value to a character
                char_idx = min(int(Z[i, j] * len(chars)), len(chars) - 1)
                line += chars[char_idx]
            print(line)
        print("\n")

def main():
    """Main function"""
    # Create the app launcher
    launcher = AppLauncherBase("RFT Visualizer", "fa5s.wave-square")
    
    # Check if the GUI should be disabled
    if "--no-gui" in sys.argv:
        launcher.launch_terminal(RFTVisualizerTerminal)
    else:
        launcher.launch_gui(RFTVisualizer)

if __name__ == "__main__":
    main()
