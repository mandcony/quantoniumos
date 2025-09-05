#!/usr/bin/env python3
"""
QuantoniumOS Desktop Manager - Scientific Minimal Design
======================================================
PhD-level UI/UX for quantum computing research platform
"""

import sys
import os
import math
import subprocess
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QGraphicsView, QGraphicsScene, 
                            QGraphicsEllipseItem, QGraphicsTextItem, QGraphicsProxyWidget,
                            QPushButton, QLabel, QDesktopWidget, QGraphicsRectItem)
from PyQt5.QtCore import Qt, QTimer, QRectF, QPointF, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import (QPalette, QColor, QFont, QPainter, QPen, QBrush, QCursor, 
                        QLinearGradient, QRadialGradient)
from PyQt5.QtSvg import QSvgWidget, QGraphicsSvgItem

class QuantoniumDesktop(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QuantoniumOS - Quantum Computing Research Platform")
        
        # Golden Ratio mathematical constants
        self.phi = 1.618033988749894  # Golden Ratio φ
        self.phi_inv = 0.618033988749894  # 1/φ
        self.phi_sq = 2.618033988749894  # φ²
        
        # Base unit derived from screen dimensions and golden ratio
        screen = QDesktopWidget().screenGeometry()
        self.base_unit = min(screen.width(), screen.height()) / (self.phi ** 6)  # Mathematical base
        
        # Theme state
        self.is_dark_theme = True  # Start with dark theme by default
        
        # Scientific minimal color scheme
        self.bg_color = "#fafafa"  # Ultra light background
        self.primary_color = "#2c3e50"  # Deep blue-gray
        self.accent_color = "#3498db"  # Scientific blue
        self.text_color = "#34495e"  # Dark gray
        self.surface_color = "#ffffff"  # Pure white
        
        self.setStyleSheet(f"background-color: {self.bg_color};")
        
        # Remove window frame and make it fullscreen
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.showMaximized()
        
        # Load the appropriate stylesheet
        self.load_styles()
        
        # Create graphics view and scene
        self.view = QGraphicsView(self)
        self.scene = QGraphicsScene(self)
        self.view.setScene(self.scene)
        self.setCentralWidget(self.view)
        
        # Configure view for minimal design
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setStyleSheet("background: transparent; border: none;")
        self.view.setRenderHint(QPainter.Antialiasing)
        
        # Get screen dimensions
        screen = QDesktopWidget().screenGeometry()
        self.scene.setSceneRect(0, 0, screen.width(), screen.height())
        
        # Initialize scientific interface
        self.is_arch_expanded = False
        self.app_items = []
        self.arch_background = None
        self.panel_trigger = self.create_panel_trigger()
        self.quantum_logo = self.create_quantum_logo()
        self.system_time = self.create_system_time()
        self.system_status = self.create_system_status()
        self.theme_toggle = self.create_theme_toggle()
        
        # Initialize timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_time)
        self.timer.start(1000)
        self.update_time()
        
        self.load_scientific_apps()
    
    def create_panel_trigger(self):
        """Create quantum logo as expandable arch trigger"""
        screen = QDesktopWidget().screenGeometry()
        
        # Create circular trigger zone around the quantum logo
        trigger = QGraphicsEllipseItem(screen.width()//2 - 60, screen.height()//2 - 60, 120, 120)
        trigger.setBrush(QBrush(QColor(0, 0, 0, 0)))  # Transparent
        trigger.setPen(QPen(QColor(0, 0, 0, 0), 0))    # No border
        
        # Add interaction
        trigger.setAcceptedMouseButtons(Qt.LeftButton)
        trigger.mousePressEvent = self.toggle_arch
        trigger.setCursor(QCursor(Qt.PointingHandCursor))
        
        self.scene.addItem(trigger)
        return trigger
        
    def create_quantum_logo(self):
        """Create mathematically precise Q logo using Golden Ratio proportions"""
        screen = QDesktopWidget().screenGeometry()
        
        # Golden ratio positioning - center point
        center_x = screen.width() / 2
        center_y = screen.height() / 2
        
        # Logo size based on golden ratio
        logo_size = self.base_unit * self.phi_sq  # φ² scaling
        
        # Create clickable Q text as the main trigger
        q_text = QGraphicsTextItem("Q")
        q_text.setDefaultTextColor(QColor("#2c3e50"))
        
        # Font size based on golden ratio
        font_size = int(self.base_unit * self.phi)
        font = QFont("SF Pro Display", font_size, QFont.Light)
        if not font.exactMatch():
            font = QFont("Segoe UI", font_size, QFont.Light)
        font.setLetterSpacing(QFont.AbsoluteSpacing, self.base_unit * 0.1)
        
        q_text.setFont(font)
        
        # Position Q with mathematical precision
        q_bounds = q_text.boundingRect()
        q_x = center_x - (q_bounds.width() / 2)
        q_y = center_y - (q_bounds.height() / 2)
        q_text.setPos(q_x, q_y)
        
        # Make Q text visible but not clickable (panel trigger handles clicks)
        q_text.setFlag(QGraphicsTextItem.ItemIsSelectable, False)
        
        # Add subtle hover effect indicator with golden ratio proportions
        hover_radius = logo_size * self.phi_inv * 0.8
        hover_circle = QGraphicsEllipseItem(
            center_x - hover_radius, 
            center_y - hover_radius, 
            hover_radius * 2, 
            hover_radius * 2
        )
        hover_circle.setBrush(QBrush(QColor(0, 0, 0, 0)))  # Transparent
        hover_circle.setPen(QPen(QColor(52, 152, 219, 30), int(self.base_unit * 0.05)))  # Golden ratio border
        
        self.scene.addItem(hover_circle)
        self.scene.addItem(q_text)
        
        return q_text
    
    def create_system_time(self):
        """Create mathematically precise system time display using Golden Ratio"""
        screen = QDesktopWidget().screenGeometry()
        
        time_text = QGraphicsTextItem()
        time_text.setDefaultTextColor(QColor("#34495e"))
        
        # Font size based on golden ratio proportions
        time_font_size = int(self.base_unit * self.phi_inv)  # 1/φ scaling
        font = QFont("SF Mono", time_font_size, QFont.Normal)
        if not font.exactMatch():
            font = QFont("Consolas", time_font_size, QFont.Normal)
        
        time_text.setFont(font)
        
        # Position with golden ratio margins
        margin_x = self.base_unit * self.phi_sq  # φ² margin from edge
        margin_y = self.base_unit * self.phi_inv  # 1/φ margin from top
        time_text.setPos(screen.width() - margin_x - 120, margin_y)
        
        self.scene.addItem(time_text)
        return time_text
        
    def create_theme_toggle(self):
        """Create theme toggle button"""
        screen = QDesktopWidget().screenGeometry()
        
        # Create a button for theme toggle
        theme_btn = QPushButton()
        theme_btn.setFixedSize(40, 40)
        theme_btn.setCursor(QCursor(Qt.PointingHandCursor))
        
        # Set icon and tooltip based on current theme
        icon_text = "🌙" if self.is_dark_theme else "☀️"
        theme_btn.setText(icon_text)
        theme_btn.setToolTip("Switch to Light Mode" if self.is_dark_theme else "Switch to Dark Mode")
        
        # Style the button
        theme_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(255, 255, 255, 0.2);
                border: 1px solid rgba(255, 255, 255, 0.3);
                border-radius: 20px;
                color: white;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 0.3);
            }
        """)
        
        # Connect the button to the theme toggle function
        theme_btn.clicked.connect(self.toggle_theme)
        
        # Add to scene
        theme_proxy = self.scene.addWidget(theme_btn)
        theme_proxy.setPos(screen.width() - 60, 20)  # Top-right corner
        
        return theme_proxy
        
    def create_system_status(self):
        """Create mathematically precise system status using Golden Ratio"""
        screen = QDesktopWidget().screenGeometry()
        center_x = screen.width() / 2
        center_y = screen.height() / 2
        
        status_text = QGraphicsTextItem("QUANTONIUMOS")
        status_text.setDefaultTextColor(QColor("#3498db"))
        
        # Font size based on golden ratio
        status_font_size = int(self.base_unit * self.phi_inv * 0.8)  # Smaller than main logo
        font = QFont("SF Pro Display", status_font_size, QFont.Medium)
        if not font.exactMatch():
            font = QFont("Segoe UI", status_font_size, QFont.Normal)
        font.setLetterSpacing(QFont.AbsoluteSpacing, self.base_unit * 0.08)
        
        status_text.setFont(font)
        
        # Position at the bottom of the screen, near the footer
        status_bounds = status_text.boundingRect()
        status_x = center_x - (status_bounds.width() / 2)  # Center horizontally
        status_y = screen.height() - status_bounds.height() - (self.base_unit * 2.5)  # Near bottom but above taskbar
        status_text.setPos(status_x, status_y)
        
        # Add subtle interaction hint with precise positioning
        hint_text = QGraphicsTextItem("Click Q to show apps")
        hint_text.setDefaultTextColor(QColor(52, 73, 94, 120))  # Semi-transparent
        
        hint_font_size = int(self.base_unit * self.phi_inv * 0.6)  # Even smaller
        hint_font = QFont("SF Pro Display", hint_font_size, QFont.Normal)
        if not hint_font.exactMatch():
            hint_font = QFont("Segoe UI", hint_font_size, QFont.Normal)
        
        hint_text.setFont(hint_font)
        
        # Position hint below the Q logo in the center
        hint_bounds = hint_text.boundingRect()
        hint_x = center_x - (hint_bounds.width() / 2)
        hint_y = center_y + (self.base_unit * self.phi)
        
        self.scene.addItem(status_text)
        self.scene.addItem(hint_text)
        
        return status_text
    
    def update_time(self):
        current_time = datetime.now().strftime("%H:%M")
        self.system_time.setPlainText(current_time)
    
    def load_styles(self):
        """Load the appropriate stylesheet based on theme"""
        base_path = os.path.dirname(os.path.dirname(__file__))
        
        if self.is_dark_theme:
            qss_path = os.path.join(base_path, "ui", "styles_dark.qss")
        else:
            qss_path = os.path.join(base_path, "ui", "styles_light.qss")
        
        if os.path.exists(qss_path):
            with open(qss_path, 'r') as f:
                self.setStyleSheet(f.read())
        else:
            print(f"Warning: Theme file not found at {qss_path}")
    
    def toggle_theme(self):
        """Toggle between light and dark themes"""
        self.is_dark_theme = not self.is_dark_theme
        self.load_styles()
        
        # Update status with theme change
        theme_name = "DARK" if self.is_dark_theme else "LIGHT"
        self.system_status.setPlainText(f"THEME: {theme_name}")
        
        # Reset status after delay using golden ratio timing
        reset_delay = int(self.phi * 1000)  # φ seconds in milliseconds
        QTimer.singleShot(reset_delay, lambda: self.system_status.setPlainText("QUANTONIUMOS"))
    
    def load_scientific_apps(self):
        """Load applications with scientific metadata"""
        base_path = os.path.dirname(os.path.dirname(__file__))
        self.apps = [
            {
                "name": "RFT Validation Suite", 
                "path": os.path.join(base_path, "apps", "rft_validation_suite.py"), 
                "category": "ANALYSIS",
                "description": "Mathematical validation framework"
            },
            {
                "name": "RFT Visualizer", 
                "path": os.path.join(base_path, "apps", "rft_visualizer.py"), 
                "category": "VISUALIZATION", 
                "description": "Real-time signal processing"
            },
            {
                "name": "Quantum Simulator", 
                "path": os.path.join(base_path, "apps", "quantum_simulator.py"), 
                "category": "SIMULATION",
                "description": "Quantum circuit modeling"
            },
            {
                "name": "Quantum Cryptography", 
                "path": os.path.join(base_path, "apps", "quantum_crypto.py"), 
                "category": "SECURITY",
                "description": "Cryptographic protocols"
            },
            {
                "name": "System Monitor", 
                "path": os.path.join(base_path, "apps", "qshll_system_monitor.py"), 
                "category": "SYSTEM",
                "description": "Resource monitoring"
            },
            {
                "name": "Q-Notes", 
                "path": os.path.join(base_path, "apps", "q_notes.py"), 
                "category": "RESEARCH",
                "description": "Research documentation"
            },
            {
                "name": "Q-Vault", 
                "path": os.path.join(base_path, "apps", "q_vault.py"), 
                "category": "DATA",
                "description": "Secure data storage"
            }
        ]
    
    def toggle_arch(self, event):
        """Toggle expandable arch display"""
        print(f"Toggle arch called. Current state: expanded={self.is_arch_expanded}")
        if not self.is_arch_expanded:
            print("Expanding arch...")
            self.expand_arch()
        else:
            print("Collapsing arch...")
            self.collapse_arch()
    
    def expand_arch(self):
        """Expand applications in arch formation with colored square icons and SVG graphics"""
        print("DEBUG: expand_arch function called")
        self.is_arch_expanded = True
        
        # Clear existing items
        for item in self.app_items:
            self.scene.removeItem(item)
        self.app_items.clear()
        
        screen = QDesktopWidget().screenGeometry()
        center_x = screen.width() // 2
        center_y = screen.height() // 2
        
        # Dimensions to match the provided image
        arch_radius = 280  # Adjusted radius to match image
        button_size = 80   # Square app buttons
        font_size = 10     # Better readable font size
        
        # Create outer circle background
        arch_bg = QGraphicsEllipseItem(center_x - arch_radius, center_y - arch_radius, 
                                      arch_radius * 2, arch_radius * 2)
        arch_bg.setBrush(QBrush(QColor(255, 255, 255, 10)))  # Very subtle overlay
        arch_bg.setPen(QPen(QColor(52, 152, 219, 30), 1))    # Subtle border
        
        # Create inner circle for aesthetics (matching image)
        inner_radius = arch_radius * 0.4
        inner_circle = QGraphicsEllipseItem(center_x - inner_radius, center_y - inner_radius,
                                           inner_radius * 2, inner_radius * 2)
        inner_circle.setBrush(QBrush(QColor(255, 255, 255, 5)))  # Very subtle fill
        inner_circle.setPen(QPen(QColor(52, 152, 219, 20), 1))   # Very subtle border
        
        self.scene.addItem(arch_bg)
        self.scene.addItem(inner_circle)
        self.app_items.append(arch_bg)
        self.app_items.append(inner_circle)
        self.arch_background = arch_bg
        
        # Position apps in circular layout around Q, matching the image provided
        num_apps = len(self.apps)
        
        # Using a partial circle for apps (top half plus sides)
        angle_step = math.pi * 1.5 / (num_apps - 1) if num_apps > 1 else 0
        start_angle = math.pi * 1.75  # Start from top-left (315 degrees)
        
        # App configurations with minimal styling
        app_configs = {
            "RFT Validation": {
                "display_name": "RFT Validator"
            },
            "RFT Visual": {
                "display_name": "RFT Visualizer"
            },
            "Quantum Simulator": {
                "display_name": "Quantum Simulator"
            },
            "Quantum Crypto": {
                "display_name": "Quantum Crypto"
            },
            "System Monitor": {
                "display_name": "System Monitor"
            },
            "Q-Notes": {
                "display_name": "Q-Notes"
            },
            "Q-Vault": {
                "display_name": "Q-Vault"
            }
        }
        
        for i, app in enumerate(self.apps):
            # Calculate position on the circle
            angle = start_angle - i * angle_step
            x = center_x + arch_radius * math.cos(angle)
            y = center_y + arch_radius * math.sin(angle)
            
            # Get app name and determine configuration
            app_name = app["name"]
            
            # Determine the app config key to use
            config_key = None
            if "RFT" in app_name and "Validation" in app_name:
                config_key = "RFT Validation"
            elif "RFT" in app_name and "Visual" in app_name:
                config_key = "RFT Visual"
            elif "Quantum" in app_name and "Simulator" in app_name:
                config_key = "Quantum Simulator"
            elif "Quantum" in app_name and "Crypto" in app_name:
                config_key = "Quantum Crypto"
            elif "System" in app_name:
                config_key = "System Monitor"
            elif "Notes" in app_name:
                config_key = "Q-Notes"
            elif "Vault" in app_name:
                config_key = "Q-Vault"
            else:
                config_key = "Q-Vault"  # Default
                
            # Get the configuration for this app
            config = app_configs[config_key]
            display_name = config["display_name"]
            
            # Create modern styled app button
            app_btn = QPushButton()
            app_btn.setFixedSize(button_size, button_size)
            
            # Create minimal, clean button style without colored backgrounds
            app_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: rgba(255, 255, 255, 0.5);
                    border: 1px solid rgba(52, 152, 219, 0.2);
                    border-radius: 8px;
                }}
                QPushButton:hover {{
                    border: 2px solid #3498db;
                    background-color: rgba(255, 255, 255, 0.8);
                }}
                QPushButton:pressed {{
                    background-color: rgba(52, 152, 219, 0.1);
                }}
            """)
            
            # Add SVG icon to button
            icon_name = ""
            if "RFT Validation" in config_key:
                icon_name = "rft_validator.svg"
            elif "RFT Visual" in config_key:
                icon_name = "rft_visualizer.svg"
            elif "Quantum Simulator" in config_key:
                icon_name = "quantum_simulator.svg"
            elif "Quantum Crypto" in config_key:
                icon_name = "quantum_crypto.svg"
            elif "System Monitor" in config_key:
                icon_name = "system_monitor.svg"
            elif "Q-Notes" in config_key:
                icon_name = "q_notes.svg"
            elif "Q-Vault" in config_key:
                icon_name = "q_vault.svg"
            
            # Create SVG widget and add to button layout
            if icon_name:
                icon_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ui", "icons", icon_name)
                if os.path.exists(icon_path):
                    svg_widget = QSvgWidget(icon_path)
                    svg_widget.setFixedSize(button_size - 20, button_size - 20)  # Smaller than button for padding
                    
                    # Create a layout for the button to center the SVG
                    layout = app_btn.layout()
                    if not layout:
                        from PyQt5.QtWidgets import QVBoxLayout
                        layout = QVBoxLayout(app_btn)
                        layout.setContentsMargins(10, 10, 10, 10)
                        layout.setAlignment(Qt.AlignCenter)
                    
                    layout.addWidget(svg_widget)
                else:
                    print(f"Icon not found: {icon_path}")
            
            app_btn.setText("")  # No text on button itself
            app_btn.clicked.connect(lambda checked, path=app["path"], name=app["name"]: 
                                   self.launch_app({"path": path, "name": name}))
            
            # Add button to scene
            proxy = self.scene.addWidget(app_btn)
            proxy.setPos(x - button_size/2, y - button_size/2)
            self.app_items.append(proxy)
            
            # Add text label below icon
            label = QGraphicsTextItem(display_name)
            label.setDefaultTextColor(QColor("#2c3e50"))
            
            # Font settings
            font = QFont("Segoe UI", font_size)
            font.setWeight(QFont.Medium)
            label.setFont(font)
            
            # Center text relative to icon position
            label_bounds = label.boundingRect()
            label_x = x - (label_bounds.width() / 2)
            
            # Position labels consistently below icons for clarity
            label_y = y + (button_size / 2) + 5
            label.setPos(label_x, label_y)
            
            self.scene.addItem(label)
            self.app_items.append(label)
        
        print(f"Expanded arch with {len(self.apps)} applications - minimal version")
        
    def collapse_arch(self):
        """Collapse arch formation"""
        self.is_arch_expanded = False
        
        for item in self.app_items:
            self.scene.removeItem(item)
        self.app_items.clear()
        self.arch_background = None
        
        print("Collapsed application arch")
    
    def launch_app(self, app_data):
        """Launch application with improved error handling using Golden Ratio feedback"""
        app_name = app_data["name"]
        app_path = app_data["path"]
        
        print(f"Launching: {app_name}")
        self.collapse_arch()
        
        try:
            # Update status with mathematical precision
            self.system_status.setPlainText(f"LAUNCHING {app_name.upper()}")
            
            # Launch with proper environment
            base_dir = os.path.dirname(os.path.dirname(__file__))
            process = subprocess.Popen(
                [sys.executable, app_path], 
                cwd=base_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            print(f"Successfully launched {app_name} (PID: {process.pid})")
            
            # Reset status after delay using golden ratio timing (φ seconds)
            reset_delay = int(self.phi * 1000)  # φ seconds in milliseconds
            QTimer.singleShot(reset_delay, lambda: self.system_status.setPlainText("QUANTONIUMOS"))
            
        except Exception as e:
            print(f"Error launching {app_name}: {e}")
            self.system_status.setPlainText("LAUNCH ERROR")
            QTimer.singleShot(int(self.phi_sq * 1000), lambda: self.system_status.setPlainText("QUANTONIUMOS"))
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.setWindowFlags(Qt.Window)
            self.showNormal()
        super().keyPressEvent(event)

def main():
    app = QApplication(sys.argv)
    
    # Set application-wide font
    font = QFont("SF Pro Display", 10)
    if not font.exactMatch():
        font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    desktop = QuantoniumDesktop()
    desktop.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
