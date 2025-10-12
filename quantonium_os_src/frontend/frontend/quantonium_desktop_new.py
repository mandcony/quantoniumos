#!/usr/bin/env python3
"""
QuantoniumOS Desktop Manager - Scientific Minimal Design with SVG Icons
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
                            QPushButton, QLabel, QDesktopWidget, QGraphicsRectItem, QVBoxLayout)
from PyQt5.QtCore import Qt, QTimer, QRectF, QPointF, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import (QPalette, QColor, QFont, QPainter, QPen, QBrush, QCursor, 
                        QLinearGradient, QRadialGradient)
from PyQt5.QtSvg import QSvgWidget, QGraphicsSvgItem

class QuantoniumDesktop(QMainWindow):
    """
    Scientific Desktop Manager for Quantum Computation Environment
    with mathematically precise Golden Ratio proportions
    """
    
    def __init__(self):
        super().__init__()
        
        # Mathematical constants for design precision (Golden Ratio)
        self.phi = 1.618033988749895  # φ (Golden Ratio)
        self.phi_sq = self.phi * self.phi  # φ²
        self.phi_inv = 1 / self.phi  # 1/φ
        self.base_unit = 16  # Base unit for scaling (multiply by φ powers for harmony)
        
        # Setup UI
        self.setWindowTitle("QuantoniumOS")
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.showFullScreen()
        
        # Graphics View setup with absolute precision
        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene, self)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setRenderHint(QPainter.SmoothPixmapTransform)
        self.view.setRenderHint(QPainter.TextAntialiasing)
        
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
        
        # Center coordinates
        center_x = screen.width() / 2
        center_y = screen.height() / 2
        
        # Q Logo dimensions scaled by Golden Ratio
        outer_radius = self.base_unit * self.phi_sq  # φ² scaling for outer circle
        inner_radius = outer_radius * self.phi_inv   # 1/φ scaling for inner circle
        stroke_width = self.base_unit * 0.15         # Precise stroke scaling
        
        # Create outer circle
        outer_circle = QGraphicsEllipseItem(
            center_x - outer_radius, 
            center_y - outer_radius,
            outer_radius * 2, 
            outer_radius * 2
        )
        
        # Create inner circle (precise proportion of outer)
        inner_circle = QGraphicsEllipseItem(
            center_x - inner_radius, 
            center_y - inner_radius,
            inner_radius * 2, 
            inner_radius * 2
        )
        
        # Create diagonal line (mathematically placed)
        line_length = outer_radius * self.phi_inv  # 1/φ scaling
        line_angle = math.radians(45)  # 45 degrees in radians
        
        # Calculate line endpoints with precise geometric harmony
        line_dx = math.cos(line_angle) * line_length
        line_dy = math.sin(line_angle) * line_length
        
        # Apply Golden Ratio offset for the line
        offset = inner_radius * self.phi_inv * 0.5
        
        # Styling with minimal, precise aesthetics
        pen = QPen(QColor("#3498db"), stroke_width)
        pen.setCapStyle(Qt.RoundCap)
        
        outer_circle.setPen(pen)
        outer_circle.setBrush(QBrush(QColor(0, 0, 0, 0)))  # Transparent fill
        
        inner_circle.setPen(pen)
        inner_circle.setBrush(QBrush(QColor(0, 0, 0, 0)))  # Transparent fill
        
        # Add to scene with perfect layering
        self.scene.addItem(outer_circle)
        self.scene.addItem(inner_circle)
        
        # Return for reference
        return outer_circle
    
    def create_system_time(self):
        """Create mathematically precise time display using Golden Ratio"""
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
    
    def load_scientific_apps(self):
        """Load applications with scientific metadata"""
        base_path = os.path.dirname(os.path.dirname(__file__))
        self.apps = [
            {
                "name": "RFT Validation Suite", 
                "path": os.path.join(base_path, "src", "apps", "rft_validation_suite.py"), 
                "category": "ANALYSIS",
                "description": "Mathematical validation framework",
                "icon": "rft_validator.svg"
            },
            {
                "name": "AI Chat", 
                "path": os.path.join(base_path, "src", "apps", "qshll_chatbox.py"), 
                "category": "AI", 
                "description": "Quantum-enhanced AI assistant",
                "icon": "ai_chat.svg"
            },
            {
                "name": "Quantum Simulator", 
                "path": os.path.join(base_path, "src", "apps", "quantum_simulator.py"), 
                "category": "SIMULATION",
                "description": "Quantum circuit modeling",
                "icon": "quantum_simulator.svg"
            },
            {
                "name": "Quantum Cryptography", 
                "path": os.path.join(base_path, "src", "apps", "quantum_crypto.py"), 
                "category": "SECURITY",
                "description": "Cryptographic protocols",
                "icon": "quantum_crypto.svg"
            },
            {
                "name": "System Monitor", 
                "path": os.path.join(base_path, "src", "apps", "qshll_system_monitor.py"), 
                "category": "SYSTEM",
                "description": "Resource monitoring",
                "icon": "system_monitor.svg"
            },
            {
                "name": "Q-Notes", 
                "path": os.path.join(base_path, "src", "apps", "q_notes.py"), 
                "category": "RESEARCH",
                "description": "Research documentation",
                "icon": "q_notes.svg"
            },
            {
                "name": "Q-Vault", 
                "path": os.path.join(base_path, "src", "apps", "q_vault.py"), 
                "category": "DATA",
                "description": "Secure data storage",
                "icon": "q_vault.svg"
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
        """Expand applications in arch formation with SVG icons"""
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
        
        # Position apps in circular layout around Q
        num_apps = len(self.apps)
        
        # Using a partial circle for apps (top half plus sides)
        angle_step = math.pi * 1.5 / (num_apps - 1) if num_apps > 1 else 0
        start_angle = math.pi * 1.75  # Start from top-left (315 degrees)
        
        for i, app in enumerate(self.apps):
            # Calculate position on the circle
            angle = start_angle - i * angle_step
            x = center_x + arch_radius * math.cos(angle)
            y = center_y + arch_radius * math.sin(angle)
            
            # Get app name and information
            app_name = app["name"]
            app_path = app["path"]
            app_icon = app.get("icon", None)
            display_name = app_name.split()[-1] if len(app_name.split()) > 1 else app_name
            
            # Create app button with background
            app_rect = QGraphicsRectItem(x - button_size/2, y - button_size/2, button_size, button_size)
            app_rect.setBrush(QBrush(QColor(255, 255, 255, 128)))
            app_rect.setPen(QPen(QColor(52, 152, 219, 51), 1))
            app_rect.setAcceptedMouseButtons(Qt.LeftButton)
            app_rect.setCursor(QCursor(Qt.PointingHandCursor))
            app_rect.mousePressEvent = lambda event, path=app_path, name=app_name: self.launch_app({"path": path, "name": name})
            
            self.scene.addItem(app_rect)
            self.app_items.append(app_rect)
            
            # Add SVG icon if available
            if app_icon:
                icon_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "ui", "icons", app_icon)
                if os.path.exists(icon_path):
                    # Create SVG item
                    svg_item = QGraphicsSvgItem(icon_path)
                    icon_size = button_size * 0.7  # Slightly smaller than button
                    
                    # Calculate scale factor to fit icon within the desired size
                    svg_bounds = svg_item.boundingRect()
                    scale_factor = min(icon_size / svg_bounds.width(), icon_size / svg_bounds.height())
                    svg_item.setScale(scale_factor)
                    
                    # Center the icon on the button
                    svg_x = x - (svg_bounds.width() * scale_factor / 2)
                    svg_y = y - (svg_bounds.height() * scale_factor / 2)
                    svg_item.setPos(svg_x, svg_y)
                    
                    self.scene.addItem(svg_item)
                    self.app_items.append(svg_item)
                else:
                    print(f"Icon not found: {icon_path}")
            
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
        
        print(f"Expanded arch with {len(self.apps)} applications - SVG version")
        
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
        import subprocess  # Import here to ensure it's available
        app_name = app_data["name"]
        app_path = app_data["path"]
        
        print(f"Launching: {app_name}")
        self.collapse_arch()
        
        try:
            # Update status with mathematical precision
            self.system_status.setPlainText(f"LAUNCHING {app_name.upper()}")
            
            # Launch with proper environment
            base_dir = os.path.dirname(os.path.dirname(__file__))
            
            # Launch process with proper handling
            process = subprocess.Popen(
                [sys.executable, app_path], 
                cwd=base_dir,
                stdout=subprocess.DEVNULL if "qshll_chatbox" in app_path.lower() else subprocess.PIPE,
                stderr=subprocess.DEVNULL if "qshll_chatbox" in app_path.lower() else subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == 'win32' and "qshll_chatbox" in app_path.lower() else 0
            )
            
            print(f"Successfully launched {app_name} (PID: {process.pid})")
            
            # Reset status after delay using golden ratio timing (φ seconds)
            reset_delay = int(self.phi * 1000)  # φ seconds in milliseconds
            QTimer.singleShot(reset_delay, lambda: self.system_status.setPlainText("QUANTONIUMOS"))
            
        except Exception as e:
            print(f"Error launching {app_name}: {e}")
            import traceback
            traceback.print_exc()
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
