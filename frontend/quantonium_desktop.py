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

class QuantoniumDesktop(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QuantoniumOS - Quantum Computing Research Platform")
        
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
        self.panel_trigger = self.create_panel_trigger()
        self.app_items = []
        self.arch_background = None
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
        trigger.mousePressEvent = self.toggle_arch
        trigger.setCursor(QCursor(Qt.PointingHandCursor))
        
        self.scene.addItem(trigger)
        return trigger
        
    def create_quantum_logo(self):
        """Create interactive quantum logo as arch trigger"""
        screen = QDesktopWidget().screenGeometry()
        
        # Create container for logo elements
        logo_group = QGraphicsRectItem(screen.width()//2 - 60, screen.height()//2 - 60, 120, 120)
        logo_group.setBrush(QBrush(QColor(0, 0, 0, 0)))  # Transparent
        logo_group.setPen(QPen(QColor(0, 0, 0, 0), 0))   # No border
        
        # "Q" with scientific typography
        q_text = QGraphicsTextItem("Q")
        q_text.setDefaultTextColor(QColor(self.primary_color))
        
        # Scientific font selection
        font = QFont("SF Pro Display", 64, QFont.Light)
        if not font.exactMatch():
            font = QFont("Segoe UI", 64, QFont.Light)
        font.setLetterSpacing(QFont.AbsoluteSpacing, 2)
        
        q_text.setFont(font)
        q_text.setPos(screen.width()//2 - 30, screen.height()//2 - 80)
        
        # Add subtle hover effect indicator
        hover_circle = QGraphicsEllipseItem(screen.width()//2 - 45, screen.height()//2 - 45, 90, 90)
        hover_circle.setBrush(QBrush(QColor(0, 0, 0, 0)))  # Transparent
        hover_circle.setPen(QPen(QColor(self.accent_color, 30), 2))  # Very subtle border
        
        self.scene.addItem(hover_circle)
        self.scene.addItem(q_text)
        
        return q_text
        
    def create_system_time(self):
        """Create minimal system time display"""
        screen = QDesktopWidget().screenGeometry()
        
        time_text = QGraphicsTextItem()
        time_text.setDefaultTextColor(QColor(self.text_color))
        
        # Monospace font for time
        font = QFont("SF Mono", 16, QFont.Normal)
        if not font.exactMatch():
            font = QFont("Consolas", 16, QFont.Normal)
        
        time_text.setFont(font)
        time_text.setPos(screen.width() - 120, 30)
        
        self.scene.addItem(time_text)
        return time_text
        
    def create_system_status(self):
        """Create minimal system status with interaction hint"""
        screen = QDesktopWidget().screenGeometry()
        
        status_text = QGraphicsTextItem("QUANTUM READY")
        status_text.setDefaultTextColor(QColor(self.accent_color))
        
        font = QFont("SF Pro Display", 11, QFont.Medium)
        if not font.exactMatch():
            font = QFont("Segoe UI", 11, QFont.Normal)
        font.setLetterSpacing(QFont.AbsoluteSpacing, 1.2)
        
        status_text.setFont(font)
        status_text.setPos(screen.width()//2 - 50, screen.height()//2 + 20)
        
        # Add subtle interaction hint
        hint_text = QGraphicsTextItem("Click Q to expand")
        hint_text.setDefaultTextColor(QColor(self.text_color, 120))  # Semi-transparent
        
        hint_font = QFont("SF Pro Display", 9, QFont.Normal)
        if not hint_font.exactMatch():
            hint_font = QFont("Segoe UI", 9, QFont.Normal)
        
        hint_text.setFont(hint_font)
        hint_text.setPos(screen.width()//2 - 40, screen.height()//2 + 45)
        
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
        if not self.is_arch_expanded:
            self.expand_arch()
        else:
            self.collapse_arch()
    
    def expand_arch(self):
        """Expand applications in arch formation"""
        self.is_arch_expanded = True
        
        # Clear existing items
        for item in self.app_items:
            self.scene.removeItem(item)
        self.app_items.clear()
        
        screen = QDesktopWidget().screenGeometry()
        center_x = screen.width() // 2
        center_y = screen.height() // 2
        
        # Create subtle arch background
        arch_radius = 200
        arch_bg = QGraphicsEllipseItem(center_x - arch_radius, center_y - arch_radius, 
                                      arch_radius * 2, arch_radius * 2)
        arch_bg.setBrush(QBrush(QColor(255, 255, 255, 20)))  # Very subtle white overlay
        arch_bg.setPen(QPen(QColor(52, 152, 219, 50), 2))    # Subtle blue border
        
        self.scene.addItem(arch_bg)
        self.app_items.append(arch_bg)
        self.arch_background = arch_bg
        
        # Position apps in half-circle arch
        num_apps = len(self.apps)
        start_angle = math.pi  # Start at left (180 degrees)
        end_angle = 0         # End at right (0 degrees)
        angle_step = math.pi / (num_apps - 1) if num_apps > 1 else 0
        
        for i, app in enumerate(self.apps):
            # Calculate position on arch
            angle = start_angle - (i * angle_step)
            x = center_x + (arch_radius - 40) * math.cos(angle)
            y = center_y + (arch_radius - 40) * math.sin(angle)
            
            # Create app icon/button
            app_btn = QPushButton()
            app_btn.setFixedSize(80, 80)
            app_btn.setStyleSheet(f"""
                QPushButton {{
                    background: rgba(255, 255, 255, 0.9);
                    border: 2px solid rgba(52, 152, 219, 0.3);
                    border-radius: 40px;
                    font-family: "SF Pro Display", "Segoe UI";
                    font-size: 9px;
                    font-weight: 600;
                    color: {self.primary_color};
                    text-align: center;
                }}
                QPushButton:hover {{
                    background: rgba(52, 152, 219, 0.1);
                    border: 2px solid {self.accent_color};
                }}
                QPushButton:pressed {{
                    background: rgba(52, 152, 219, 0.2);
                }}
            """)
            
            # Set button text (abbreviated app name)
            app_name = app['name']
            if len(app_name) > 12:
                app_name = app_name.split()[0]  # Use first word
            
            app_btn.setText(app_name)
            app_btn.clicked.connect(lambda checked, path=app['path'], name=app['name']: self.launch_app(path, name))
            
            # Add to scene
            proxy = self.scene.addWidget(app_btn)
            proxy.setPos(x - 40, y - 40)  # Center the button
            self.app_items.append(proxy)
            
        print(f"Expanded arch with {len(self.apps)} applications")
    
    def collapse_arch(self):
        """Collapse arch formation"""
        self.is_arch_expanded = False
        
        for item in self.app_items:
            self.scene.removeItem(item)
        self.app_items.clear()
        self.arch_background = None
        
        print("Collapsed application arch")
    
    def launch_app(self, app_path, app_name):
        """Launch application with improved error handling"""
        print(f"Launching: {app_name}")
        self.collapse_arch()
        
        try:
            # Update status
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
            
            # Reset status after delay
            QTimer.singleShot(2000, lambda: self.system_status.setPlainText("QUANTUM READY"))
            
        except Exception as e:
            print(f"Error launching {app_name}: {e}")
            self.system_status.setPlainText("LAUNCH ERROR")
            QTimer.singleShot(3000, lambda: self.system_status.setPlainText("QUANTUM READY"))
    
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
