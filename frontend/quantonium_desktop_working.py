#!/usr/bin/env python3
"""
QuantoniumOS Desktop Manager - Working Version
=============================================
Simple expandable arch with Q logo and clock
"""

import sys
import os
import math
import subprocess
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QGraphicsView, QGraphicsScene, 
                            QGraphicsEllipseItem, QGraphicsTextItem, QGraphicsProxyWidget,
                            QPushButton, QLabel, QDesktopWidget)
from PyQt5.QtCore import Qt, QTimer, QRectF, QPointF
from PyQt5.QtGui import QPalette, QColor, QFont, QPainter, QPen, QBrush, QCursor

class QuantoniumDesktop(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QuantoniumOS")
        self.setStyleSheet("background-color: #f8f6f0;")
        
        # Remove window frame and make it fullscreen
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.showMaximized()
        
        # Create graphics view and scene
        self.view = QGraphicsView(self)
        self.scene = QGraphicsScene(self)
        self.view.setScene(self.scene)
        self.setCentralWidget(self.view)
        
        # Remove scrollbars and set transparent background
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setStyleSheet("background: transparent; border: none;")
        
        # Get screen dimensions
        screen = QDesktopWidget().screenGeometry()
        self.scene.setSceneRect(0, 0, screen.width(), screen.height())
        
        # 3) Initialize UI elements
        self.is_arch_expanded = False
        self.arch, self.arrow_proxy = self.add_shaded_arch()
        self.app_proxies = []
        self.q_logo = self.add_q_logo()
        self.clock_text = self.add_clock()
        self.update_time()

        # Timer for clock
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_time)
        self.timer.start(1000)

        self.load_apps()
        
    def add_shaded_arch(self):
        # Create arch at left center of screen
        screen = QDesktopWidget().screenGeometry()
        arch_x = 10
        arch_y = screen.height() // 2 - 50
        
        # Draw the arch (half circle)
        arch = QGraphicsEllipseItem(arch_x, arch_y, 60, 100)
        arch.setBrush(QBrush(QColor("#f0c040")))
        arch.setPen(QPen(QColor("#d4af37"), 4))
        arch.setStartAngle(90 * 16)  # Start at 90 degrees
        arch.setSpanAngle(180 * 16)  # Span 180 degrees (half circle)
        
        # Add click handler
        arch.mousePressEvent = self.toggle_arch
        arch.setCursor(QCursor(Qt.PointingHandCursor))
        
        self.scene.addItem(arch)
        
        # Add arrow in the arch
        arrow_label = QLabel("►")
        arrow_label.setStyleSheet("color: #8B4513; font-size: 20px; font-weight: bold; background: transparent;")
        arrow_proxy = self.scene.addWidget(arrow_label)
        arrow_proxy.setPos(arch_x + 25, arch_y + 35)
        
        return arch, arrow_proxy
    
    def add_q_logo(self):
        # Add Q logo in center
        screen = QDesktopWidget().screenGeometry()
        q_text = QGraphicsTextItem("Q")
        q_text.setDefaultTextColor(QColor("#d4af37"))
        font = QFont("Arial", 72, QFont.Bold)
        q_text.setFont(font)
        q_text.setPos(screen.width()//2 - 40, screen.height()//2 - 60)
        self.scene.addItem(q_text)
        return q_text
    
    def add_clock(self):
        # Add clock in top right
        screen = QDesktopWidget().screenGeometry()
        clock_text = QGraphicsTextItem()
        clock_text.setDefaultTextColor(QColor("#d4af37"))
        font = QFont("Arial", 18)
        clock_text.setFont(font)
        clock_text.setPos(screen.width() - 150, 20)
        self.scene.addItem(clock_text)
        return clock_text
    
    def update_time(self):
        current_time = datetime.now().strftime("%H:%M:%S")
        self.clock_text.setPlainText(current_time)
    
    def load_apps(self):
        # App data with absolute paths
        base_path = os.path.dirname(os.path.dirname(__file__))
        self.apps = [
            {"name": "Q-Notes", "path": os.path.join(base_path, "apps", "q_notes.py"), "icon": "📝"},
            {"name": "Q-Vault", "path": os.path.join(base_path, "apps", "q_vault.py"), "icon": "🔐"},
            {"name": "System Monitor", "path": os.path.join(base_path, "apps", "qshll_system_monitor.py"), "icon": "📊"}
        ]
    
    def toggle_arch(self, event):
        print(f"Arch clicked! Expanded: {self.is_arch_expanded}")
        if not self.is_arch_expanded:
            self.expand_arch()
        else:
            self.collapse_arch()
    
    def expand_arch(self):
        print("Expanding arch...")
        self.is_arch_expanded = True
        
        # Remove existing app proxies
        for proxy in self.app_proxies:
            self.scene.removeItem(proxy)
        self.app_proxies.clear()
        
        # Create app buttons in arc formation
        screen = QDesktopWidget().screenGeometry()
        center_x = 200
        center_y = screen.height() // 2
        radius = 100
        
        total_angle = math.pi
        if len(self.apps) > 1:
            angle_step = total_angle / (len(self.apps) - 1)
        else:
            angle_step = 0
            
        for i, app in enumerate(self.apps):
            # Create app button
            btn = QPushButton(f"{app['icon']}")
            btn.setFixedSize(50, 50)
            btn.setStyleSheet("""
                QPushButton {
                    background: rgba(212, 175, 55, 0.3);
                    border: 2px solid #d4af37;
                    border-radius: 25px;
                    color: #d4af37;
                    font-size: 24px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background: rgba(212, 175, 55, 0.5);
                }
            """)
            btn.clicked.connect(lambda checked, path=app['path']: self.launch_app(path))
            
            # Position in arc
            angle = i * angle_step
            x = center_x + radius * math.cos(math.pi - angle) - 25
            y = center_y - radius * math.sin(math.pi - angle) - 25
            
            proxy = self.scene.addWidget(btn)
            proxy.setPos(x, y)
            self.app_proxies.append(proxy)
            
            print(f"Added app {app['name']} at ({x}, {y})")
    
    def collapse_arch(self):
        print("Collapsing arch...")
        self.is_arch_expanded = False
        
        # Remove app proxies
        for proxy in self.app_proxies:
            self.scene.removeItem(proxy)
        self.app_proxies.clear()
    
    def launch_app(self, app_path):
        print(f"Launching: {app_path}")
        self.collapse_arch()
        try:
            base_dir = os.path.dirname(os.path.dirname(__file__))
            subprocess.Popen([sys.executable, app_path], cwd=base_dir)
            print(f"Successfully launched {app_path}")
        except Exception as e:
            print(f"Error launching {app_path}: {e}")
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            # Exit fullscreen, show window controls
            self.setWindowFlags(Qt.Window)
            self.showNormal()
        super().keyPressEvent(event)

def main():
    app = QApplication(sys.argv)
    desktop = QuantoniumDesktop()
    desktop.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
