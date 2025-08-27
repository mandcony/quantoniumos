#!/usr/bin/env python3
"""
QuantoniumOS Desktop Manager - Expandable Tab System
===================================================
Precise expandable half-circle tab on left that expands into full arc with apps
"""

import sys
import os
import math
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, 
                            QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                            QFrame, QDesktopWidget, QGraphicsDropShadowEffect)
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, pyqtSignal, QRect
from PyQt5.QtGui import QPalette, QColor, QFont, QPainter, QPen, QBrush, QPixmap, QIcon, QPolygon, QCursor

class QuantoniumDesktop(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QuantoniumOS")
        self.setStyleSheet("background-color: #f8f6f0;")
        
        # Remove window frame and make it fullscreen
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.showMaximized()
        
        # Main widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Create the expandable tab system
        self.tab_widget = ExpandableTab()
        self.tab_widget.setParent(self.central_widget)
        
        # Position the tab on the left center
        self.position_tab()
        
    def position_tab(self):
        # Get screen dimensions
        screen = QDesktopWidget().screenGeometry()
        
        # Position tab at left center
        tab_height = 100
        y = (screen.height() - tab_height) // 2
        
        self.tab_widget.setGeometry(0, y, 60, tab_height)
        self.tab_widget.show()

class ExpandableTab(QWidget):
    def __init__(self):
        super().__init__()
        self.expanded = False
        self.setFixedSize(60, 100)
        self.setCursor(QCursor(Qt.PointingHandCursor))
        
        # App data
        self.apps = [
            {"name": "Q-Notes", "path": "apps/q_notes.py", "icon": "📝"},
            {"name": "Q-Vault", "path": "apps/q_vault.py", "icon": "🔐"},
            {"name": "System Monitor", "path": "apps/qshll_system_monitor.py", "icon": "📊"}
        ]
        
        # Animation for expansion
        self.animation = QPropertyAnimation(self, b"geometry")
        self.animation.setDuration(300)
        self.animation.setEasingCurve(QEasingCurve.OutCubic)
        
        self.setup_ui()
        
    def setup_ui(self):
        # Create app buttons (initially hidden)
        self.app_buttons = []
        
        for i, app in enumerate(self.apps):
            btn = QPushButton(f"{app['icon']}")
            btn.setFixedSize(40, 40)
            btn.setStyleSheet("""
                QPushButton {
                    background: transparent;
                    border: none;
                    color: #d4af37;
                    font-size: 20px;
                    font-weight: bold;
                    border-radius: 20px;
                }
                QPushButton:hover {
                    color: #f0c040;
                    background: rgba(212, 175, 55, 0.1);
                }
            """)
            btn.clicked.connect(lambda checked, path=app['path']: self.launch_app(path))
            btn.setParent(self)
            btn.hide()
            self.app_buttons.append(btn)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.toggle_expansion()
    
    def toggle_expansion(self):
        if not self.expanded:
            # Expand to show full arc
            self.expand()
        else:
            # Collapse back to tab
            self.collapse()
    
    def expand(self):
        self.expanded = True
        
        # Animate to expanded size
        expanded_rect = QRect(0, self.y(), 400, 200)
        self.animation.setStartValue(self.geometry())
        self.animation.setEndValue(expanded_rect)
        self.animation.finished.connect(self.show_apps)
        self.animation.start()
    
    def collapse(self):
        self.expanded = False
        
        # Hide app buttons first
        for btn in self.app_buttons:
            btn.hide()
        
        # Animate back to tab size
        tab_rect = QRect(0, self.parent().height()//2 - 50, 60, 100)
        self.animation.setStartValue(self.geometry())
        self.animation.setEndValue(tab_rect)
        self.animation.start()
    
    def show_apps(self):
        # Position and show app icons in arc formation
        if self.expanded:
            self.position_icons_in_arc()
            for btn in self.app_buttons:
                btn.show()
    
    def position_icons_in_arc(self):
        # Position icons along the arch curve
        center_x = 200
        center_y = 180
        radius = 100
        
        # Calculate angles for even distribution along half circle
        total_angle = math.pi  # 180 degrees
        if len(self.apps) > 1:
            angle_step = total_angle / (len(self.apps) - 1)
        else:
            angle_step = 0
            
        for i, btn in enumerate(self.app_buttons):
            angle = i * angle_step
            x = center_x + radius * math.cos(math.pi - angle) - 20  # Center icon
            y = center_y - radius * math.sin(math.pi - angle) - 20
            btn.move(int(x), int(y))
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        if not self.expanded:
            # Draw compact tab with embedded arrow
            self.draw_tab(painter)
        else:
            # Draw expanded arc
            self.draw_expanded_arc(painter)
    
    def draw_tab(self, painter):
        # Draw the small half-circle tab
        pen = QPen(QColor("#d4af37"), 4)
        painter.setPen(pen)
        brush = QBrush(QColor("#f8f6f0"))
        painter.setBrush(brush)
        
        # Create half circle tab shape
        tab_rect = QRect(5, 20, 50, 60)
        painter.drawChord(tab_rect, 90*16, 180*16)  # Right half circle
        
        # Draw embedded arrow pointing right
        painter.setPen(QPen(QColor("#d4af37"), 2))
        # Arrow shape: >
        points = [
            (25, 40),  # Top point
            (35, 50),  # Right point
            (25, 60)   # Bottom point
        ]
        
        for i in range(len(points)-1):
            painter.drawLine(points[i][0], points[i][1], points[i+1][0], points[i+1][1])
    
    def draw_expanded_arc(self, painter):
        # Draw the full half-circle arch
        pen = QPen(QColor("#d4af37"), 4)
        painter.setPen(pen)
        
        # Half circle arc
        arc_rect = QRect(50, 50, 300, 150)
        painter.drawArc(arc_rect, 0, 180 * 16)  # 180 degrees
        
        # Add subtle glow effect
        glow_pen = QPen(QColor("#f0c040"), 2)
        painter.setPen(glow_pen)
        painter.drawArc(arc_rect.adjusted(2, 2, -2, -2), 0, 180 * 16)
    
    def launch_app(self, app_path):
        print(f"Launching: {app_path}")
        # Collapse after launching
        self.collapse()
        os.system(f"python {app_path}")

def main():
    app = QApplication(sys.argv)
    desktop = QuantoniumDesktop()
    desktop.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
