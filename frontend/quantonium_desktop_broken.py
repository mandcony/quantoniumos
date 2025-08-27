#!/usr/bin/env python3
"""
QuantoniumOS Desktop Manager - Expandable Tab System
===================================================
Precise expandable half-circle tab on left that expands into full arc with apps
"""

import sys
import os
import math
import subprocess
from datetime import datetime
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
        
        # Create Q logo in center
        self.q_logo = QLabel("Q")
        self.q_logo.setParent(self.central_widget)
        self.q_logo.setStyleSheet("""
            QLabel {
                color: #d4af37;
                font-size: 72px;
                font-weight: bold;
                background: transparent;
            }
        """)
        self.q_logo.setAlignment(Qt.AlignCenter)
        
        # Create time display
        self.time_label = QLabel()
        self.time_label.setParent(self.central_widget)
        self.time_label.setStyleSheet("""
            QLabel {
                color: #d4af37;
                font-size: 18px;
                font-weight: normal;
                background: transparent;
            }
        """)
        self.time_label.setAlignment(Qt.AlignCenter)
        
        # Timer for updating time
        self.time_timer = QTimer()
        self.time_timer.timeout.connect(self.update_time)
        self.time_timer.start(1000)  # Update every second
        self.update_time()
        
        # Position the tab on the left center
        self.position_tab()
        self.position_center_elements()
        
    def position_tab(self):
        # Get screen dimensions
        screen = QDesktopWidget().screenGeometry()
        
        # Position tab at left center
        tab_height = 100
        y = (screen.height() - tab_height) // 2
        
        self.tab_widget.setGeometry(0, y, 60, tab_height)
        self.tab_widget.show()
    
    def position_center_elements(self):
        # Get screen dimensions
        screen = QDesktopWidget().screenGeometry()
        
        # Position Q logo in center
        self.q_logo.setGeometry(screen.width()//2 - 50, screen.height()//2 - 50, 100, 100)
        self.q_logo.show()
        
        # Position time in top right
        self.time_label.setGeometry(screen.width() - 150, 20, 140, 30)
        self.time_label.show()
    
    def update_time(self):
        current_time = datetime.now().strftime("%H:%M:%S")
        self.time_label.setText(current_time)
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            # Exit fullscreen, show window controls
            self.setWindowFlags(Qt.Window)
            self.showNormal()
        super().keyPressEvent(event)

class ExpandableTab(QWidget):
    def __init__(self):
        super().__init__()
        self.expanded = False
        self.setFixedSize(60, 100)
        self.setCursor(QCursor(Qt.PointingHandCursor))
        
        # App data with absolute paths
        base_path = os.path.dirname(os.path.dirname(__file__))
        self.apps = [
            {"name": "Q-Notes", "path": os.path.join(base_path, "apps", "q_notes.py"), "icon": "📝"},
            {"name": "Q-Vault", "path": os.path.join(base_path, "apps", "q_vault.py"), "icon": "🔐"},
            {"name": "System Monitor", "path": os.path.join(base_path, "apps", "qshll_system_monitor.py"), "icon": "📊"}
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
            print(f"Tab clicked! Currently expanded: {self.expanded}")
            self.toggle_expansion()
    
    def toggle_expansion(self):
        print(f"Toggling expansion - current state: {self.expanded}")
        if not self.expanded:
            print("Expanding...")
            # Expand to show full arc
            self.expand()
        else:
            print("Collapsing...")
            # Collapse back to tab
            self.collapse()
    
    def expand(self):
        self.expanded = True
        print(f"Expanding to size: {self.geometry()} -> will be 400x200")
        
        # Disconnect any previous connections
        try:
            self.animation.finished.disconnect()
        except:
            pass
        
        # Animate to expanded size
        current_rect = self.geometry()
        expanded_rect = QRect(0, current_rect.y() - 50, 400, 200)
        print(f"Animation: {current_rect} -> {expanded_rect}")
        
        self.animation.setStartValue(current_rect)
        self.animation.setEndValue(expanded_rect)
        self.animation.finished.connect(self.show_apps)
        self.animation.start()
        
        # Force immediate update for testing
        self.setGeometry(expanded_rect)
        self.show_apps()
        self.update()
    
    def collapse(self):
        self.expanded = False
        print("Collapsing back to tab size")
        
        # Hide app buttons first
        for btn in self.app_buttons:
            btn.hide()
        
        # Disconnect any previous connections
        try:
            self.animation.finished.disconnect()
        except:
            pass
        
        # Animate back to tab size
        screen = QDesktopWidget().screenGeometry()
        tab_rect = QRect(0, (screen.height() - 100) // 2, 60, 100)
        self.animation.setStartValue(self.geometry())
        self.animation.setEndValue(tab_rect)
        self.animation.start()
        
        # Force immediate update for testing
        self.setGeometry(tab_rect)
        self.update()
    
    def show_apps(self):
        # Position and show app icons in arc formation
        print("show_apps called!")
        if self.expanded:
            print("Widget is expanded, positioning icons...")
            self.position_icons_in_arc()
            for i, btn in enumerate(self.app_buttons):
                print(f"Showing app button {i}: {self.apps[i]['name']}")
                btn.show()
                btn.raise_()  # Bring to front
            print("All app buttons should now be visible!")
        else:
            print("Widget not expanded, not showing apps")
    
    def position_icons_in_arc(self):
        # Position icons along the arch curve WITHIN widget bounds
        center_x = 200
        center_y = 150  # FIXED: Move center down so icons are visible!
        radius = 80    # Smaller radius to fit
        
        print(f"Positioning {len(self.app_buttons)} icons in arc")
        print(f"Center: ({center_x}, {center_y}), Radius: {radius}")
        
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
            print(f"Icon {i} position: ({int(x)}, {int(y)})")
            btn.move(int(x), int(y))
            btn.setStyleSheet("""
                QPushButton {
                    background: rgba(212, 175, 55, 0.3);
                    border: 2px solid #d4af37;
                    color: #d4af37;
                    font-size: 20px;
                    font-weight: bold;
                    border-radius: 20px;
                }
                QPushButton:hover {
                    color: #f0c040;
                    background: rgba(212, 175, 55, 0.5);
                }
            """)
    
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
        # Draw the small half-circle tab with more visibility
        pen = QPen(QColor("#d4af37"), 6)  # Thicker border
        painter.setPen(pen)
        brush = QBrush(QColor("#f0c040"))  # More visible fill
        painter.setBrush(brush)
        
        # Create half circle tab shape - bigger and more visible
        tab_rect = QRect(2, 15, 55, 70)
        painter.drawChord(tab_rect, 90*16, 180*16)  # Right half circle
        
        # Draw embedded arrow pointing right - bigger and more visible
        painter.setPen(QPen(QColor("#8B4513"), 4))  # Brown arrow, thicker
        # Arrow shape: >
        points = [
            (20, 35),  # Top point
            (40, 50),  # Right point  
            (20, 65)   # Bottom point
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
        try:
            # Use subprocess to launch app properly with correct working directory
            base_dir = os.path.dirname(os.path.dirname(__file__))
            subprocess.Popen([sys.executable, app_path], cwd=base_dir)
            print(f"Successfully launched {app_path}")
        except Exception as e:
            print(f"Error launching {app_path}: {e}")

def main():
    app = QApplication(sys.argv)
    desktop = QuantoniumDesktop()
    desktop.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
