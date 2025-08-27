#!/usr/bin/env python3
"""
QuantoniumOS Desktop Manager
===========================
Sleek, minimal desktop with expandable side arch and your original cream design
"""

import sys
import os
import subprocess
from datetime import datetime
import math
from typing import List, Dict, Tuple, Optional

# PyQt5 imports
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QGraphicsView, QGraphicsScene, QGraphicsProxyWidget,
    QLabel, QGraphicsEllipseItem, QGraphicsTextItem, QWidget, QVBoxLayout
)
from PyQt5.QtGui import QFont, QColor, QBrush, QPen, QPainter, QPixmap
from PyQt5.QtCore import Qt, QTimer, QRectF, QPointF

# Optional imports
try:
    import qtawesome as qta
    HAS_QTAWESOME = True
except ImportError:
    HAS_QTAWESOME = False

try:
    import pytz
    HAS_PYTZ = True
except ImportError:
    HAS_PYTZ = False

# Get paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Try to import RFT Assembly
try:
    from ASSEMBLY.python_bindings.unitary_rft import RFTProcessor
    HAS_RFT = True
except ImportError:
    HAS_RFT = False


class AppIconLabel(QLabel):
    """Sleek app icon that launches apps on click"""
    
    def __init__(self, icon_name, app_name, script_path, position):
        super().__init__()
        self.icon_name = icon_name
        self.app_name = app_name
        self.script_path = script_path
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(64, 64)
        self.setMaximumSize(64, 64)
        
        # Your original sleek icon style
        self.setStyleSheet("""
            QLabel {
                background: rgba(100, 166, 255, 100);
                border: 2px solid rgba(100, 166, 255, 200);
                border-radius: 32px;
                color: white;
                font-weight: bold;
                font-size: 12px;
            }
            QLabel:hover {
                background: rgba(150, 200, 255, 150);
                border: 2px solid rgba(150, 200, 255, 255);
            }
        """)
        self.update_icon()

    def update_icon(self):
        """Update icon display"""
        if HAS_QTAWESOME:
            try:
                icon = qta.icon(self.icon_name, color=QColor("white"))
                self.setPixmap(icon.pixmap(48, 48))
            except Exception:
                # Fallback to text
                self.setText(self.app_name[:2].upper())
        else:
            # Text-based icon
            self.setText(self.app_name[:2].upper())

    def mousePressEvent(self, event):
        """Launch app on click"""
        if event.button() == Qt.LeftButton:
            self.launch_app()

    def launch_app(self):
        """Launch the app script"""
        full_path = os.path.join(BASE_DIR, "apps", self.script_path)
        if os.path.exists(full_path):
            try:
                subprocess.Popen([sys.executable, full_path], start_new_session=True)
                print(f"✅ Launched: {self.app_name}")
            except Exception as e:
                print(f"❌ Failed to launch {self.app_name}: {e}")
        else:
            print(f"❌ App not found: {full_path}")


class QuantoniumDesktop(QMainWindow):
    """Main QuantoniumOS Desktop with your original sleek design"""
    
    def __init__(self):
        super().__init__()
        self.setObjectName("QuantoniumMainWindow")
        self.setWindowTitle("QuantoniumOS")

        # Full screen setup
        screen = QApplication.primaryScreen().geometry()
        self.screen_width = screen.width()
        self.screen_height = screen.height()
        self.setGeometry(0, 0, self.screen_width, self.screen_height)

        # Your original beautiful gradient background
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #0a0a1a, stop:0.3 #1a1a2e, 
                    stop:0.7 #16213e, stop:1.0 #0f3460);
            }
        """)

        # Setup graphics scene
        self.scene = QGraphicsScene(0, 0, self.screen_width, self.screen_height)
        self.view = QGraphicsView(self.scene, self)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setStyleSheet("QGraphicsView { border: none; }")
        self.setCentralWidget(self.view)

        # Initialize RFT if available
        self.rft = None
        if HAS_RFT:
            try:
                self.rft = RFTProcessor()
                print("🚀 RFT Engine initialized")
            except Exception as e:
                print(f"⚠ RFT Engine failed to initialize: {e}")

        # UI elements
        self.is_arch_expanded = False
        self.arch, self.arrow_proxy = self.add_shaded_arch()
        self.app_proxies = []
        self.q_logo = self.add_q_logo()
        self.clock_text = self.add_clock()
        self.update_time()

        # Clock timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_time)
        self.timer.start(1000)

        # Load applications
        self.load_apps()

    def add_q_logo(self):
        """Add the central Q logo"""
        q_text = QGraphicsTextItem("Q")
        
        # Large, bold Q
        font = QFont("Arial Black", 200)
        font.setBold(True)
        q_text.setFont(font)
        
        # Your original cream/blue color
        q_text.setDefaultTextColor(QColor(100, 166, 255))
        q_text.setOpacity(0.8)
        
        # Center the Q
        text_width = q_text.boundingRect().width()
        text_height = q_text.boundingRect().height()
        q_text.setPos((self.screen_width - text_width) / 2, (self.screen_height - text_height) / 2)
        
        self.scene.addItem(q_text)
        return q_text

    def add_clock(self):
        """Add real-time clock in top-right corner"""
        clock_item = QGraphicsTextItem()
        
        font = QFont("Courier New", 16)
        font.setBold(True)
        clock_item.setFont(font)
        clock_item.setDefaultTextColor(QColor("white"))
        
        # Position in top-right
        clock_item.setPos(self.screen_width - 200, 20)
        self.scene.addItem(clock_item)
        return clock_item

    def add_shaded_arch(self):
        """Add the expandable side arch - your original design"""
        # Collapsed state dimensions
        tab_width = self.screen_width * 0.015
        tab_height = self.screen_height * 0.075
        tab_y = self.screen_height / 2 - tab_height / 2
        
        # Create arch shape
        arch_rect = QRectF(-tab_width, tab_y, tab_width * 2, tab_height)
        arch = QGraphicsEllipseItem(arch_rect)
        
        # Your original arch styling
        color = QColor(100, 166, 255)
        color.setAlphaF(0.6)
        arch.setBrush(QBrush(color))
        arch.setPen(QPen(Qt.NoPen))
        arch.setAcceptHoverEvents(True)
        arch.setAcceptedMouseButtons(Qt.LeftButton)
        
        # Connect mouse events
        arch.mousePressEvent = self.toggle_arch
        self.scene.addItem(arch)

        # Add arrow indicator
        arrow_label = QLabel()
        if HAS_QTAWESOME:
            try:
                arrow_icon = qta.icon("mdi.arrow-right", color=QColor("white"))
                arrow_size = int(tab_height * 0.4)
                arrow_label.setPixmap(arrow_icon.pixmap(arrow_size, arrow_size))
            except Exception:
                arrow_label.setText("→")
                arrow_label.setStyleSheet("color: white; font-weight: bold;")
        else:
            arrow_label.setText("→")
            arrow_label.setStyleSheet("color: white; font-weight: bold;")
        
        arrow_label.setAlignment(Qt.AlignCenter)
        arrow_proxy = self.scene.addWidget(arrow_label)
        
        # Position arrow
        arrow_width = 24
        arrow_height = 24
        arrow_x = -tab_width + (tab_width * 2 - arrow_width) / 2
        arrow_y = tab_y + (tab_height - arrow_height) / 2
        arrow_proxy.setPos(arrow_x, arrow_y)
        
        # Allow arrow to toggle arch too
        arrow_label.mousePressEvent = self.toggle_arch

        return arch, arrow_proxy

    def toggle_arch(self, event):
        """Expand/collapse the arch with smooth animation"""
        if not self.is_arch_expanded:
            # Expand the arch
            arch_width = self.screen_width * 0.12
            arch_height = self.screen_height * 0.8
            center_y = self.screen_height / 2 - arch_height / 2
            self.arch.setRect(QRectF(-arch_width, center_y, arch_width * 2, arch_height))
            
            # Show app icons
            for proxy in self.app_proxies:
                proxy.setVisible(True)
            
            # Update arrow to point left
            arrow_label = self.arrow_proxy.widget()
            if HAS_QTAWESOME:
                try:
                    arrow_icon = qta.icon("mdi.arrow-left", color=QColor("white"))
                    arrow_size = int(arch_height * 0.04)
                    arrow_label.setPixmap(arrow_icon.pixmap(arrow_size, arrow_size))
                except Exception:
                    arrow_label.setText("←")
            else:
                arrow_label.setText("←")
            
            # Reposition arrow
            arrow_width = 24
            arrow_height = 24
            arrow_x = -arch_width + (arch_width * 2 - arrow_width) / 2
            arrow_y = center_y + (arch_height - arrow_height) / 2
            self.arrow_proxy.setPos(arrow_x, arrow_y)
            
        else:
            # Collapse the arch
            tab_width = self.screen_width * 0.015
            tab_height = self.screen_height * 0.075
            tab_y = self.screen_height / 2 - tab_height / 2
            self.arch.setRect(QRectF(-tab_width, tab_y, tab_width * 2, tab_height))
            
            # Hide app icons
            for proxy in self.app_proxies:
                proxy.setVisible(False)
            
            # Update arrow to point right
            arrow_label = self.arrow_proxy.widget()
            if HAS_QTAWESOME:
                try:
                    arrow_icon = qta.icon("mdi.arrow-right", color=QColor("white"))
                    arrow_size = int(tab_height * 0.4)
                    arrow_label.setPixmap(arrow_icon.pixmap(arrow_size, arrow_size))
                except Exception:
                    arrow_label.setText("→")
            else:
                arrow_label.setText("→")
            
            # Reposition arrow
            arrow_width = 24
            arrow_height = 24
            arrow_x = -tab_width + (tab_width * 2 - arrow_width) / 2
            arrow_y = tab_y + (tab_height - arrow_height) / 2
            self.arrow_proxy.setPos(arrow_x, arrow_y)

        self.is_arch_expanded = not self.is_arch_expanded
        self.scene.update()

    def load_apps(self):
        """Load your essential apps with proper spacing"""
        # Your actual apps only
        apps = [
            {"name": "Q-Notes", "script": "q_notes.py", "icon": "mdi.note"},
            {"name": "Q-Vault", "script": "q_vault.py", "icon": "mdi.lock"},
            {"name": "System Monitor", "script": "qshll_system_monitor.py", "icon": "mdi.monitor"},
        ]
        
        # Calculate positioning
        icon_size = 64
        spacing = 20
        start_y = (self.screen_height - (len(apps) * (icon_size + spacing))) / 2
        x = -icon_size - 10  # Position in the collapsed arch area
        
        for i, app in enumerate(apps):
            y = start_y + i * (icon_size + spacing)
            
            # Create app icon
            icon_label = AppIconLabel(
                icon_name=app["icon"],
                app_name=app["name"],
                script_path=app["script"],
                position=(x, y)
            )
            
            proxy_icon = self.scene.addWidget(icon_label)
            proxy_icon.setPos(x, y)

            # Create app name label
            name_label = QLabel(app["name"])
            name_label.setStyleSheet("""
                QLabel {
                    color: white;
                    font-size: 10px;
                    font-weight: bold;
                    background: transparent;
                }
            """)
            name_label.setAlignment(Qt.AlignCenter)
            name_label.setMaximumWidth(icon_size + spacing)
            
            proxy_name = self.scene.addWidget(name_label)
            name_x = x
            name_y = y + icon_size + 5
            proxy_name.setPos(name_x, name_y)

            # Store proxies for show/hide
            self.app_proxies.append(proxy_icon)
            self.app_proxies.append(proxy_name)

        # Initially hide all apps (arch starts collapsed)
        for proxy in self.app_proxies:
            proxy.setVisible(False)

    def update_time(self):
        """Update clock display"""
        if HAS_PYTZ:
            try:
                est = pytz.timezone("America/New_York")
                now = datetime.now(est)
            except:
                now = datetime.now()
        else:
            now = datetime.now()
        
        time_str = now.strftime("%I:%M %p\n%b %d, %Y")
        self.clock_text.setPlainText(time_str)

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts"""
        if event.key() == Qt.Key_Escape:
            self.close()
        elif event.key() == Qt.Key_Space:
            self.toggle_arch(None)
        super().keyPressEvent(event)


def main():
    """Launch QuantoniumOS Desktop"""
    app = QApplication(sys.argv)
    
    # Create and show the desktop
    desktop = QuantoniumDesktop()
    desktop.showFullScreen()
    
    # Run the application
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
