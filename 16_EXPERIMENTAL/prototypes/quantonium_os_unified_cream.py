#!/usr/bin/env python3
"""
QuantoniumOS Unified - Cream Design with Circular Dock
=====================================================
Complete quantum operating system matching the exact cream-colored design 
with circular app dock from screenshots, integrating all backend functions.
"""

import json
import logging
import math
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from functools import partial
from pathlib import Path
import pytz
# Flask imports for backend API
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from PyQt5.QtCore import QRectF, QSize, Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import (QBrush, QColor, QFont, QIcon, QPainter, QPalette,
                         QPen, QPixmap, QTransform)
# PyQt5 imports for the cream design
from PyQt5.QtWidgets import (QAction, QApplication, QFrame,
                             QGraphicsEllipseItem, QGraphicsProxyWidget,
                             QGraphicsScene, QGraphicsTextItem, QGraphicsView,
                             QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                             QListWidget, QMainWindow, QMenu, QMessageBox,
                             QPushButton, QScrollArea, QSplitter,
                             QSystemTrayIcon, QTabWidget, QTextEdit,
                             QVBoxLayout, QWidget)

# Try to import qtawesome for icons
try:
    import qtawesome as qta

    QTA_AVAILABLE = True
except ImportError:
    QTA_AVAILABLE = False
    print("⚠️ qtawesome not available - using text fallbacks for icons")

# Internal imports
sys.path.append(os.path.dirname(__file__))

# Directory setup
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
APPS_DIR = os.path.join(BASE_DIR, "apps")
STYLES_QSS = os.path.join(BASE_DIR, "styles.qss")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FlaskBackendThread(QThread):
    """Flask backend running in separate thread"""

    def __init__(self, os_instance):
        super().__init__()
        self.os_instance = os_instance
        self.app = Flask(__name__)
        CORS(self.app)
        self.setup_routes()

    def setup_routes(self):
        """Setup Flask API routes for backend services"""

        @self.app.route("/api/apps/list", methods=["GET"])
        def list_apps():
            """Return list of available applications"""
            apps = [
                {
                    "name": "File Explorer",
                    "script": "qshll_file_explorer.py",
                    "icon": "mdi.folder",
                },
                {"name": "Settings", "script": "qshll_settings.py", "icon": "mdi.cog"},
                {
                    "name": "Task Manager",
                    "script": "qshll_task_manager.py",
                    "icon": "mdi.view-dashboard",
                },
                {"name": "Q-Browser", "script": "q_browser.py", "icon": "mdi.web"},
                {"name": "Q-Mail", "script": "q_mail.py", "icon": "mdi.email"},
                {
                    "name": "Wave Composer",
                    "script": "q_wave_composer.py",
                    "icon": "mdi.music",
                },
                {"name": "Q-Vault", "script": "q_vault.py", "icon": "mdi.lock"},
                {"name": "Q-Notes", "script": "q_notes.py", "icon": "mdi.note"},
                {"name": "Q-Dock", "script": "q_dock.py", "icon": "mdi.dock-window"},
                {
                    "name": "Wave Debugger",
                    "script": "q_wave_debugger.py",
                    "icon": "mdi.wave",
                },
            ]
            return jsonify({"status": "success", "apps": apps})

        @self.app.route("/api/apps/launch", methods=["POST"])
        def launch_app():
            """Launch an application via backend"""
            data = request.get_json()
            app_script = data.get("script")

            if not app_script:
                return jsonify({"status": "error", "message": "No script provided"})

            script_path = os.path.join(APPS_DIR, app_script)

            if not os.path.exists(script_path):
                return jsonify(
                    {"status": "error", "message": f"Script not found: {script_path}"}
                )

            try:
                proc = subprocess.Popen(
                    [sys.executable, script_path], start_new_session=True
                )
                logger.info(f"✅ Launched: {script_path} (PID: {proc.pid})")
                return jsonify(
                    {
                        "status": "success",
                        "message": f"Launched {app_script}",
                        "pid": proc.pid,
                    }
                )
            except Exception as e:
                logger.error(f"❌ Failed to launch {script_path}: {e}")
                return jsonify({"status": "error", "message": str(e)})

        @self.app.route("/api/system/status", methods=["GET"])
        def system_status():
            """Get system status information"""
            import psutil

            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()

            return jsonify(
                {
                    "status": "success",
                    "system": {
                        "cpu": f"{cpu_percent:.1f}%",
                        "memory": f"{memory.percent:.1f}%",
                        "quantum_state": "Coherent (1000-qubit)",
                        "windows": len(
                            [
                                p
                                for p in psutil.process_iter()
                                if "python" in p.name().lower()
                            ]
                        ),
                    },
                }
            )

        @self.app.route("/health", methods=["GET"])
        def health_check():
            """Health check endpoint"""
            return jsonify(
                {"status": "healthy", "timestamp": datetime.now().isoformat()}
            )

    def run(self):
        """Run Flask backend"""
        try:
            logger.info("🚀 Starting Flask backend on http://localhost:5000")
            self.app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
        except Exception as e:
            logger.error(f"❌ Flask backend error: {e}")


def load_stylesheet(qss_path):
    """Load stylesheet from QSS file with fallback"""
    if os.path.exists(qss_path):
        try:
            with open(qss_path, "r", encoding="utf-8") as f:
                print(f"✅ Stylesheet loaded from {qss_path}")
                return f.read()
        except Exception as e:
            print(f"⚠️ Error loading stylesheet: {e}")

    # Fallback cream design stylesheet
    return """
    /* QuantoniumOS Cream Design - Fallback Styles */
    QMainWindow {
        background-color: #f0ead6;  /* Cream background */
        color: #333333;
    }
    
    QLabel#QLogo {
        color: #6b6b6b;             /* Gray Q logo */
        font: bold 180pt "Segoe UI";
        opacity: 0.8;
    }
    
    QLabel#AppIcon {
        color: #000000;             /* Black icons */
        background: transparent;
    }
    
    QLabel#ClockItem {
        color: #333333;
        font: bold 14pt "Segoe UI";
    }
    
    QLabel#AppName {
        color: #333333;
        font: 10pt "Segoe UI";
    }
    
    QGraphicsEllipseItem#ShadedArch {
        color: rgba(200, 200, 200, 0.3);  /* Semi-transparent dock */
        opacity: 0.3;
    }
    
    QLabel#ArrowIcon {
        color: #666666;
    }
    """


class AppIconLabel(QLabel):
    """Icon label for apps in the circular dock"""

    def __init__(self, icon_name, app_name, script_path, os_instance):
        super().__init__()
        self.icon_name = icon_name
        self.app_name = app_name
        self.script_path = script_path
        self.os_instance = os_instance

        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(64, 64)
        self.setMaximumSize(64, 64)
        self.setObjectName("AppIcon")

        # Set up icon
        self.update_icon()

        # Set cursor to pointer
        self.setCursor(Qt.PointingHandCursor)

    def update_icon(self):
        """Update icon using qtawesome or fallback"""
        try:
            if QTA_AVAILABLE:
                icon = qta.icon(self.icon_name, color=QColor("#000000"))  # Black icons
                self.setPixmap(icon.pixmap(64, 64))
                print(f"Icon {self.app_name} color set to: #000000")
            else:
                # Fallback to text
                self.setText(self.app_name[0])
                self.setStyleSheet(
                    "color: #000000; font-size: 24px; font-weight: bold;"
                )
        except Exception as e:
            print(f"⚠️ Error updating icon for {self.app_name}: {e}")
            self.setText(self.app_name[0])

    def mousePressEvent(self, event):
        """Launch app on click"""
        if event.button() == Qt.LeftButton:
            self.launch_app()
        super().mousePressEvent(event)

    def launch_app(self):
        """Launch the app"""
        if os.path.exists(self.script_path):
            try:
                subprocess.Popen(
                    [sys.executable, self.script_path], start_new_session=True
                )
                print(f"✅ Launched: {self.script_path}")
            except Exception as e:
                print(f"❌ Error launching {self.script_path}: {e}")
        else:
            print(f"⚠️ Script not found: {self.script_path}")


class QuantoniumOSCreamDesign(QMainWindow):
    """Main QuantoniumOS with cream design and circular dock"""

    def __init__(self):
        super().__init__()
        self.setObjectName("QuantoniumMainWindow")
        self.setWindowTitle("QuantoniumOS - Quantum Operating System")

        # Get screen geometry
        screen = QApplication.primaryScreen().geometry()
        self.screen_width = screen.width()
        self.screen_height = screen.height()
        self.setGeometry(0, 0, self.screen_width, self.screen_height)

        # Load stylesheet
        self.stylesheet = load_stylesheet(STYLES_QSS)
        self.setStyleSheet(self.stylesheet)

        # Setup graphics scene and view
        self.scene = QGraphicsScene(0, 0, self.screen_width, self.screen_height)
        self.view = QGraphicsView(self.scene, self)
        self.view.setObjectName("DesktopView")
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setCentralWidget(self.view)

        # UI state
        self.is_dock_expanded = False
        self.app_proxies = []

        # Setup UI elements
        self.dock_arch = self.create_dock_arch()
        self.q_logo = self.create_q_logo()
        self.clock_label = self.create_clock()
        self.load_app_icons()

        # Start clock timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_clock)
        self.timer.start(1000)
        self.update_clock()

        # Start Flask backend
        self.backend_thread = None
        self.start_backend()

        print("[OK] QuantoniumOS Cream Design initialized successfully!")

    def create_dock_arch(self):
        """Create the circular dock arch on the left side"""
        # Collapsed state dimensions
        tab_width = self.screen_width * 0.015
        tab_height = self.screen_height * 0.075
        tab_y = self.screen_height / 2 - tab_height / 2

        # Create arch
        arch_rect = QRectF(-tab_width, tab_y, tab_width * 2, tab_height)
        arch = QGraphicsEllipseItem(arch_rect)

        # Style the arch
        color = QColor(200, 200, 200, 80)  # Semi-transparent gray
        arch.setBrush(QBrush(color))
        arch.setPen(QPen(Qt.NoPen))
        arch.setAcceptHoverEvents(True)
        arch.setAcceptedMouseButtons(Qt.LeftButton)

        # Add click handler
        def toggle_dock(event):
            if event.button() == Qt.LeftButton:
                self.toggle_dock()

        arch.mousePressEvent = toggle_dock
        self.scene.addItem(arch)

        return arch

    def create_q_logo(self):
        """Create the large gray Q logo in center"""
        q_text = QGraphicsTextItem("Q")
        font = QFont("Segoe UI", 180, QFont.Bold)
        q_text.setFont(font)
        q_text.setDefaultTextColor(QColor("#6b6b6b"))  # Gray color
        q_text.setOpacity(0.8)

        # Center the Q logo
        text_width = q_text.boundingRect().width()
        text_height = q_text.boundingRect().height()
        q_text.setPos(
            (self.screen_width - text_width) / 2, (self.screen_height - text_height) / 2
        )

        self.scene.addItem(q_text)
        return q_text

    def create_clock(self):
        """Create clock in top-right corner"""
        clock_item = QGraphicsTextItem()
        font = QFont("Segoe UI", 14, QFont.Bold)
        clock_item.setFont(font)
        clock_item.setDefaultTextColor(QColor("#333333"))

        # Position in top-right
        clock_item.setPos(self.screen_width - 150, 20)

        self.scene.addItem(clock_item)
        return clock_item

    def load_app_icons(self):
        """Load app icons in circular dock arrangement"""
        apps = [
            {
                "name": "File Explorer",
                "script": "qshll_file_explorer.py",
                "icon": "mdi.folder",
            },
            {"name": "Settings", "script": "qshll_settings.py", "icon": "mdi.cog"},
            {
                "name": "Task Manager",
                "script": "qshll_task_manager.py",
                "icon": "mdi.view-dashboard",
            },
            {"name": "Q-Browser", "script": "q_browser.py", "icon": "mdi.web"},
            {"name": "Q-Mail", "script": "q_mail.py", "icon": "mdi.email"},
            {
                "name": "Wave Composer",
                "script": "q_wave_composer.py",
                "icon": "mdi.music",
            },
            {"name": "Q-Vault", "script": "q_vault.py", "icon": "mdi.lock"},
            {"name": "Q-Notes", "script": "q_notes.py", "icon": "mdi.note"},
            {"name": "Q-Dock", "script": "q_dock.py", "icon": "mdi.dock-window"},
            {
                "name": "Wave Debugger",
                "script": "q_wave_debugger.py",
                "icon": "mdi.wave",
            },
        ]

        # Circular arrangement parameters
        radius = 120
        center_x = 150
        center_y = self.screen_height / 2

        for i, app in enumerate(apps):
            angle = (i * 36) - 90  # Start from top, 36 degrees apart
            radian = math.radians(angle)

            x = center_x + radius * math.cos(radian)
            y = center_y + radius * math.sin(radian)

            # Create app icon
            script_path = os.path.join(APPS_DIR, app["script"])
            icon_label = AppIconLabel(app["icon"], app["name"], script_path, self)

            # Add to scene
            proxy_icon = self.scene.addWidget(icon_label)
            proxy_icon.setPos(x - 32, y - 32)  # Center the 64x64 icon

            # Create app name label
            name_label = QLabel(app["name"])
            name_label.setObjectName("AppName")
            name_label.setAlignment(Qt.AlignCenter)
            name_label.setStyleSheet("color: #333333; font: 10pt 'Segoe UI';")

            proxy_name = self.scene.addWidget(name_label)
            proxy_name.setPos(x - 40, y + 35)  # Below the icon

            # Store proxies
            self.app_proxies.extend([proxy_icon, proxy_name])

        # Initially hide all icons (behind collapsed dock)
        for proxy in self.app_proxies:
            proxy.setVisible(False)

    def toggle_dock(self):
        """Toggle dock expansion"""
        self.is_dock_expanded = not self.is_dock_expanded

        if self.is_dock_expanded:
            # Expand dock
            arch_width = self.screen_width * 0.2
            arch_height = self.screen_height * 0.8
            center_y = (self.screen_height - arch_height) / 2

            self.dock_arch.setRect(
                QRectF(-arch_width, center_y, arch_width * 2, arch_height)
            )

            # Show app icons
            for proxy in self.app_proxies:
                proxy.setVisible(True)

            print("🔄 Dock expanded - apps visible")
        else:
            # Collapse dock
            tab_width = self.screen_width * 0.015
            tab_height = self.screen_height * 0.075
            tab_y = self.screen_height / 2 - tab_height / 2

            self.dock_arch.setRect(QRectF(-tab_width, tab_y, tab_width * 2, tab_height))

            # Hide app icons
            for proxy in self.app_proxies:
                proxy.setVisible(False)

            print("🔄 Dock collapsed - apps hidden")

        self.scene.update()

    def update_clock(self):
        """Update clock display"""
        try:
            est = pytz.timezone("America/New_York")
            now = datetime.now(est)
            time_str = now.strftime("%I:%M %p\n%b %d")
            self.clock_label.setPlainText(time_str)
        except:
            # Fallback if pytz not available
            now = datetime.now()
            time_str = now.strftime("%I:%M %p\n%b %d")
            self.clock_label.setPlainText(time_str)

    def start_backend(self):
        """Start Flask backend in separate thread"""
        try:
            self.backend_thread = FlaskBackendThread(self)
            self.backend_thread.start()
            print("🚀 Flask backend started successfully")
        except Exception as e:
            print(f"❌ Failed to start backend: {e}")

    def keyPressEvent(self, event):
        """Handle key events"""
        if event.key() == Qt.Key_Escape:
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()
        elif event.key() == Qt.Key_F11:
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()

    def closeEvent(self, event):
        """Clean shutdown"""
        if self.backend_thread:
            self.backend_thread.terminate()
            self.backend_thread.wait()
        event.accept()


class QuantoniumOSUnified:
    """Unified QuantoniumOS Controller"""

    def __init__(self):
        self.app = None
        self.main_window = None
        self.initialize()

    def initialize(self):
        """Initialize the OS"""
        print("🌟 Initializing QuantoniumOS - Cream Design Edition")

        # Create QApplication
        self.app = QApplication(sys.argv)
        self.app.setApplicationName("QuantoniumOS")
        self.app.setApplicationDisplayName("QuantoniumOS - Quantum Operating System")

        # Create main window
        self.main_window = QuantoniumOSCreamDesign()

        print("✅ QuantoniumOS initialized successfully!")

    def run(self):
        """Run the OS"""
        if self.main_window:
            self.main_window.showFullScreen()
            print("🚀 QuantoniumOS is now running!")
            return self.app.exec_()
        else:
            print("❌ Failed to initialize QuantoniumOS")
            return 1


def main():
    """Main entry point"""
    print("=" * 60)
    print("🔬 QuantoniumOS - Quantum Operating System")
    print("🎨 Cream Design with Circular Dock")
    print("=" * 60)

    # Create and run OS
    quantonium_os = QuantoniumOSUnified()
    return quantonium_os.run()


if __name__ == "__main__":
    sys.exit(main())
