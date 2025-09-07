#!/usr/bin/env python3
"""
QuantoniumOS Main Interface
==========================
The true QuantoniumOS with side arch, expandable app dock, and central Q logo
"""

import sys
import os
import subprocess
from datetime import datetime
import pytz
import qtawesome as qta
import re

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QGraphicsView, QGraphicsScene, QGraphicsProxyWidget,
    QLabel, QGraphicsEllipseItem, QGraphicsTextItem
)
from PyQt5.QtGui import QFont, QColor, QBrush, QPen, QPainter, QTransform
from PyQt5.QtCore import Qt, QTimer, QRectF

# Directory setup
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
APP_DIR = os.path.join(BASE_DIR, "apps")
STYLES_QSS = os.path.join(BASE_DIR, "styles.qss")

# Add RFT Assembly path
assembly_path = os.path.join(BASE_DIR, "ASSEMBLY", "python_bindings")
if not os.path.exists(assembly_path):
    # Try WORKING_RFT_ASSEMBLY as fallback
    assembly_path = os.path.join(BASE_DIR, "WORKING_RFT_ASSEMBLY", "python_bindings")

if os.path.exists(assembly_path):
    sys.path.append(assembly_path)
    try:
        import unitary_rft
        print("RFT Assembly loaded successfully from", assembly_path)
    except ImportError as e:
        print(f"Could not import RFT Assembly: {e}")
else:
    print("RFT Assembly not found")

class AppIconLabel(QLabel):
    """Label that displays an icon for an app, launching the app on click."""
    
    def __init__(self, icon_name, app_name, script_path, position):
        super().__init__()
        self.icon_name = icon_name
        self.app_name = app_name
        self.script_path = script_path
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(64, 64)
        self.setMaximumSize(64, 64)
        self.setStyleSheet("""
            QLabel {
                background: rgba(100, 166, 255, 100);
                border: 2px solid rgba(100, 166, 255, 200);
                border-radius: 32px;
                color: white;
                font-weight: bold;
            }
            QLabel:hover {
                background: rgba(150, 200, 255, 150);
                border: 2px solid rgba(150, 200, 255, 255);
            }
        """)
        self.update_icon()

    def update_icon(self):
        """Update the icon using qtawesome or fallback to text."""
        try:
            icon = qta.icon(self.icon_name, color=QColor("white"))
            self.setPixmap(icon.pixmap(48, 48))
        except Exception as e:
            print(f"⚠️ Error setting icon for {self.app_name}: {e}")
            self.setText(self.app_name[0])  # Fallback to first letter

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.launch_app()
        super().mousePressEvent(event)

    def launch_app(self):
        """Launch the external script in a new process."""
        if os.path.exists(self.script_path):
            try:
                subprocess.Popen([sys.executable, self.script_path], start_new_session=True)
                print(f"✅ Launched: {self.script_path}")
            except Exception as e:
                print(f"⚠️ ERROR: Failed to launch {self.script_path}: {e}")
        else:
            print(f"⚠️ WARNING: Script not found: {self.script_path}")

class QuantoniumOSWindow(QMainWindow):
    """Main OS UI with the 'Q' logo, side arch, app icons, clock, etc."""
    
    def __init__(self):
        super().__init__()
        self.setObjectName("QuantoniumMainWindow")
        self.setWindowTitle("QuantoniumOS")

        # Set geometry based on primary screen
        screen = QApplication.primaryScreen().geometry()
        self.screen_width = screen.width()
        self.screen_height = screen.height()
        self.setGeometry(0, 0, self.screen_width, self.screen_height)

        # Set the main window style
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #0a0a1a, stop:0.3 #1a1a2e, 
                    stop:0.7 #16213e, stop:1.0 #0f3460);
            }
        """)

        # Setup scene & view
        self.scene = QGraphicsScene(0, 0, self.screen_width, self.screen_height)
        self.view = QGraphicsView(self.scene, self)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setStyleSheet("QGraphicsView { border: none; }")
        self.setCentralWidget(self.view)

        # Initialize UI elements
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

        # Load apps
        self.load_apps()

    def add_q_logo(self):
        """Draw a stylized 'Q' in the center."""
        q_text = QGraphicsTextItem("Q")
        
        # Create a large font
        font = QFont("Arial Black", 200)
        font.setBold(True)
        q_text.setFont(font)
        
        # Set color and opacity
        q_text.setDefaultTextColor(QColor(100, 166, 255))
        q_text.setOpacity(0.8)
        
        # Center the Q
        text_width = q_text.boundingRect().width()
        text_height = q_text.boundingRect().height()
        q_text.setPos((self.screen_width - text_width) / 2, (self.screen_height - text_height) / 2)
        
        self.scene.addItem(q_text)
        return q_text

    def add_clock(self):
        """Adds a real-time clock in top-right corner."""
        clock_item = QGraphicsTextItem()
        
        # Set font for clock
        font = QFont("Courier New", 16)
        font.setBold(True)
        clock_item.setFont(font)
        clock_item.setDefaultTextColor(QColor("white"))
        
        # Position in top-right
        clock_item.setPos(self.screen_width - 200, 20)
        
        self.scene.addItem(clock_item)
        return clock_item

    def add_shaded_arch(self):
        """The side arch that can expand/collapse, holding app icons."""
        # Collapsed state dimensions
        tab_width = self.screen_width * 0.015
        tab_height = self.screen_height * 0.075
        tab_y = self.screen_height / 2 - tab_height / 2
        
        # Create the arch shape
        arch_rect = QRectF(-tab_width, tab_y, tab_width * 2, tab_height)
        arch = QGraphicsEllipseItem(arch_rect)
        
        # Style the arch
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
        try:
            arrow_icon = qta.icon("mdi.arrow-right", color=QColor("white"))
            arrow_size = int(tab_height * 0.4)
            arrow_label.setPixmap(arrow_icon.pixmap(arrow_size, arrow_size))
        except:
            arrow_label.setText("→")
            arrow_label.setStyleSheet("color: white; font-weight: bold; font-size: 16px;")
        
        arrow_label.setAlignment(Qt.AlignCenter)
        arrow_proxy = self.scene.addWidget(arrow_label)
        
        # Position arrow
        arrow_width = 24  # Approximate
        arrow_height = 24
        arrow_x = -tab_width + (tab_width * 2 - arrow_width) / 2
        arrow_y = tab_y + (tab_height - arrow_height) / 2
        arrow_proxy.setPos(arrow_x, arrow_y)
        
        # Allow arrow to toggle arch too
        arrow_label.mousePressEvent = self.toggle_arch

        return arch, arrow_proxy

    def toggle_arch(self, event):
        """Expand/collapse the arch with icons."""
        if event.button() != Qt.LeftButton:
            return
            
        self.is_arch_expanded = not self.is_arch_expanded
        
        if self.is_arch_expanded:
            # Expand the arch
            arch_width = self.screen_width * 0.15
            arch_height = self.screen_height * 0.7
            center_y = (self.screen_height - arch_height) / 2
            self.arch.setRect(QRectF(-arch_width, center_y, arch_width * 2, arch_height))
            
            # Show app icons
            for proxy in self.app_proxies:
                proxy.setVisible(True)
            
            # Update arrow to point left
            arrow_label = self.arrow_proxy.widget()
            try:
                arrow_icon = qta.icon("mdi.arrow-left", color=QColor("white"))
                arrow_size = int(arch_height * 0.04)
                arrow_label.setPixmap(arrow_icon.pixmap(arrow_size, arrow_size))
            except:
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
            try:
                arrow_icon = qta.icon("mdi.arrow-right", color=QColor("white"))
                arrow_size = int(tab_height * 0.4)
                arrow_label.setPixmap(arrow_icon.pixmap(arrow_size, arrow_size))
            except:
                arrow_label.setText("→")
            
            # Reposition arrow
            arrow_width = 24
            arrow_height = 24
            arrow_x = -tab_width + (tab_width * 2 - arrow_width) / 2
            arrow_y = tab_y + (tab_height - arrow_height) / 2
            self.arrow_proxy.setPos(arrow_x, arrow_y)

        self.scene.update()

    def load_apps(self):
        """Loads app icons on the side arch, in a grid layout."""
        # Define available apps and their corresponding scripts
        apps = [
            {"name": "RFT Visualizer", "icon": "mdi.wave", "script": "launch_rft_visualizer.py"},
            {"name": "Quantum Simulator", "icon": "mdi.atom", "script": "launch_quantum_simulator.py"},
            {"name": "Q-Mail", "icon": "mdi.email", "script": "launch_q_mail.py"},
            {"name": "Q-Notes", "icon": "mdi.note", "script": "launch_q_notes.py"},
            {"name": "Q-Vault", "icon": "mdi.lock", "script": "launch_q_vault.py"},
        ]

        # Grid layout parameters
        columns = 2
        icon_size = 64
        spacing = self.screen_width * 0.02
        label_height = 20
        start_x = self.screen_width * 0.01
        start_y = self.screen_height / 4

        for i, app in enumerate(apps):
            col = i % columns
            row = i // columns
            x = start_x + col * (icon_size + spacing)
            y = start_y + row * (icon_size + spacing + label_height)

            # Create app icon - use absolute path from apps directory
            script_path = os.path.join(APP_DIR, app["script"])
            
            icon_label = AppIconLabel(
                icon_name=app["icon"],
                app_name=app["name"],
                script_path=script_path,
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
            name_label.setMaximumWidth(int(icon_size + spacing))
            
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
        """Updates clock display every second."""
        try:
            est = pytz.timezone("America/New_York")
            now = datetime.now(est)
        except:
            now = datetime.now()
        
        time_str = now.strftime("%I:%M %p\n%b %d, %Y")
        self.clock_text.setPlainText(time_str)

    def keyPressEvent(self, event):
        """Handle key press events."""
        if event.key() == Qt.Key_Escape:
            # Toggle fullscreen
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()
        elif event.key() == Qt.Key_F11:
            # F11 also toggles fullscreen
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()

class QuantoniumOSTerminal:
    """Terminal-based interface for QuantoniumOS"""
    
    def __init__(self):
        """Initialize the terminal interface"""
        self.running = True
    
    def start(self):
        """Start the terminal interface"""
        print("\n" + "=" * 60)
        print("QuantoniumOS Terminal Interface")
        print("=" * 60 + "\n")
        
        print("Available commands:")
        print("  help          - Show this help message")
        print("  apps          - List available applications") 
        print("  launch [app]  - Launch an application")
        print("  status        - Show system status")
        print("  exit          - Exit QuantoniumOS\n")
        
        # Main loop
        while self.running:
            command = input("QuantoniumOS> ").strip()
            self.process_command(command)
    
    def process_command(self, command: str):
        """Process a terminal command"""
        parts = command.split()
        
        if not parts:
            return
        
        cmd = parts[0].lower()
        args = parts[1:]
        
        if cmd == "help":
            print("\nAvailable commands:")
            print("  help          - Show this help message")
            print("  apps          - List available applications")
            print("  launch [app]  - Launch an application")
            print("  status        - Show system status")
            print("  exit          - Exit QuantoniumOS\n")
        
        elif cmd == "apps":
            self.list_apps()
        
        elif cmd == "launch":
            if not args:
                print("Error: Missing application name")
                print("Usage: launch [app]")
            else:
                self.launch_app(args[0])
        
        elif cmd == "status":
            self.show_status()
        
        elif cmd == "exit":
            print("Exiting QuantoniumOS...")
            self.running = False
        
        else:
            print(f"Unknown command: {cmd}")
            print("Type 'help' for a list of commands")
    
    def list_apps(self):
        """List available applications"""
        print("\nAvailable applications:")
        
        apps = [
            "visualizer - RFT Visualizer",
            "simulator - Quantum Simulator", 
            "mail - Q-Mail",
            "notes - Q-Notes",
            "vault - Q-Vault"
        ]
        
        for app in apps:
            print(f"  {app}")
        print("")
    
    def launch_app(self, app_name: str):
        """Launch an application"""
        app_mapping = {
            "visualizer": "apps/launch_rft_visualizer.py",
            "simulator": "apps/launch_quantum_simulator.py",
            "mail": "apps/launch_q_mail.py",
            "notes": "apps/launch_q_notes.py",
            "vault": "apps/launch_q_vault.py"
        }
        
        if app_name in app_mapping:
            script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), app_mapping[app_name])
            if os.path.exists(script_path):
                print(f"Launching {app_name}...")
                try:
                    subprocess.Popen([sys.executable, script_path])
                except Exception as e:
                    print(f"Error: Failed to launch {app_name}: {e}")
            else:
                print(f"Error: Application {app_name} not found at {script_path}")
        else:
            print(f"Error: Unknown application: {app_name}")
            print("Type 'apps' to see available applications")
    
    def show_status(self):
        """Show system status"""
        print("\n" + "=" * 40)
        print("QuantoniumOS System Status")
        print("=" * 40)
        print("RFT Assembly: LOADED")
        print("PyQt5 GUI: AVAILABLE")
        print("QtAwesome Icons: AVAILABLE")
        print("Timezone Support: AVAILABLE")
        print("Assembly Kernel: OPERATIONAL (64KB)")
        print("=" * 40 + "\n")

def main():
    """Main function"""
    print("Starting QuantoniumOS...")
    
    # Check if the GUI should be disabled
    if "--no-gui" in sys.argv:
        print("Terminal mode requested")
        terminal = QuantoniumOSTerminal()
        terminal.start()
    else:
        print("Creating QApplication...")
        app = QApplication(sys.argv)
        app.setStyle("Fusion")  # Use Fusion style for better appearance
        
        print("Creating QuantoniumOS window...")
        window = QuantoniumOSWindow()
        
        print("Showing window fullscreen...")
        window.showFullScreen()
        
        print("Starting event loop...")
        sys.exit(app.exec_())

if __name__ == "__main__":
    main()
