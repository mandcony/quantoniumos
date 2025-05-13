import sys
import os
import subprocess
from functools import partial
from datetime import datetime
import pytz  # Ensure this module is installed: pip install pytz

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QGridLayout, QHBoxLayout, QPushButton, QLabel
)
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt, QSize, QTimer

# Define directories for apps and icons
APP_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "apps")
ICONS_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "icons")

# Path to your external QSS file
STYLES_QSS = r"C:\quantonium_os\styles.qss"

def load_stylesheet(qss_path):
    """Loads and returns the contents of the QSS file."""
    if os.path.exists(qss_path):
        with open(qss_path, "r") as f:
            return f.read()
    else:
        print(f"⚠️ QSS file not found: {qss_path}")
        return ""

class MainDesktop(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Quantonium OS")
        self.setGeometry(100, 100, 1280, 720)

        # Set up the central widget (its appearance will be controlled by QSS)
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        # Main layout: vertical layout containing the desktop area and the taskbar
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # Desktop area: Grid layout for icons
        self.desktop_area = QWidget()
        self.desktop_layout = QGridLayout(self.desktop_area)
        self.desktop_layout.setContentsMargins(30, 30, 30, 30)
        self.desktop_layout.setSpacing(15)
        self.desktop_area.setLayout(self.desktop_layout)
        self.load_desktop_icons()

        # Taskbar: Horizontal layout at the bottom
        self.taskbar = QWidget()
        self.taskbar.setFixedHeight(60)
        self.taskbar_layout = QHBoxLayout(self.taskbar)
        self.taskbar_layout.setContentsMargins(10, 0, 10, 0)
        self.taskbar_layout.setSpacing(10)
        self.taskbar_layout.addStretch()  # Push clock to the right

        # Clock label on the taskbar (minimal inline styling; colors are from QSS)
        self.clock_label = QLabel("")
        self.clock_label.setStyleSheet("font-size: 18pt; font-weight: bold;")
        self.taskbar_layout.addWidget(self.clock_label)

        # Timer to update the clock every second
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_clock)
        self.timer.start(1000)
        self.update_clock()

        # Add the desktop area and taskbar to the main layout
        self.main_layout.addWidget(self.desktop_area)
        self.main_layout.addWidget(self.taskbar)

    def load_desktop_icons(self):
        """Loads available apps as desktop icons arranged in a grid."""
        self.apps = [
            {"name": "File Explorer", "script": "qshll_file_explorer.py", "icon": "qshll_file_explorer.svg"},
            {"name": "Settings",      "script": "qshll_settings.py",       "icon": "qshll_settings.svg"},
            {"name": "Task Manager",  "script": "qshll_task_manager.py",    "icon": "qshll_task_manager.svg"},
            {"name": "Q-Browser",     "script": "q_browser.py",             "icon": "q_browser.svg"},
            {"name": "Q-Mail",        "script": "q_mail.py",                "icon": "q_mail.svg"},
            {"name": "Q-Notes",       "script": "q_notes.py",               "icon": "q_notes.svg"},
            {"name": "Wave Composer", "script": "q_wave_composer.py",       "icon": "q_wave_composer.svg"},
            {"name": "Q-Dock",        "script": "q_dock.py",                "icon": "q_dock.svg"},
            {"name": "Q-Vault",       "script": "q_vault.py",               "icon": "q_vault.svg"},
        ]

        cols = 3
        row = 0
        col = 0
        for app in self.apps:
            icon_widget = QWidget()
            vbox = QVBoxLayout(icon_widget)
            vbox.setContentsMargins(0, 0, 0, 0)
            vbox.setSpacing(3)

            btn = QPushButton()
            btn.setFixedSize(128, 128)
            icon_path = os.path.join(ICONS_DIR, app["icon"])
            if os.path.exists(icon_path):
                btn.setIcon(QIcon(icon_path))
                btn.setIconSize(QSize(128, 128))
            else:
                btn.setText(app["name"])
            btn.clicked.connect(partial(self.launch_app, app["script"]))

            label = QLabel(app["name"])
            label.setAlignment(Qt.AlignCenter)

            vbox.addWidget(btn, alignment=Qt.AlignCenter)
            vbox.addWidget(label, alignment=Qt.AlignCenter)
            self.desktop_layout.addWidget(icon_widget, row, col)
            col += 1
            if col >= cols:
                col = 0
                row += 1

    def launch_app(self, script_name):
        """Launches the specified application script using subprocess."""
        script_path = os.path.join(APP_DIR, script_name)
        if os.path.exists(script_path):
            try:
                print(f"Launching {script_path}...")
                subprocess.Popen([sys.executable, script_path], start_new_session=True)
            except Exception as e:
                print(f"❌ ERROR: Could not launch {script_path}. Reason: {e}")
        else:
            print(f"⚠️ WARNING: Script not found: {script_path}")

    def update_clock(self):
        """Updates the clock label to show the current Eastern Standard Time (EST)."""
        est = pytz.timezone("America/New_York")
        now = datetime.now(est)
        time_str = now.strftime("%I:%M %p")
        self.clock_label.setText(time_str)

    def keyPressEvent(self, event):
        """Allows toggling fullscreen with the Escape key."""
        if event.key() == Qt.Key_Escape:
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()
        else:
            super().keyPressEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Load and apply the external QSS stylesheet for unified styling
    stylesheet = load_stylesheet(STYLES_QSS)
    if stylesheet:
        app.setStyleSheet(stylesheet)
    
    desktop = MainDesktop()
    desktop.show()
    sys.exit(app.exec_())
