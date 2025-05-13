import sys
import os
import subprocess
from functools import partial
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QWidget, QVBoxLayout, QHBoxLayout, QLabel
)
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import QSize, Qt

# Set Directories
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
APPS_DIR = os.path.join(ROOT_DIR, "apps")
ICONS_DIR = os.path.join(ROOT_DIR, "icons")

class QuantumDesktop(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Quantonium OS")
        self.setGeometry(100, 100, 1280, 720)

        # Central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # Dock (Bottom bar for app icons)
        self.dock_widget = QWidget()
        self.dock_layout = QHBoxLayout(self.dock_widget)
        self.dock_layout.setAlignment(Qt.AlignCenter)
        self.dock_layout.setSpacing(15)

        # Define Apps
        self.apps = [
            {"name": "File Explorer", "script": "qshll_file_explorer.py", "icon": "qshll_file_explorer.svg"},
            {"name": "Settings", "script": "qshll_settings.py", "icon": "qshll_settings.svg"},
            {"name": "Task Manager", "script": "qshll_task_manager.py", "icon": "qshll_task_manager.svg"},
            {"name": "Q-Browser", "script": "q_browser.py", "icon": "q_browser.svg"},
            {"name": "Q-Mail", "script": "q_mail.py", "icon": "q_mail.svg"},
            {"name": "Q-Notes", "script": "q_notes.py", "icon": "q_notes.svg"},
            {"name": "Wave Composer", "script": "q_wave_composer.py", "icon": "q_wave_composer.svg"},
            {"name": "Q-Dock", "script": "q_dock.py", "icon": "q_dock.svg"},
            {"name": "Q-Vault", "script": "q_vault.py", "icon": "q_vault.svg"},
        ]

        # Add App Buttons to Dock
        for app in self.apps:
            btn = QPushButton()
            icon_path = os.path.join(ICONS_DIR, app["icon"])
            if os.path.exists(icon_path):
                btn.setIcon(QIcon(icon_path))
                btn.setIconSize(QSize(48, 48))
            else:
                btn.setText(app["name"])
            btn.setFixedSize(80, 80)
            btn.setStyleSheet("border-radius: 10px; background-color: #3b82f6; color: white;")
            btn.clicked.connect(partial(self.launch_app, app["script"]))
            self.dock_layout.addWidget(btn)

        self.main_layout.addStretch()
        self.main_layout.addWidget(self.dock_widget, alignment=Qt.AlignBottom | Qt.AlignHCenter)

        # Allow Escape (`Esc`) key to toggle fullscreen
        self.showFullScreen()

    def keyPressEvent(self, event):
        """Allow ESC to toggle fullscreen."""
        if event.key() == Qt.Key_Escape:
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()
        else:
            super().keyPressEvent(event)

    def launch_app(self, script_name):
        """Launch an app from the dock."""
        script_path = os.path.join(APPS_DIR, script_name)
        if os.path.exists(script_path):
            try:
                subprocess.Popen([sys.executable, script_path])
                print(f"✅ Launched: {script_path}")
            except Exception as e:
                print(f"❌ Error launching {script_path}: {e}")
        else:
            print(f"⚠️ Script not found: {script_path}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    desktop = QuantumDesktop()
    desktop.show()
    sys.exit(app.exec_())
