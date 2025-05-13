import sys
import os
import subprocess
from PyQt5.QtWidgets import (
    QApplication, QFrame, QHBoxLayout, QPushButton, QWidget
)
from PyQt5.QtGui import QIcon, QCloseEvent
from PyQt5.QtCore import Qt, QSize, QRect

# Define directories
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))  # C:\quantonium_os\apps
BASE_DIR = os.path.dirname(ROOT_DIR)  # Go up one level to C:\quantonium_os
APPS_DIR = os.path.join(BASE_DIR, "apps")  # Correct path: C:\quantonium_os\apps
ICONS_DIR = os.path.join(BASE_DIR, "icons")  # Correct path: C:\quantonium_os\icons
STYLES_QSS = os.path.join(BASE_DIR, "styles.qss")  # Correct path: C:\quantonium_os\styles.qss

def load_stylesheet(qss_path):
    """Load the stylesheet from the given path, with fallback if not found."""
    if os.path.exists(qss_path):
        try:
            with open(qss_path, "r", encoding="utf-8") as f:
                print(f"✅ Stylesheet loaded from {qss_path}")
                return f.read()
        except UnicodeDecodeError as e:
            print(f"⚠️ Error decoding stylesheet from {qss_path}: {e}")
            print(f"Position: {e.start}, Character: {e.object[e.start]}")
            return ""
    print(f"⚠️ Stylesheet not found: {qss_path}")
    return ""

class QDock(QFrame):
    def __init__(self):
        super().__init__()
        self.setObjectName("QDockPanel")
        self.setWindowTitle("QDock")
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)  # Always on top, no title bar

        # Load stylesheet
        self.stylesheet = load_stylesheet(STYLES_QSS)
        if not self.stylesheet:
            raise ValueError("Stylesheet could not be loaded; cannot proceed without styles.")
        self.setStyleSheet(self.stylesheet)

        # Set up layout
        self.layout = QHBoxLayout(self)
        self.layout.setAlignment(Qt.AlignCenter)
        self.layout.setSpacing(self.extract_dimension_from_stylesheet("QFrame#QDockPanel", "spacing", 15))

        # Define the apps for this dock (only Browser, Email, Notes)
        self.apps = [
            {"name": "Q-Browser", "script": "q_browser.py"},
            {"name": "Q-Mail", "script": "q_mail.py"},
            {"name": "Q-Notes", "script": "q_notes.py"},
        ]

        # Add app buttons
        for app in self.apps:
            btn = QPushButton()
            btn.setObjectName("DockButton")
            icon_path = os.path.join(ICONS_DIR, f"{app['name'].lower().replace(' ', '_')}.svg")
            if os.path.exists(icon_path):
                btn.setIcon(QIcon(icon_path))
                icon_size = self.extract_dimension_from_stylesheet("QPushButton#DockButton", "icon-size", 48)
                btn.setIconSize(QSize(icon_size, icon_size))
            else:
                btn.setText(app["name"])
            
            btn_width = self.extract_dimension_from_stylesheet("QPushButton#DockButton", "fixed-width", 80)
            btn_height = self.extract_dimension_from_stylesheet("QPushButton#DockButton", "fixed-height", 80)
            btn.setFixedSize(btn_width, btn_height)

            btn.setStyleSheet(self.stylesheet)
            btn.clicked.connect(lambda _, s=app["script"]: self.launch_app(s))
            self.layout.addWidget(btn)

        # Add close button with a distinct object name
        close_btn = QPushButton("X")
        close_btn.setObjectName("CloseButton")  # Changed from DockButton to CloseButton
        close_btn.setFixedSize(btn_width, btn_height)
        close_btn.setStyleSheet(self.stylesheet)
        close_btn.clicked.connect(self.close)
        self.layout.addWidget(close_btn)

        # Position the dock at the bottom-right corner
        self.resize(self.sizeHint())  # Initial size based on layout
        self.position_dock()

    def extract_dimension_from_stylesheet(self, selector, property_name, default_value):
        """Extract a dimension (e.g., icon-size, fixed-width) from the stylesheet."""
        if not self.styleSheet():
            print(f"⚠️ Stylesheet is empty, using default {property_name}: {default_value}")
            return default_value
        import re
        pattern = rf"{selector}\s*\{{[^}}]*{property_name}:\s*([^;}}]+)"
        match = re.search(pattern, self.styleSheet())
        if match:
            dimension_str = match.group(1).strip()
            try:
                dimension_str = dimension_str.replace("px", "")
                return int(dimension_str)
            except Exception as e:
                print(f"⚠️ Error parsing {property_name} for {selector}: {e}")
        print(f"⚠️ {property_name} not found for {selector}, using default: {default_value}")
        return default_value

    def launch_app(self, script_name):
        """Launch the specified app script."""
        script_path = os.path.join(APPS_DIR, script_name)
        if os.path.exists(script_path):
            subprocess.Popen([sys.executable, script_path])
            print(f"✅ Launched: {script_path}")
        else:
            print(f"⚠️ WARNING: Script not found: {script_path}")

    def position_dock(self):
        """Position the dock at the bottom-right corner of the screen."""
        screen = QApplication.primaryScreen().geometry()
        dock_width = self.width()
        dock_height = self.extract_dimension_from_stylesheet("QFrame#QDockPanel", "fixed-height", 100)
        x = screen.width() - dock_width
        y = screen.height() - dock_height
        self.setGeometry(x, y, dock_width, dock_height)

    def closeEvent(self, event: QCloseEvent):
        """Handle the close event to hide the window instead of quitting the app."""
        event.ignore()  # Prevent the app from closing
        self.hide()    # Hide the dock instead

    def showEvent(self, event):
        """Reposition the dock when shown."""
        super().showEvent(event)
        self.position_dock()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    stylesheet = load_stylesheet(STYLES_QSS)
    app.setStyleSheet(stylesheet)
    dock = QDock()
    dock.show()
    sys.exit(app.exec_())