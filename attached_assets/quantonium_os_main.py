import sys
import os
import subprocess
from datetime import datetime
import pytz
import qtawesome as qta

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
            return ""  # Return empty string as fallback
    print(f"⚠️ Stylesheet not found: {qss_path}")
    return ""  # Return empty string as fallback

class AppIconLabel(QLabel):
    """Label that displays an icon for an app, launching the app on click."""
    def __init__(self, icon_name, app_name, script_path, position, stylesheet):
        super().__init__()
        self.icon_name = icon_name
        self.app_name = app_name
        self.script_path = script_path
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(64, 64)
        self.setMaximumSize(64, 64)
        self.setObjectName("AppIcon")
        self.setStyleSheet(stylesheet)  # Let the OS styling apply
        self._color = self.extract_color_from_stylesheet(stylesheet, "QLabel#AppIcon")
        self.update_icon()

    def extract_color_from_stylesheet(self, stylesheet, selector, property_name="color"):
        """Extract a color from the stylesheet for a given selector."""
        if not stylesheet:
            print(f"⚠️ Stylesheet is empty, using fallback color for {selector}")
            return QColor("white")  # Fallback color
        import re
        pattern = rf"{selector}\s*\{{[^}}]*{property_name}:\s*([^;}}]+)"
        match = re.search(pattern, stylesheet)
        if match:
            color_str = match.group(1).strip()
            try:
                return QColor(color_str)
            except Exception as e:
                print(f"⚠️ Error parsing {property_name} for {selector}: {e}")
                return QColor("white")  # Fallback color
        print(f"⚠️ Could not extract {property_name} for {selector}, using fallback color")
        return QColor("white")  # Fallback color

    def update_icon(self):
        """Update the icon using qtawesome with the extracted color."""
        try:
            icon = qta.icon(self.icon_name, color=self._color)
            self.setPixmap(icon.pixmap(64, 64))
            print(f"Icon {self.app_name} color set to: {self._color.name()}")
        except Exception as e:
            print(f"⚠️ Error updating icon for {self.app_name}: {e}")
            self.setText(self.app_name[0])  # Fallback to first letter if icon fails

    def setStyleSheet(self, stylesheet):
        """If the stylesheet changes, re-extract the color to update icon color too."""
        super().setStyleSheet(stylesheet)
        style = self.styleSheet()
        if "color:" in style:
            color_str = style.split("color:")[1].split(";")[0].strip()
            try:
                self._color = QColor(color_str)
                self.update_icon()
                print(f"Updated {self.app_name} icon color to: {self._color.name()} from stylesheet")
            except Exception as e:
                print(f"⚠️ Error parsing color from stylesheet for {self.app_name}: {e}")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.launch_app()
        super().mousePressEvent(event)

    def launch_app(self):
        """Launch the external script (if it exists) in a new process."""
        if os.path.exists(self.script_path):
            try:
                subprocess.Popen([sys.executable, self.script_path], start_new_session=True)
                print(f"✅ Launched: {self.script_path}")
            except Exception as e:
                print(f"⚠️ ERROR: Failed to launch {self.script_path}: {e}")
        else:
            print(f"⚠️ WARNING: Script not found: {self.script_path}")

class QuantoniumUI(QMainWindow):
    """Main OS UI with the 'Q' logo, side arch, app icons, clock, etc."""
    def __init__(self):
        super().__init__()
        self.setObjectName("QuantoniumMainWindow")
        self.setWindowTitle("Quantonium OS")

        # Set geometry based on primary screen
        screen = QApplication.primaryScreen().geometry()
        self.screen_width = screen.width()
        self.screen_height = screen.height()
        self.setGeometry(0, 0, self.screen_width, self.screen_height)

        # 1) Load stylesheet
        self.stylesheet = load_stylesheet(STYLES_QSS)
        if not self.stylesheet:
            print("⚠️ Stylesheet could not be loaded; proceeding with default styles.")
            self.stylesheet = "QLabel#AppIcon { color: white; }"  # Minimal fallback
        self.setStyleSheet(self.stylesheet)

        # 2) Setup scene & view
        self.scene = QGraphicsScene(0, 0, self.screen_width, self.screen_height)
        self.view = QGraphicsView(self.scene, self)
        self.view.setObjectName("DesktopView")
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setCentralWidget(self.view)

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

    # The following methods are identical to your original code, just condensed.

    def extract_color_from_stylesheet(self, selector, property_name="color"):
        # ... (same logic as above)
        import re
        if not self.stylesheet:
            return QColor("white")
        pattern = rf"{selector}\s*\{{[^}}]*{property_name}:\s*([^;}}]+)"
        match = re.search(pattern, self.stylesheet)
        if match:
            color_str = match.group(1).strip()
            try:
                return QColor(color_str)
            except:
                return QColor("white")
        return QColor("white")

    def extract_font_from_stylesheet(self, selector):
        # ... (parses "font: bold 16pt" etc.)
        import re
        if not self.stylesheet:
            return QFont("Arial", 12)
        pattern = rf"{selector}\s*\{{[^}}]*font:\s*([^;}}]+)"
        match = re.search(pattern, self.stylesheet)
        font = QFont("Arial", 12)
        if not match:
            return font
        font_str = match.group(1).strip()
        try:
            parts = font_str.split()
            if len(parts) >= 2:
                font.setFamily(" ".join(parts[:-2]))
                size = int(parts[-2].replace("pt", "")) if "pt" in parts[-2] else 10
                font.setPointSize(size)
                if "bold" in font_str.lower():
                    font.setBold(True)
        except:
            pass
        return font

    def extract_opacity_from_stylesheet(self, selector):
        import re
        if not self.stylesheet:
            return 0.2
        pattern = rf"{selector}\s*\{{[^}}]*opacity:\s*([^;}}]+)"
        match = re.search(pattern, self.stylesheet)
        if match:
            try:
                return float(match.group(1).strip())
            except:
                return 0.2
        return 0.2

    def extract_rotation_from_stylesheet(self, selector):
        import re
        if not self.stylesheet:
            return 0.0
        pattern = rf"{selector}\s*\{{[^}}]*rotation:\s*([^;}}]+)"
        match = re.search(pattern, self.stylesheet)
        if match:
            try:
                return float(match.group(1).strip())
            except:
                return 0.0
        return 0.0

    def extract_scale_factor_from_stylesheet(self, selector):
        import re
        if not self.stylesheet:
            return 0.8
        pattern = rf"{selector}\s*\{{[^}}]*scale-factor:\s*([^;}}]+)"
        match = re.search(pattern, self.stylesheet)
        if match:
            try:
                return float(match.group(1).strip())
            except:
                return 0.8
        return 0.8

    def add_q_logo(self):
        """Draw a stylized 'Q' in the center."""
        q_text = QGraphicsTextItem("Q")
        font = self.extract_font_from_stylesheet("QLabel#QLogo")
        scale_factor = self.extract_scale_factor_from_stylesheet("QLabel#QLogo")
        target_size = min(self.screen_width, self.screen_height) * scale_factor

        q_text.setFont(font)
        base_width = q_text.boundingRect().width()
        base_height = q_text.boundingRect().height()
        base_larger_dimension = max(base_width, base_height)
        if base_larger_dimension > 0:
            font_scale_factor = target_size / base_larger_dimension
        else:
            font_scale_factor = 1.0

        base_font_size = font.pointSize()
        new_font_size = int(base_font_size * font_scale_factor)
        font.setPointSize(new_font_size)
        q_text.setFont(font)

        color = self.extract_color_from_stylesheet("QLabel#QLogo")
        q_text.setDefaultTextColor(color)
        opacity = self.extract_opacity_from_stylesheet("QLabel#QLogo")
        q_text.setOpacity(opacity)
        rotation = self.extract_rotation_from_stylesheet("QLabel#QLogo")
        if rotation != 0:
            transform = QTransform()
            transform.rotate(rotation)
            q_text.setTransform(transform)

        text_width = q_text.boundingRect().width()
        text_height = q_text.boundingRect().height()
        q_text.setPos((self.screen_width - text_width) / 2, (self.screen_height - text_height) / 2)
        self.scene.addItem(q_text)
        return q_text

    def add_clock(self):
        """Adds a real-time clock in top-right corner."""
        clock_item = QGraphicsTextItem()
        font = self.extract_font_from_stylesheet("QLabel#ClockItem")
        clock_item.setFont(font)
        color = self.extract_color_from_stylesheet("QLabel#ClockItem")
        clock_item.setDefaultTextColor(color)
        clock_width = clock_item.boundingRect().width()
        clock_item.setPos(self.screen_width - clock_width - self.screen_width * 0.05,
                          self.screen_height * 0.01)
        self.scene.addItem(clock_item)
        return clock_item

    def add_shaded_arch(self):
        """The side arch that can expand/collapse, holding app icons."""
        tab_width = self.screen_width * 0.015
        tab_height = self.screen_height * 0.075
        tab_y = self.screen_height / 2 - tab_height / 2
        arch_rect = QRectF(-tab_width, tab_y, tab_width * 2, tab_height)
        arch = QGraphicsEllipseItem(arch_rect)
        color = self.extract_color_from_stylesheet("QGraphicsEllipseItem#ShadedArch")
        opacity = self.extract_opacity_from_stylesheet("QGraphicsEllipseItem#ShadedArch")
        color.setAlphaF(opacity)
        arch.setBrush(QBrush(color))
        arch.setPen(QPen(Qt.NoPen))
        arch.setAcceptHoverEvents(True)
        arch.setAcceptedMouseButtons(Qt.LeftButton)
        arch.hoverEnterEvent = lambda event: self.scene.update()
        arch.mousePressEvent = self.toggle_arch
        self.scene.addItem(arch)

        arrow_label = QLabel()
        arrow_label.setObjectName("ArrowIcon")
        arrow_icon = qta.icon("mdi.arrow-right", color=self.extract_color_from_stylesheet("QLabel#ArrowIcon"))
        arrow_size = int(tab_height * 0.4)
        arrow_label.setPixmap(arrow_icon.pixmap(arrow_size, arrow_size))
        arrow_label.setAlignment(Qt.AlignCenter)
        arrow_proxy = self.scene.addWidget(arrow_label)
        arrow_width = arrow_label.pixmap().width()
        arrow_height = arrow_label.pixmap().height()
        arrow_x = -tab_width + (tab_width * 2 - arrow_width) / 2
        arrow_y = tab_y + (tab_height - arrow_height) / 2
        arrow_proxy.setPos(arrow_x, arrow_y)
        arrow_label.mousePressEvent = self.toggle_arch

        return arch, arrow_proxy

    def toggle_arch(self, event):
        """Expand/collapse the arch with icons."""
        if event.button() != Qt.LeftButton:
            return
        self.is_arch_expanded = not self.is_arch_expanded
        if self.is_arch_expanded:
            arch_width = self.screen_width * 0.15
            arch_height = self.screen_height * 0.7
            center_y = (self.screen_height - arch_height) / 2
            self.arch.setRect(QRectF(-arch_width, center_y, arch_width * 2, arch_height))
            for proxy in self.app_proxies:
                proxy.setVisible(True)
            # Rotate arrow to point left
            transform = QTransform()
            transform.rotate(180)
            arrow_label = self.arrow_proxy.widget()
            arrow_size = int(arch_height * 0.4)
            arrow_pixmap = qta.icon("mdi.arrow-right",
                                    color=self.extract_color_from_stylesheet("QLabel#ArrowIcon")
                                   ).pixmap(arrow_size, arrow_size)
            arrow_label.setPixmap(arrow_pixmap.transformed(transform))
            arrow_width = arrow_label.pixmap().width()
            arrow_height = arrow_label.pixmap().height()
            arrow_x = -arch_width + (arch_width * 2 - arrow_width) / 2
            arrow_y = center_y + (arch_height - arrow_height) / 2
            self.arrow_proxy.setPos(arrow_x, arrow_y)
        else:
            tab_width = self.screen_width * 0.015
            tab_height = self.screen_height * 0.075
            tab_y = self.screen_height / 2 - tab_height / 2
            self.arch.setRect(QRectF(-tab_width, tab_y, tab_width * 2, tab_height))
            for proxy in self.app_proxies:
                proxy.setVisible(False)
            # Arrow back to normal
            transform = QTransform()
            arrow_label = self.arrow_proxy.widget()
            arrow_size = int(tab_height * 0.4)
            arrow_pixmap = qta.icon("mdi.arrow-right",
                                    color=self.extract_color_from_stylesheet("QLabel#ArrowIcon")
                                   ).pixmap(arrow_size, arrow_size)
            arrow_label.setPixmap(arrow_pixmap)
            arrow_width = arrow_label.pixmap().width()
            arrow_height = arrow_label.pixmap().height()
            arrow_x = -tab_width + (tab_width * 2 - arrow_width) / 2
            arrow_y = tab_y + (tab_height - arrow_height) / 2
            self.arrow_proxy.setPos(arrow_x, arrow_y)
        self.scene.update()

    def load_apps(self):
        """Loads app icons on the side arch, in a grid layout."""
        app_icons = {
            "File Explorer": "mdi.folder",
            "Settings": "mdi.cog",
            "Task Manager": "mdi.view-dashboard",
            "Q-Browser": "mdi.web",
            "Q-Mail": "mdi.email",
            "Wave Composer": "mdi.music",
            "Q-Vault": "mdi.lock",
            "Q-Notes": "mdi.note",
            "Q-Dock": "mdi.dock-window",
            "Wave Debugger": "mdi.wave"
        }

        apps = [
            {"name": "File Explorer", "script": "qshll_file_explorer.py"},
            {"name": "Settings", "script": "qshll_settings.py"},
            {"name": "Task Manager", "script": "qshll_task_manager.py"},
            {"name": "Q-Browser", "script": "q_browser.py"},
            {"name": "Q-Mail", "script": "q_mail.py"},
            {"name": "Wave Composer", "script": "q_wave_composer.py"},
            {"name": "Q-Vault", "script": "q_vault.py"},
            {"name": "Q-Notes", "script": "q_notes.py"},
            {"name": "Q-Dock", "script": "q_dock.py"},
            {"name": "Wave Debugger", "script": "q_wave_debugger.py"}
        ]

        columns = 2
        icon_size = 64
        spacing = self.screen_width * 0.02
        label_height = self.screen_height * 0.01
        start_x = self.screen_width * 0.01
        start_y = self.screen_height / 4

        for i, app in enumerate(apps):
            col = i % columns
            row = i // columns
            x = start_x + col * (icon_size + spacing)
            y = start_y + row * (icon_size + spacing + label_height)

            script_path = os.path.join(APP_DIR, app["script"])
            icon_name = app_icons[app["name"]]

            icon_label = AppIconLabel(
                icon_name=icon_name,
                app_name=app["name"],
                script_path=script_path,
                position=(x, y),
                stylesheet=self.stylesheet
            )
            proxy_icon = self.scene.addWidget(icon_label)
            proxy_icon.setPos(x, y)

            name_label = QLabel(app["name"])
            name_label.setObjectName("AppName")
            name_label.setAlignment(Qt.AlignCenter)
            proxy_name = self.scene.addWidget(name_label)
            name_width = name_label.sizeHint().width()
            name_x = x + (icon_size - name_width) / 2
            proxy_name.setPos(name_x, y + icon_size + self.screen_height * 0.005)

            self.app_proxies.append(proxy_icon)
            self.app_proxies.append(proxy_name)

        # By default, hide them behind the arch until toggled
        for proxy in self.app_proxies:
            proxy.setVisible(False)

    def update_time(self):
        """Updates clock display every second."""
        est = pytz.timezone("America/New_York")
        now = datetime.now(est)
        time_str = now.strftime("%I:%M %p\n%b %d")
        self.clock_text.setPlainText(time_str)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            # Toggle fullscreen vs normal
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 1) Load the global QSS
    stylesheet = load_stylesheet(STYLES_QSS)
    # 2) Apply it to the entire app
    app.setStyleSheet(stylesheet)

    # 3) Create and show main OS window
    window = QuantoniumUI()
    window.showFullScreen()
    sys.exit(app.exec_())
