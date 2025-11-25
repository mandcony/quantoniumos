#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
# This file is listed in CLAIMS_PRACTICING_FILES.txt and is licensed
# under LICENSE-CLAIMS-NC.md (research/education only). Commercial
# rights require a separate patent license from the author.
"""
QuantoniumOS Desktop Environment
Golden Ratio UI with Q Logo Launcher
Wave-Space Computing Interface
"""

import sys
import os
import json
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFrame, QScrollArea,
                             QGridLayout, QMessageBox, QSystemTrayIcon, QMenu, QStatusBar)
from PyQt5.QtCore import Qt, QSize, QTimer, pyqtSignal, QPoint, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QIcon, QPalette, QColor, QFont, QPainter, QPen, QPixmap, QPainterPath

# Golden ratio constant
PHI = 1.618033988749

# Import middleware engine
try:
    # Add parent directory to path for imports
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from quantonium_os_src.engine.middleware_transform import MiddlewareTransformEngine
    MIDDLEWARE_AVAILABLE = True
except ImportError:
    MIDDLEWARE_AVAILABLE = False

class QLogo(QWidget):
    """Animated Q Logo - Central Launcher"""
    
    clicked = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(200, 200)
        self.rotation = 0
        self.hover = False
        self.setMouseTracking(True)
        
        # Animation timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.rotate)
        self.timer.start(50)  # 20 FPS
        
    def rotate(self):
        if self.hover:
            self.rotation = (self.rotation + 2) % 360
        else:
            self.rotation = (self.rotation + 0.5) % 360
        self.update()
        
    def enterEvent(self, event):
        self.hover = True
        
    def leaveEvent(self, event):
        self.hover = False
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
            
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Center point
        center = QPoint(100, 100)
        
        # Rotate around center
        painter.translate(center)
        painter.rotate(self.rotation)
        painter.translate(-center)
        
        # Draw outer circle (quantum field)
        pen = QPen(QColor(0, 170, 255) if not self.hover else QColor(0, 255, 170), 3)
        painter.setPen(pen)
        painter.drawEllipse(center, 80, 80)
        
        # Draw Q letter
        font = QFont('Arial', 72, QFont.Bold)
        painter.setFont(font)
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(50, 130, 'Q')
        
        # Draw resonance waves
        for i in range(3):
            radius = 80 - (i * 15)
            alpha = int(255 * (1 - i/3) * (0.3 + 0.7 * self.hover))
            pen = QPen(QColor(0, 200, 255, alpha), 2)
            painter.setPen(pen)
            painter.drawEllipse(center, radius, radius)


class AppLauncher(QPushButton):
    """Individual App Launcher Button"""
    
    def __init__(self, app_name, app_config, parent=None):
        super().__init__(parent)
        self.app_name = app_name
        self.app_config = app_config
        
        # Golden ratio sizing
        width = int(120 * PHI)
        height = 120
        self.setFixedSize(width, height)
        
        # Setup UI
        self.setText(app_config.get('name', app_name))
        self.setToolTip(app_config.get('description', ''))
        
        # Style
        self.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2a2a2a, stop:1 #1a1a1a);
                border: 2px solid #00aaff;
                border-radius: 10px;
                color: white;
                font-size: 11pt;
                font-weight: bold;
                padding: 10px;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3a3a3a, stop:1 #2a2a2a);
                border: 2px solid #00ffaa;
            }}
            QPushButton:pressed {{
                background: #00aaff;
            }}
        """)


class QuantoniumDesktop(QMainWindow):
    """Main QuantoniumOS Desktop Environment"""
    
    def __init__(self):
        super().__init__()
        self.apps = {}
        self.app_windows = {}
        self.load_apps()
        self.init_ui()
        
    def load_apps(self):
        """Load app registry"""
        registry_path = Path(__file__).parent.parent.parent / "data" / "config" / "app_registry.json"
        
        if registry_path.exists():
            with open(registry_path, 'r') as f:
                data = json.load(f)
                self.apps = data.get('apps', {})
        else:
            # Default apps if registry not found
            self.apps = {
                'quantum_simulator': {
                    'name': 'Quantum Simulator',
                    'description': 'Quantum circuit simulation',
                    'enabled': True
                },
                'quantum_crypto': {
                    'name': 'Quantum Crypto',
                    'description': 'QKD and quantum cryptography',
                    'enabled': True
                },
                'q_notes': {
                    'name': 'Q Notes',
                    'description': 'Note taking application',
                    'enabled': True
                },
                'q_vault': {
                    'name': 'Q Vault',
                    'description': 'Secure storage',
                    'enabled': True
                },
                'rft_validator': {
                    'name': 'RFT Validator',
                    'description': 'RFT mathematical validation',
                    'enabled': True
                },
                'rft_visualizer': {
                    'name': 'RFT Visualizer',
                    'description': 'RFT data visualization',
                    'enabled': True
                },
                'system_monitor': {
                    'name': 'System Monitor',
                    'description': 'System performance monitoring',
                    'enabled': True
                }
            }
    
    def init_ui(self):
        """Initialize the UI"""
        self.setWindowTitle('QuantoniumOS')
        self.setGeometry(100, 100, 1200, 800)
        
        # Set dark theme
        self.set_dark_theme()
        
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Top bar
        top_bar = self.create_top_bar()
        layout.addWidget(top_bar)
        
        # Main content area
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setAlignment(Qt.AlignCenter)
        
        # Q Logo in center
        self.q_logo = QLogo()
        self.q_logo.clicked.connect(self.toggle_app_grid)
        content_layout.addWidget(self.q_logo, alignment=Qt.AlignCenter)
        
        # Title below logo
        title = QLabel('QuantoniumOS')
        title.setStyleSheet('color: #00aaff; font-size: 24pt; font-weight: bold;')
        title.setAlignment(Qt.AlignCenter)
        content_layout.addWidget(title)
        
        subtitle = QLabel('Quantum-Inspired Operating System')
        subtitle.setStyleSheet('color: #888888; font-size: 12pt;')
        subtitle.setAlignment(Qt.AlignCenter)
        content_layout.addWidget(subtitle)
        
        content_layout.addSpacing(30)
        
        # App grid (hidden initially)
        self.app_grid_widget = self.create_app_grid()
        self.app_grid_widget.setVisible(False)
        content_layout.addWidget(self.app_grid_widget)
        
        layout.addWidget(content, stretch=1)
        
        # Status bar
        status = self.statusBar()
        status.setStyleSheet('background: #1a1a1a; color: #00aaff; padding: 5px;')
        
        # Show middleware status
        if MIDDLEWARE_AVAILABLE:
            engine = MiddlewareTransformEngine()
            variant_count = len(engine.list_all_variants())
            status.showMessage(f'QuantoniumOS v1.0 | {len(self.apps)} apps | {variant_count} wave transforms | Ready')
        else:
            status.showMessage(f'QuantoniumOS v1.0 | {len(self.apps)} apps loaded | Ready')
        
    def set_dark_theme(self):
        """Apply dark theme with quantum blue accents"""
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(20, 20, 20))
        palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
        palette.setColor(QPalette.Base, QColor(30, 30, 30))
        palette.setColor(QPalette.AlternateBase, QColor(40, 40, 40))
        palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
        palette.setColor(QPalette.ToolTipText, QColor(0, 0, 0))
        palette.setColor(QPalette.Text, QColor(255, 255, 255))
        palette.setColor(QPalette.Button, QColor(40, 40, 40))
        palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
        palette.setColor(QPalette.Highlight, QColor(0, 170, 255))
        palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
        self.setPalette(palette)
        
    def create_top_bar(self):
        """Create top menu bar"""
        bar = QFrame()
        bar.setFixedHeight(50)
        bar.setStyleSheet('background: #1a1a1a; border-bottom: 2px solid #00aaff;')
        
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(20, 5, 20, 5)
        
        # Logo/Title
        logo_label = QLabel('⚛ QuantoniumOS')
        logo_label.setStyleSheet('color: #00aaff; font-size: 16pt; font-weight: bold;')
        layout.addWidget(logo_label)
        
        layout.addStretch()
        
        # System buttons
        btn_minimize = QPushButton('_')
        btn_maximize = QPushButton('□')
        btn_close = QPushButton('✕')
        
        for btn in [btn_minimize, btn_maximize, btn_close]:
            btn.setFixedSize(40, 40)
            btn.setStyleSheet('''
                QPushButton {
                    background: transparent;
                    color: #00aaff;
                    border: none;
                    font-size: 16pt;
                }
                QPushButton:hover {
                    background: #2a2a2a;
                }
            ''')
        
        btn_minimize.clicked.connect(self.showMinimized)
        btn_maximize.clicked.connect(self.toggle_maximize)
        btn_close.clicked.connect(self.close)
        
        layout.addWidget(btn_minimize)
        layout.addWidget(btn_maximize)
        layout.addWidget(btn_close)
        
        return bar
        
    def create_app_grid(self):
        """Create application launcher grid"""
        container = QWidget()
        layout = QGridLayout(container)
        layout.setSpacing(20)
        layout.setContentsMargins(50, 20, 50, 20)
        
        # Create launchers for each app
        row, col = 0, 0
        cols = 4  # 4 apps per row
        
        for app_id, app_config in self.apps.items():
            if not app_config.get('enabled', True):
                continue
                
            launcher = AppLauncher(app_id, app_config)
            launcher.clicked.connect(lambda aid=app_id: self.launch_app(aid))
            
            layout.addWidget(launcher, row, col)
            
            col += 1
            if col >= cols:
                col = 0
                row += 1
        
        return container
        
    def toggle_app_grid(self):
        """Show/hide app grid"""
        self.app_grid_widget.setVisible(not self.app_grid_widget.isVisible())
        
    def toggle_maximize(self):
        """Toggle maximize/restore"""
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()
            
    def launch_app(self, app_id):
        """Launch an application"""
        if app_id in self.app_windows and self.app_windows[app_id].isVisible():
            self.app_windows[app_id].raise_()
            self.app_windows[app_id].activateWindow()
            return
            
        app_config = self.apps.get(app_id, {})
        
        try:
            # Dynamic import of app
            module_path = f"quantonium_os_src.apps.{app_id}"
            module = __import__(module_path, fromlist=[''])
            
            # Get main class
            class_name = app_config.get('class_name', 'MainWindow')
            if hasattr(module, class_name):
                app_class = getattr(module, class_name)
                window = app_class()
                window.show()
                self.app_windows[app_id] = window
            else:
                QMessageBox.information(self, 'Coming Soon', 
                    f"{app_config.get('name', app_id)} is being developed!")
                
        except ImportError as e:
            # Show placeholder for now
            QMessageBox.information(self, 'Coming Soon', 
                f"{app_config.get('name', app_id)} is being developed!\n\n{app_config.get('description', '')}")


def main():
    app = QApplication(sys.argv)
    app.setApplicationName('QuantoniumOS')
    app.setOrganizationName('QuantoniumOS')
    
    desktop = QuantoniumDesktop()
    desktop.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
