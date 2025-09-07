#!/usr/bin/env python3
"""
QuantoniumOS - Modern Entry Point
=================================
Professional launcher with centralized path management and configuration.
Replaces quantonium_os_main.py with a clean, maintainable architecture.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

# Import utilities
try:
    from utils import initialize_quantonium, paths, config, imports
    from utils.imports import get_app
except ImportError as e:
    print(f"Error: Could not import QuantoniumOS utilities: {e}")
    print("Make sure you're running from the project root directory.")
    sys.exit(1)

# PyQt5 imports
try:
    from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout
    from PyQt5.QtWidgets import QPushButton, QLabel, QGraphicsView, QGraphicsScene
    from PyQt5.QtGui import QFont, QPixmap, QIcon
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal
    import qtawesome as qta
except ImportError as e:
    print(f"Error: PyQt5 not available: {e}")
    print("Install with: pip install PyQt5")
    sys.exit(1)

from datetime import datetime
import pytz


class AppLauncher(QWidget):
    """Widget for launching applications"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.app_registry = config.get_app_registry()
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the app launcher interface"""
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("QuantoniumOS Applications")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # App buttons
        apps_layout = QHBoxLayout()
        
        for app_name, app_config in self.app_registry.items():
            if app_config.enabled:
                btn = QPushButton(app_config.name)
                btn.setToolTip(app_config.description)
                btn.clicked.connect(lambda checked, name=app_name: self.launch_app(name))
                
                # Try to set icon
                icon_path = config.get_icon_path(app_config.icon)
                if icon_path.exists():
                    btn.setIcon(QIcon(str(icon_path)))
                    
                apps_layout.addWidget(btn)
                
        layout.addLayout(apps_layout)
        
    def launch_app(self, app_name: str):
        """Launch the specified application"""
        try:
            app_config = config.get_app_config(app_name)
            if not app_config:
                print(f"App '{app_name}' not found in registry")
                return
                
            print(f"Launching {app_config.name}...")
            
            # Import the app module
            module_name = app_config.module.split('.')[-1]  # Get last part for file name
            app_module = get_app(module_name)
            
            if app_module:
                # Get the app class
                app_class = getattr(app_module, app_config.class_name, None)
                if app_class:
                    # Create and show the app
                    self.current_app = app_class()
                    self.current_app.show()
                    print(f"✓ {app_config.name} launched successfully")
                else:
                    print(f"✗ Class '{app_config.class_name}' not found in {module_name}")
            else:
                print(f"✗ Could not import module '{module_name}'")
                
        except Exception as e:
            print(f"✗ Failed to launch {app_name}: {e}")


class SystemStatusWidget(QWidget):
    """Widget showing system status"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.setup_timer()
        
    def setup_ui(self):
        """Setup status display"""
        layout = QVBoxLayout(self)
        
        # Time display
        self.time_label = QLabel()
        self.time_label.setFont(QFont("Arial", 12))
        self.time_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.time_label)
        
        # Status display
        self.status_label = QLabel()
        self.status_label.setFont(QFont("Arial", 10))
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        self.update_display()
        
    def setup_timer(self):
        """Setup update timer"""
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_display)
        self.timer.start(1000)  # Update every second
        
    def update_display(self):
        """Update time and status display"""
        # Update time
        now = datetime.now(pytz.timezone('US/Eastern'))
        time_str = now.strftime("%H:%M:%S %Z")
        self.time_label.setText(time_str)
        
        # Update status
        kernel_status = "✓" if imports.import_kernel() else "✗"
        self.status_label.setText(f"Kernel: {kernel_status} | Apps: {len(config.get_enabled_apps())}")


class QuantoniumMainWindow(QMainWindow):
    """
    Main QuantoniumOS window with modern architecture
    """
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QuantoniumOS")
        self.setMinimumSize(800, 600)
        
        # Load theme
        self.load_theme()
        
        # Setup UI
        self.setup_ui()
        
        # Initialize system
        self.initialize_system()
        
    def load_theme(self):
        """Load application theme"""
        try:
            theme_path = config.get_theme_path()
            if theme_path.exists():
                with open(theme_path, 'r') as f:
                    stylesheet = f.read()
                self.setStyleSheet(stylesheet)
                print(f"✓ Theme loaded from {theme_path}")
            else:
                print(f"⚠ Theme file not found: {theme_path}")
        except Exception as e:
            print(f"⚠ Could not load theme: {e}")
            
    def setup_ui(self):
        """Setup the main user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        
        # Header with logo and status
        header_layout = QHBoxLayout()
        
        # Logo
        logo_label = QLabel("Q")
        logo_label.setFont(QFont("Arial", 24, QFont.Bold))
        logo_label.setAlignment(Qt.AlignCenter)
        logo_label.setStyleSheet("color: #00ff00; border: 2px solid #00ff00; border-radius: 25px; width: 50px; height: 50px;")
        logo_label.setFixedSize(50, 50)
        header_layout.addWidget(logo_label)
        
        # Spacer
        header_layout.addStretch()
        
        # Status widget
        self.status_widget = SystemStatusWidget()
        header_layout.addWidget(self.status_widget)
        
        layout.addLayout(header_layout)
        
        # App launcher
        self.app_launcher = AppLauncher()
        layout.addWidget(self.app_launcher)
        
        # Footer
        footer = QLabel("QuantoniumOS - Quantum Operating System")
        footer.setAlignment(Qt.AlignCenter)
        footer.setFont(QFont("Arial", 10))
        layout.addWidget(footer)
        
    def initialize_system(self):
        """Initialize QuantoniumOS system"""
        print("Initializing QuantoniumOS...")
        
        success = initialize_quantonium()
        if success:
            print("✓ QuantoniumOS initialized successfully")
            self.setWindowTitle("QuantoniumOS - Ready")
        else:
            print("⚠ QuantoniumOS initialization had issues")
            self.setWindowTitle("QuantoniumOS - Limited Mode")
            
        # Show system info
        print(f"Project root: {paths.project_root}")
        print(f"Available apps: {len(config.get_enabled_apps())}")
        
        # Validate configuration
        issues = config.validate_config()
        if issues:
            print("Configuration issues found:")
            for issue in issues:
                print(f"  ⚠ {issue}")


def main():
    """
    Main entry point for QuantoniumOS
    """
    print("=" * 50)
    print("QuantoniumOS - Quantum Operating System")
    print("=" * 50)
    
    # Create QApplication
    app = QApplication(sys.argv)
    app.setApplicationName("QuantoniumOS")
    app.setApplicationVersion("1.0.0")
    
    # Set application properties
    app.setQuitOnLastWindowClosed(True)
    
    try:
        # Create main window
        window = QuantoniumMainWindow()
        window.show()
        
        print("✓ QuantoniumOS started successfully")
        print("Close the window to exit.")
        
        # Run application
        sys.exit(app.exec_())
        
    except Exception as e:
        print(f"✗ QuantoniumOS startup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
