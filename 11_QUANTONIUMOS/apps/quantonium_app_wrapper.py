#!/usr/bin/env python3
"""
QuantoniumOS App Wrapper with Unified Design System
Provides consistent styling for all QuantoniumOS applications
"""

import os
import sys

# Add paths for design system and apps
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
apps_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, parent_dir)
sys.path.insert(0, apps_dir)

try:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget

    from 11_QUANTONIUMOS.quantonium_design_system import (apply_unified_style,
                                          get_app_geometry, get_design_system)

    class QuantoniumAppWrapper(QMainWindow):
        """Wrapper class that applies unified design to any QuantoniumOS app"""

        def __init__(self, app_widget, app_title="QuantoniumOS Application"):
            super().__init__()
            self.setWindowTitle(f"{app_title} - QuantoniumOS")

            # Apply unified design system
            apply_unified_style(self)

            # Set geometry using design system
            geometry = get_app_geometry()
            self.setGeometry(
                int(geometry["x"]),
                int(geometry["y"]),
                int(geometry["width"]),
                int(geometry["height"]),
            )

            # Apply styling to the app widget if it exists
            if app_widget:
                apply_unified_style(app_widget)
                self.setCentralWidget(app_widget)

            # Optional: Add status bar with unified styling
            self.statusBar().showMessage(f"Ready - {app_title}")

    def launch_app(app_class, app_name, app_title):
        """Launch any QuantoniumOS app with unified design"""
        try:
            app = QApplication(sys.argv)

            # Create the app instance
            if hasattr(app_class, "__call__"):
                app_widget = app_class()
            else:
                app_widget = app_class

            # Check if the app widget has a show method (standalone) or needs wrapping
            if hasattr(app_widget, "show") and isinstance(app_widget, QMainWindow):
                # App is already a main window, just apply styling
                apply_unified_style(app_widget)
                app_widget.setWindowTitle(f"{app_title} - QuantoniumOS")

                # Set geometry using design system
                geometry = get_app_geometry()
                app_widget.setGeometry(
                    int(geometry["x"]),
                    int(geometry["y"]),
                    int(geometry["width"]),
                    int(geometry["height"]),
                )

                app_widget.show()
            else:
                # Wrap the widget in our unified window
                main_window = QuantoniumAppWrapper(app_widget, app_title)
                main_window.show()

            print(f"✅ Launched {app_title} with unified design")
            return app.exec_()

        except Exception as e:
            print(f"❌ Error launching {app_name}: {e}")
            return 1

except ImportError as e:
    print(f"❌ Design system import error: {e}")

    # Fallback launcher without design system
    def launch_app(app_class, app_name, app_title):
        """Fallback launcher without unified design"""
        try:
            from PyQt5.QtWidgets import QApplication

            app = QApplication(sys.argv)
            app_widget = app_class()

            if hasattr(app_widget, "show"):
                app_widget.show()
                return app.exec_()
            else:
                print(f"❌ {app_name} cannot be displayed")
                return 1

        except Exception as e:
            print(f"❌ Error launching {app_name}: {e}")
            return 1
