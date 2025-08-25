"""
QuantoniumOS Window Manager
Advanced window management with VS Code integration and quantum animations
"""

import ctypes
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import win32api
import win32con
import win32gui
from PyQt5.QtCore import (QEasingCurve, QObject, QPoint, QPropertyAnimation,
                          QRect, QTimer, pyqtSignal)
from PyQt5.QtWidgets import QApplication, QGraphicsOpacityEffect, QWidget


class QuantumWindowManager(QObject):
    """Advanced window management with VS Code integration"""

    # Signals
    window_created = pyqtSignal(str, dict)
    window_destroyed = pyqtSignal(str)
    window_focused = pyqtSignal(str)
    window_minimized = pyqtSignal(str)
    window_restored = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.windows: Dict[str, QWidget] = {}
        self.window_states: Dict[str, dict] = {}
        self.vs_code_handle = None
        self.monitor_timer = QTimer()
        self.monitor_timer.timeout.connect(self._monitor_windows)
        self.monitor_timer.start(100)  # Monitor every 100ms

        # Window animations
        self.animation_enabled = True
        self.animation_duration = 250  # ms

        # Load quantum stylesheet
        self.load_quantum_styles()

        # Initialize VS Code detection
        self._detect_vscode()

    def load_quantum_styles(self):
        """Load the quantum master stylesheet"""
        try:
            style_path = (
                Path(__file__).parent.parent / "styles" / "quantonium_master.qss"
            )
            if style_path.exists():
                with open(style_path, "r") as f:
                    self.quantum_stylesheet = f.read()
            else:
                self.quantum_stylesheet = ""
                print(f"Warning: Could not find stylesheet at {style_path}")
        except Exception as e:
            print(f"Error loading quantum stylesheet: {e}")
            self.quantum_stylesheet = ""

    def _detect_vscode(self):
        """Detect if VS Code is running and get handle"""

        def enum_windows_callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):
                window_text = win32gui.GetWindowText(hwnd)
                if "Visual Studio Code" in window_text:
                    windows.append(hwnd)
            return True

        try:
            vscode_windows = []
            win32gui.EnumWindows(enum_windows_callback, vscode_windows)
            if vscode_windows:
                self.vs_code_handle = vscode_windows[0]
                print(f"✅ VS Code detected: {hex(self.vs_code_handle)}")
            else:
                print("⚠️ VS Code not detected")
        except Exception as e:
            print(f"Error detecting VS Code: {e}")

    def create_quantum_window(
        self,
        app_name: str,
        widget: QWidget,
        position: Optional[Tuple[int, int]] = None,
        size: Optional[Tuple[int, int]] = None,
        flags: Optional[int] = None,
    ) -> str:
        """Create a new quantum window with proper styling"""

        window_id = f"quantum_{app_name}_{id(widget)}"

        # Apply quantum styling
        if self.quantum_stylesheet:
            widget.setStyleSheet(self.quantum_stylesheet)

        # Set window properties
        if flags:
            widget.setWindowFlags(flags)
        else:
            widget.setWindowFlags(
                widget.windowFlags()
                | widget.Qt.Window
                | widget.Qt.CustomizeWindowHint
                | widget.Qt.WindowTitleHint
                | widget.Qt.WindowCloseButtonHint
                | widget.Qt.WindowMinimizeButtonHint
                | widget.Qt.WindowMaximizeButtonHint
            )

        # Set quantum window title
        widget.setWindowTitle(f"🌌 QuantoniumOS - {app_name}")

        # Set position and size
        if position:
            widget.move(position[0], position[1])
        else:
            # Smart positioning relative to VS Code if available
            if self.vs_code_handle:
                try:
                    rect = win32gui.GetWindowRect(self.vs_code_handle)
                    vs_x, vs_y, vs_right, vs_bottom = rect
                    widget.move(vs_right + 20, vs_y + len(self.windows) * 40)
                except:
                    widget.move(
                        100 + len(self.windows) * 30, 100 + len(self.windows) * 30
                    )
            else:
                widget.move(100 + len(self.windows) * 30, 100 + len(self.windows) * 30)

        if size:
            widget.resize(size[0], size[1])
        else:
            widget.resize(800, 600)

        # Add quantum glass effect
        self._apply_quantum_effects(widget)

        # Store window reference
        self.windows[window_id] = widget
        self.window_states[window_id] = {
            "app_name": app_name,
            "visible": False,
            "minimized": False,
            "maximized": False,
            "position": (widget.x(), widget.y()),
            "size": (widget.width(), widget.height()),
            "z_order": len(self.windows),
        }

        # Connect window signals
        widget.closeEvent = lambda event: self._on_window_close(window_id, event)

        # Emit signal
        self.window_created.emit(window_id, self.window_states[window_id])

        return window_id

    def _apply_quantum_effects(self, widget: QWidget):
        """Apply quantum visual effects to widget"""
        try:
            # Set window opacity for glass effect
            widget.setWindowOpacity(0.95)

            # Add quantum glow effect through stylesheet classes
            current_style = widget.styleSheet()
            widget.setStyleSheet(
                current_style
                + "\nQWidget { border: 1px solid rgba(0, 255, 204, 0.3); }"
            )

        except Exception as e:
            print(f"Error applying quantum effects: {e}")

    def show_window(self, window_id: str, animated: bool = True):
        """Show window with optional animation"""
        if window_id not in self.windows:
            return

        widget = self.windows[window_id]

        if animated and self.animation_enabled:
            self._animate_show(widget)
        else:
            widget.show()

        self.window_states[window_id]["visible"] = True
        widget.raise_()
        widget.activateWindow()

    def _animate_show(self, widget: QWidget):
        """Animate window appearance"""
        try:
            # Fade in animation
            widget.setWindowOpacity(0.0)
            widget.show()

            self.fade_animation = QPropertyAnimation(widget, b"windowOpacity")
            self.fade_animation.setDuration(self.animation_duration)
            self.fade_animation.setStartValue(0.0)
            self.fade_animation.setEndValue(0.95)
            self.fade_animation.setEasingCurve(QEasingCurve.OutCubic)
            self.fade_animation.start()
        except Exception as e:
            print(f"Error in show animation: {e}")
            widget.show()

    def minimize_window(self, window_id: str):
        """Minimize window"""
        if window_id in self.windows:
            self.windows[window_id].showMinimized()
            self.window_states[window_id]["minimized"] = True
            self.window_minimized.emit(window_id)

    def maximize_window(self, window_id: str):
        """Maximize window"""
        if window_id in self.windows:
            self.windows[window_id].showMaximized()
            self.window_states[window_id]["maximized"] = True

    def restore_window(self, window_id: str):
        """Restore window from minimized/maximized state"""
        if window_id in self.windows:
            self.windows[window_id].showNormal()
            self.window_states[window_id]["minimized"] = False
            self.window_states[window_id]["maximized"] = False
            self.window_restored.emit(window_id)

    def close_window(self, window_id: str, animated: bool = True):
        """Close window with optional animation"""
        if window_id not in self.windows:
            return

        widget = self.windows[window_id]

        if animated and self.animation_enabled:
            self._animate_close(widget, window_id)
        else:
            self._finish_close(widget, window_id)

    def _animate_close(self, widget: QWidget, window_id: str):
        """Animate window closing"""
        try:
            self.close_animation = QPropertyAnimation(widget, b"windowOpacity")
            self.close_animation.setDuration(self.animation_duration)
            self.close_animation.setStartValue(0.95)
            self.close_animation.setEndValue(0.0)
            self.close_animation.setEasingCurve(QEasingCurve.InCubic)
            self.close_animation.finished.connect(
                lambda: self._finish_close(widget, window_id)
            )
            self.close_animation.start()
        except Exception as e:
            print(f"Error in close animation: {e}")
            self._finish_close(widget, window_id)

    def _finish_close(self, widget: QWidget, window_id: str):
        """Complete the close operation after animation"""
        try:
            widget.close()
            if window_id in self.windows:
                del self.windows[window_id]
                del self.window_states[window_id]
            self.window_destroyed.emit(window_id)
        except Exception as e:
            print(f"Error finishing close: {e}")

    def _on_window_close(self, window_id: str, event):
        """Handle window close event"""
        self.close_window(window_id, animated=True)
        event.accept()

    def arrange_cascade(self):
        """Arrange windows in cascade pattern"""
        offset = 30
        x, y = 50, 50

        for window_id in sorted(
            self.windows.keys(), key=lambda k: self.window_states[k]["z_order"]
        ):
            widget = self.windows[window_id]
            widget.move(x, y)
            x += offset
            y += offset
            if x > 300 or y > 300:
                x, y = 50, 50

    def arrange_tile_horizontal(self):
        """Tile windows horizontally"""
        if not self.windows:
            return

        screen = QApplication.desktop().screenGeometry()
        window_count = len(self.windows)
        width = screen.width() // window_count

        x = 0
        for window_id in self.windows:
            widget = self.windows[window_id]
            widget.setGeometry(x, 0, width, screen.height() - 100)
            x += width

    def arrange_tile_vertical(self):
        """Tile windows vertically"""
        if not self.windows:
            return

        screen = QApplication.desktop().screenGeometry()
        window_count = len(self.windows)
        height = (screen.height() - 100) // window_count

        y = 0
        for window_id in self.windows:
            widget = self.windows[window_id]
            widget.setGeometry(0, y, screen.width(), height)
            y += height

    def _monitor_windows(self):
        """Monitor window states and handle focus changes"""
        for window_id, widget in list(self.windows.items()):
            try:
                if widget.isActiveWindow() and not self.window_states[window_id].get(
                    "focused"
                ):
                    self.window_states[window_id]["focused"] = True
                    self.window_focused.emit(window_id)
                    # Update z-order
                    for wid in self.window_states:
                        if wid != window_id:
                            self.window_states[wid]["focused"] = False
            except RuntimeError:
                # Widget was deleted
                if window_id in self.windows:
                    del self.windows[window_id]
                if window_id in self.window_states:
                    del self.window_states[window_id]

    def integrate_with_vscode(self):
        """Integrate with VS Code window if detected"""
        if self.vs_code_handle:
            try:
                # Get VS Code window position
                rect = win32gui.GetWindowRect(self.vs_code_handle)
                vs_x, vs_y, vs_right, vs_bottom = rect

                # Position quantum windows relative to VS Code
                offset_x = vs_right + 20
                offset_y = vs_y

                for i, window_id in enumerate(self.windows):
                    widget = self.windows[window_id]
                    widget.move(offset_x, offset_y + (i * 40))

                print("✅ Windows positioned relative to VS Code")
            except Exception as e:
                print(f"Error integrating with VS Code: {e}")

    def save_session(self, filepath: str):
        """Save current window session"""
        try:
            session_data = {"windows": {}, "arrangement": "custom", "version": "2.0"}

            for window_id, state in self.window_states.items():
                if window_id in self.windows:
                    widget = self.windows[window_id]
                    session_data["windows"][window_id] = {
                        "app_name": state["app_name"],
                        "position": (widget.x(), widget.y()),
                        "size": (widget.width(), widget.height()),
                        "visible": state["visible"],
                        "minimized": state["minimized"],
                        "maximized": state["maximized"],
                    }

            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, "w") as f:
                json.dump(session_data, f, indent=2)

            print(f"✅ Session saved to {filepath}")
            return True
        except Exception as e:
            print(f"Error saving session: {e}")
            return False

    def load_session(self, filepath: str):
        """Load a saved window session"""
        try:
            with open(filepath, "r") as f:
                session_data = json.load(f)

            print(f"✅ Session loaded from {filepath}")
            return session_data
        except Exception as e:
            print(f"Error loading session: {e}")
            return {}


# Global window manager instance
window_manager = QuantumWindowManager()
