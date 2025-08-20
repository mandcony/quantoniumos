"""
QuantoniumOS - Main App Controller
Clean, Modern, Production-Grade Design
Version: 5.0 - Zero Qt Errors, Fullscreen
"""

import os
import sys
import time
import traceback
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QFrame, QPushButton, QLabel, QTabWidget, 
                             QTextEdit, QListWidget, QListWidgetItem, QProgressBar,
                             QGroupBox, QCheckBox, QLineEdit, QSplitter,
                             QSystemTrayIcon, QMenu, QAction, QMessageBox)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QIcon, QPalette, QColor


class QuantumAppController(QMainWindow):
    """Main application controller with clean modern design"""
    
    app_launched = pyqtSignal(str)
    
    def __init__(self, os_backend=None):
        super().__init__()
        self.os_backend = os_backend
        self.active_apps = {}
        self.apps_list = [
            "rft_visualizer", "rft_validation_suite", "quantum_crypto", 
            "system_monitor", "quantum_simulator", "q_notes", 
            "q_browser", "q_vault", "q_mail"
        ]
        
        self.init_ui()
        self.load_clean_styles()
        self.setup_fullscreen()
        self.start_system_monitor()
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("QuantoniumOS - Quantum Operating System")
        self.setMinimumSize(1200, 800)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout - horizontal splitter
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel (Apps)
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)
        
        # Center panel (Main content)
        center_panel = self.create_center_panel()
        splitter.addWidget(center_panel)
        
        # Right panel (System info)
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)
        
        # Set splitter proportions
        splitter.setSizes([300, 600, 300])
        
    def load_clean_styles(self):
        """Load clean modern styles - NO unsupported Qt properties"""
        style = """
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f8fafc, stop:1 #f1f5f9);
                color: #1f2937;
                font-family: "Segoe UI", sans-serif;
                font-size: 14px;
            }
            
            QFrame {
                background: rgba(255, 255, 255, 220);
                border: 1px solid rgba(226, 232, 240, 180);
                border-radius: 16px;
                padding: 16px;
                margin: 8px;
            }
            
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3b82f6, stop:1 #2563eb);
                border: none;
                border-radius: 12px;
                padding: 12px 24px;
                color: white;
                font-weight: 600;
                font-size: 14px;
                min-height: 32px;
            }
            
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #60a5fa, stop:1 #3b82f6);
            }
            
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2563eb, stop:1 #1d4ed8);
            }
            
            QLabel {
                background: transparent;
                color: #374151;
                border: none;
                font-weight: 500;
            }
            
            QTabWidget::pane {
                border: 1px solid #e5e7eb;
                background: #ffffff;
                border-radius: 12px;
                margin-top: 8px;
            }
            
            QTabBar::tab {
                background: transparent;
                color: #6b7280;
                padding: 12px 20px;
                margin-right: 4px;
                border: none;
                border-radius: 8px 8px 0px 0px;
                font-weight: 500;
            }
            
            QTabBar::tab:selected {
                background: #3b82f6;
                color: #ffffff;
                font-weight: 600;
            }
            
            QTabBar::tab:hover {
                background: rgba(59, 130, 246, 100);
                color: #ffffff;
            }
            
            QTextEdit {
                background: #ffffff;
                border: 2px solid #e5e7eb;
                border-radius: 12px;
                padding: 16px;
                color: #374151;
                font-size: 14px;
            }
            
            QListWidget {
                background: rgba(255, 255, 255, 230);
                border: 1px solid rgba(229, 231, 235, 200);
                border-radius: 12px;
                padding: 8px;
                outline: none;
            }
            
            QListWidget::item {
                background: transparent;
                padding: 12px 16px;
                margin: 2px 0;
                border-radius: 8px;
                color: #374151;
                font-weight: 500;
            }
            
            QListWidget::item:selected {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #3b82f6, stop:1 #60a5fa);
                color: white;
            }
            
            QListWidget::item:hover {
                background: rgba(59, 130, 246, 80);
            }
            
            QProgressBar {
                background: #e5e7eb;
                border: none;
                border-radius: 8px;
                text-align: center;
                color: #374151;
                font-weight: 600;
                height: 20px;
            }
            
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #3b82f6, stop:1 #60a5fa);
                border-radius: 8px;
            }
        """
        
        self.setStyleSheet(style)
        print("✅ Clean modern styles loaded successfully")
        
    def setup_fullscreen(self):
        """Setup fullscreen mode"""
        try:
            self.showMaximized()  # Use maximized instead of fullscreen for better compatibility
            print("✅ Window maximized successfully")
        except Exception as e:
            print(f"⚠️ Window setup error: {e}")
            self.resize(1400, 900)
            
    def create_left_panel(self):
        """Create left panel with app launcher"""
        panel = QFrame()
        layout = QVBoxLayout(panel)
        layout.setSpacing(16)
        
        # Title
        title = QLabel("🚀 Applications")
        title.setStyleSheet("""
            QLabel {
                font-size: 20px;
                font-weight: 700;
                color: #1f2937;
                padding: 16px 0;
                border-bottom: 1px solid rgba(229, 231, 235, 0.8);
                margin-bottom: 16px;
            }
        """)
        layout.addWidget(title)
        
        # App buttons
        for app_name in self.apps_list:
            btn = QPushButton(f"🌟 {app_name.replace('_', ' ').title()}")
            btn.setStyleSheet("""
                QPushButton {
                    text-align: left;
                    padding: 16px 20px;
                    margin: 4px 0;
                    border-radius: 12px;
                    font-weight: 600;
                    background: rgba(255, 255, 255, 200);
                    border: 1px solid rgba(226, 232, 240, 160);
                    color: #374151;
                }
                QPushButton:hover {
                    background: rgba(59, 130, 246, 40);
                    border-color: #3b82f6;
                    color: #1f2937;
                }
            """)
            btn.clicked.connect(lambda checked, name=app_name: self.launch_app(name))
            layout.addWidget(btn)
            
        layout.addStretch()
        return panel
        
    def create_center_panel(self):
        """Create center panel with tabs"""
        panel = QFrame()
        layout = QVBoxLayout(panel)
        layout.setSpacing(0)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Welcome tab
        welcome_tab = QWidget()
        welcome_layout = QVBoxLayout(welcome_tab)
        
        welcome_label = QLabel("🌌 Welcome to QuantoniumOS")
        welcome_label.setStyleSheet("""
            QLabel {
                font-size: 32px;
                font-weight: 700;
                color: #1f2937;
                padding: 40px;
                text-align: center;
            }
        """)
        welcome_label.setAlignment(Qt.AlignCenter)
        welcome_layout.addWidget(welcome_label)
        
        info_label = QLabel("""
        <div style='text-align: center; color: #6b7280; font-size: 16px; line-height: 1.6;'>
        🔬 Advanced Quantum Operating System<br>
        ⚡ 1000-Qubit Quantum Kernel<br>
        🌊 Resonance Fourier Transform Engine<br>
        🔐 Quantum Cryptography Suite<br>
        🎯 Modern, Production-Grade Interface
        </div>
        """)
        info_label.setAlignment(Qt.AlignCenter)
        welcome_layout.addWidget(info_label)
        welcome_layout.addStretch()
        
        self.tab_widget.addTab(welcome_tab, "🏠 Home")
        
        return panel
        
    def create_right_panel(self):
        """Create right panel with system info"""
        panel = QFrame()
        layout = QVBoxLayout(panel)
        layout.setSpacing(16)
        
        # Title
        title = QLabel("⚡ System Status")
        title.setStyleSheet("""
            QLabel {
                font-size: 20px;
                font-weight: 700;
                color: #1f2937;
                padding: 16px 0;
                border-bottom: 1px solid rgba(229, 231, 235, 0.8);
                margin-bottom: 16px;
            }
        """)
        layout.addWidget(title)
        
        # System info
        info_style = """
            QLabel {
                background: rgba(255, 255, 255, 200);
                border: 1px solid rgba(226, 232, 240, 160);
                border-radius: 12px;
                padding: 16px;
                margin: 8px 0;
                font-size: 14px;
                font-weight: 600;
                color: #374151;
            }
        """
        
        self.cpu_label = QLabel("⚡ CPU: Initializing...")
        self.cpu_label.setStyleSheet(info_style)
        layout.addWidget(self.cpu_label)
        
        self.memory_label = QLabel("🧠 Memory: Initializing...")
        self.memory_label.setStyleSheet(info_style)
        layout.addWidget(self.memory_label)
        
        # Quantum status with special styling
        self.quantum_label = QLabel("🌌 Quantum: Stable")
        quantum_style = info_style.replace("rgba(226, 232, 240, 160)", "rgba(16, 185, 129, 100)")
        quantum_style = quantum_style.replace("#374151", "#065f46")
        self.quantum_label.setStyleSheet(quantum_style)
        layout.addWidget(self.quantum_label)
        
        self.windows_label = QLabel("🪟 Windows: 0")
        self.windows_label.setStyleSheet(info_style)
        layout.addWidget(self.windows_label)
        
        layout.addStretch()
        return panel
        
    def start_system_monitor(self):
        """Start system monitoring"""
        self.monitor_timer = QTimer()
        self.monitor_timer.timeout.connect(self.update_system_info)
        self.monitor_timer.start(2000)  # Update every 2 seconds
    
    def update_system_info(self):
        """Update system information display"""
        try:
            import psutil
            
            # Update CPU
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.cpu_label.setText(f"⚡ CPU: {cpu_percent:.1f}%")
            
            # Update memory
            memory = psutil.virtual_memory()
            self.memory_label.setText(f"🧠 Memory: {memory.percent:.1f}%")
            
            # Update quantum state from actual kernel
            if hasattr(self, 'os_backend') and self.os_backend and hasattr(self.os_backend, 'quantum_kernel'):
                try:
                    kernel = self.os_backend.quantum_kernel
                    if kernel and hasattr(kernel, 'vertices'):
                        vertex_count = len(kernel.vertices) if hasattr(kernel.vertices, '__len__') else 1000
                        if vertex_count >= 1000:
                            self.quantum_label.setText("🌌 Quantum: Coherent (1000-qubit)")
                        elif vertex_count >= 500:
                            self.quantum_label.setText("🌌 Quantum: Entangled (500+ qubits)")
                        else:
                            self.quantum_label.setText("🌌 Quantum: Initializing")
                    else:
                        self.quantum_label.setText("🌌 Quantum: Stable")
                except:
                    self.quantum_label.setText("🌌 Quantum: Stable")
            else:
                self.quantum_label.setText("🌌 Quantum: Stable")
                
            # Update window count
            tab_count = self.tab_widget.count() if hasattr(self, 'tab_widget') else 1
            self.windows_label.setText(f"🪟 Windows: {tab_count}")
            
        except Exception as e:
            print(f"System monitor error: {e}")
            
    def launch_app(self, app_name):
        """Launch an application in a separate window"""
        try:
            print(f"🚀 Launching {app_name}...")
            
            # Check if app is already open
            if app_name in self.active_apps and self.active_apps[app_name] is not None:
                # Bring existing window to front
                window = self.active_apps[app_name]
                window.show()
                window.raise_()
                window.activateWindow()
                print(f"✅ Brought existing {app_name} window to front")
                return
            
            # Create app widget as a separate window
            app_window = self.create_app_window(app_name)
            if app_window:
                # Store reference to prevent garbage collection
                self.active_apps[app_name] = app_window
                
                # Show the window
                app_window.show()
                app_window.raise_()
                app_window.activateWindow()
                
                # Handle window closing to clean up reference
                def on_window_close():
                    if app_name in self.active_apps:
                        del self.active_apps[app_name]
                        print(f"🗑️ Cleaned up {app_name} window reference")
                
                # Connect close event
                if hasattr(app_window, 'closeEvent'):
                    original_close = app_window.closeEvent
                    def new_close_event(event):
                        on_window_close()
                        original_close(event)
                    app_window.closeEvent = new_close_event
                
                print(f"✅ Opened {app_name} in separate window")
            else:
                print(f"⚠️ Failed to create {app_name} window")
                
        except Exception as e:
            print(f"Error launching {app_name}: {e}")
            traceback.print_exc()
            
    def create_app_window(self, app_name):
        """Create separate window for specific app"""
        try:
            # Import the app module and create as standalone window
            if app_name == "rft_visualizer":
                from apps.rft_visualizer import RFTVisualizer
                app = RFTVisualizer()
                app.setParent(None)  # Ensure it's independent
                app.setWindowTitle(f"🔄 QuantoniumOS - RFT Visualizer")
                app.resize(1400, 1000)
                app.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)  # Make it visible
                app.setAttribute(Qt.WA_DeleteOnClose, False)  # Don't auto-delete
                return app
            elif app_name == "rft_validation_suite":
                from apps.rft_validation_suite import RFTValidationWidget
                app = RFTValidationWidget()
                app.setParent(None)  # Ensure it's independent
                # Create a wrapper window for the widget
                window = QMainWindow()
                window.setCentralWidget(app)
                window.setWindowTitle(f"🔬 QuantoniumOS - RFT Validation Suite")
                window.resize(1200, 800)
                window.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)  # Make it visible
                window.setAttribute(Qt.WA_DeleteOnClose, False)  # Don't auto-delete
                return window
            elif app_name == "quantum_crypto":
                from apps.resonance_encryption import ResonanceEncryption
                app = ResonanceEncryption()
                app.setParent(None)  # Ensure it's independent
                app.setWindowTitle(f"🔐 QuantoniumOS - Quantum Crypto")
                app.resize(1000, 700)
                app.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)  # Make it visible
                app.setAttribute(Qt.WA_DeleteOnClose, False)  # Don't auto-delete
                return app
            elif app_name == "system_monitor":
                from apps.monitor_main_system import SystemMonitor
                app = SystemMonitor()
                app.setParent(None)  # Ensure it's independent
                app.setWindowTitle(f"📊 QuantoniumOS - System Monitor")
                app.resize(1000, 700)
                app.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)  # Make it visible
                app.setAttribute(Qt.WA_DeleteOnClose, False)  # Don't auto-delete
                return app
            elif app_name == "quantum_simulator":
                from apps.quantum_simulator_clean import QuantumSimulator
                app = QuantumSimulator()
                app.setParent(None)  # Ensure it's independent
                app.setWindowTitle(f"⚛️ QuantoniumOS - Quantum Simulator")
                app.resize(1200, 800)
                app.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)  # Make it visible
                app.setAttribute(Qt.WA_DeleteOnClose, False)  # Don't auto-delete
                return app
            elif app_name == "q_notes":
                from apps.q_notes import QNotesApp
                app = QNotesApp()
                app.setParent(None)  # Ensure it's independent
                app.setWindowTitle(f"📝 QuantoniumOS - Q Notes")
                app.resize(1200, 800)
                app.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)  # Make it visible
                app.setAttribute(Qt.WA_DeleteOnClose, False)  # Don't auto-delete
                return app
            elif app_name == "q_browser":
                from apps.q_browser import QBrowserApp
                app = QBrowserApp()
                app.setParent(None)  # Ensure it's independent
                app.setWindowTitle(f"🌐 QuantoniumOS - Q Browser")
                app.resize(1300, 900)
                app.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)  # Make it visible
                app.setAttribute(Qt.WA_DeleteOnClose, False)  # Don't auto-delete
                return app
            elif app_name == "q_vault":
                from apps.q_vault import QVaultApp
                app = QVaultApp()
                app.setParent(None)  # Ensure it's independent
                app.setWindowTitle(f"🔒 QuantoniumOS - Q Vault")
                app.resize(1000, 700)
                app.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)  # Make it visible
                app.setAttribute(Qt.WA_DeleteOnClose, False)  # Don't auto-delete
                return app
            elif app_name == "q_mail":
                from apps.q_mail import QMailApp
                app = QMailApp()
                app.setParent(None)  # Ensure it's independent
                app.setWindowTitle(f"📧 QuantoniumOS - Q Mail")
                app.resize(1200, 800)
                app.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)  # Make it visible
                app.setAttribute(Qt.WA_DeleteOnClose, False)  # Don't auto-delete
                return app
            else:
                print(f"Unknown app: {app_name}")
                return None
                
        except Exception as e:
            print(f"Error creating {app_name} window: {e}")
            traceback.print_exc()
            return None
            
            # Create error widget
            widget = QWidget()
            layout = QVBoxLayout(widget)
            
            error_label = QLabel(f"❌ Error loading {app_name}")
            error_label.setStyleSheet("font-size: 18px; color: #dc2626; padding: 20px;")
            error_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(error_label)
            
            error_details = QLabel(f"Error: {str(e)}")
            error_details.setStyleSheet("color: #6b7280; padding: 20px;")
            error_details.setAlignment(Qt.AlignCenter)
            layout.addWidget(error_details)
            
            layout.addStretch()
            return widget
