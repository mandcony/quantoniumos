"""
QuantoniumOS Application Controller
Main controller for all Quantum applications with integrated window management
"""

import sys
import os
import importlib
from pathlib import Path
from typing import Dict, Optional, List

try:
    from PyQt5.QtWidgets import (QMainWindow, QApplication, QWidget, 
                                QVBoxLayout, QHBoxLayout, QPushButton,
                                QTabWidget, QDockWidget, QToolBar,
                                QAction, QMenu, QMenuBar, QStatusBar,
                                QLabel, QSplitter, QFrame)
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
    from PyQt5.QtGui import QIcon, QKeySequence, QPixmap
    PYQT5_AVAILABLE = True
except ImportError:
    print("⚠️ PyQt5 not available - using fallback mode")
    PYQT5_AVAILABLE = False

# Add QuantoniumOS paths
sys.path.append(str(Path(__file__).parent.parent.parent))

if PYQT5_AVAILABLE:
    try:
        from window_manager import window_manager
    except ImportError:
        print("⚠️ Window manager not available")
        window_manager = None

class QuantumAppController(QMainWindow if PYQT5_AVAILABLE else object):
    """Main controller for all Quantum applications"""
    
    def __init__(self):
        if not PYQT5_AVAILABLE:
            print("❌ PyQt5 required for QuantumOS frontend")
            return
            
        super().__init__()
        self.apps: Dict[str, object] = {}
        self.app_threads: Dict[str, QThread] = {}
        self.init_ui()
        self.load_apps()
        
    def init_ui(self):
        """Initialize the main UI"""
        self.setWindowTitle("🌌 QuantoniumOS - Quantum Operating System")
        
        # START FULLSCREEN - 2025 MODERN AESTHETIC
        self.showMaximized()  # Start in fullscreen mode
        
        # Apply quantum master stylesheet
        self.load_quantum_styles()
        
        # Central widget with splitter
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)  # Modern minimal margins
        
        # Create splitter
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel - Quick Launch
        self.create_left_panel(splitter)
        
        # Center - Tab system
        self.central_tabs = QTabWidget()
        self.central_tabs.setTabsClosable(True)
        self.central_tabs.tabCloseRequested.connect(self.close_tab)
        splitter.addWidget(self.central_tabs)
        
        # Right panel - System info
        self.create_right_panel(splitter)
        
        # Set splitter proportions
        splitter.setSizes([300, 1400, 300])  # Better proportions for fullscreen
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create tool bar
        self.create_tool_bar()
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("🚀 QuantoniumOS v3.0 - Modern Quantum Interface Ready")
        
        # Start system monitor
        self.start_system_monitor()
    
    def load_quantum_styles(self):
        """Load clean modern QuantoniumOS design - zero Qt errors"""
        try:
            # Clean, modern QuantoniumOS design system - NO unsupported properties
            modern_sleek_style = """
                QMainWindow {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #f8fafc, stop:1 #f1f5f9);
                    color: #1f2937;
                    font-family: "Segoe UI", "SF Pro Display", sans-serif;
                    font-size: 14px;
                }
                
                QFrame {
                    background: rgba(255, 255, 255, 220);
                    border: 1px solid rgba(226, 232, 240, 180);
                    border-radius: 16px;
                    padding: 20px;
                    margin: 8px;
                }
                
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #3b82f6, stop:1 #2563eb);
                    border: none;
                    border-radius: 12px;
                    padding: 14px 24px;
                    color: white;
                    font-weight: 600;
                    font-size: 14px;
                    min-height: 36px;
                }
                
                QPushButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #60a5fa, stop:1 #3b82f6);
                }
                
                QPushButton:pressed {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #2563eb, stop:1 #1d4ed8);
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
                
                QLabel {
                    background: transparent;
                    color: #374151;
                    border: none;
                    font-weight: 500;
                }
                
                QLineEdit {
                    background: #ffffff;
                    border: 2px solid #e5e7eb;
                    border-radius: 8px;
                    padding: 12px 16px;
                    color: #374151;
                    font-size: 14px;
                }
                
                QLineEdit:focus {
                    border-color: #3b82f6;
                }
                
                QTextEdit {
                    background: #ffffff;
                    border: 2px solid #e5e7eb;
                    border-radius: 12px;
                    padding: 16px;
                    color: #374151;
                    font-size: 14px;
                }
                
                QTextEdit:focus {
                    border-color: #3b82f6;
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
                
                QGroupBox {
                    background: rgba(255, 255, 255, 200);
                    border: 2px solid #e5e7eb;
                    border-radius: 12px;
                    padding: 16px;
                    margin-top: 20px;
                    font-weight: 600;
                    color: #374151;
                }
                
                QProgressBar {
                    background: #e5e7eb;
                    border: none;
                    border-radius: 8px;
                    text-align: center;
                    color: #374151;
                    font-weight: 600;
                    height: 24px;
                }
                
                QProgressBar::chunk {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #3b82f6, stop:1 #60a5fa);
                    border-radius: 8px;
                }
            """
            
            self.setStyleSheet(modern_sleek_style)
            print("✅ Clean modern QuantoniumOS design loaded successfully")
            
        except Exception as e:
            print(f"Error loading stylesheet: {e}")
            self.setStyleSheet("QWidget { font-family: 'Segoe UI'; }")

    def setup_fullscreen_ui(self):
        """Setup fullscreen UI with proper window management"""
        try:
            # Make window truly fullscreen
                }
                
                QFrame {
                    background: rgba(255, 255, 255, 220);
                    border: 1px solid rgba(226, 232, 240, 180);
                    border-radius: 16px;
                    padding: 20px;
                    margin: 8px;
                }
                
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #3b82f6, stop:1 #2563eb);
                    border: none;
                    border-radius: 12px;
                    padding: 14px 24px;
                    color: white;
                    font-weight: 600;
                    font-size: 14px;
                    min-height: 36px;
                }
                
                QPushButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #60a5fa, stop:1 #3b82f6);
                }
                
                QPushButton:pressed {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #2563eb, stop:1 #1d4ed8);
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
                
                QLabel {
                    background: transparent;
                    color: #374151;
                    border: none;
                    font-weight: 500;
                }
                
                QLineEdit {
                    background: #ffffff;
                    border: 2px solid #e5e7eb;
                    border-radius: 8px;
                    padding: 12px 16px;
                    color: #374151;
                    font-size: 14px;
                }
                
                QLineEdit:focus {
                    border-color: #3b82f6;
                }
                
                QTextEdit {
                    background: #ffffff;
                    border: 2px solid #e5e7eb;
                    border-radius: 12px;
                    padding: 16px;
                    color: #374151;
                    font-size: 14px;
                }
                
                QTextEdit:focus {
                    border-color: #3b82f6;
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
                
                QGroupBox {
                    background: rgba(255, 255, 255, 200);
                    border: 2px solid #e5e7eb;
                    border-radius: 12px;
                    padding: 16px;
                    margin-top: 20px;
                    font-weight: 600;
                    color: #374151;
                }
                
                QProgressBar {
                    background: #e5e7eb;
                    border: none;
                    border-radius: 8px;
                    text-align: center;
                    color: #374151;
                    font-weight: 600;
                    height: 24px;
                }
                
                QProgressBar::chunk {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #3b82f6, stop:1 #60a5fa);
                    border-radius: 8px;
                }
            """
            
            self.setStyleSheet(modern_sleek_style)
            print("✅ Clean modern QuantoniumOS design loaded successfully")
            
        except Exception as e:
            print(f"Error loading stylesheet: {e}")
            self.setStyleSheet("QWidget { font-family: 'Segoe UI'; }")
                
                /* Modern Typography */
                QLabel {
                    color: #374151;
                    font-weight: 500;
                    background: transparent;
                }
                
                /* Elegant Tab System */
                QTabWidget::pane {
                    background: rgba(255, 255, 255, 0.9);
                    border: 1px solid rgba(229, 231, 235, 0.8);
                    border-radius: 16px;
                    margin-top: 12px;
                }
                
                QTabBar::tab {
                    background: transparent;
                    color: #6b7280;
                    padding: 12px 24px;
                    margin-right: 8px;
                    border: none;
                    border-radius: 12px 12px 0px 0px;
                    font-weight: 500;
                    min-width: 100px;
                }
                
                QTabBar::tab:selected {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #4f46e5, stop:1 #3730a3);
                    color: white;
                    font-weight: 600;
                }
                
                QTabBar::tab:hover:!selected {
                    background: rgba(75, 70, 229, 0.1);
                    color: #374151;
                }
                
                /* Clean Menu Bar */
                QMenuBar {
                    background: rgba(255, 255, 255, 0.95);
                    color: #374151;
                    border: none;
                    padding: 8px 16px;
                    font-weight: 500;
                }
                
                QMenuBar::item {
                    background: transparent;
                    padding: 8px 16px;
                    border-radius: 8px;
                    font-weight: 500;
                }
                
                QMenuBar::item:selected {
                    background: rgba(75, 70, 229, 0.15);
                    color: #4f46e5;
                }
                
                /* Elegant Dropdown Menus */
                QMenu {
                    background: rgba(255, 255, 255, 0.95);
                    color: #374151;
                    border: 1px solid rgba(229, 231, 235, 0.8);
                    border-radius: 12px;
                    padding: 8px;
                    backdrop-filter: blur(20px);
                }
                
                QMenu::item {
                    background: transparent;
                    padding: 10px 16px;
                    border-radius: 8px;
                    font-weight: 500;
                }
                
                QMenu::item:selected {
                    background: rgba(75, 70, 229, 0.15);
                    color: #4f46e5;
                }
                
                QMenu::separator {
                    height: 1px;
                    background: rgba(229, 231, 235, 0.8);
                    margin: 8px 16px;
                }
                
                /* Modern Status Bar */
                QStatusBar {
                    background: rgba(255, 255, 255, 0.95);
                    color: #6b7280;
                    border: none;
                    padding: 8px 16px;
                    font-size: 13px;
                    font-weight: 500;
                }
                
                QStatusBar::item {
                    border: none;
                }
                
                /* Clean Toolbar */
                QToolBar {
                    background: rgba(255, 255, 255, 0.95);
                    border: none;
                    spacing: 12px;
                    padding: 12px 16px;
                    border-radius: 12px;
                }
                
                QToolBar::handle {
                    background: rgba(229, 231, 235, 0.8);
                    width: 3px;
                    border-radius: 2px;
                    margin: 4px;
                }
                
                QToolBar QToolButton {
                    background: transparent;
                    border: none;
                    border-radius: 8px;
                    padding: 10px;
                    color: #6b7280;
                    font-weight: 500;
                }
                
                QToolBar QToolButton:hover {
                    background: rgba(75, 70, 229, 0.15);
                    color: #4f46e5;
                }
                
                QToolBar QToolButton:pressed {
                    background: rgba(75, 70, 229, 0.25);
                    color: #3730a3;
                }
                
                /* Modern Splitter */
                QSplitter::handle {
                    background: rgba(229, 231, 235, 0.8);
                    width: 2px;
                    height: 2px;
                    border-radius: 1px;
                }
                
                QSplitter::handle:hover {
                    background: #4f46e5;
                }
                
                /* Clean Text Inputs */
                QLineEdit {
                    background: rgba(255, 255, 255, 0.9);
                    border: 2px solid rgba(229, 231, 235, 0.8);
                    border-radius: 12px;
                    padding: 12px 16px;
                    color: #374151;
                    font-size: 14px;
                    font-weight: 500;
                }
                
                QLineEdit:focus {
                    border: 2px solid #4f46e5;
                    background: white;
                }
                
                QTextEdit {
                    background: rgba(255, 255, 255, 0.9);
                    border: 2px solid rgba(229, 231, 235, 0.8);
                    border-radius: 12px;
                    padding: 16px;
                    color: #374151;
                    font-size: 14px;
                    font-weight: 500;
                    line-height: 1.6;
                }
                
                QTextEdit:focus {
                    border: 2px solid #4f46e5;
                    background: white;
                }
                
                /* Modern Progress Bar */
                QProgressBar {
                    background: rgba(229, 231, 235, 0.8);
                    border: none;
                    border-radius: 8px;
                    text-align: center;
                    color: #374151;
                    font-weight: 600;
                    height: 24px;
                }
                
                QProgressBar::chunk {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #4f46e5, stop:1 #6366f1);
                    border-radius: 6px;
                }
                
                /* Elegant Scrollbars */
                QScrollBar:vertical {
                    background: rgba(229, 231, 235, 0.3);
                    width: 12px;
                    border-radius: 6px;
                    border: none;
                }
                
                QScrollBar::handle:vertical {
                    background: rgba(156, 163, 175, 0.6);
                    border-radius: 6px;
                    min-height: 40px;
                }
                
                QScrollBar::handle:vertical:hover {
                    background: rgba(107, 114, 128, 0.8);
                }
                
                QScrollBar:horizontal {
                    background: rgba(229, 231, 235, 0.3);
                    height: 12px;
                    border-radius: 6px;
                    border: none;
                }
                
                QScrollBar::handle:horizontal {
                    background: rgba(156, 163, 175, 0.6);
                    border-radius: 6px;
                    min-width: 40px;
                }
                
                QScrollBar::handle:horizontal:hover {
                    background: rgba(107, 114, 128, 0.8);
                }
                
                QScrollBar::add-line, QScrollBar::sub-line {
                    border: none;
                    background: none;
                }
                
                /* Modern List Widgets */
                QListWidget {
                    background: rgba(255, 255, 255, 0.9);
                    border: 1px solid rgba(229, 231, 235, 0.8);
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
                        stop:0 #4f46e5, stop:1 #6366f1);
                    color: white;
                }
                
                QListWidget::item:hover {
                    background: rgba(75, 70, 229, 0.1);
                }
                
                /* Modern Checkboxes */
                QCheckBox {
                    color: #374151;
                    spacing: 12px;
                    font-weight: 500;
                }
                
                QCheckBox::indicator {
                    width: 20px;
                    height: 20px;
                    border: 2px solid rgba(156, 163, 175, 0.8);
                    border-radius: 6px;
                    background: rgba(255, 255, 255, 0.9);
                }
                
                QCheckBox::indicator:checked {
                    background: #4f46e5;
                    border-color: #4f46e5;
                }
                
                QCheckBox::indicator:hover {
                    border-color: #4f46e5;
                }
                
                /* Modern Radio Buttons */
                QRadioButton {
                    color: #374151;
                    spacing: 12px;
                    font-weight: 500;
                }
                
                QRadioButton::indicator {
                    width: 20px;
                    height: 20px;
                    border: 2px solid rgba(156, 163, 175, 0.8);
                    border-radius: 10px;
                    background: rgba(255, 255, 255, 0.9);
                }
                
                QRadioButton::indicator:checked {
                    background: radial-gradient(circle, #4f46e5 30%, rgba(255, 255, 255, 0.9) 30%);
                    border-color: #4f46e5;
                }
                
                QRadioButton::indicator:hover {
                    border-color: #4f46e5;
                }
                
                /* Modern Sliders */
                QSlider::groove:horizontal {
                    border: none;
                    height: 6px;
                    background: rgba(229, 231, 235, 0.8);
                    border-radius: 3px;
                }
                
                QSlider::handle:horizontal {
                    background: #4f46e5;
                    border: none;
                    width: 20px;
                    height: 20px;
                    border-radius: 10px;
                    margin: -7px 0;
                }
                
                QSlider::handle:horizontal:hover {
                    background: #6366f1;
                }
                
                /* Modern Dock Widgets */
                QDockWidget {
                    color: #374151;
                    border: 1px solid rgba(229, 231, 235, 0.8);
                    border-radius: 12px;
                }
                
                QDockWidget::title {
                    background: rgba(255, 255, 255, 0.95);
                    padding: 16px;
                    border-top-left-radius: 12px;
                    border-top-right-radius: 12px;
                    font-weight: 600;
                    font-size: 15px;
                    color: #374151;
                }
            """
            
            self.setStyleSheet(modern_sleek_style)
            print("✅ Modern sleek QuantoniumOS design loaded successfully")
            
        except Exception as e:
            print(f"Error loading stylesheet: {e}")
            # Fallback to no styling if there's an error
            self.setStyleSheet("")
    
    def create_left_panel(self, parent):
        """Create left panel with quick launch - Modern sleek design"""
        left_frame = QFrame()
        left_frame.setFrameStyle(QFrame.StyledPanel)
        left_layout = QVBoxLayout(left_frame)
        left_layout.setSpacing(20)  # More generous spacing
        
        # Title - Modern typography
        title = QLabel("🚀 Applications")
        title.setStyleSheet("""
            QLabel {
                font-size: 22px;
                font-weight: 700;
                color: #1f2937;
                padding: 20px 0;
                border-bottom: 1px solid rgba(229, 231, 235, 0.8);
                margin-bottom: 16px;
                background: transparent;
            }
        """)
        left_layout.addWidget(title)
        
        # Application buttons - Modern card-like design
        apps = [
            ("🔬 RFT Engine", "rft_visualizer"),
            ("🧪 RFT Validation", "rft_validation_suite"),
            ("🔐 Quantum Crypto", "quantum_crypto"),
            ("📊 System Monitor", "system_monitor"),
            ("🌌 Quantum Simulator", "quantum_simulator"),
            ("📝 Notes", "q_notes"),
            ("🌐 Browser", "q_browser"),
            ("🔒 Vault", "q_vault"),
            ("📧 Mail", "q_mail")
        ]
        
        for app_name, app_id in apps:
            btn = QPushButton(app_name)
            btn.clicked.connect(lambda checked, aid=app_id: self.launch_app(aid))
            btn.setStyleSheet("""
                QPushButton {
                    text-align: left;
                    padding: 16px 20px;
                    margin: 6px 0;
                    font-size: 15px;
                    font-weight: 600;
                    border-radius: 12px;
                    border: 1px solid rgba(229, 231, 235, 0.5);
                    background: rgba(255, 255, 255, 0.7);
                    color: #374151;
                }
                QPushButton:hover {
                    background: rgba(255, 255, 255, 0.95);
                    border: 1px solid rgba(75, 70, 229, 0.3);
                    color: #4f46e5;
                }
                QPushButton:pressed {
                    background: rgba(75, 70, 229, 0.1);
                }
            """)
            left_layout.addWidget(btn)
        
        left_layout.addStretch()
        parent.addWidget(left_frame)
    
    def create_right_panel(self, parent):
        """Create right panel with system info - Modern sleek design"""
        right_frame = QFrame()
        right_frame.setFrameStyle(QFrame.StyledPanel)
        right_layout = QVBoxLayout(right_frame)
        right_layout.setSpacing(20)  # More generous spacing
        
        # Title - Modern typography
        title = QLabel("⚡ System")
        title.setStyleSheet("""
            QLabel {
                font-size: 22px;
                font-weight: 700;
                color: #1f2937;
                padding: 20px 0;
                border-bottom: 1px solid rgba(229, 231, 235, 0.8);
                margin-bottom: 16px;
                background: transparent;
            }
        """)
        right_layout.addWidget(title)
        
        # System info labels - Modern card design
        info_style = """
            QLabel {
                background: rgba(255, 255, 255, 0.8);
                border: 1px solid rgba(229, 231, 235, 0.6);
                border-radius: 12px;
                padding: 16px;
                margin: 8px 0;
                font-size: 14px;
                font-weight: 600;
                color: #374151;
            }
        """
        
        self.cpu_label = QLabel("CPU: Initializing...")
        self.cpu_label.setStyleSheet(info_style)
        
        self.memory_label = QLabel("Memory: Initializing...")
        self.memory_label.setStyleSheet(info_style)
        
        self.quantum_label = QLabel("Quantum State: Stable")
        quantum_style = info_style.replace("rgba(229, 231, 235, 0.6)", "rgba(16, 185, 129, 0.3)")
        quantum_style = quantum_style.replace("#374151", "#065f46")
        self.quantum_label.setStyleSheet(quantum_style)  # Green theme for quantum
        
        self.windows_label = QLabel("Active Windows: 0")
        self.windows_label.setStyleSheet(info_style)
        
        for label in [self.cpu_label, self.memory_label, self.quantum_label, self.windows_label]:
            right_layout.addWidget(label)
        
        right_layout.addStretch()
        parent.addWidget(right_frame)
    
    def create_menu_bar(self):
        """Create application menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('&File')
        
        new_action = QAction('&New Window', self)
        new_action.setShortcut(QKeySequence.New)
        new_action.triggered.connect(self.new_window)
        file_menu.addAction(new_action)
        
        open_action = QAction('&Open App', self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self.open_app_dialog)
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        save_session_action = QAction('&Save Session', self)
        save_session_action.setShortcut(QKeySequence.Save)
        save_session_action.triggered.connect(self.save_session)
        file_menu.addAction(save_session_action)
        
        load_session_action = QAction('&Load Session', self)
        load_session_action.triggered.connect(self.load_session)
        file_menu.addAction(load_session_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('&Exit', self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu('&View')
        
        if window_manager:
            cascade_action = QAction('&Cascade Windows', self)
            cascade_action.triggered.connect(window_manager.arrange_cascade)
            view_menu.addAction(cascade_action)
            
            tile_h_action = QAction('Tile &Horizontal', self)
            tile_h_action.triggered.connect(window_manager.arrange_tile_horizontal)
            view_menu.addAction(tile_h_action)
            
            tile_v_action = QAction('Tile &Vertical', self)
            tile_v_action.triggered.connect(window_manager.arrange_tile_vertical)
            view_menu.addAction(tile_v_action)
        
        # Quantum menu
        quantum_menu = menubar.addMenu('&Quantum')
        
        visualizer_action = QAction('🔬 &RFT Visualizer', self)
        visualizer_action.triggered.connect(lambda: self.launch_app('rft_visualizer'))
        quantum_menu.addAction(visualizer_action)
        
        crypto_action = QAction('🔐 &Quantum Crypto', self)
        crypto_action.triggered.connect(lambda: self.launch_app('quantum_crypto'))
        quantum_menu.addAction(crypto_action)
        
        simulator_action = QAction('🌌 &Quantum Simulator', self)
        simulator_action.triggered.connect(lambda: self.launch_app('quantum_simulator'))
        quantum_menu.addAction(simulator_action)
        
        quantum_menu.addSeparator()
        
        notes_action = QAction('📝 &Quantum Notes', self)
        notes_action.triggered.connect(lambda: self.launch_app('q_notes'))
        quantum_menu.addAction(notes_action)
        
        browser_action = QAction('🌐 &Quantum Browser', self)
        browser_action.triggered.connect(lambda: self.launch_app('q_browser'))
        quantum_menu.addAction(browser_action)
        
        vault_action = QAction('🔐 &Quantum Vault', self)
        vault_action.triggered.connect(lambda: self.launch_app('q_vault'))
        quantum_menu.addAction(vault_action)
        
        mail_action = QAction('📧 &Quantum Mail', self)
        mail_action.triggered.connect(lambda: self.launch_app('q_mail'))
        quantum_menu.addAction(mail_action)
        
        # Help menu
        help_menu = menubar.addMenu('&Help')
        
        about_action = QAction('&About QuantoniumOS', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def create_tool_bar(self):
        """Create main toolbar"""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        
        # Quick access tools
        tools = [
            ("🔬", "RFT Visualizer", lambda: self.launch_app('rft_visualizer')),
            ("🔐", "Quantum Crypto", lambda: self.launch_app('quantum_crypto')),
            ("📊", "System Monitor", lambda: self.launch_app('system_monitor')),
            ("🌌", "Quantum Simulator", lambda: self.launch_app('quantum_simulator'))
        ]
        
        for icon, tooltip, action in tools:
            btn = QAction(icon, self)
            btn.setToolTip(tooltip)
            btn.triggered.connect(action)
            toolbar.addAction(btn)
            
        toolbar.addSeparator()
        
        # Window management
        if window_manager:
            cascade_action = QAction("⚡", self)
            cascade_action.setToolTip("Cascade Windows")
            cascade_action.triggered.connect(window_manager.arrange_cascade)
            toolbar.addAction(cascade_action)
    
    def load_apps(self):
        """Load available quantum applications"""
        # App module mapping
        app_modules = {
            'rft_visualizer': 'apps.rft_visualizer',
            'rft_validation_suite': 'apps.rft_validation_suite',
            'quantum_crypto': 'apps.resonance_encryption',
            'system_monitor': 'apps.monitor_main_system',
            'quantum_simulator': 'apps.multi_qubit_state',
            'q_notes': 'apps.q_notes',
            'q_browser': 'apps.q_browser',
            'q_vault': 'apps.q_vault',
            'q_mail': 'apps.q_mail'
        }
        
        for app_name, module_name in app_modules.items():
            try:
                module = importlib.import_module(module_name)
                self.apps[app_name] = module
                print(f"✅ Loaded app: {app_name}")
            except ImportError as e:
                print(f"⚠️ Failed to load {app_name}: {e}")
                # Create placeholder
                self.apps[app_name] = None
    
    def launch_app(self, app_name: str):
        """Launch a quantum application directly in a tab"""
        try:
            print(f"🚀 Launching {app_name}...")
            
            if app_name not in self.apps:
                self.status_bar.showMessage(f"❌ App {app_name} not found", 3000)
                return
            
            app_module = self.apps[app_name]
            if app_module is None:
                self.status_bar.showMessage(f"❌ App {app_name} not available", 3000)
                return
            
            # Create application window/widget DIRECTLY
            app_widget = self.create_app_widget_direct(app_name, app_module)
            if app_widget is None:
                self.status_bar.showMessage(f"❌ Failed to create {app_name} widget", 3000)
                return
            
            # Add directly to central tabs WITHOUT intermediate launch dialogs
            tab_name = app_name.replace('_', ' ').title()
            tab_index = self.central_tabs.addTab(app_widget, f"🚀 {tab_name}")
            self.central_tabs.setCurrentIndex(tab_index)
            
            self.status_bar.showMessage(f"✅ {tab_name} launched in tab", 2000)
            self.update_window_count()
            
        except Exception as e:
            error_msg = f"❌ Error launching {app_name}: {str(e)}"
            print(error_msg)
            self.status_bar.showMessage(error_msg, 5000)
    
    def create_app_widget_direct(self, app_name: str, app_module) -> Optional[QWidget]:
        """Create application widget directly - NO launch dialogs"""
        try:
            # PRIORITY 1: Try to create actual Qt widget classes
            widget_classes = [
                'RFTVisualizer', 'RFTValidationWidget', 'QuantumGateVisualizer', 'MainWindow',
                'ResonanceEncryption', 'SystemMonitor', 'QuantumSimulator',
                'QuantumNotes', 'QuantumBrowser', 'QuantumVault', 'QuantumMail'
            ]
            
            for widget_class in widget_classes:
                if hasattr(app_module, widget_class):
                    try:
                        widget_instance = getattr(app_module, widget_class)()
                        print(f"✅ Created {widget_class} widget for {app_name}")
                        return widget_instance
                    except Exception as e:
                        print(f"⚠️ Failed to create {widget_class}: {e}")
                        continue
            
            # PRIORITY 2: For function-based apps, create embedded widget
            if hasattr(app_module, 'main'):
                return self.create_embedded_app_widget(app_name, app_module.main)
            elif hasattr(app_module, 'run'):
                return self.create_embedded_app_widget(app_name, app_module.run)
            
            # PRIORITY 3: Create functional placeholder with actual app data
            return self.create_functional_widget(app_name, app_module)
                
        except Exception as e:
            print(f"Error creating {app_name} widget: {e}")
            return self.create_functional_widget(app_name, app_module)
    
    def create_embedded_app_widget(self, app_name: str, app_function) -> QWidget:
        """Create widget that embeds the actual running application"""
        from PyQt5.QtWidgets import QTextEdit, QVBoxLayout, QHBoxLayout, QPushButton
        
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Header
        header_layout = QHBoxLayout()
        title = QLabel(f"🌌 {app_name.replace('_', ' ').title()}")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #60a5fa; padding: 8px;")
        header_layout.addWidget(title)
        
        # Control buttons
        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.setMaximumWidth(100)
        refresh_btn.clicked.connect(lambda: self.run_embedded_app(app_function, output_area))
        header_layout.addWidget(refresh_btn)
        
        header_layout.addStretch()
        layout.addLayout(header_layout)
        
        # Output area for the app
        output_area = QTextEdit()
        output_area.setStyleSheet("""
            QTextEdit {
                background: #1a1a1a;
                color: #e8e8e8;
                border: 1px solid rgba(255, 255, 255, 26);
                border-radius: 8px;
                padding: 12px;
                font-family: 'Consolas', monospace;
                font-size: 12px;
            }
        """)
        output_area.setPlainText(f"🚀 Running {app_name}...\n")
        layout.addWidget(output_area)
        
        # Run the app immediately
        self.run_embedded_app(app_function, output_area)
        
        return widget
    
    def run_embedded_app(self, app_function, output_widget):
        """Run app function and capture output"""
        import sys
        from io import StringIO
        
        try:
            # Capture stdout
            old_stdout = sys.stdout
            sys.stdout = captured_output = StringIO()
            
            # Run the app function
            result = app_function()
            
            # Restore stdout
            sys.stdout = old_stdout
            
            # Display captured output
            output_text = captured_output.getvalue()
            if output_text:
                output_widget.append(output_text)
            else:
                output_widget.append(f"✅ Application executed successfully\nResult: {result}")
                
        except Exception as e:
            sys.stdout = old_stdout
            output_widget.append(f"❌ Error: {str(e)}")
    
    def create_functional_widget(self, app_name: str, app_module) -> QWidget:
        """Create functional widget with app info and controls"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Header
        title = QLabel(f"🌌 {app_name.replace('_', ' ').title()}")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #60a5fa; padding: 16px;")
        layout.addWidget(title)
        
        # App info
        info_text = f"""
📱 <b>Application:</b> {app_name.replace('_', ' ').title()}<br>
🔧 <b>Module:</b> {app_module.__name__ if hasattr(app_module, '__name__') else 'Unknown'}<br>
📂 <b>Status:</b> Loaded and Ready<br>
⚡ <b>Type:</b> {'Qt Widget' if any(hasattr(app_module, cls) for cls in ['MainWindow', 'QWidget']) else 'Function-based'}
        """
        
        info_label = QLabel(info_text)
        info_label.setStyleSheet("""
            QLabel {
                background: rgba(37, 99, 235, 51);
                border: 1px solid rgba(37, 99, 235, 76);
                border-radius: 12px;
                padding: 16px;
                margin: 8px;
            }
        """)
        layout.addWidget(info_label)
        
        # Available methods
        methods = [attr for attr in dir(app_module) if not attr.startswith('_')]
        if methods:
            methods_label = QLabel(f"🛠️ <b>Available Methods:</b><br>" + "<br>".join(f"• {method}" for method in methods[:10]))
            methods_label.setStyleSheet("""
                QLabel {
                    background: rgba(16, 185, 129, 51);
                    border: 1px solid rgba(16, 185, 129, 76);
                    border-radius: 8px;
                    padding: 12px;
                    margin: 8px;
                }
            """)
            layout.addWidget(methods_label)
        
        layout.addStretch()
        return widget
    
    def create_wrapper_widget(self, app_name: str, app_function) -> QWidget:
        """Create a wrapper widget for function-based apps"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        title = QLabel(f"🌌 {app_name.replace('_', ' ').title()}")
        title.setProperty("class", "quantum-glow")
        layout.addWidget(title)
        
        launch_btn = QPushButton(f"Launch {app_name}")
        launch_btn.clicked.connect(lambda: self.run_app_function(app_function))
        layout.addWidget(launch_btn)
        
        output_label = QLabel("Ready to launch...")
        layout.addWidget(output_label)
        
        layout.addStretch()
        return widget
    
    def create_placeholder_widget(self, app_name: str) -> QWidget:
        """Create placeholder widget for unavailable apps"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        title = QLabel(f"🌌 {app_name.replace('_', ' ').title()}")
        title.setProperty("class", "quantum-glow")
        layout.addWidget(title)
        
        status_label = QLabel(f"⚠️ Application {app_name} is being integrated...")
        layout.addWidget(status_label)
        
        info_label = QLabel("This quantum application will be available in the next update.")
        layout.addWidget(info_label)
        
        layout.addStretch()
        return widget
    
    def run_app_function(self, app_function):
        """Run application function safely"""
        try:
            result = app_function()
            print(f"App function result: {result}")
        except Exception as e:
            print(f"Error running app function: {e}")
    
    def close_tab(self, index: int):
        """Close tab and associated window"""
        widget = self.central_tabs.widget(index)
        self.central_tabs.removeTab(index)
        
        if widget and window_manager:
            # Find and close associated window
            for window_id, window_widget in window_manager.windows.items():
                if window_widget == widget:
                    window_manager.close_window(window_id)
                    break
        
        self.update_window_count()
    
    def start_system_monitor(self):
        """Start system monitoring"""
        self.monitor_timer = QTimer()
        self.monitor_timer.timeout.connect(self.update_system_info)
        self.monitor_timer.start(1000)  # Update every second
    
    def update_system_info(self):
        """Update system information display"""
        try:
            import psutil
            
            # Update CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.cpu_label.setText(f"⚡ CPU: {cpu_percent:.1f}%")
            
            # Update memory usage
            memory = psutil.virtual_memory()
            self.memory_label.setText(f"🧠 Memory: {memory.percent:.1f}%")
            
            # Update quantum state from actual quantum kernel
            if hasattr(self, 'os_backend') and self.os_backend and hasattr(self.os_backend, 'quantum_kernel'):
                try:
                    # Get actual quantum kernel status
                    kernel = self.os_backend.quantum_kernel
                    if kernel and hasattr(kernel, 'vertices'):
                        vertex_count = len(kernel.vertices) if hasattr(kernel.vertices, '__len__') else 1000
                        if vertex_count >= 1000:
                            self.quantum_label.setText("Quantum: 🌌 Coherent (1000-qubit)")
                        elif vertex_count >= 500:
                            self.quantum_label.setText("Quantum: 🔀 Entangled (500+ qubits)")
                        else:
                            self.quantum_label.setText("Quantum: ⚡ Initializing")
                    else:
                        self.quantum_label.setText("Quantum: 🌊 Stable")
                except:
                    self.quantum_label.setText("Quantum: 🌊 Stable")
            else:
                self.quantum_label.setText("Quantum: 🌊 Stable")
            
            self.update_window_count()
            
        except ImportError:
            self.cpu_label.setText("⚡ CPU: N/A (psutil not available)")
            self.memory_label.setText("🧠 Memory: N/A")
        except Exception as e:
            print(f"Error updating system info: {e}")
    
    def update_window_count(self):
        """Update active window count"""
        if window_manager:
            count = len(window_manager.windows)
            self.windows_label.setText(f"🪟 Windows: {count}")
        else:
            tab_count = self.central_tabs.count()
            self.windows_label.setText(f"📑 Tabs: {tab_count}")
    
    def new_window(self):
        """Create new window"""
        self.status_bar.showMessage("🆕 New window functionality coming soon", 2000)
    
    def open_app_dialog(self):
        """Open application selection dialog"""
        try:
            from PyQt5.QtWidgets import QInputDialog
            app_list = list(self.apps.keys())
            app_name, ok = QInputDialog.getItem(
                self, 
                "🚀 Launch Quantum Application",
                "Select application:",
                [name.replace('_', ' ').title() for name in app_list],
                0,
                False
            )
            if ok and app_name:
                # Convert back to app_id
                app_id = app_name.lower().replace(' ', '_')
                self.launch_app(app_id)
        except Exception as e:
            print(f"Error in app dialog: {e}")
    
    def save_session(self):
        """Save current session"""
        if window_manager:
            session_path = Path.home() / ".quantonium" / "session.json"
            session_path.parent.mkdir(exist_ok=True)
            if window_manager.save_session(str(session_path)):
                self.status_bar.showMessage("💾 Session saved", 2000)
            else:
                self.status_bar.showMessage("❌ Failed to save session", 3000)
    
    def load_session(self):
        """Load saved session"""
        if window_manager:
            session_path = Path.home() / ".quantonium" / "session.json"
            if session_path.exists():
                session_data = window_manager.load_session(str(session_path))
                if session_data:
                    self.status_bar.showMessage("📂 Session loaded", 2000)
                else:
                    self.status_bar.showMessage("❌ Failed to load session", 3000)
            else:
                self.status_bar.showMessage("❌ No saved session found", 3000)
    
    def show_about(self):
        """Show about dialog"""
        try:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.about(
                self,
                "About QuantoniumOS",
                """<h2>🌌 QuantoniumOS v2.0</h2>
                <p><b>Advanced Quantum Operating System</b></p>
                <p>Developed by Ana - 1000X Dev Helper</p>
                <p>Powered by quantum computing principles and advanced UI design</p>
                <br>
                <p><b>Features:</b></p>
                <ul>
                <li>🔬 RFT Transform Visualization</li>
                <li>🔐 Quantum Cryptography</li>
                <li>🌌 Multi-qubit Simulation</li>
                <li>📊 Real-time System Monitoring</li>
                <li>🪟 Advanced Window Management</li>
                </ul>
                """
            )
        except Exception as e:
            print(f"Error showing about dialog: {e}")
    
    def closeEvent(self, event):
        """Handle application close event"""
        try:
            # Save session
            if window_manager:
                session_path = Path.home() / ".quantonium" / "session.json"
                session_path.parent.mkdir(exist_ok=True)
                window_manager.save_session(str(session_path))
                
                # Close all windows
                for window_id in list(window_manager.windows.keys()):
                    window_manager.close_window(window_id, animated=False)
            
            print("👋 QuantoniumOS shutting down...")
            event.accept()
        except Exception as e:
            print(f"Error during shutdown: {e}")
            event.accept()

def main():
    """Main entry point for QuantoniumOS frontend"""
    if not PYQT5_AVAILABLE:
        print("❌ PyQt5 required for QuantoniumOS frontend")
        print("Install with: pip install PyQt5")
        return
    
    # Create application
    app = QApplication(sys.argv)
    
    # Set application metadata
    app.setApplicationName("QuantoniumOS")
    app.setApplicationDisplayName("🌌 QuantoniumOS v2.0")
    app.setOrganizationName("Quantum Computing Labs")
    
    # Set application style
    app.setStyle('Fusion')
    
    # Enable high DPI support
    try:
        app.setAttribute(Qt.AA_EnableHighDpiScaling)
        app.setAttribute(Qt.AA_UseHighDpiPixmaps)
    except:
        pass
    
    # Create main window
    print("🚀 Starting QuantoniumOS...")
    main_window = QuantumAppController()
    main_window.show()
    
    # Integrate with VS Code if available
    if window_manager:
        window_manager.integrate_with_vscode()
    
    print("✅ QuantoniumOS frontend ready!")
    
    # Start event loop
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
