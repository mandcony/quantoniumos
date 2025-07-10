#!/usr/bin/env python3
"""
Example Qt Application with Optimized QSS Styling
Use this in your VSCode environment for your Python Qt project
"""

import os
import sys

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtWidgets import (QApplication, QFrame, QGraphicsScene,
                             QGraphicsView, QHBoxLayout, QLabel, QMainWindow,
                             QMenuBar, QPushButton, QScrollArea, QStatusBar,
                             QTabWidget, QTextEdit, QTreeView, QVBoxLayout,
                             QWidget)


class QuantoniumOSMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setObjectName("QSHLLFileExplorer")
        self.setWindowTitle("QuantoniumOS - Enhanced Interface")
        self.setMinimumSize(800, 600)

        # Load the optimized QSS
        self.load_stylesheet()

        # Setup the UI
        self.setup_ui()

    def load_stylesheet(self):
        """Load the optimized QSS file"""
        try:
            qss_file = "optimized_qss_style.qss"
            if os.path.exists(qss_file):
                with open(qss_file, "r", encoding="utf-8") as file:
                    self.setStyleSheet(file.read())
                print(f"✓ Loaded QSS stylesheet: {qss_file}")
            else:
                print(f"⚠ QSS file not found: {qss_file}")
                # Fallback inline styles
                self.setStyleSheet(self.get_fallback_qss())
        except Exception as e:
            print(f"✗ Error loading QSS: {e}")
            self.setStyleSheet(self.get_fallback_qss())

    def get_fallback_qss(self):
        """Fallback QSS in case file loading fails"""
        return """
        QMainWindow {
            background-color: #F8E8C6;
            color: #FFF5E1;
        }
        QPushButton {
            background-color: rgba(163, 111, 90, 0.8);
            color: #FFF5E1;
            border: 2px solid rgba(208, 181, 155, 0.8);
            border-radius: 8px;
            padding: 8px;
            font: bold 10pt "Segoe UI";
        }
        QPushButton:hover {
            background-color: rgba(163, 111, 90, 1.0);
        }
        """

    def setup_ui(self):
        """Setup the main UI components"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout(central_widget)

        # Header section
        header_layout = self.create_header_section()
        main_layout.addLayout(header_layout)

        # Content area with tabs
        content_area = self.create_content_area()
        main_layout.addWidget(content_area)

        # Dock panel
        dock_panel = self.create_dock_panel()
        main_layout.addWidget(dock_panel)

        # Setup menu bar and status bar
        self.setup_menu_bar()
        self.setup_status_bar()

    def create_header_section(self):
        """Create the header section with logo and clock"""
        header_layout = QHBoxLayout()

        # Q Logo
        logo_label = QLabel("Q")
        logo_label.setObjectName("QLogo")
        logo_label.setAlignment(Qt.AlignCenter)
        logo_label.setFixedSize(100, 100)
        header_layout.addWidget(logo_label)

        # Clock item
        clock_label = QLabel("12:34 PM")
        clock_label.setObjectName("ClockItem")
        clock_label.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(clock_label)

        # Update clock every second
        self.clock_timer = QTimer()
        self.clock_timer.timeout.connect(self.update_clock)
        self.clock_timer.start(1000)

        return header_layout

    def create_content_area(self):
        """Create the main content area with tabs"""
        tab_widget = QTabWidget()

        # Desktop View tab
        desktop_tab = QWidget()
        desktop_layout = QVBoxLayout(desktop_tab)

        desktop_view = QGraphicsView()
        desktop_view.setObjectName("DesktopView")
        desktop_scene = QGraphicsScene()
        desktop_view.setScene(desktop_scene)
        desktop_layout.addWidget(desktop_view)

        tab_widget.addTab(desktop_tab, "Desktop")

        # File Explorer tab
        explorer_tab = QWidget()
        explorer_layout = QVBoxLayout(explorer_tab)

        arch_view = QTreeView()
        arch_view.setObjectName("ArchView")
        explorer_layout.addWidget(arch_view)

        tab_widget.addTab(explorer_tab, "Explorer")

        # Text Editor tab (QVault)
        editor_tab = QWidget()
        editor_layout = QVBoxLayout(editor_tab)

        text_editor = QTextEdit()
        text_editor.setObjectName("TextEditor")
        text_editor.setPlaceholderText("Enter your text here...")
        editor_layout.addWidget(text_editor)

        # Save button
        save_button = QPushButton("Save Document")
        save_button.setObjectName("SaveButton")
        save_button.clicked.connect(self.save_document)
        editor_layout.addWidget(save_button)

        tab_widget.addTab(editor_tab, "QVault")

        return tab_widget

    def create_dock_panel(self):
        """Create the dock panel with app buttons"""
        dock_frame = QFrame()
        dock_frame.setObjectName("QDockPanel")
        dock_frame.setFixedHeight(120)

        dock_layout = QHBoxLayout(dock_frame)
        dock_layout.setSpacing(15)

        # App buttons
        app_names = ["Home", "Settings", "Tools", "Help"]
        for app_name in app_names:
            app_button = QPushButton(app_name)
            app_button.setObjectName("DockButton")
            app_button.setFixedSize(80, 80)
            app_button.clicked.connect(
                lambda checked, name=app_name: self.launch_app(name)
            )
            dock_layout.addWidget(app_button)

        # Close button
        close_button = QPushButton("✕")
        close_button.setObjectName("CloseButton")
        close_button.setFixedSize(80, 80)
        close_button.clicked.connect(self.close)
        dock_layout.addWidget(close_button)

        dock_layout.addStretch()

        return dock_frame

    def setup_menu_bar(self):
        """Setup the menu bar"""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")
        file_menu.addAction("New", self.new_file)
        file_menu.addAction("Open", self.open_file)
        file_menu.addAction("Save", self.save_document)
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close)

        # View menu
        view_menu = menubar.addMenu("View")
        view_menu.addAction("Desktop", lambda: self.switch_tab(0))
        view_menu.addAction("Explorer", lambda: self.switch_tab(1))
        view_menu.addAction("QVault", lambda: self.switch_tab(2))

        # Help menu
        help_menu = menubar.addMenu("Help")
        help_menu.addAction("About", self.show_about)

    def setup_status_bar(self):
        """Setup the status bar"""
        status_bar = self.statusBar()
        status_bar.showMessage("QuantoniumOS Ready")

    def update_clock(self):
        """Update the clock display"""
        from datetime import datetime

        current_time = datetime.now().strftime("%I:%M %p")
        clock_label = self.findChild(QLabel, "ClockItem")
        if clock_label:
            clock_label.setText(current_time)

    def launch_app(self, app_name):
        """Handle app launch"""
        self.statusBar().showMessage(f"Launching {app_name}...")
        print(f"Launching app: {app_name}")

    def save_document(self):
        """Handle document saving"""
        self.statusBar().showMessage("Document saved successfully")
        print("Document saved")

    def new_file(self):
        """Create new file"""
        text_editor = self.findChild(QTextEdit, "TextEditor")
        if text_editor:
            text_editor.clear()
        self.statusBar().showMessage("New document created")

    def open_file(self):
        """Open file dialog"""
        self.statusBar().showMessage("Open file dialog...")
        print("Open file functionality would go here")

    def switch_tab(self, index):
        """Switch to specific tab"""
        tab_widget = self.findChild(QTabWidget)
        if tab_widget:
            tab_widget.setCurrentIndex(index)

    def show_about(self):
        """Show about dialog"""
        from PyQt5.QtWidgets import QMessageBox

        QMessageBox.about(
            self,
            "About QuantoniumOS",
            "QuantoniumOS Enhanced Interface\nOptimized for VSCode Development",
        )


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)

    # Set application properties
    app.setApplicationName("QuantoniumOS")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("Quantonium Systems")

    # Create and show main window
    window = QuantoniumOSMainWindow()
    window.show()

    # Center the window on screen
    screen = app.primaryScreen().geometry()
    window.move(
        (screen.width() - window.width()) // 2, (screen.height() - window.height()) // 2
    )

    print("✓ QuantoniumOS application started")
    print("✓ QSS styling applied")
    print("✓ Ready for development in VSCode")

    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
