#!/usr/bin/env python3
"""
Q-Notes - Quantum Text Editor
============================
Integrated with QuantoniumOS RFT system
"""

import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTextEdit, QPushButton, 
    QVBoxLayout, QWidget, QFileDialog, QStatusBar, QMenuBar
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

# Get base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UI_DIR = os.path.join(BASE_DIR, "ui")

class QNotes(QMainWindow):
    """Quantum-enhanced text editor"""
    
    def __init__(self):
        super().__init__()
        self.setObjectName("AppWindow")
        self.setWindowTitle("Q-Notes - Quantum Text Editor")
        self.setGeometry(200, 200, 800, 600)
        
        self.init_ui()
        self.load_styles()
        
    def init_ui(self):
        """Initialize the user interface"""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Text editor
        self.text_editor = QTextEdit()
        self.text_editor.setFont(QFont("Consolas", 12))
        layout.addWidget(self.text_editor)
        
        # Save button
        self.save_button = QPushButton("Save Document")
        self.save_button.clicked.connect(self.save_document)
        layout.addWidget(self.save_button)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
    def load_styles(self):
        """Load QuantoniumOS stylesheet"""
        qss_path = os.path.join(UI_DIR, "styles.qss")
        if os.path.exists(qss_path):
            with open(qss_path, 'r') as f:
                self.setStyleSheet(f.read())
                
    def save_document(self):
        """Save the current document"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Document", "", "Text Files (*.txt);;All Files (*)"
        )
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.text_editor.toPlainText())
                self.status_bar.showMessage(f"Saved: {file_path}")
            except Exception as e:
                self.status_bar.showMessage(f"Error saving: {e}")

def main():
    app = QApplication(sys.argv)
    notes = QNotes()
    notes.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
