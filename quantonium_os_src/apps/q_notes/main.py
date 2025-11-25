#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
# This file is listed in CLAIMS_PRACTICING_FILES.txt and is licensed
# under LICENSE-CLAIMS-NC.md (research/education only). Commercial
# rights require a separate patent license from the author.
"""Q Notes - Quantum-Enhanced Note Taking"""

from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QTextEdit, QFileDialog, QLabel, QMenuBar, QMenu, QAction)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

class MainWindow(QMainWindow):
    """Q Notes Application"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Q Notes - QuantoniumOS")
        self.setGeometry(100, 100, 900, 600)
        
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        title = QLabel("📝 Q Notes - Quantum Note Taking")
        title.setFont(QFont("Sans Serif", 16, QFont.Bold))
        title.setStyleSheet("color: #00aaff; padding: 10px;")
        layout.addWidget(title)
        
        # Editor
        self.editor = QTextEdit()
        self.editor.setPlaceholderText("Start typing your quantum notes...")
        layout.addWidget(self.editor)
        
        # Buttons
        btn_layout = QHBoxLayout()
        self.save_btn = QPushButton("💾 Save")
        self.load_btn = QPushButton("📂 Load")
        self.clear_btn = QPushButton("🗑️ Clear")
        self.save_btn.clicked.connect(self.save_note)
        self.load_btn.clicked.connect(self.load_note)
        self.clear_btn.clicked.connect(self.editor.clear)
        btn_layout.addWidget(self.save_btn)
        btn_layout.addWidget(self.load_btn)
        btn_layout.addWidget(self.clear_btn)
        layout.addLayout(btn_layout)
        
        self.statusBar().showMessage("Ready")
        self.set_dark_theme()
    
    def set_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #1a1a1a; color: #ffffff; }
            QPushButton { background-color: #00aaff; color: #fff; border: none; padding: 8px 16px; border-radius: 4px; }
            QPushButton:hover { background-color: #00ffaa; }
            QTextEdit { background-color: #2a2a2a; color: #fff; border: 1px solid #00aaff; padding: 10px; font-size: 12pt; }
        """)
    
    def save_note(self):
        fname, _ = QFileDialog.getSaveFileName(self, "Save Note", "", "Text Files (*.txt);;All Files (*)")
        if fname:
            with open(fname, 'w') as f:
                f.write(self.editor.toPlainText())
            self.statusBar().showMessage(f"Saved: {fname}")
    
    def load_note(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Load Note", "", "Text Files (*.txt);;All Files (*)")
        if fname:
            with open(fname, 'r') as f:
                self.editor.setText(f.read())
            self.statusBar().showMessage(f"Loaded: {fname}")
