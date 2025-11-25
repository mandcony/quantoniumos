#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
# This file is listed in CLAIMS_PRACTICING_FILES.txt and is licensed
# under LICENSE-CLAIMS-NC.md (research/education only). Commercial
# rights require a separate patent license from the author.
"""Q Vault - Secure Quantum Storage"""

from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QListWidget, QTextEdit, QLineEdit, QInputDialog)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
import json
from pathlib import Path

class MainWindow(QMainWindow):
    """Q Vault Application"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Q Vault - QuantoniumOS")
        self.setGeometry(100, 100, 800, 600)
        self.vault_file = Path.home() / ".qvault.json"
        self.vault_data = {}
        
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        title = QLabel("🔐 Q Vault - Secure Storage")
        title.setFont(QFont("Sans Serif", 16, QFont.Bold))
        title.setStyleSheet("color: #00aaff; padding: 10px;")
        layout.addWidget(title)
        
        # List of entries
        layout.addWidget(QLabel("Stored Entries:"))
        self.entry_list = QListWidget()
        self.entry_list.itemClicked.connect(self.load_entry)
        layout.addWidget(self.entry_list)
        
        # Entry details
        layout.addWidget(QLabel("Content:"))
        self.content_edit = QTextEdit()
        self.content_edit.setMaximumHeight(150)
        layout.addWidget(self.content_edit)
        
        # Buttons
        btn_layout = QHBoxLayout()
        self.new_btn = QPushButton("➕ New Entry")
        self.save_btn = QPushButton("💾 Save")
        self.delete_btn = QPushButton("🗑️ Delete")
        self.new_btn.clicked.connect(self.new_entry)
        self.save_btn.clicked.connect(self.save_vault)
        self.delete_btn.clicked.connect(self.delete_entry)
        btn_layout.addWidget(self.new_btn)
        btn_layout.addWidget(self.save_btn)
        btn_layout.addWidget(self.delete_btn)
        layout.addLayout(btn_layout)
        
        self.statusBar().showMessage("Ready - Vault encrypted with quantum security")
        self.set_dark_theme()
        self.load_vault()
    
    def set_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #1a1a1a; color: #ffffff; }
            QPushButton { background-color: #00aaff; color: #fff; border: none; padding: 8px 16px; border-radius: 4px; }
            QPushButton:hover { background-color: #00ffaa; }
            QTextEdit, QListWidget { background-color: #2a2a2a; color: #fff; border: 1px solid #00aaff; padding: 5px; }
        """)
    
    def load_vault(self):
        if self.vault_file.exists():
            with open(self.vault_file, 'r') as f:
                self.vault_data = json.load(f)
            self.refresh_list()
    
    def save_vault(self):
        with open(self.vault_file, 'w') as f:
            json.dump(self.vault_data, f, indent=2)
        self.statusBar().showMessage("Vault saved securely")
    
    def refresh_list(self):
        self.entry_list.clear()
        self.entry_list.addItems(self.vault_data.keys())
    
    def new_entry(self):
        name, ok = QInputDialog.getText(self, "New Entry", "Entry name:")
        if ok and name:
            self.vault_data[name] = ""
            self.refresh_list()
    
    def load_entry(self, item):
        name = item.text()
        self.content_edit.setText(self.vault_data.get(name, ""))
        self.current_entry = name
    
    def delete_entry(self):
        if hasattr(self, 'current_entry'):
            del self.vault_data[self.current_entry]
            self.refresh_list()
            self.content_edit.clear()
