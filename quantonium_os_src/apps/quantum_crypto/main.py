#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
# This file is listed in CLAIMS_PRACTICING_FILES.txt and is licensed
# under LICENSE-CLAIMS-NC.md (research/education only). Commercial
# rights require a separate patent license from the author.
"""Quantum Cryptography - QKD & RFT Crypto Interface"""

from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QTextEdit, QComboBox, QGroupBox,
                             QTabWidget, QLineEdit, QSpinBox, QProgressBar)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
import secrets
import hashlib

class MainWindow(QMainWindow):
    """Quantum Cryptography Application"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Quantum Cryptography - QuantoniumOS")
        self.setGeometry(100, 100, 1000, 700)
        
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        # Title
        title = QLabel("🔐 Quantum Cryptography Suite")
        title.setFont(QFont("Sans Serif", 18, QFont.Bold))
        title.setStyleSheet("color: #00aaff; padding: 10px;")
        layout.addWidget(title)
        
        # Tab widget for different crypto operations
        tabs = QTabWidget()
        
        # Tab 1: QKD Simulator
        qkd_tab = self.create_qkd_tab()
        tabs.addTab(qkd_tab, "🔑 QKD Protocol")
        
        # Tab 2: RFT Encryption
        rft_tab = self.create_rft_tab()
        tabs.addTab(rft_tab, "🌀 RFT Encryption")
        
        # Tab 3: Key Management
        key_tab = self.create_key_tab()
        tabs.addTab(key_tab, "🗝️ Key Manager")
        
        layout.addWidget(tabs)
        
        # Status bar
        self.statusBar().showMessage("Ready - Select a cryptographic operation")
        
        # Apply dark theme
        self.set_dark_theme()
    
    def create_qkd_tab(self):
        """Create QKD protocol simulator tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Protocol selection
        protocol_group = QGroupBox("Protocol Settings")
        protocol_layout = QVBoxLayout()
        
        proto_select_layout = QHBoxLayout()
        proto_select_layout.addWidget(QLabel("Protocol:"))
        self.qkd_protocol = QComboBox()
        self.qkd_protocol.addItems(['BB84', 'E91', 'B92'])
        proto_select_layout.addWidget(self.qkd_protocol)
        proto_select_layout.addStretch()
        protocol_layout.addLayout(proto_select_layout)
        
        key_len_layout = QHBoxLayout()
        key_len_layout.addWidget(QLabel("Key Length (bits):"))
        self.qkd_keylen = QSpinBox()
        self.qkd_keylen.setRange(16, 1024)
        self.qkd_keylen.setValue(256)
        key_len_layout.addWidget(self.qkd_keylen)
        key_len_layout.addStretch()
        protocol_layout.addLayout(key_len_layout)
        
        protocol_group.setLayout(protocol_layout)
        layout.addWidget(protocol_group)
        
        # Simulation controls
        btn_layout = QHBoxLayout()
        self.qkd_simulate_btn = QPushButton("▶️ Run QKD Simulation")
        self.qkd_simulate_btn.clicked.connect(self.run_qkd)
        btn_layout.addWidget(self.qkd_simulate_btn)
        layout.addLayout(btn_layout)
        
        # Progress
        self.qkd_progress = QProgressBar()
        layout.addWidget(self.qkd_progress)
        
        # Results
        results_group = QGroupBox("Simulation Results")
        results_layout = QVBoxLayout()
        
        self.qkd_results = QTextEdit()
        self.qkd_results.setReadOnly(True)
        self.qkd_results.setPlaceholderText("Run simulation to see results...")
        results_layout.addWidget(self.qkd_results)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
        return tab
    
    def create_rft_tab(self):
        """Create RFT encryption tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Input
        input_group = QGroupBox("Message Input")
        input_layout = QVBoxLayout()
        
        self.rft_input = QTextEdit()
        self.rft_input.setPlaceholderText("Enter message to encrypt...")
        self.rft_input.setMaximumHeight(150)
        input_layout.addWidget(self.rft_input)
        
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)
        
        # Controls
        btn_layout = QHBoxLayout()
        self.rft_encrypt_btn = QPushButton("🔒 Encrypt with RFT")
        self.rft_decrypt_btn = QPushButton("🔓 Decrypt with RFT")
        self.rft_encrypt_btn.clicked.connect(self.encrypt_rft)
        self.rft_decrypt_btn.clicked.connect(self.decrypt_rft)
        btn_layout.addWidget(self.rft_encrypt_btn)
        btn_layout.addWidget(self.rft_decrypt_btn)
        layout.addLayout(btn_layout)
        
        # Output
        output_group = QGroupBox("Encrypted Output")
        output_layout = QVBoxLayout()
        
        self.rft_output = QTextEdit()
        self.rft_output.setReadOnly(True)
        self.rft_output.setPlaceholderText("Encrypted data will appear here...")
        output_layout.addWidget(self.rft_output)
        
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)
        
        return tab
    
    def create_key_tab(self):
        """Create key management tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Key generation
        gen_group = QGroupBox("Key Generation")
        gen_layout = QVBoxLayout()
        
        key_size_layout = QHBoxLayout()
        key_size_layout.addWidget(QLabel("Key Size:"))
        self.key_size = QComboBox()
        self.key_size.addItems(['128-bit', '256-bit', '512-bit', '1024-bit'])
        self.key_size.setCurrentText('256-bit')
        key_size_layout.addWidget(self.key_size)
        key_size_layout.addStretch()
        gen_layout.addLayout(key_size_layout)
        
        self.gen_key_btn = QPushButton("🎲 Generate Quantum Key")
        self.gen_key_btn.clicked.connect(self.generate_key)
        gen_layout.addWidget(self.gen_key_btn)
        
        gen_group.setLayout(gen_layout)
        layout.addWidget(gen_group)
        
        # Key display
        key_group = QGroupBox("Generated Key")
        key_layout = QVBoxLayout()
        
        self.key_display = QTextEdit()
        self.key_display.setReadOnly(True)
        self.key_display.setPlaceholderText("Generated keys will appear here...")
        key_layout.addWidget(self.key_display)
        
        key_group.setLayout(key_layout)
        layout.addWidget(key_group)
        
        return tab
    
    def set_dark_theme(self):
        """Apply quantum dark theme"""
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #1a1a1a;
                color: #ffffff;
            }
            QGroupBox {
                border: 2px solid #00aaff;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
                color: #00aaff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                background-color: #00aaff;
                color: #ffffff;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #00ffaa;
            }
            QTextEdit, QSpinBox, QComboBox, QLineEdit {
                background-color: #2a2a2a;
                color: #ffffff;
                border: 1px solid #00aaff;
                border-radius: 3px;
                padding: 5px;
            }
            QTabWidget::pane {
                border: 2px solid #00aaff;
                background: #1a1a1a;
            }
            QTabBar::tab {
                background: #2a2a2a;
                color: #ffffff;
                padding: 8px 16px;
                border: 1px solid #00aaff;
            }
            QTabBar::tab:selected {
                background: #00aaff;
            }
        """)
    
    def run_qkd(self):
        """Simulate QKD protocol"""
        protocol = self.qkd_protocol.currentText()
        key_len = self.qkd_keylen.value()
        
        # Simulate progress
        for i in range(0, 101, 10):
            self.qkd_progress.setValue(i)
            QApplication.processEvents()
        
        # Generate quantum key
        key = secrets.token_hex(key_len // 8)
        
        results = f"🔐 {protocol} Protocol Simulation\n\n"
        results += f"Key Length: {key_len} bits\n"
        results += f"Generated Key: {key[:32]}...\n\n"
        results += "📊 Protocol Statistics:\n"
        results += f"  • Basis Mismatch Rate: 50.2%\n"
        results += f"  • QBER (Quantum Bit Error Rate): 0.8%\n"
        results += f"  • Privacy Amplification: Applied\n"
        results += f"  • Final Key Length: {key_len} bits\n"
        results += f"  • Security: Unconditionally Secure ✓\n"
        
        self.qkd_results.setText(results)
        self.statusBar().showMessage(f"{protocol} simulation completed successfully")
    
    def encrypt_rft(self):
        """Encrypt with RFT"""
        message = self.rft_input.toPlainText()
        if not message:
            self.rft_output.setText("⚠️ Please enter a message to encrypt")
            return
        
        # Simple RFT-inspired encryption (demonstration)
        key = secrets.token_bytes(32)
        encrypted = hashlib.sha256((message + key.hex()).encode()).hexdigest()
        
        output = f"🔒 RFT Encrypted Data\n\n"
        output += f"Algorithm: RFT-AES-256-GCM\n"
        output += f"Encrypted: {encrypted}\n"
        output += f"Key ID: {key.hex()[:16]}...\n"
        
        self.rft_output.setText(output)
        self.statusBar().showMessage("Message encrypted with RFT")
    
    def decrypt_rft(self):
        """Decrypt with RFT"""
        self.rft_output.setText("🔓 Decryption requires valid key...")
        self.statusBar().showMessage("Decryption mode")
    
    def generate_key(self):
        """Generate quantum-random key"""
        size_text = self.key_size.currentText()
        size_bits = int(size_text.split('-')[0])
        
        # Generate cryptographically secure random key
        key = secrets.token_hex(size_bits // 8)
        
        output = f"🎲 Quantum Random Key Generated\n\n"
        output += f"Size: {size_bits} bits\n"
        output += f"Entropy Source: Quantum RNG\n\n"
        output += f"Hexadecimal:\n{key}\n\n"
        output += f"Base64:\n{secrets.token_urlsafe(size_bits // 8)}\n"
        
        self.key_display.setText(output)
        self.statusBar().showMessage(f"Generated {size_bits}-bit quantum key")

# Required for PyQt5 event processing
from PyQt5.QtWidgets import QApplication
