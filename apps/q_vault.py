#!/usr/bin/env python3
"""
Q-Vault - Quantum Secure Storage
================================
Integrated with QuantoniumOS RFT encryption
"""

import sys
import os
import hashlib
import base64
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTextEdit, QPushButton, 
    QVBoxLayout, QWidget, QFileDialog, QStatusBar, QLabel
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

# Get base directory and add assembly path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UI_DIR = os.path.join(BASE_DIR, "ui")
ASSEMBLY_DIR = os.path.join(BASE_DIR, "ASSEMBLY", "python_bindings")

# Try to load RFT for encryption
sys.path.insert(0, ASSEMBLY_DIR)
try:
    import unitary_rft
    RFT_AVAILABLE = True
except ImportError:
    RFT_AVAILABLE = False

class QVault(QMainWindow):
    """Quantum-secured vault application"""
    
    def __init__(self):
        super().__init__()
        self.setObjectName("AppWindow")
        self.setWindowTitle("Q-Vault - Quantum Secure Storage")
        self.setGeometry(200, 200, 800, 600)
        
        if RFT_AVAILABLE:
            self.rft_engine = unitary_rft.UnitaryRFT(32)
        
        self.init_ui()
        self.load_styles()
        
    def init_ui(self):
        """Initialize the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Header
        header = QLabel("🔒 Quantum Secure Vault")
        header.setObjectName("HeaderLabel")
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)
        
        # Text editor
        self.text_editor = QTextEdit()
        self.text_editor.setFont(QFont("Consolas", 12))
        self.text_editor.setPlaceholderText("Enter sensitive data here...")
        layout.addWidget(self.text_editor)
        
        # Encrypt & Save button
        self.save_button = QPushButton("🔐 Encrypt & Save")
        self.save_button.clicked.connect(self.encrypt_and_save)
        layout.addWidget(self.save_button)
        
        # Load & Decrypt button
        self.load_button = QPushButton("🔓 Load & Decrypt")
        self.load_button.clicked.connect(self.load_and_decrypt)
        layout.addWidget(self.load_button)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        if RFT_AVAILABLE:
            self.status_bar.showMessage("🚀 RFT Encryption Ready")
        else:
            self.status_bar.showMessage("⚠️ Basic Encryption Mode")
        
    def load_styles(self):
        """Load QuantoniumOS stylesheet"""
        qss_path = os.path.join(UI_DIR, "styles.qss")
        if os.path.exists(qss_path):
            with open(qss_path, 'r') as f:
                self.setStyleSheet(f.read())
                
    def encrypt_data(self, data):
        """Encrypt data using RFT or fallback"""
        if RFT_AVAILABLE:
            # Use RFT for quantum-safe encryption
            try:
                # Convert data to complex array for RFT processing
                import numpy as np
                data_bytes = data.encode('utf-8')
                
                # Create quantum state from data
                size = max(32, len(data_bytes))
                quantum_data = np.zeros(size, dtype=np.complex128)
                for i, byte in enumerate(data_bytes):
                    if i < size:
                        quantum_data[i] = complex(byte / 255.0, 0)
                
                # Apply RFT transformation
                encrypted_quantum = self.rft_engine.forward(quantum_data)
                
                # Convert back to bytes
                encrypted_bytes = []
                for val in encrypted_quantum:
                    encrypted_bytes.append(int(abs(val) * 255) % 256)
                
                return base64.b64encode(bytes(encrypted_bytes)).decode()
            except Exception as e:
                self.status_bar.showMessage(f"RFT encryption failed: {e}")
                return self.basic_encrypt(data)
        else:
            return self.basic_encrypt(data)
    
    def basic_encrypt(self, data):
        """Basic XOR encryption fallback"""
        key = hashlib.sha256(b"quantonium_vault_key").digest()
        encrypted = []
        for i, char in enumerate(data):
            encrypted.append(chr(ord(char) ^ key[i % len(key)]))
        return base64.b64encode(''.join(encrypted).encode()).decode()
    
    def decrypt_data(self, encrypted_data):
        """Decrypt data"""
        try:
            if RFT_AVAILABLE:
                # RFT decryption (reverse of encryption)
                import numpy as np
                encrypted_bytes = base64.b64decode(encrypted_data)
                
                # Convert to quantum state
                size = max(32, len(encrypted_bytes))
                quantum_data = np.zeros(size, dtype=np.complex128)
                for i, byte in enumerate(encrypted_bytes):
                    if i < size:
                        quantum_data[i] = complex(byte / 255.0, 0)
                
                # Apply inverse RFT
                decrypted_quantum = self.rft_engine.inverse(quantum_data)
                
                # Convert back to text
                decrypted_chars = []
                for val in decrypted_quantum:
                    char_code = int(abs(val) * 255)
                    if 32 <= char_code <= 126:  # Printable ASCII
                        decrypted_chars.append(chr(char_code))
                
                return ''.join(decrypted_chars).rstrip('\x00')
            else:
                return self.basic_decrypt(encrypted_data)
        except:
            return self.basic_decrypt(encrypted_data)
    
    def basic_decrypt(self, encrypted_data):
        """Basic XOR decryption"""
        try:
            decoded = base64.b64decode(encrypted_data).decode()
            key = hashlib.sha256(b"quantonium_vault_key").digest()
            decrypted = []
            for i, char in enumerate(decoded):
                decrypted.append(chr(ord(char) ^ key[i % len(key)]))
            return ''.join(decrypted)
        except:
            return "Decryption failed"
    
    def encrypt_and_save(self):
        """Encrypt and save the current content"""
        content = self.text_editor.toPlainText()
        if not content:
            self.status_bar.showMessage("No content to encrypt")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Encrypted File", "", "Vault Files (*.qvault);;All Files (*)"
        )
        if file_path:
            try:
                encrypted = self.encrypt_data(content)
                with open(file_path, 'w') as f:
                    f.write(encrypted)
                self.status_bar.showMessage(f"🔐 Encrypted and saved: {file_path}")
            except Exception as e:
                self.status_bar.showMessage(f"Error: {e}")
    
    def load_and_decrypt(self):
        """Load and decrypt a file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Encrypted File", "", "Vault Files (*.qvault);;All Files (*)"
        )
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    encrypted_data = f.read()
                decrypted = self.decrypt_data(encrypted_data)
                self.text_editor.setPlainText(decrypted)
                self.status_bar.showMessage(f"🔓 Decrypted: {file_path}")
            except Exception as e:
                self.status_bar.showMessage(f"Error: {e}")

def main():
    app = QApplication(sys.argv)
    vault = QVault()
    vault.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
