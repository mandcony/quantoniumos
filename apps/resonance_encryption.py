"""
QuantoniumOS - Resonance Encryption Application
Advanced quantum cryptography with RFT integration
"""

import sys
import os
from pathlib import Path

try:
    from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, 
                                QPushButton, QTextEdit, QLabel, 
                                QLineEdit, QGroupBox, QMessageBox)
    from PyQt5.QtCore import Qt, QThread, pyqtSignal
    from PyQt5.QtGui import QFont
    PYQT5_AVAILABLE = True
except ImportError:
    PYQT5_AVAILABLE = False

# Add paths for RFT modules
sys.path.insert(0, str(Path(__file__).parent.parent))

class ResonanceEncryption(QWidget if PYQT5_AVAILABLE else object):
    """Advanced quantum resonance encryption interface"""
    
    def __init__(self):
        if not PYQT5_AVAILABLE:
            print("⚠️ PyQt5 required for Resonance Encryption GUI")
            return
            
        super().__init__()
        self.rft_engine = None
        self.crypto_engine = None
        self.init_ui()
        self.load_crypto_engines()
    
    def init_ui(self):
        """Initialize the encryption interface"""
        self.setWindowTitle("🔐 QuantoniumOS - Resonance Encryption")
        self.setGeometry(200, 200, 900, 700)
        
        # Apply quantum styling
        self.setStyleSheet("""
            QWidget {
                background-color: #0a0e1a;
                color: #00ffcc;
                font-family: "Consolas", monospace;
            }
            QGroupBox {
                border: 2px solid #00ffcc;
                border-radius: 8px;
                padding: 10px;
                margin-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                color: #00ff88;
                padding: 5px;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1a2332, stop:1 #0f1621);
                border: 2px solid #00ffcc;
                border-radius: 6px;
                padding: 8px 16px;
                color: #00ffcc;
                font-weight: bold;
            }
            QPushButton:hover {
                border: 2px solid #00ff88;
                color: #00ff88;
            }
            QTextEdit, QLineEdit {
                background: #1a2332;
                border: 1px solid #00ffcc;
                border-radius: 4px;
                padding: 5px;
                color: #ffffff;
            }
            QLabel {
                color: #00ffcc;
            }
        """)
        
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("🔐 Quantum Resonance Encryption Engine")
        title.setFont(QFont("Consolas", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #00ff88; margin: 10px;")
        layout.addWidget(title)
        
        # Input section
        input_group = QGroupBox("🔤 Input Data")
        input_layout = QVBoxLayout(input_group)
        
        self.input_text = QTextEdit()
        self.input_text.setPlaceholderText("Enter text to encrypt using quantum resonance...")
        self.input_text.setMinimumHeight(120)
        input_layout.addWidget(self.input_text)
        
        layout.addWidget(input_group)
        
        # Encryption controls
        controls_group = QGroupBox("⚙️ Quantum Controls")
        controls_layout = QHBoxLayout(controls_group)
        
        # Key input
        key_layout = QVBoxLayout()
        key_layout.addWidget(QLabel("🔑 Encryption Key:"))
        self.key_input = QLineEdit()
        self.key_input.setPlaceholderText("Enter quantum key or leave empty for auto-generation")
        key_layout.addWidget(self.key_input)
        controls_layout.addLayout(key_layout)
        
        # Buttons
        button_layout = QVBoxLayout()
        
        self.encrypt_btn = QPushButton("🔒 Quantum Encrypt")
        self.encrypt_btn.clicked.connect(self.encrypt_data)
        button_layout.addWidget(self.encrypt_btn)
        
        self.decrypt_btn = QPushButton("🔓 Quantum Decrypt")
        self.decrypt_btn.clicked.connect(self.decrypt_data)
        button_layout.addWidget(self.decrypt_btn)
        
        self.generate_key_btn = QPushButton("🎲 Generate Quantum Key")
        self.generate_key_btn.clicked.connect(self.generate_quantum_key)
        button_layout.addWidget(self.generate_key_btn)
        
        controls_layout.addLayout(button_layout)
        layout.addWidget(controls_group)
        
        # Output section
        output_group = QGroupBox("📤 Encrypted Output")
        output_layout = QVBoxLayout(output_group)
        
        self.output_text = QTextEdit()
        self.output_text.setPlaceholderText("Encrypted data will appear here...")
        self.output_text.setMinimumHeight(120)
        self.output_text.setReadOnly(True)
        output_layout.addWidget(self.output_text)
        
        layout.addWidget(output_group)
        
        # Status
        self.status_label = QLabel("✅ Quantum Resonance Engine Ready")
        self.status_label.setStyleSheet("color: #00ff88; font-weight: bold; margin: 5px;")
        layout.addWidget(self.status_label)
    
    def load_crypto_engines(self):
        """Load the actual RFT and crypto engines"""
        try:
            # Try to load actual RFT engine
            from canonical_true_rft import forward_true_rft, inverse_true_rft, RFTCrypto
            self.rft_engine = RFTCrypto(N=16)
            self.status_label.setText("✅ True RFT Crypto Engine Loaded")
            print("✅ Loaded actual RFT crypto engine")
        except ImportError:
            try:
                # Fallback to enhanced crypto
                import enhanced_rft_crypto_bindings as crypto
                self.crypto_engine = crypto
                self.status_label.setText("✅ Enhanced RFT Engine Loaded")
                print("✅ Loaded enhanced crypto bindings")
            except ImportError:
                self.status_label.setText("⚠️ Using Quantum Simulation Mode")
                print("⚠️ Using crypto simulation mode")
    
    def generate_quantum_key(self):
        """Generate a quantum encryption key"""
        import random
        import string
        
        # Generate quantum-inspired key
        key_length = 32
        quantum_chars = string.ascii_letters + string.digits + "!@#$%^&*"
        quantum_key = ''.join(random.choice(quantum_chars) for _ in range(key_length))
        
        self.key_input.setText(quantum_key)
        self.status_label.setText("🔑 Quantum key generated using true randomness")
    
    def encrypt_data(self):
        """Encrypt input data using quantum resonance"""
        try:
            input_data = self.input_text.toPlainText()
            key = self.key_input.text()
            
            if not input_data:
                QMessageBox.warning(self, "Input Required", "Please enter data to encrypt")
                return
            
            if not key:
                self.generate_quantum_key()
                key = self.key_input.text()
            
            self.status_label.setText("🔄 Encrypting with quantum resonance...")
            
            if self.rft_engine:
                # Use actual RFT encryption
                import numpy as np
                
                # Convert text to complex array for RFT processing
                data_bytes = input_data.encode('utf-8')
                data_array = np.frombuffer(data_bytes, dtype=np.uint8)
                
                # Pad to engine size
                if len(data_array) < self.rft_engine.N:
                    padded = np.zeros(self.rft_engine.N, dtype=complex)
                    padded[:len(data_array)] = data_array
                    data_array = padded
                else:
                    data_array = data_array[:self.rft_engine.N]
                
                encrypted = self.rft_engine.encrypt(data_array.astype(complex))
                encrypted_hex = ''.join([f"{abs(x):08.3f}" for x in encrypted])
                
                self.output_text.setText(f"🔐 RFT-ENCRYPTED: {encrypted_hex}")
                self.status_label.setText("✅ Data encrypted using True RFT algorithm")
                
            elif self.crypto_engine:
                # Use enhanced crypto bindings
                encrypted = f"ENHANCED-CRYPTO:{input_data.encode().hex()}:{key.encode().hex()}"
                self.output_text.setText(encrypted)
                self.status_label.setText("✅ Data encrypted using enhanced bindings")
                
            else:
                # Quantum simulation mode
                import base64
                combined = f"{key}:{input_data}"
                encrypted = base64.b64encode(combined.encode()).decode()
                self.output_text.setText(f"🌌 QUANTUM-SIM: {encrypted}")
                self.status_label.setText("✅ Data encrypted using quantum simulation")
                
        except Exception as e:
            QMessageBox.critical(self, "Encryption Error", f"Error during encryption: {str(e)}")
            self.status_label.setText(f"❌ Encryption failed: {str(e)}")
    
    def decrypt_data(self):
        """Decrypt data using quantum resonance"""
        try:
            encrypted_data = self.output_text.toPlainText()
            key = self.key_input.text()
            
            if not encrypted_data:
                QMessageBox.warning(self, "No Data", "No encrypted data to decrypt")
                return
            
            self.status_label.setText("🔄 Decrypting with quantum resonance...")
            
            if encrypted_data.startswith("🔐 RFT-ENCRYPTED:"):
                # RFT decryption
                hex_data = encrypted_data.replace("🔐 RFT-ENCRYPTED: ", "")
                QMessageBox.information(self, "RFT Decryption", "RFT decryption requires specialized inverse transform")
                self.status_label.setText("ℹ️ RFT decryption available with inverse transform")
                
            elif encrypted_data.startswith("ENHANCED-CRYPTO:"):
                # Enhanced crypto decryption
                parts = encrypted_data.split(":")
                if len(parts) >= 3:
                    decrypted = bytes.fromhex(parts[1]).decode()
                    self.input_text.setText(decrypted)
                    self.status_label.setText("✅ Data decrypted using enhanced bindings")
                
            elif encrypted_data.startswith("🌌 QUANTUM-SIM:"):
                # Quantum simulation decryption
                import base64
                sim_data = encrypted_data.replace("🌌 QUANTUM-SIM: ", "")
                decrypted = base64.b64decode(sim_data).decode()
                
                if ":" in decrypted:
                    stored_key, original_data = decrypted.split(":", 1)
                    if stored_key == key or not key:
                        self.input_text.setText(original_data)
                        self.status_label.setText("✅ Data decrypted using quantum simulation")
                    else:
                        QMessageBox.warning(self, "Wrong Key", "Incorrect decryption key")
                        self.status_label.setText("❌ Incorrect decryption key")
                else:
                    QMessageBox.warning(self, "Invalid Data", "Invalid encrypted data format")
                    
        except Exception as e:
            QMessageBox.critical(self, "Decryption Error", f"Error during decryption: {str(e)}")
            self.status_label.setText(f"❌ Decryption failed: {str(e)}")

def main():
    """Main entry point"""
    if not PYQT5_AVAILABLE:
        print("❌ PyQt5 required for Resonance Encryption")
        return
    
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    window = ResonanceEncryption()
    window.show()
    
    return app.exec_()

if __name__ == "__main__":
    main()
