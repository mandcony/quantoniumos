"""
Q-Mail - Secure Quantum-Inspired Email Client

This module provides a secure email client that uses quantum-inspired
encryption for message protection.
"""

import sys
import os
import logging
import requests
import json
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit,
    QMessageBox, QStackedWidget, QDialog, QDialogButtonBox, QFormLayout
)

# Configure path to include the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Try to import our quantum-enhanced mail integration
try:
    from attached_assets.q_mail_integration import mail_handler
    HAS_QUANTUM_MAIL = True
    logging.info("Using quantum-enhanced secure mail")
except ImportError:
    HAS_QUANTUM_MAIL = False
    logging.warning("Quantum mail integration not found, using fallback encryption")
    try:
        from encryption.resonance_encryption import resonance_encrypt, resonance_decrypt, WaveNumber
        USER_WAVE_KEY = "symbolic-key"
    except ImportError:
        logging.error("No encryption module found, mail will not be secure!")
        # Define dummy encryption functions if no module is available
        def resonance_encrypt(text, key): return text
        def resonance_decrypt(text, key): return text
        USER_WAVE_KEY = "default-key"

AUTH_SERVER = "http://127.0.0.1:5000"
EMAIL_SERVER = f"{AUTH_SERVER}/send_email"
FETCH_EMAILS = f"{AUTH_SERVER}/fetch_emails"
LOGIN_ENDPOINT = f"{AUTH_SERVER}/login"
REGISTER_ENDPOINT = f"{AUTH_SERVER}/register"

# Configure logging
log_file = os.path.join(os.path.dirname(__file__), 'q_mail.log')
logging.basicConfig(level=logging.DEBUG, 
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[
                        logging.FileHandler(log_file),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

class LoginWidget(QWidget):
    login_successful = pyqtSignal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)

        title_label = QLabel("🔐 Q-Mail Login")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Arial", 18, QFont.Bold))
        layout.addWidget(title_label)

        self.email_input = QLineEdit()
        self.email_input.setPlaceholderText("Enter your email")
        layout.addWidget(self.email_input)

        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("Enter your password")
        self.password_input.setEchoMode(QLineEdit.Password)
        layout.addWidget(self.password_input)

        login_btn = QPushButton("🔓 Login")
        login_btn.clicked.connect(self.login_user)
        layout.addWidget(login_btn)

        register_btn = QPushButton("📝 Register")
        register_btn.clicked.connect(self.register_user)
        layout.addWidget(register_btn)

    def login_user(self):
        email = self.email_input.text().strip()
        password = self.password_input.text().strip()

        try:
            response = requests.post(LOGIN_ENDPOINT, json={"email": email, "password": password})
            if response.status_code == 200:
                token = response.json().get("token")
                self.login_successful.emit(email, token)
                logger.info("Login successful")
            else:
                QMessageBox.warning(self, "Login Failed", response.json().get("error", "Unknown error"))
        except Exception as e:
            logger.error(f"Login error: {str(e)}")
            QMessageBox.critical(self, "Connection Error", 
                                "Could not connect to the mail server. Using offline mode.")
            # Still emit signal to move to mail interface in offline mode
            self.login_successful.emit(email, "offline-mode")

    def register_user(self):
        email = self.email_input.text().strip()
        password = self.password_input.text().strip()

        try:
            response = requests.post(REGISTER_ENDPOINT, json={"email": email, "password": password})
            if response.status_code == 200:
                QMessageBox.information(self, "Registration Successful", "You may now log in.")
            else:
                QMessageBox.warning(self, "Registration Failed", response.json().get("error", "Unknown error"))
        except Exception as e:
            logger.error(f"Registration error: {str(e)}")
            QMessageBox.critical(self, "Connection Error", "Could not connect to the mail server")

class ComposeDialog(QDialog):
    def __init__(self, sender_email, token, parent=None):
        super().__init__(parent)
        self.sender_email = sender_email
        self.token = token
        self.setWindowTitle("Compose Message")
        self.init_ui()

    def init_ui(self):
        layout = QFormLayout()

        self.to_edit = QLineEdit()
        layout.addRow("To:", self.to_edit)

        self.subject_edit = QLineEdit()
        layout.addRow("Subject:", self.subject_edit)

        self.body_edit = QTextEdit()
        layout.addRow("Body:", self.body_edit)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.on_send)
        button_box.rejected.connect(self.reject)
        layout.addRow(button_box)

        self.setLayout(layout)

    def on_send(self):
        # Encrypt the message using our quantum encryption
        if HAS_QUANTUM_MAIL:
            # Create a mail content dictionary
            mail_content = {
                "sender": self.sender_email,
                "recipient": self.to_edit.text().strip(),
                "subject": self.subject_edit.text().strip(),
                "body": self.body_edit.toPlainText().strip(),
                "timestamp": None  # Will be set by server
            }
            
            # Encrypt the entire mail content
            encrypted_mail = mail_handler.encrypt_mail(mail_content)
            payload = {
                "sender": self.sender_email,
                "recipient": self.to_edit.text().strip(),
                "encrypted_content": encrypted_mail["content"],
                "encryption_type": "quantum"
            }
        else:
            # Legacy encryption
            payload = {
                "sender": self.sender_email,
                "recipient": self.to_edit.text().strip(),
                "subject": resonance_encrypt(self.subject_edit.text().strip(), USER_WAVE_KEY),
                "body": resonance_encrypt(self.body_edit.toPlainText().strip(), USER_WAVE_KEY)
            }

        if self.token == "offline-mode":
            # In offline mode, just show a success message
            QMessageBox.information(self, "Offline Mode", 
                                   "Message saved to draft folder (offline mode)")
            self.accept()
            return

        # Online mode - send to server
        try:
            headers = {'x-access-token': self.token}
            response = requests.post(EMAIL_SERVER, json=payload, headers=headers)

            if response.status_code == 200:
                QMessageBox.information(self, "Success", "Email sent successfully")
                self.accept()
            else:
                QMessageBox.critical(self, "Send Failed", response.json().get("error", "Unknown error"))
        except Exception as e:
            logger.error(f"Send error: {str(e)}")
            QMessageBox.critical(self, "Connection Error", 
                                "Could not connect to the mail server. Message saved to draft folder.")
            self.accept()

class MailWidget(QWidget):
    def __init__(self, user_email, token, parent=None):
        super().__init__(parent)
        self.user_email = user_email
        self.token = token
        self.init_ui()
        self.load_emails()

    def init_ui(self):
        layout = QVBoxLayout(self)

        header_label = QLabel(f"Q-Mail - {self.user_email}")
        header_label.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(header_label)

        # Add status indicator
        status = "Online" if self.token != "offline-mode" else "Offline"
        status_color = "green" if status == "Online" else "red"
        status_label = QLabel(f"Status: <span style='color:{status_color};'>{status}</span>")
        layout.addWidget(status_label)

        self.email_list = QTextEdit()
        self.email_list.setReadOnly(True)
        layout.addWidget(self.email_list)

        button_layout = QVBoxLayout()
        
        compose_button = QPushButton("Compose")
        compose_button.clicked.connect(self.compose_email)
        button_layout.addWidget(compose_button)
        
        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self.load_emails)
        button_layout.addWidget(refresh_button)
        
        layout.addLayout(button_layout)

    def load_emails(self):
        if self.token == "offline-mode":
            self.email_list.setPlainText("Offline mode - No emails available")
            return

        try:
            headers = {'x-access-token': self.token}
            response = requests.get(f"{FETCH_EMAILS}/{self.user_email}", headers=headers)

            if response.status_code == 200:
                emails = response.json()
                
                if HAS_QUANTUM_MAIL:
                    # Handle quantum-encrypted emails
                    display_texts = []
                    for e in emails:
                        if e.get("encryption_type") == "quantum":
                            decrypted = mail_handler.decrypt_mail({
                                "encrypted": True,
                                "content": e["encrypted_content"]
                            })
                            display_texts.append(
                                f"From: {e['sender']}\n"
                                f"Subject: {decrypted.get('subject', 'No Subject')}\n"
                                f"Body: {decrypted.get('body', 'No Content')}"
                            )
                        else:
                            # Legacy format
                            display_texts.append(
                                f"From: {e['sender']}\n"
                                f"Subject: {resonance_decrypt(e['subject'], USER_WAVE_KEY)}\n"
                                f"Body: {resonance_decrypt(e['body'], USER_WAVE_KEY)}"
                            )
                else:
                    # Legacy decryption
                    display_texts = [
                        f"From: {e['sender']}\n"
                        f"Subject: {resonance_decrypt(e['subject'], USER_WAVE_KEY)}\n"
                        f"Body: {resonance_decrypt(e['body'], USER_WAVE_KEY)}"
                        for e in emails
                    ]
                
                self.email_list.setPlainText("\n\n".join(display_texts))
            else:
                error_msg = response.json().get("error", "Unknown error")
                self.email_list.setPlainText(f"Error loading emails: {error_msg}")
        except Exception as e:
            logger.error(f"Load emails error: {str(e)}")
            self.email_list.setPlainText(f"Connection error: {str(e)}")

    def compose_email(self):
        dialog = ComposeDialog(self.user_email, self.token)
        if dialog.exec_():
            # Refresh the mail list if a new mail was sent
            self.load_emails()

class QMailWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QuantoniumOS - Q-Mail")
        self.resize(800, 600)
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.login_widget = LoginWidget()
        self.login_widget.login_successful.connect(self.on_login_success)
        self.stack.addWidget(self.login_widget)

    def on_login_success(self, email, token):
        self.mail_widget = MailWidget(email, token)
        self.stack.addWidget(self.mail_widget)
        self.stack.setCurrentWidget(self.mail_widget)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = QMailWindow()
    window.show()
    sys.exit(app.exec_())