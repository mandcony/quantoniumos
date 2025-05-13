import sys
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

# Use relative imports based on context
import os
import sys
# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now we can import from the root project
from encryption.resonance_encryption import resonance_encrypt, resonance_decrypt 
from encryption.wave_primitives import WaveNumber

AUTH_SERVER = "http://127.0.0.1:5000"
EMAIL_SERVER = f"{AUTH_SERVER}/send_email"
FETCH_EMAILS = f"{AUTH_SERVER}/fetch_emails"
LOGIN_ENDPOINT = f"{AUTH_SERVER}/login"
REGISTER_ENDPOINT = f"{AUTH_SERVER}/register"
AUTH_SERVER = "http://127.0.0.1:5000"
EMAIL_SERVER = f"{AUTH_SERVER}/send_email"
FETCH_EMAILS = f"{AUTH_SERVER}/fetch_emails"
LOGIN_ENDPOINT = f"{AUTH_SERVER}/login"
REGISTER_ENDPOINT = f"{AUTH_SERVER}/register"

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

USER_WAVE_KEY = WaveNumber(amplitude=1.23, phase=0.45)

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

        response = requests.post(LOGIN_ENDPOINT, json={"email": email, "password": password})
        if response.status_code == 200:
            token = response.json().get("token")
            self.login_successful.emit(email, token)
            logger.info("Login successful")
        else:
            QMessageBox.warning(self, "Login Failed", response.json().get("error"))

    def register_user(self):
        email = self.email_input.text().strip()
        password = self.password_input.text().strip()

        response = requests.post(REGISTER_ENDPOINT, json={"email": email, "password": password})
        if response.status_code == 200:
            QMessageBox.information(self, "Registration Successful", "You may now log in.")
        else:
            QMessageBox.warning(self, "Registration Failed", response.json().get("error"))

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
        payload = {
            "sender": self.sender_email,
            "recipient": self.to_edit.text().strip(),
            "subject": resonance_encrypt(self.subject_edit.text().strip(), USER_WAVE_KEY),
            "body": resonance_encrypt(self.body_edit.toPlainText().strip(), USER_WAVE_KEY)
        }

        headers = {'x-access-token': self.token}
        response = requests.post(EMAIL_SERVER, json=payload, headers=headers)

        if response.status_code == 200:
            QMessageBox.information(self, "Success", "Email sent successfully")
            self.accept()
        else:
            QMessageBox.critical(self, "Send Failed", response.json().get("error"))

class MailWidget(QWidget):
    def __init__(self, user_email, token, parent=None):
        super().__init__(parent)
        self.user_email = user_email
        self.token = token
        self.init_ui()
        self.load_emails()

    def init_ui(self):
        layout = QVBoxLayout(self)

        self.email_list = QTextEdit()
        layout.addWidget(self.email_list)

        compose_button = QPushButton("Compose")
        compose_button.clicked.connect(self.compose_email)
        layout.addWidget(compose_button)

    def load_emails(self):
        headers = {'x-access-token': self.token}
        response = requests.get(f"{FETCH_EMAILS}/{self.user_email}", headers=headers)

        if response.status_code == 200:
            emails = response.json()
            display_text = "\n\n".join([
                f"From: {e['sender']}\nSubject: {resonance_decrypt(e['subject'], USER_WAVE_KEY)}\nBody: {resonance_decrypt(e['body'], USER_WAVE_KEY)}"
                for e in emails
            ])
            self.email_list.setPlainText(display_text)
        else:
            QMessageBox.critical(self, "Error", response.json().get("error"))

    def compose_email(self):
        dialog = ComposeDialog(self.user_email, self.token)
        dialog.exec_()

class QMailWindow(QMainWindow):
    def __init__(self):
        super().__init__()
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
    # Import and use the headless environment setup
    from attached_assets import setup_headless_environment
    env_config = setup_headless_environment()
    print(f"Running on {env_config['platform']} in {'headless' if env_config['headless'] else 'windowed'} mode")
    
    app = QApplication(sys.argv)
    window = QMailWindow()
    window.show()
    sys.exit(app.exec_())
