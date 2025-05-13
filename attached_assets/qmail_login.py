import sys
import requests
import json
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, 
    QLineEdit, QMessageBox, QLabel
)

AUTH_SERVER = "http://127.0.0.1:5000"
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "user_config.json")

class QMailLogin(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Q-Mail Login / Registration")
        self.setGeometry(100, 100, 400, 250)

        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)
        self.mainLayout = QVBoxLayout(self.centralWidget)

        self.infoLabel = QLabel("Enter your email:")
        self.mainLayout.addWidget(self.infoLabel)

        self.emailInput = QLineEdit()
        self.emailInput.setPlaceholderText("e.g., user@quantonium.os")
        self.mainLayout.addWidget(self.emailInput)

        self.passwordInput = QLineEdit()
        self.passwordInput.setPlaceholderText("Enter your password")
        self.passwordInput.setEchoMode(QLineEdit.Password)
        self.mainLayout.addWidget(self.passwordInput)

        self.loginBtn = QPushButton("🔐 Login")
        self.loginBtn.clicked.connect(self.loginUser)
        self.mainLayout.addWidget(self.loginBtn)

        self.registerBtn = QPushButton("📝 Register")
        self.registerBtn.clicked.connect(self.registerUser)
        self.mainLayout.addWidget(self.registerBtn)

    def loginUser(self):
        """Logs in the user and stores session"""
        email, password = self.emailInput.text().strip(), self.passwordInput.text().strip()
        if not email or not password:
            QMessageBox.warning(self, "Login Failed", "Please enter email and password")
            return

        try:
            response = requests.post(f"{AUTH_SERVER}/login", json={"email": email, "password": password})
            
            if response.status_code == 200:
                data = response.json()
                if "message" in data:
                    with open(CONFIG_PATH, "w") as f:
                        json.dump({"user_email": email}, f)
                    QMessageBox.information(self, "Success", "Login successful!")
                    self.close()
                    os.system("python q_mail.py")  # Launch Q-Mail App
                else:
                    QMessageBox.critical(self, "Login Failed", "Unexpected server response.")
            else:
                error_message = response.json().get("error", "Invalid credentials")
                QMessageBox.critical(self, "Login Failed", error_message)
        
        except requests.exceptions.RequestException as e:
            QMessageBox.critical(self, "Server Error", f"Could not connect to email server: {e}")

    def registerUser(self):
        """Registers a new user"""
        email, password = self.emailInput.text().strip(), self.passwordInput.text().strip()
        if not email or not password:
            QMessageBox.warning(self, "Error", "Please enter email and password")
            return

        try:
            response = requests.post(f"{AUTH_SERVER}/register", json={"email": email, "password": password})
            
            if response.status_code == 201:
                QMessageBox.information(self, "Success", "User registered successfully!")
            else:
                error_message = response.json().get("error", "Unknown error")
                QMessageBox.critical(self, "Error", error_message)
        
        except requests.exceptions.RequestException as e:
            QMessageBox.critical(self, "Server Error", f"Could not connect to email server: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    loginWindow = QMailLogin()
    loginWindow.show()
    sys.exit(app.exec_())
