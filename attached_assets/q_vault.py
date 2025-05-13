import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTextEdit, QPushButton, QVBoxLayout, QWidget, QFileDialog, QMessageBox
)

# Set path to styles.qss in the same directory
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
STYLES_QSS = os.path.join(ROOT_DIR, "styles.qss")

def load_stylesheet(qss_path):
    """Load the stylesheet from the given path, with fallback if not found."""
    if os.path.exists(qss_path):
        try:
            with open(qss_path, "r", encoding="utf-8") as f:
                print(f"✅ Stylesheet loaded from {qss_path}")
                return f.read()
        except UnicodeDecodeError as e:
            print(f"⚠️ Error decoding stylesheet from {qss_path}: {e}")
            print(f"Position: {e.start}, Character: {e.object[e.start]}")
            return ""
    print(f"⚠️ Stylesheet not found: {qss_path}")
    return ""

class QVault(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setObjectName("QVault")  # Set object name for QSS targeting
        self.setWindowTitle("Q-Vault")

        # Load stylesheet
        self.stylesheet = load_stylesheet(STYLES_QSS)
        if not self.stylesheet:
            print("⚠️ No stylesheet available, proceeding with default styles")
        else:
            self.setStyleSheet(self.stylesheet)

        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)
        self.mainLayout = QVBoxLayout(self.centralWidget)

        self.textEditor = QTextEdit()
        self.textEditor.setObjectName("TextEditor")  # Set object name for QSS
        self.mainLayout.addWidget(self.textEditor)

        self.saveButton = QPushButton("Encrypt & Save")
        self.saveButton.setObjectName("SaveButton")  # Set object name for QSS
        self.saveButton.clicked.connect(self.encryptAndSave)
        self.mainLayout.addWidget(self.saveButton)

    def encryptAndSave(self):
        text = self.textEditor.toPlainText()
        encrypted_text = "".join(chr(ord(c) + 3) for c in text)  # Simple Caesar Cipher encryption

        options = QFileDialog.Options()
        filePath, _ = QFileDialog.getSaveFileName(self, "Save Encrypted File", "", "Vault Files (*.vault);;All Files (*)", options=options)
        if filePath:
            with open(filePath, "w", encoding="utf-8") as file:
                file.write(encrypted_text)
            QMessageBox.information(self, "Success", "File saved and encrypted successfully!")

if __name__ == "__main__":
    # Import and use the headless environment setup
    from attached_assets import setup_headless_environment
    env_config = setup_headless_environment()
    print(f"Running on {env_config['platform']} in {'headless' if env_config['headless'] else 'windowed'} mode")
    
    app = QApplication(sys.argv)
    stylesheet = load_stylesheet(STYLES_QSS)
    app.setStyleSheet(stylesheet)
    vaultApp = QVault()
    vaultApp.show()
    sys.exit(app.exec_())
