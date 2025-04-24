import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QTextEdit, QPushButton

# Add the root directory (C:\quantonium_os\) to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

from config import Config

def simple_resonance_encrypt(data: str) -> str:
    cfg = Config()
    amplitude = cfg.data.get("resonance_frequency", 0.8)
    val = int(abs(amplitude) * 255)
    return "".join(chr(ord(c) ^ val) for c in data)

class QVault(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Q-Vault - Resonance-Encrypted Storage")
        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)
        self.mainLayout = QVBoxLayout(self.centralWidget)
        self.mainLayout.addWidget(QLabel("Store secrets safely"))
        self.secretText = QTextEdit()
        self.mainLayout.addWidget(self.secretText)
        self.saveBtn = QPushButton("Encrypt & Save")
        self.saveBtn.clicked.connect(self.encryptAndSave)
        self.mainLayout.addWidget(self.saveBtn)
        self.setStyleSheet(open(os.path.join(ROOT_DIR, "styles.qss"), "r").read())

    def encryptAndSave(self):
        data = self.secretText.toPlainText()
        enc = simple_resonance_encrypt(data)
        print(f"[Q-Vault] Data stored (encrypted): {enc[:50]}...")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    vault = QVault()
    vault.show()
    sys.exit(app.exec_())