import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit, QPushButton, QVBoxLayout, QWidget, QFileDialog

class QNotes(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QNotes")
        self.setGeometry(100, 100, 800, 600)

        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)
        self.mainLayout = QVBoxLayout(self.centralWidget)

        self.textEditor = QTextEdit()
        self.mainLayout.addWidget(self.textEditor)

        self.saveButton = QPushButton("Save Notes")
        self.mainLayout.addWidget(self.saveButton)
        self.saveButton.clicked.connect(self.save_notes)

    def save_notes(self):
        filePath, _ = QFileDialog.getSaveFileName(self, "Save Note", "", "Notes (*.txt);;All Files (*)")
        if filePath:
            with open(filePath, 'w', encoding='utf-8') as f:
                f.write(self.textEditor.toPlainText())

if __name__ == "__main__":
    # Import and use the headless environment setup
    from attached_assets import setup_headless_environment
    env_config = setup_headless_environment()
    print(f"Running on {env_config['platform']} in {'headless' if env_config['headless'] else 'windowed'} mode")
    
    app = QApplication(sys.argv)

    # Apply external stylesheet using relative path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    qss_path = os.path.join(script_dir, "styles.qss")
    if os.path.exists(qss_path):
        with open(qss_path, 'r') as file:
            app.setStyleSheet(file.read())
        print("✅ Stylesheet loaded successfully.")
    else:
        print(f"⚠️ Stylesheet could not be loaded. Path: {qss_path}")

    notesApp = QNotes()
    notesApp.show()
    sys.exit(app.exec_())
