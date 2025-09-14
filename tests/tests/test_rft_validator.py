#!/usr/bin/env python3
"""
Test minimal RFT Validation Suite
"""

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget

class TestRFTValidator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TEST RFT Validator")
        self.resize(800, 600)
        
        cw = QWidget()
        self.setCentralWidget(cw)
        layout = QVBoxLayout(cw)
        
        label = QLabel("RFT Validator Test - If you see this, PyQt5 is working!")
        layout.addWidget(label)
        
        print("✓ Test RFT Validator window created successfully")

def main():
    print("Starting test RFT validator...")
    app = QApplication(sys.argv)
    window = TestRFTValidator()
    window.show()
    print("✓ Test window shown")
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
