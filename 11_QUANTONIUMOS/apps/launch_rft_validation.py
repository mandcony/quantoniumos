#!/usr/bin/env python3
"""
RFT Validation Suite App Launcher
"""
import os
import sys

# Add the app directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from PyQt5.QtWidgets import QApplication, QMainWindow
    from rft_validation_suite import RFTValidationWidget

    def main():
        app = QApplication(sys.argv)

        # Create main window to hold the widget
        main_window = QMainWindow()
        main_window.setWindowTitle("RFT Validation Suite")
        main_window.setGeometry(100, 100, 1200, 800)

        # Create and set the RFT widget as central widget
        rft_widget = RFTValidationWidget()
        main_window.setCentralWidget(rft_widget)

        main_window.show()
        return app.exec_()

    if __name__ == "__main__":
        sys.exit(main())

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure PyQt5 and required packages are installed")
