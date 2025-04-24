import sys
import os
import threading
from PyQt5.QtWidgets import QApplication, QDialog, QVBoxLayout, QWidget, QTableWidget, QTableWidgetItem, QPushButton, QLabel
from PyQt5.QtCore import QTimer, Qt

# Modernized QSS stylesheet
modern_stylesheet = """
QDialog {
    background: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, 
                                stop:0 #1a1a2e, stop:1 #2d2d4a);
}

QPushButton {
    background-color: #3b82f6;
    color: #ffffff;
    border: none;
    border-radius: 8px;
    padding: 12px;
    font: bold 13pt "Segoe UI";
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
}

QPushButton:hover {
    background-color: #60a5fa;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    transition: all 0.2s ease;
}

QPushButton:pressed {
    background-color: #2563eb;
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.2);
}

QLabel {
    color: #e2e8f0;
    font: 11pt "Roboto";
}

QTableWidget {
    background-color: #2d2d4a;
    color: #e2e8f0;
    border: 1px solid #4b5563;
    border-radius: 6px;
    gridline-color: #4b5563;
}

QTableWidget::item {
    color: #e2e8f0;
    padding: 4px;
}

QTableWidget::item:selected {
    background-color: #3b82f6;
    color: #ffffff;
}

QTableWidget::horizontalHeader {
    background-color: #3b82f6;
    color: #ffffff;
    font: bold 11pt "Roboto";
    border: none;
    padding: 8px;
}
"""

class QSHLLTaskManager(QDialog):  
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QSHLL Task Manager")
        self.setGeometry(100, 100, 600, 400)

        # Layout setup
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)  # Add padding
        layout.setSpacing(8)  # Space between widgets

        # Title label
        title_label = QLabel("Task Manager - Monitoring Processes")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # Process table
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["PID", "CPU%", "Memory"])
        self.table.horizontalHeader().setStretchLastSection(True)  # Stretch last column
        self.table.setSelectionBehavior(QTableWidget.SelectRows)  # Select whole rows
        layout.addWidget(self.table)

        # Stop button
        self.stopBtn = QPushButton("Stop Monitoring")
        self.stopBtn.clicked.connect(self.stopMonitoring)
        layout.addWidget(self.stopBtn)

        # Timer for polling data
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.pollData)
        self.timer.start(1000)  # Update every 1 second

    def pollData(self):
        # Placeholder data (replace with actual process monitoring logic)
        self.table.setRowCount(1)
        self.table.setItem(0, 0, QTableWidgetItem("1234"))
        self.table.setItem(0, 1, QTableWidgetItem("12.5%"))
        self.table.setItem(0, 2, QTableWidgetItem("256MB"))

    def stopMonitoring(self):
        self.timer.stop()
        self.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(modern_stylesheet)  # Apply modern stylesheet
    manager = QSHLLTaskManager()
    manager.exec_()