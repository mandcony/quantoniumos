#!/usr/bin/env python3
"""
QuantoniumOS System Monitor
==========================
Monitors RFT assembly and quantum kernels
"""

import sys
import os
import psutil
import time
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, 
    QLabel, QProgressBar, QTableWidget, QTableWidgetItem
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont

# Get paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UI_DIR = os.path.join(BASE_DIR, "ui")
ASSEMBLY_DIR = os.path.join(BASE_DIR, "ASSEMBLY", "python_bindings")

# Try RFT integration
sys.path.insert(0, ASSEMBLY_DIR)
try:
    import unitary_rft
    RFT_AVAILABLE = True
except ImportError:
    RFT_AVAILABLE = False

class SystemMonitor(QMainWindow):
    """QuantoniumOS System Monitor"""
    
    def __init__(self):
        super().__init__()
        self.setObjectName("AppWindow")
        self.setWindowTitle("QuantoniumOS System Monitor")
        self.setGeometry(100, 100, 900, 600)
        
        self.init_ui()
        self.load_styles()
        self.setup_timer()
        
    def init_ui(self):
        """Initialize UI components"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Header
        header = QLabel("🖥️ QuantoniumOS System Monitor")
        header.setObjectName("HeaderLabel")
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)
        
        # System stats
        stats_layout = QHBoxLayout()
        layout.addLayout(stats_layout)
        
        # CPU usage
        cpu_widget = QWidget()
        cpu_layout = QVBoxLayout(cpu_widget)
        cpu_layout.addWidget(QLabel("CPU Usage"))
        self.cpu_progress = QProgressBar()
        self.cpu_label = QLabel("0%")
        cpu_layout.addWidget(self.cpu_progress)
        cpu_layout.addWidget(self.cpu_label)
        stats_layout.addWidget(cpu_widget)
        
        # Memory usage
        mem_widget = QWidget()
        mem_layout = QVBoxLayout(mem_widget)
        mem_layout.addWidget(QLabel("Memory Usage"))
        self.mem_progress = QProgressBar()
        self.mem_label = QLabel("0%")
        mem_layout.addWidget(self.mem_progress)
        mem_layout.addWidget(self.mem_label)
        stats_layout.addWidget(mem_widget)
        
        # RFT Status
        rft_widget = QWidget()
        rft_layout = QVBoxLayout(rft_widget)
        rft_layout.addWidget(QLabel("RFT Assembly"))
        self.rft_status = QLabel("🔴 Offline" if not RFT_AVAILABLE else "🟢 Online")
        self.rft_status.setAlignment(Qt.AlignCenter)
        rft_layout.addWidget(self.rft_status)
        stats_layout.addWidget(rft_widget)
        
        # Process table
        layout.addWidget(QLabel("Running Processes:"))
        self.process_table = QTableWidget()
        self.process_table.setColumnCount(4)
        self.process_table.setHorizontalHeaderLabels(["PID", "Name", "CPU%", "Memory"])
        layout.addWidget(self.process_table)
        
    def load_styles(self):
        """Load stylesheet"""
        qss_path = os.path.join(UI_DIR, "styles.qss")
        if os.path.exists(qss_path):
            with open(qss_path, 'r') as f:
                self.setStyleSheet(f.read())
                
    def setup_timer(self):
        """Setup update timer"""
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_stats)
        self.timer.start(2000)  # Update every 2 seconds
        self.update_stats()  # Initial update
        
    def update_stats(self):
        """Update system statistics"""
        # CPU usage
        cpu_percent = psutil.cpu_percent()
        self.cpu_progress.setValue(int(cpu_percent))
        self.cpu_label.setText(f"{cpu_percent:.1f}%")
        
        # Memory usage
        memory = psutil.virtual_memory()
        mem_percent = memory.percent
        self.mem_progress.setValue(int(mem_percent))
        self.mem_label.setText(f"{mem_percent:.1f}%")
        
        # Update process table
        self.update_process_table()
        
    def update_process_table(self):
        """Update the process table"""
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        # Sort by CPU usage
        processes.sort(key=lambda x: x['cpu_percent'] or 0, reverse=True)
        
        # Update table
        self.process_table.setRowCount(min(10, len(processes)))  # Show top 10
        for i, proc in enumerate(processes[:10]):
            self.process_table.setItem(i, 0, QTableWidgetItem(str(proc['pid'])))
            self.process_table.setItem(i, 1, QTableWidgetItem(proc['name'] or 'N/A'))
            self.process_table.setItem(i, 2, QTableWidgetItem(f"{proc['cpu_percent'] or 0:.1f}%"))
            self.process_table.setItem(i, 3, QTableWidgetItem(f"{proc['memory_percent'] or 0:.1f}%"))

def main():
    app = QApplication(sys.argv)
    monitor = SystemMonitor()
    monitor.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
