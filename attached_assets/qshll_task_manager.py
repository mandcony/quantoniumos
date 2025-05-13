# File: qshll_task_manager.py

import sys
import os
import psutil
import datetime
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QTableWidget, QTableWidgetItem,
    QTabWidget, QWidget, QProgressBar, QHeaderView, QPushButton
)
from PyQt5.QtCore import QTimer, Qt
import logging
import ctypes

# Set up paths
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
LOGS_DIR = os.path.join(ROOT_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

# Set up logging
logging.basicConfig(level=logging.DEBUG, 
                    filename=os.path.join(LOGS_DIR, "task_manager.log"), 
                    filemode="a",
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
logger.addHandler(console_handler)

# Path to external QSS file
STYLES_QSS = os.path.join(ROOT_DIR, "styles.qss")

def load_stylesheet(qss_path):
    """Load the stylesheet from the given path."""
    if os.path.exists(qss_path):
        try:
            with open(qss_path, "r", encoding="utf-8") as f:
                logger.info(f"✅ Stylesheet loaded from {qss_path}")
                return f.read()
        except Exception as e:
            logger.error(f"❌ Error loading stylesheet: {e}")
    logger.warning(f"⚠️ Stylesheet not found: {qss_path}")
    return ""

class QSHLLTaskManager(QDialog):
    def __init__(self):
        super().__init__()
        
        # Set up the UI
        self.setWindowTitle("QSHLL Task Manager")
        self.setGeometry(100, 100, 800, 600)
        self.setMinimumSize(600, 400)
        
        # Create the tab widget and main layout
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        tab_widget = QTabWidget()
        layout.addWidget(tab_widget)
        
        # Create tabs
        processes_tab = QWidget()
        performance_tab = QWidget()
        tab_widget.addTab(processes_tab, "Processes")
        tab_widget.addTab(performance_tab, "Performance")
        
        # Set up processes tab
        processes_layout = QVBoxLayout(processes_tab)
        self.processes_table = QTableWidget()
        self.processes_table.setColumnCount(5)
        self.processes_table.setHorizontalHeaderLabels(["PID", "Name", "CPU %", "Memory %", "Status"])
        header = self.processes_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        processes_layout.addWidget(self.processes_table)
        
        # Add terminate button
        self.terminate_btn = QPushButton("Terminate Process")
        self.terminate_btn.clicked.connect(self.terminate_process)
        processes_layout.addWidget(self.terminate_btn)
        
        # Set up performance tab
        performance_layout = QVBoxLayout(performance_tab)
        
        # CPU Usage
        self.cpu_label = QLabel("CPU Usage:")
        performance_layout.addWidget(self.cpu_label)
        self.cpu_bar = QProgressBar()
        self.cpu_bar.setRange(0, 100)
        performance_layout.addWidget(self.cpu_bar)
        
        # Memory Usage
        self.mem_label = QLabel("Memory Usage:")
        performance_layout.addWidget(self.mem_label)
        self.mem_bar = QProgressBar()
        self.mem_bar.setRange(0, 100)
        performance_layout.addWidget(self.mem_bar)
        
        # Disk Usage
        self.disk_label = QLabel("Disk Usage:")
        performance_layout.addWidget(self.disk_label)
        self.disk_bar = QProgressBar()
        self.disk_bar.setRange(0, 100)
        performance_layout.addWidget(self.disk_bar)
        
        # Export button
        self.export_btn = QPushButton("Export Performance Data")
        self.export_btn.clicked.connect(self.export_data_to_csv)
        performance_layout.addWidget(self.export_btn)
        
        # Set up data structures
        self.system_df = pd.DataFrame(columns=["Timestamp", "CPU", "Memory", "Disk"])
        
        # Set up update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.updateData)
        self.timer.start(1000)  # Update every second
        
        # Initial update
        self.updateData()
        
        # Load stylesheet
        self.stylesheet = load_stylesheet(STYLES_QSS)
        if self.stylesheet:
            self.setStyleSheet(self.stylesheet)
    
    def updateData(self):
        try:
            # Update system metrics
            cpu_percent = psutil.cpu_percent()
            mem_percent = psutil.virtual_memory().percent
            disk_percent = psutil.disk_usage('/').percent
            
            # Update performance tab
            self.cpu_bar.setValue(int(cpu_percent))
            self.cpu_label.setText(f"CPU Usage: {cpu_percent:.1f}%")
            
            self.mem_bar.setValue(int(mem_percent))
            self.mem_label.setText(f"Memory Usage: {mem_percent:.1f}%")
            
            self.disk_bar.setValue(int(disk_percent))
            self.disk_label.setText(f"Disk Usage: {disk_percent:.1f}%")
            
            # Log system metrics
            timestamp = datetime.datetime.now()
            self.system_df = self.system_df.append({
                "Timestamp": timestamp,
                "CPU": cpu_percent,
                "Memory": mem_percent,
                "Disk": disk_percent
            }, ignore_index=True)
            
            # Only keep the last 3600 entries (1 hour at 1 second intervals)
            if len(self.system_df) > 3600:
                self.system_df = self.system_df.iloc[-3600:]
                
            # Update processes table
            self.updateTable()
                
        except Exception as e:
            logger.error(f"Error updating data: {e}")
    
    def terminate_process(self):
        try:
            selected_rows = self.processes_table.selectedItems()
            if not selected_rows:
                return
                
            # Get the first selected row
            row = selected_rows[0].row()
            pid = int(self.processes_table.item(row, 0).text())
            
            # Confirm termination
            try:
                process = psutil.Process(pid)
                process.terminate()
                logger.info(f"Terminated process with PID {pid}")
            except Exception as e:
                logger.error(f"Failed to terminate process: {e}")
                
        except Exception as e:
            logger.error(f"Error in terminate_process: {e}")
    
    def updateTable(self):
        try:
            # Get processes info
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'status']):
                try:
                    # Get process info
                    proc_info = proc.info
                    processes.append([
                        str(proc_info['pid']),
                        proc_info['name'],
                        f"{proc_info['cpu_percent']:.1f}",
                        f"{proc_info['memory_percent']:.1f}" if proc_info['memory_percent'] else "0.0",
                        proc_info['status']
                    ])
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass
            
            # Sort by CPU usage (descending)
            processes.sort(key=lambda x: float(x[2]), reverse=True)
            
            # Update table
            self.processes_table.setRowCount(len(processes))
            for row, proc in enumerate(processes):
                for col, value in enumerate(proc):
                    self.processes_table.setItem(row, col, QTableWidgetItem(value))
                
        except Exception as e:
            logger.error(f"Error in updateTable: {e}")

    def export_data_to_csv(self):
        try:
            csv_path = os.path.join(ROOT_DIR, "quantonium_performance_metrics.csv")
            self.system_df.to_csv(csv_path, index=False)
            logger.info(f"Performance data exported to {csv_path}")
        except Exception as e:
            logger.error(f"Error exporting data: {e}")

if __name__ == "__main__":
    try:
        # Import and use the headless environment setup
        from attached_assets import setup_headless_environment
        env_config = setup_headless_environment()
        logger.info(f"Running on {env_config['platform']} in {'headless' if env_config['headless'] else 'windowed'} mode")
        
        app = QApplication(sys.argv)
        
        # Load and apply the stylesheet
        style_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "styles.qss")
        if os.path.exists(style_path):
            with open(style_path, 'r') as f:
                app.setStyleSheet(f.read())
            logger.info("✅ Stylesheet loaded successfully.")
        else:
            logger.warning(f"⚠️ Stylesheet not found at {style_path}")
            
        win = QSHLLTaskManager()
        win.show()
        sys.exit(app.exec_())
    except Exception as e:
        logger.error(f"Application failed to start: {e}")
        sys.exit(1)
