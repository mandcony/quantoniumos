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

# Set up logging
logging.basicConfig(level=logging.DEBUG, filename="task_manager.log", filemode="a",
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
logger.addHandler(console_handler)

# Ensure correct paths and DLL loading
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'apps'))
BIN_DIR = os.path.join(ROOT_DIR, 'bin')
if os.path.exists(BIN_DIR):
    os.add_dll_directory(BIN_DIR)
else:
    logger.warning(f"Bin directory not found: {BIN_DIR}")

# Attempt to load engine_core.dll
try:
    ENGINE_DLL = ctypes.CDLL(os.path.join(BIN_DIR, 'engine_core.dll'))
    logger.debug("Successfully loaded engine_core.dll")
except Exception as e:
    logger.warning(f"Failed to load engine_core.dll: {e}")
    ENGINE_DLL = None

# Import custom modules with fallback
try:
    from system_resonance_manager import Process, monitor_resonance_states
except ImportError as e:
    logger.warning(f"Failed to import system_resonance_manager: {e}")
    Process = None
    monitor_resonance_states = None

class QSHLLTaskManager(QDialog):
    def __init__(self):
        super().__init__()
        logger.debug("Starting Task Manager initialization")
        try:
            self.setWindowTitle("QSHLL Task Manager")
            self.setGeometry(100, 100, 1000, 600)

            # Load stylesheet
            qss_path = os.path.join(ROOT_DIR, 'styles.qss')
            if os.path.exists(qss_path):
                with open(qss_path, 'r') as file:
                    self.setStyleSheet(file.read())
            else:
                logger.warning(f"Stylesheet not found: {qss_path}")
                self.setStyleSheet("QWidget { background-color: #2E2E2E; color: white; }")

            # Main layout
            layout = QVBoxLayout(self)
            self.tabs = QTabWidget()
            layout.addWidget(self.tabs)

            # Processes Tab
            self.processes_tab = QWidget()
            self.tabs.addTab(self.processes_tab, "Processes")
            process_layout = QVBoxLayout(self.processes_tab)
            self.process_table = QTableWidget(0, 4)
            self.process_table.setHorizontalHeaderLabels(["Process Name", "PID", "CPU%", "Memory%"])
            self.process_table.setSortingEnabled(True)
            self.process_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            process_layout.addWidget(self.process_table)

            # Performance Tab
            self.performance_tab = QWidget()
            self.tabs.addTab(self.performance_tab, "Performance")
            perf_layout = QVBoxLayout(self.performance_tab)
            self.system_table = QTableWidget(0, 7)
            self.system_table.setHorizontalHeaderLabels(["Time", "CPU%", "Memory%", "Disk%", "Topo CPU", "Topo Memory", "Topo Disk"])
            self.system_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            perf_layout.addWidget(self.system_table)

            self.export_button = QPushButton("Export Data to CSV")
            self.export_button.clicked.connect(self.export_data_to_csv)
            perf_layout.addWidget(self.export_button)

            # Data storage
            self.data_columns = ["Time", "CPU%", "Memory%", "Disk%", "Topo CPU", "Topo Memory", "Topo Disk"]
            self.system_df = pd.DataFrame(columns=self.data_columns)

            self.boot_time = psutil.boot_time()
            self.last_disk_io = psutil.disk_io_counters()
            self.last_disk_io_time = datetime.datetime.now()

            # Initialize topological processes if available
            if Process and monitor_resonance_states:
                vertices = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
                self.topo_processes = [Process(i, priority=1.0, amplitude=complex(1.0, 0), vertices=vertices) for i in range(3)]
            else:
                self.topo_processes = None
                logger.warning("Topological metrics unavailable due to missing system_resonance_manager")

            # Start polling
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.pollData)
            self.timer.start(1000)

            logger.debug("Task Manager initialization complete")
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise

    def pollData(self):
        try:
            # Update process table
            processes = []
            for proc in psutil.process_iter(attrs=['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    processes.append((proc.info['name'], proc.info['pid'], proc.info['cpu_percent'], proc.info['memory_percent']))
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue

            self.process_table.setRowCount(len(processes))
            for i, proc in enumerate(processes):
                self.process_table.setItem(i, 0, QTableWidgetItem(proc[0]))
                self.process_table.setItem(i, 1, QTableWidgetItem(str(proc[1])))

                cpu_bar = QProgressBar()
                cpu_bar.setMaximum(100)
                cpu_bar.setValue(min(int(proc[2]), 100))
                cpu_bar.setFormat(f"{proc[2]:.1f}%")
                self.process_table.setCellWidget(i, 2, cpu_bar)

                mem_bar = QProgressBar()
                mem_bar.setMaximum(100)
                mem_bar.setValue(min(int(proc[3]), 100))
                mem_bar.setFormat(f"{proc[3]:.1f}%")
                self.process_table.setCellWidget(i, 3, mem_bar)

            # Update performance metrics
            current_time = len(self.system_df)
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            if self.topo_processes and monitor_resonance_states:
                try:
                    self.topo_processes = monitor_resonance_states(self.topo_processes, dt=0.1)
                    avg_priority = sum(p.priority.amplitude for p in self.topo_processes) / len(self.topo_processes) * 50
                    avg_amplitude = sum(abs(p.amplitude.real) for p in self.topo_processes) / len(self.topo_processes) * 50
                    avg_resonance = sum(p.resonance for p in self.topo_processes) / len(self.topo_processes) * 100
                except Exception as e:
                    logger.error(f"Error in monitor_resonance_states: {e}")
                    avg_priority = avg_amplitude = avg_resonance = 0
            else:
                avg_priority = avg_amplitude = avg_resonance = 0

            new_row = pd.DataFrame([[current_time, cpu_percent, memory.percent, disk.percent, avg_priority, avg_amplitude, avg_resonance]],
                                   columns=self.data_columns)
            self.system_df = pd.concat([self.system_df, new_row], ignore_index=True)

            if len(self.system_df) > 60:
                self.system_df = self.system_df.tail(60)

            self.updateTable()
        except Exception as e:
            logger.error(f"Error in pollData: {e}")

    def updateTable(self):
        try:
            self.system_table.setRowCount(len(self.system_df))
            for i, row in self.system_df.iterrows():
                for j, value in enumerate(row):
                    self.system_table.setItem(i, j, QTableWidgetItem(str(round(value, 2) if isinstance(value, float) else value)))
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
        app = QApplication(sys.argv)
        win = QSHLLTaskManager()
        win.show()
        sys.exit(app.exec_())
    except Exception as e:
        logger.error(f"Application failed to start: {e}")
        sys.exit(1)