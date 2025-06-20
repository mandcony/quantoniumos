"""
qshll_task_manager.py - Silent Version for Quantonium v2
=======================================================
Completely suppresses all resonance manager output using context manager approach.
"""

from __future__ import annotations

import contextlib
import datetime
import io
import logging
import os
import sys
import threading
from pathlib import Path
from typing import List, Tuple, Optional, Any

# Conditional imports with fallbacks
try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import psutil
except ImportError:
    psutil = None

try:
    from PyQt5.QtCore import QTimer, Qt
    from PyQt5.QtWidgets import (
        QApplication, QDialog, QHeaderView, QProgressBar, QPushButton,
        QTabWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
    )
    PYQT5_AVAILABLE = True
except ImportError:
    PYQT5_AVAILABLE = False

######################################################################
# Silent Output Context Manager
######################################################################

class SilentContext:
    """Context manager that completely silences all output from resonance manager."""
    
    def __init__(self):
        self.original_stdout = None
        self.original_stderr = None
        self.devnull = None
        
    def __enter__(self):
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.devnull = open(os.devnull, 'w')
        sys.stdout = self.devnull
        sys.stderr = self.devnull
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        if self.devnull:
            self.devnull.close()

######################################################################
# Logging Setup
######################################################################

LOG_PATH = Path(__file__).with_suffix(".log")
logging.basicConfig(
    level=logging.INFO,
    filename=str(LOG_PATH),
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

######################################################################
# Resonance Manager Integration
######################################################################

# Try to import resonance manager with graceful fallback
Process = None
monitor_resonance_states = None

try:
    # Add paths for resonance manager
    resman_dir = Path(r"C:\quantonium_v2\orchestration")
    if resman_dir.exists():
        sys.path.append(str(resman_dir))
    
    from resonance_manager import Process, monitor_resonance_states
    logger.info("Successfully imported resonance_manager")
except ImportError as exc:
    logger.warning(f"Resonance manager not available: {exc}")

######################################################################
# Task Manager GUI
######################################################################

if PYQT5_AVAILABLE:
    class QSHLLTaskManager(QDialog):
        """Silent PyQt5 task manager with topological metrics."""

        DATA_COLUMNS = ["Time", "CPU%", "Memory%", "Disk%", "Topo CPU", "Topo Memory", "Topo Disk"]

        def __init__(self):
            super().__init__()
            self.setWindowTitle("QSHLL Task Manager (Silent)")
            self.resize(1000, 600)

            # Main layout
            main_layout = QVBoxLayout(self)
            self.tabs = QTabWidget()
            main_layout.addWidget(self.tabs)

            # Processes tab
            self._setup_processes_tab()
            
            # Performance tab
            self._setup_performance_tab()

            # Data storage
            if pd:
                self.system_df = pd.DataFrame(columns=self.DATA_COLUMNS)
            else:
                self.system_data = []

            # Initialize topological processes
            self._setup_topological_processes()

            # Timer for updates
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.poll_data)
            self.timer.start(2000)  # Update every 2 seconds
            
            logger.info("Task Manager initialized successfully")

        def _setup_processes_tab(self):
            """Setup the processes monitoring tab."""
            self.processes_tab = QWidget()
            self.tabs.addTab(self.processes_tab, "Processes")
            
            proc_layout = QVBoxLayout(self.processes_tab)
            self.process_table = QTableWidget(0, 4)
            self.process_table.setHorizontalHeaderLabels(["Process Name", "PID", "CPU%", "Memory%"])
            self.process_table.setSortingEnabled(True)
            
            if self.process_table.horizontalHeader():
                self.process_table.horizontalHeader().setStretchLastSection(True)
            
            proc_layout.addWidget(self.process_table)

        def _setup_performance_tab(self):
            """Setup the performance monitoring tab."""
            self.performance_tab = QWidget()
            self.tabs.addTab(self.performance_tab, "Performance")
            
            perf_layout = QVBoxLayout(self.performance_tab)
            self.system_table = QTableWidget(0, len(self.DATA_COLUMNS))
            self.system_table.setHorizontalHeaderLabels(self.DATA_COLUMNS)
            
            if self.system_table.horizontalHeader():
                self.system_table.horizontalHeader().setStretchLastSection(True)
            
            perf_layout.addWidget(self.system_table)

            # Export button
            self.export_button = QPushButton("Export Data to CSV")
            self.export_button.clicked.connect(self.export_data_to_csv)
            perf_layout.addWidget(self.export_button)

        def _setup_topological_processes(self):
            """Initialize topological processes for resonance calculations."""
            self.topo_processes = None
            
            if Process and monitor_resonance_states:
                try:
                    vertices = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
                    self.topo_processes = [
                        Process(i, priority=1.0, amplitude=complex(1.0, 0), vertices=vertices)
                        for i in range(3)
                    ]
                    logger.info("Topological processes initialized")
                except Exception as exc:
                    logger.error(f"Failed to initialize topological processes: {exc}")

        def poll_data(self):
            """Update all data tables."""
            try:
                self._update_process_table()
                self._update_performance_metrics()
            except Exception as exc:
                logger.error(f"Error polling data: {exc}")

        def _update_process_table(self):
            """Update the process monitoring table."""
            if not psutil:
                return

            rows = []
            for proc in psutil.process_iter(attrs=["pid", "name", "cpu_percent", "memory_percent"]):
                try:
                    info = proc.info
                    rows.append((
                        info["name"] or "Unknown",
                        info["pid"] or 0,
                        info["cpu_percent"] or 0.0,
                        info["memory_percent"] or 0.0
                    ))
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue

            self.process_table.setRowCount(len(rows))
            for i, (name, pid, cpu, mem) in enumerate(rows):
                self.process_table.setItem(i, 0, QTableWidgetItem(str(name)))
                self.process_table.setItem(i, 1, QTableWidgetItem(str(pid)))
                self.process_table.setItem(i, 2, QTableWidgetItem(f"{cpu:.1f}%"))
                self.process_table.setItem(i, 3, QTableWidgetItem(f"{mem:.1f}%"))

        def _update_performance_metrics(self):
            """Update performance metrics including topological data."""
            if not psutil:
                return

            # Get system metrics
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            try:
                disk = psutil.disk_usage("/")
                disk_percent = disk.percent
            except:
                disk_percent = 0.0

            # Calculate topological metrics with complete silence
            topo_cpu = topo_mem = topo_disk = 0.0
            if self.topo_processes and monitor_resonance_states:
                try:
                    with SilentContext():
                        self.topo_processes = monitor_resonance_states(self.topo_processes, dt=0.1)

                    # Extract metrics from topological processes
                    if self.topo_processes:
                        topo_cpu = sum(getattr(p.priority, 'amplitude', 1.0) for p in self.topo_processes) / len(self.topo_processes) * 50
                        topo_mem = sum(abs(getattr(p.amplitude, 'real', 1.0)) for p in self.topo_processes) / len(self.topo_processes) * 50
                        topo_disk = sum(getattr(p, 'resonance', 1.0) for p in self.topo_processes) / len(self.topo_processes) * 100
                        
                except Exception as exc:
                    logger.error(f"Error in topological calculations: {exc}")

            # Create new data row
            current_time = datetime.datetime.now().strftime("%H:%M:%S")
            new_row_data = [current_time, cpu_percent, memory.percent, disk_percent, topo_cpu, topo_mem, topo_disk]

            # Update data storage
            if pd and hasattr(self, 'system_df'):
                new_row = pd.DataFrame([new_row_data], columns=self.DATA_COLUMNS)
                self.system_df = pd.concat([self.system_df, new_row], ignore_index=True).tail(60)
                data_rows = self.system_df.values.tolist()
            else:
                # Fallback to list storage
                if not hasattr(self, 'system_data'):
                    self.system_data = []
                self.system_data.append(new_row_data)
                self.system_data = self.system_data[-60:]  # Keep last 60 entries
                data_rows = self.system_data

            # Update table display
            self.system_table.setRowCount(len(data_rows))
            for i, row in enumerate(data_rows):
                for j, val in enumerate(row):
                    display_val = f"{val:.2f}" if isinstance(val, (int, float)) and j > 0 else str(val)
                    self.system_table.setItem(i, j, QTableWidgetItem(display_val))

        def export_data_to_csv(self):
            """Export performance data to CSV file."""
            csv_path = Path(__file__).with_name("quantonium_performance_metrics.csv")
            try:
                if pd and hasattr(self, 'system_df'):
                    self.system_df.to_csv(csv_path, index=False)
                else:
                    # Fallback CSV writing
                    with open(csv_path, 'w') as f:
                        f.write(','.join(self.DATA_COLUMNS) + '\n')
                        for row in getattr(self, 'system_data', []):
                            f.write(','.join(str(val) for val in row) + '\n')
                
                logger.info(f"Performance data exported to {csv_path}")
                print(f"Data exported to {csv_path}")
            except Exception as exc:
                logger.error(f"Error exporting data: {exc}")
                print(f"Export failed: {exc}")

######################################################################
# Command Line Interface
######################################################################

def main() -> None:
    """Launch the silent task manager."""
    if not PYQT5_AVAILABLE:
        print("PyQt5 not available. Please install PyQt5 to run the GUI.")
        print("Falling back to console mode...")
        console_mode()
        return

    if not psutil:
        print("psutil not available. Please install psutil for system monitoring.")
        return

    app = QApplication(sys.argv)
    window = QSHLLTaskManager()
    window.show()
    
    try:
        sys.exit(app.exec_())
    except KeyboardInterrupt:
        logger.info("Task Manager shutdown by user")
    except Exception as exc:
        logger.error(f"Unexpected error: {exc}")
        sys.exit(1)

def console_mode():
    """Simple console-based monitoring when GUI is not available."""
    if not psutil:
        print("System monitoring requires psutil. Please install it.")
        return
        
    print("QSHLL Task Manager - Console Mode")
    print("Press Ctrl+C to exit")
    
    try:
        while True:
            cpu = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            print(f"CPU: {cpu:5.1f}% | Memory: {memory.percent:5.1f}% | Available: {memory.available // (1024**3):2d}GB")
            
            import time
            time.sleep(2)
    except KeyboardInterrupt:
        print("\nTask Manager stopped.")

if __name__ == "__main__":
    main()