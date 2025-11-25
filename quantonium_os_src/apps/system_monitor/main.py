#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
# This file is listed in CLAIMS_PRACTICING_FILES.txt and is licensed
# under LICENSE-CLAIMS-NC.md (research/education only). Commercial
# rights require a separate patent license from the author.
"""System Monitor - Performance Monitoring Dashboard"""

from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QProgressBar, QGroupBox, QTableWidget, QTableWidgetItem)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
import psutil

class MainWindow(QMainWindow):
    """System Monitor Application"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("System Monitor - QuantoniumOS")
        self.setGeometry(100, 100, 900, 700)
        
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        title = QLabel("📊 System Performance Monitor")
        title.setFont(QFont("Sans Serif", 16, QFont.Bold))
        title.setStyleSheet("color: #00aaff; padding: 10px;")
        layout.addWidget(title)
        
        # CPU
        cpu_group = QGroupBox("CPU Usage")
        cpu_layout = QVBoxLayout()
        self.cpu_bar = QProgressBar()
        self.cpu_label = QLabel("0%")
        cpu_layout.addWidget(self.cpu_bar)
        cpu_layout.addWidget(self.cpu_label)
        cpu_group.setLayout(cpu_layout)
        layout.addWidget(cpu_group)
        
        # Memory
        mem_group = QGroupBox("Memory Usage")
        mem_layout = QVBoxLayout()
        self.mem_bar = QProgressBar()
        self.mem_label = QLabel("0 MB / 0 MB")
        mem_layout.addWidget(self.mem_bar)
        mem_layout.addWidget(self.mem_label)
        mem_group.setLayout(mem_layout)
        layout.addWidget(mem_group)
        
        # Disk
        disk_group = QGroupBox("Disk Usage")
        disk_layout = QVBoxLayout()
        self.disk_bar = QProgressBar()
        self.disk_label = QLabel("0 GB / 0 GB")
        disk_layout.addWidget(self.disk_bar)
        disk_layout.addWidget(self.disk_label)
        disk_group.setLayout(disk_layout)
        layout.addWidget(disk_group)
        
        # Process table
        proc_group = QGroupBox("RFT Processes")
        proc_layout = QVBoxLayout()
        self.proc_table = QTableWidget()
        self.proc_table.setColumnCount(3)
        self.proc_table.setHorizontalHeaderLabels(["Process", "CPU %", "Memory (MB)"])
        proc_layout.addWidget(self.proc_table)
        proc_group.setLayout(proc_layout)
        layout.addWidget(proc_group)
        
        # Refresh button
        self.refresh_btn = QPushButton("🔄 Refresh")
        self.refresh_btn.clicked.connect(self.update_stats)
        layout.addWidget(self.refresh_btn)
        
        self.statusBar().showMessage("Monitoring system performance")
        self.set_dark_theme()
        
        # Auto-update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_stats)
        self.timer.start(2000)  # Update every 2 seconds
        self.update_stats()
    
    def set_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #1a1a1a; color: #ffffff; }
            QGroupBox { border: 2px solid #00aaff; border-radius: 5px; margin-top: 10px; padding-top: 10px; color: #00aaff; font-weight: bold; }
            QPushButton { background-color: #00aaff; color: #fff; border: none; padding: 8px 16px; border-radius: 4px; }
            QPushButton:hover { background-color: #00ffaa; }
            QProgressBar { background-color: #2a2a2a; border: 1px solid #00aaff; text-align: center; }
            QProgressBar::chunk { background-color: #00aaff; }
            QTableWidget { background-color: #2a2a2a; color: #fff; border: 1px solid #00aaff; }
            QHeaderView::section { background-color: #00aaff; color: #fff; padding: 5px; border: none; }
        """)
    
    def update_stats(self):
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.cpu_bar.setValue(int(cpu_percent))
            self.cpu_label.setText(f"{cpu_percent:.1f}% - {psutil.cpu_count()} cores")
            
            # Memory
            mem = psutil.virtual_memory()
            self.mem_bar.setValue(int(mem.percent))
            self.mem_label.setText(f"{mem.used / 1024**3:.1f} GB / {mem.total / 1024**3:.1f} GB ({mem.percent:.1f}%)")
            
            # Disk
            disk = psutil.disk_usage('/')
            self.disk_bar.setValue(int(disk.percent))
            self.disk_label.setText(f"{disk.used / 1024**3:.1f} GB / {disk.total / 1024**3:.1f} GB ({disk.percent:.1f}%)")
            
            # Processes (top 5 by CPU)
            processes = []
            for proc in psutil.process_iter(['name', 'cpu_percent', 'memory_info']):
                try:
                    info = proc.info
                    if 'python' in info['name'].lower() or 'quantonium' in info['name'].lower():
                        processes.append((info['name'], info['cpu_percent'], info['memory_info'].rss / 1024**2))
                except:
                    pass
            
            processes.sort(key=lambda x: x[1], reverse=True)
            self.proc_table.setRowCount(min(len(processes), 5))
            
            for i, (name, cpu, mem) in enumerate(processes[:5]):
                self.proc_table.setItem(i, 0, QTableWidgetItem(name))
                self.proc_table.setItem(i, 1, QTableWidgetItem(f"{cpu:.1f}"))
                self.proc_table.setItem(i, 2, QTableWidgetItem(f"{mem:.1f}"))
        
        except Exception as e:
            self.statusBar().showMessage(f"Error updating stats: {e}")
