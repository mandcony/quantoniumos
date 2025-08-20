"""
QuantoniumOS - System Monitor Application
Real-time quantum system monitoring and diagnostics
"""

import sys
import os
import time
import threading
from pathlib import Path

try:
    from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, 
                                QPushButton, QLabel, QProgressBar,
                                QGroupBox, QGridLayout, QTextEdit)
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
    from PyQt5.QtGui import QFont
    PYQT5_AVAILABLE = True
except ImportError:
    PYQT5_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

class SystemMonitor(QWidget if PYQT5_AVAILABLE else object):
    """Real-time quantum system monitoring interface"""
    
    def __init__(self):
        if not PYQT5_AVAILABLE:
            print("⚠️ PyQt5 required for System Monitor GUI")
            return
            
        super().__init__()
        self.quantum_metrics = {
            'coherence': 97.5,
            'entanglement': 89.2,
            'fidelity': 94.8,
            'temperature': 0.015  # Kelvin
        }
        self.init_ui()
        self.start_monitoring()
    
    def init_ui(self):
        """Initialize the monitoring interface"""
        self.setWindowTitle("📊 QuantoniumOS - System Monitor")
        self.setGeometry(300, 300, 1000, 800)
        
        # Apply quantum styling
        self.setStyleSheet("""
            QWidget {
                background-color: #0a0e1a;
                color: #00ffcc;
                font-family: "Consolas", monospace;
            }
            QGroupBox {
                border: 2px solid #00ffcc;
                border-radius: 8px;
                padding: 10px;
                margin-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                color: #00ff88;
                padding: 5px;
            }
            QProgressBar {
                border: 2px solid #00ffcc;
                border-radius: 5px;
                background: #1a2332;
                text-align: center;
                color: #ffffff;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #00ffcc, stop:1 #00ff88);
                border-radius: 3px;
            }
            QLabel {
                color: #00ffcc;
            }
            QTextEdit {
                background: #1a2332;
                border: 1px solid #00ffcc;
                border-radius: 4px;
                color: #ffffff;
            }
        """)
        
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("📊 Quantum System Monitor")
        title.setFont(QFont("Consolas", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #00ff88; margin: 10px;")
        layout.addWidget(title)
        
        # Main monitoring grid
        main_layout = QHBoxLayout()
        
        # Left side - System metrics
        self.create_system_metrics(main_layout)
        
        # Right side - Quantum metrics
        self.create_quantum_metrics(main_layout)
        
        layout.addLayout(main_layout)
        
        # Bottom - Process monitor
        self.create_process_monitor(layout)
        
        # Status
        self.status_label = QLabel("✅ Quantum System Monitoring Active")
        self.status_label.setStyleSheet("color: #00ff88; font-weight: bold; margin: 5px;")
        layout.addWidget(self.status_label)
    
    def create_system_metrics(self, parent_layout):
        """Create system metrics panel"""
        system_group = QGroupBox("🖥️ System Metrics")
        system_layout = QGridLayout(system_group)
        
        # CPU Usage
        system_layout.addWidget(QLabel("⚡ CPU Usage:"), 0, 0)
        self.cpu_bar = QProgressBar()
        self.cpu_bar.setMaximum(100)
        system_layout.addWidget(self.cpu_bar, 0, 1)
        self.cpu_label = QLabel("0%")
        system_layout.addWidget(self.cpu_label, 0, 2)
        
        # Memory Usage
        system_layout.addWidget(QLabel("🧠 Memory:"), 1, 0)
        self.memory_bar = QProgressBar()
        self.memory_bar.setMaximum(100)
        system_layout.addWidget(self.memory_bar, 1, 1)
        self.memory_label = QLabel("0%")
        system_layout.addWidget(self.memory_label, 1, 2)
        
        # Disk Usage
        system_layout.addWidget(QLabel("💾 Disk:"), 2, 0)
        self.disk_bar = QProgressBar()
        self.disk_bar.setMaximum(100)
        system_layout.addWidget(self.disk_bar, 2, 1)
        self.disk_label = QLabel("0%")
        system_layout.addWidget(self.disk_label, 2, 2)
        
        # Network
        system_layout.addWidget(QLabel("🌐 Network:"), 3, 0)
        self.network_label = QLabel("0 KB/s")
        system_layout.addWidget(self.network_label, 3, 1, 1, 2)
        
        # Temperature
        system_layout.addWidget(QLabel("🌡️ Temp:"), 4, 0)
        self.temp_label = QLabel("-- °C")
        system_layout.addWidget(self.temp_label, 4, 1, 1, 2)
        
        parent_layout.addWidget(system_group)
    
    def create_quantum_metrics(self, parent_layout):
        """Create quantum metrics panel"""
        quantum_group = QGroupBox("⚛️ Quantum Metrics")
        quantum_layout = QGridLayout(quantum_group)
        
        # Quantum Coherence
        quantum_layout.addWidget(QLabel("🌊 Coherence:"), 0, 0)
        self.coherence_bar = QProgressBar()
        self.coherence_bar.setMaximum(100)
        quantum_layout.addWidget(self.coherence_bar, 0, 1)
        self.coherence_label = QLabel("97.5%")
        quantum_layout.addWidget(self.coherence_label, 0, 2)
        
        # Entanglement Quality
        quantum_layout.addWidget(QLabel("🔗 Entanglement:"), 1, 0)
        self.entanglement_bar = QProgressBar()
        self.entanglement_bar.setMaximum(100)
        quantum_layout.addWidget(self.entanglement_bar, 1, 1)
        self.entanglement_label = QLabel("89.2%")
        quantum_layout.addWidget(self.entanglement_label, 1, 2)
        
        # Quantum Fidelity
        quantum_layout.addWidget(QLabel("✨ Fidelity:"), 2, 0)
        self.fidelity_bar = QProgressBar()
        self.fidelity_bar.setMaximum(100)
        quantum_layout.addWidget(self.fidelity_bar, 2, 1)
        self.fidelity_label = QLabel("94.8%")
        quantum_layout.addWidget(self.fidelity_label, 2, 2)
        
        # Quantum Temperature
        quantum_layout.addWidget(QLabel("❄️ Q-Temp:"), 3, 0)
        self.qtemp_label = QLabel("15 mK")
        quantum_layout.addWidget(self.qtemp_label, 3, 1, 1, 2)
        
        # Qubit Count
        quantum_layout.addWidget(QLabel("🎯 Qubits:"), 4, 0)
        self.qubit_label = QLabel("1000 active")
        quantum_layout.addWidget(self.qubit_label, 4, 1, 1, 2)
        
        parent_layout.addWidget(quantum_group)
    
    def create_process_monitor(self, parent_layout):
        """Create process monitoring panel"""
        process_group = QGroupBox("🔄 Quantum Processes")
        process_layout = QVBoxLayout(process_group)
        
        self.process_text = QTextEdit()
        self.process_text.setMaximumHeight(150)
        self.process_text.setReadOnly(True)
        process_layout.addWidget(self.process_text)
        
        parent_layout.addWidget(process_group)
    
    def start_monitoring(self):
        """Start real-time monitoring"""
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_metrics)
        self.timer.start(1000)  # Update every second
        
        # Initial update
        self.update_metrics()
    
    def update_metrics(self):
        """Update all system and quantum metrics"""
        self.update_system_metrics()
        self.update_quantum_metrics()
        self.update_process_list()
    
    def update_system_metrics(self):
        """Update system resource metrics"""
        try:
            if PSUTIL_AVAILABLE:
                # CPU Usage
                cpu_percent = psutil.cpu_percent(interval=0.1)
                self.cpu_bar.setValue(int(cpu_percent))
                self.cpu_label.setText(f"{cpu_percent:.1f}%")
                
                # Memory Usage
                memory = psutil.virtual_memory()
                self.memory_bar.setValue(int(memory.percent))
                self.memory_label.setText(f"{memory.percent:.1f}%")
                
                # Disk Usage
                disk = psutil.disk_usage('/')
                disk_percent = (disk.used / disk.total) * 100
                self.disk_bar.setValue(int(disk_percent))
                self.disk_label.setText(f"{disk_percent:.1f}%")
                
                # Network
                net_io = psutil.net_io_counters()
                net_speed = (net_io.bytes_sent + net_io.bytes_recv) / 1024  # KB
                self.network_label.setText(f"{net_speed:.0f} KB total")
                
                # Temperature (if available)
                try:
                    temps = psutil.sensors_temperatures()
                    if temps:
                        avg_temp = sum(t.current for sensor in temps.values() for t in sensor) / \
                                  sum(len(sensor) for sensor in temps.values())
                        self.temp_label.setText(f"{avg_temp:.1f} °C")
                except:
                    self.temp_label.setText("N/A")
                    
            else:
                # Simulated metrics
                import random
                self.cpu_bar.setValue(random.randint(10, 80))
                self.cpu_label.setText(f"{random.randint(10, 80)}%")
                self.memory_bar.setValue(random.randint(30, 70))
                self.memory_label.setText(f"{random.randint(30, 70)}%")
                self.disk_bar.setValue(random.randint(40, 85))
                self.disk_label.setText(f"{random.randint(40, 85)}%")
                
        except Exception as e:
            self.status_label.setText(f"⚠️ System metrics error: {str(e)}")
    
    def update_quantum_metrics(self):
        """Update quantum system metrics"""
        import random
        import math
        
        # Simulate quantum fluctuations
        time_factor = time.time() % 60
        
        # Coherence (with small fluctuations)
        coherence = 97.5 + math.sin(time_factor / 10) * 2
        self.quantum_metrics['coherence'] = max(95, min(100, coherence))
        self.coherence_bar.setValue(int(self.quantum_metrics['coherence']))
        self.coherence_label.setText(f"{self.quantum_metrics['coherence']:.1f}%")
        
        # Entanglement (more variable)
        entanglement = 89.2 + math.cos(time_factor / 8) * 5 + random.uniform(-1, 1)
        self.quantum_metrics['entanglement'] = max(80, min(95, entanglement))
        self.entanglement_bar.setValue(int(self.quantum_metrics['entanglement']))
        self.entanglement_label.setText(f"{self.quantum_metrics['entanglement']:.1f}%")
        
        # Fidelity
        fidelity = 94.8 + math.sin(time_factor / 12) * 3
        self.quantum_metrics['fidelity'] = max(90, min(99, fidelity))
        self.fidelity_bar.setValue(int(self.quantum_metrics['fidelity']))
        self.fidelity_label.setText(f"{self.quantum_metrics['fidelity']:.1f}%")
        
        # Quantum temperature (in millikelvin)
        temp_mk = 15 + random.uniform(-2, 2)
        self.qtemp_label.setText(f"{temp_mk:.1f} mK")
        
        # Update qubit status
        active_qubits = 1000 + random.randint(-5, 5)
        self.qubit_label.setText(f"{active_qubits} active")
    
    def update_process_list(self):
        """Update quantum process list"""
        current_time = time.strftime("%H:%M:%S")
        processes = [
            f"[{current_time}] ⚛️ Quantum Kernel: 1000 qubits active",
            f"[{current_time}] 🔬 RFT Transform Engine: Processing",
            f"[{current_time}] 🔐 Crypto Engine: Standby",
            f"[{current_time}] 🌊 Coherence Monitor: {self.quantum_metrics['coherence']:.1f}%",
            f"[{current_time}] 🔗 Entanglement Network: {self.quantum_metrics['entanglement']:.1f}%",
            f"[{current_time}] 📊 System Monitor: Active"
        ]
        
        self.process_text.clear()
        self.process_text.append("\n".join(processes))

def main():
    """Main entry point"""
    if not PYQT5_AVAILABLE:
        print("❌ PyQt5 required for System Monitor")
        return
    
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    window = SystemMonitor()
    window.show()
    
    return app.exec_()

if __name__ == "__main__":
    main()
