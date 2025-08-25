"""
Example App Integration with Universal Export System
Demonstrates how to integrate export functionality into QuantoniumOS apps
Version: 1.0 - Production Example
"""

import os
import sys
from typing import Any, Dict, List, Optional

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (QApplication, QFrame, QGroupBox, QHBoxLayout,
                             QLabel, QProgressBar, QPushButton, QSplitter,
                             QTabWidget, QTextEdit, QVBoxLayout, QWidget)

# Add paths for our export system
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "core"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "frontend", "components"))

try:
    from quantum_export_controller import export_app_results
    from quantum_export_widget import create_export_widget

    EXPORT_AVAILABLE = True
except ImportError:
    EXPORT_AVAILABLE = False
    print("[WARNING] Export system not available")


class ExampleQuantumApp(QWidget):
    """
    Example QuantoniumOS app showing export integration
    This demonstrates the standard pattern for all apps
    """

    def __init__(self, os_backend=None):
        super().__init__()
        self.os_backend = os_backend
        self.app_name = "example_quantum_app"
        self.app_data = {}
        self.export_widget = None

        self.init_ui()
        self.load_quantum_styles()
        self.generate_sample_data()

    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Example Quantum App - Export Demo")
        self.setGeometry(200, 200, 1000, 700)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        # Header
        header = self.create_header()
        layout.addWidget(header)

        # Main content area
        content_splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(content_splitter)

        # Left panel - app functionality
        left_panel = self.create_app_panel()
        content_splitter.addWidget(left_panel)

        # Right panel - export controls
        right_panel = self.create_export_panel()
        content_splitter.addWidget(right_panel)

        content_splitter.setSizes([600, 400])

    def create_header(self) -> QFrame:
        """Create app header"""
        header = QFrame()
        header.setObjectName("quantumHeader")

        layout = QVBoxLayout(header)

        title = QLabel("🧪 Example Quantum Application")
        title.setObjectName("quantumTitle")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel("Demonstrating Universal Export & Save Integration")
        subtitle.setObjectName("quantumSubtitle")
        subtitle.setAlignment(Qt.AlignCenter)
        layout.addWidget(subtitle)

        return header

    def create_app_panel(self) -> QGroupBox:
        """Create main app functionality panel"""
        group = QGroupBox("🔬 App Functionality")
        group.setObjectName("quantumGroup")

        layout = QVBoxLayout(group)

        # Simulation controls
        controls_frame = QFrame()
        controls_layout = QHBoxLayout(controls_frame)

        self.run_simulation_btn = QPushButton("🚀 Run Quantum Simulation")
        self.run_simulation_btn.setObjectName("quantumButton")
        self.run_simulation_btn.clicked.connect(self.run_simulation)

        self.analyze_btn = QPushButton("📊 Analyze Results")
        self.analyze_btn.setObjectName("quantumButton")
        self.analyze_btn.clicked.connect(self.analyze_results)

        self.clear_btn = QPushButton("🗑️ Clear Data")
        self.clear_btn.setObjectName("quantumCancelButton")
        self.clear_btn.clicked.connect(self.clear_data)

        controls_layout.addWidget(self.run_simulation_btn)
        controls_layout.addWidget(self.analyze_btn)
        controls_layout.addWidget(self.clear_btn)
        controls_layout.addStretch()

        layout.addWidget(controls_frame)

        # Results display
        self.results_text = QTextEdit()
        self.results_text.setObjectName("quantumLog")
        self.results_text.setPlainText("Ready to run quantum simulation...")
        layout.addWidget(self.results_text)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("quantumProgressBar")
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        return group

    def create_export_panel(self) -> QGroupBox:
        """Create export controls panel"""
        group = QGroupBox("💾 Export & Save")
        group.setObjectName("quantumGroup")

        layout = QVBoxLayout(group)

        # Export instructions
        info_label = QLabel("🔮 Export your results using the Universal Export System")
        info_label.setObjectName("quantumInfo")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Quick export buttons
        quick_export_frame = QFrame()
        quick_layout = QVBoxLayout(quick_export_frame)

        self.quick_json_btn = QPushButton("⚡ Quick Export (JSON + Encryption)")
        self.quick_json_btn.setObjectName("quantumArchedButton")
        self.quick_json_btn.clicked.connect(self.quick_export_json)

        self.quick_csv_btn = QPushButton("📊 Export as CSV")
        self.quick_csv_btn.setObjectName("quantumButton")
        self.quick_csv_btn.clicked.connect(self.quick_export_csv)

        self.show_export_ui_btn = QPushButton("🎛️ Show Full Export UI")
        self.show_export_ui_btn.setObjectName("quantumButton")
        self.show_export_ui_btn.clicked.connect(self.show_export_ui)

        quick_layout.addWidget(self.quick_json_btn)
        quick_layout.addWidget(self.quick_csv_btn)
        quick_layout.addWidget(self.show_export_ui_btn)

        layout.addWidget(quick_export_frame)

        # Export status
        self.export_status = QLabel("Ready to export...")
        self.export_status.setObjectName("quantumStatus")
        layout.addWidget(self.export_status)

        # Recent exports
        recent_group = QGroupBox("📋 Recent Exports")
        recent_group.setObjectName("quantumSubGroup")
        recent_layout = QVBoxLayout(recent_group)

        self.recent_exports_text = QTextEdit()
        self.recent_exports_text.setObjectName("quantumLog")
        self.recent_exports_text.setMaximumHeight(150)
        self.recent_exports_text.setPlainText("No exports yet...")
        recent_layout.addWidget(self.recent_exports_text)

        layout.addWidget(recent_group)

        return group

    def load_quantum_styles(self):
        """Load quantum-themed styles"""
        style = """
            /* Header Styling */
            QFrame#quantumHeader {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(20, 30, 60, 240), stop:1 rgba(60, 20, 80, 240));
                border: 2px solid #64c8ff;
                border-radius: 20px;
                padding: 20px;
                margin: 10px;
            }
            
            QLabel#quantumTitle {
                color: #00ff88;
                font: bold 20px "Segoe UI";
                padding: 10px;
            }
            
            QLabel#quantumSubtitle {
                color: #64c8ff;
                font: 14px "Segoe UI";
                padding: 5px;
            }
            
            QLabel#quantumInfo {
                color: #ffffff;
                font: 12px "Segoe UI";
                padding: 10px;
                background: rgba(100, 200, 255, 20);
                border-radius: 8px;
                margin: 5px;
            }
            
            /* Groups */
            QGroupBox#quantumGroup {
                background: rgba(255, 255, 255, 30);
                border: 2px solid rgba(100, 200, 255, 100);
                border-radius: 16px;
                font: bold 14px "Segoe UI";
                color: #00ff88;
                padding-top: 20px;
                margin: 10px;
            }
            
            QGroupBox#quantumGroup::title {
                subcontrol-origin: margin;
                left: 20px;
                padding: 5px 15px;
                color: #00ff88;
                background: rgba(0, 255, 136, 20);
                border-radius: 8px;
            }
            
            QGroupBox#quantumSubGroup {
                background: rgba(255, 255, 255, 15);
                border: 1px solid rgba(100, 200, 255, 80);
                border-radius: 12px;
                font: bold 12px "Segoe UI";
                color: #64c8ff;
                padding-top: 15px;
                margin: 5px;
            }
            
            /* Arched Button */
            QPushButton#quantumArchedButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #00ff88, stop:0.5 #64c8ff, stop:1 #00ff88);
                border: 3px solid #ffffff;
                border-radius: 20px;
                padding: 12px 25px;
                color: #000000;
                font: bold 14px "Segoe UI";
                min-height: 15px;
            }
            
            QPushButton#quantumArchedButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #00ffaa, stop:0.5 #88d4ff, stop:1 #00ffaa);
                border: 3px solid #00ff88;
            }
            
            /* Standard Buttons */
            QPushButton#quantumButton {
                background: rgba(100, 200, 255, 100);
                border: 2px solid #64c8ff;
                border-radius: 10px;
                padding: 8px 16px;
                color: #ffffff;
                font: bold 11px "Segoe UI";
                min-height: 10px;
            }
            
            QPushButton#quantumButton:hover {
                background: rgba(100, 200, 255, 150);
                border: 2px solid #00ff88;
            }
            
            QPushButton#quantumCancelButton {
                background: rgba(255, 100, 100, 100);
                border: 2px solid #ff6464;
                border-radius: 10px;
                padding: 8px 16px;
                color: #ffffff;
                font: bold 11px "Segoe UI";
            }
            
            QPushButton#quantumCancelButton:hover {
                background: rgba(255, 100, 100, 150);
            }
            
            /* Text Areas */
            QTextEdit#quantumLog {
                background: rgba(0, 0, 0, 150);
                border: 1px solid #64c8ff;
                border-radius: 8px;
                color: #00ff88;
                font: 10px "Consolas";
                padding: 8px;
            }
            
            /* Progress Bar */
            QProgressBar#quantumProgressBar {
                background: rgba(255, 255, 255, 50);
                border: 2px solid #64c8ff;
                border-radius: 8px;
                text-align: center;
                color: #ffffff;
                font: bold 10px "Segoe UI";
            }
            
            QProgressBar#quantumProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #00ff88, stop:1 #64c8ff);
                border-radius: 6px;
            }
            
            /* Status Label */
            QLabel#quantumStatus {
                color: #00ff88;
                font: bold 11px "Segoe UI";
                padding: 5px;
                background: rgba(0, 255, 136, 15);
                border-radius: 5px;
            }
        """

        self.setStyleSheet(style)

    def generate_sample_data(self):
        """Generate sample quantum simulation data"""
        import random
import time

        self.app_data = {
            "simulation_id": f"quantum_sim_{int(time.time())}",
            "parameters": {
                "qubits": 8,
                "gates": ["H", "CNOT", "RZ", "RY"],
                "iterations": 1000,
                "noise_level": 0.01,
            },
            "initial_state": [random.random() + random.random() * 1j for _ in range(8)],
            "results": {},
            "metadata": {
                "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "app_version": "1.0.0",
                "quantum_backend": "QuantoniumOS Simulator",
            },
        }

    def run_simulation(self):
        """Run quantum simulation"""
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.results_text.clear()
        self.results_text.append("🚀 Starting quantum simulation...")

        # Simulate progress
        timer = QTimer()
        progress = [0]

        def update_progress():
            progress[0] += 10
            self.progress_bar.setValue(progress[0])

            if progress[0] == 30:
                self.results_text.append("🔧 Initializing quantum circuit...")
            elif progress[0] == 60:
                self.results_text.append("⚡ Applying quantum gates...")
            elif progress[0] == 90:
                self.results_text.append("📊 Computing final state...")
            elif progress[0] >= 100:
                timer.stop()
                self.simulation_completed()

        timer.timeout.connect(update_progress)
        timer.start(200)  # Update every 200ms

    def simulation_completed(self):
        """Handle simulation completion"""
        import random
import time

        # Generate simulation results
        results = {
            "final_state": [random.random() + random.random() * 1j for _ in range(8)],
            "measurement_probabilities": [random.random() for _ in range(8)],
            "fidelity": 0.95 + random.random() * 0.04,
            "execution_time": random.uniform(1.2, 3.8),
            "gate_errors": [random.uniform(0.001, 0.01) for _ in range(4)],
            "coherence_time": random.uniform(50, 100),
        }

        self.app_data["results"] = results
        self.app_data["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

        self.progress_bar.setValue(100)
        self.results_text.append("✅ Simulation completed successfully!")
        self.results_text.append(f"📈 Fidelity: {results['fidelity']:.4f}")
        self.results_text.append(f"⏱️ Execution time: {results['execution_time']:.2f}s")
        self.results_text.append(f"🧠 Coherence time: {results['coherence_time']:.1f}μs")
        self.results_text.append("")
        self.results_text.append("🔄 Ready for analysis and export!")

        QTimer.singleShot(2000, lambda: self.progress_bar.setVisible(False))

    def analyze_results(self):
        """Analyze simulation results"""
        if not self.app_data.get("results"):
            self.results_text.append(
                "❌ No simulation results to analyze. Run simulation first."
            )
            return

        self.results_text.append("")
        self.results_text.append("🔍 Analyzing results...")

        results = self.app_data["results"]

        # Add analysis data
        analysis = {
            "quantum_advantage": results["fidelity"] > 0.98,
            "error_rate": sum(results["gate_errors"]) / len(results["gate_errors"]),
            "performance_grade": "A"
            if results["fidelity"] > 0.98
            else "B"
            if results["fidelity"] > 0.95
            else "C",
            "recommendations": [],
        }

        if analysis["error_rate"] > 0.005:
            analysis["recommendations"].append("Consider error correction protocols")
        if results["coherence_time"] < 60:
            analysis["recommendations"].append("Optimize coherence preservation")
        if results["execution_time"] > 3.0:
            analysis["recommendations"].append("Performance optimization needed")

        if not analysis["recommendations"]:
            analysis["recommendations"].append(
                "Excellent performance - continue current approach"
            )

        self.app_data["analysis"] = analysis

        self.results_text.append(
            f"📊 Performance Grade: {analysis['performance_grade']}"
        )
        self.results_text.append(f"⚠️ Average Error Rate: {analysis['error_rate']:.5f}")
        self.results_text.append(
            f"🏆 Quantum Advantage: {'Yes' if analysis['quantum_advantage'] else 'No'}"
        )
        self.results_text.append("💡 Recommendations:")
        for rec in analysis["recommendations"]:
            self.results_text.append(f"   • {rec}")

        self.export_status.setText("✅ Analysis complete - Ready to export")

    def clear_data(self):
        """Clear all data"""
        self.app_data = {"metadata": self.app_data.get("metadata", {})}
        self.results_text.clear()
        self.results_text.append("🗑️ Data cleared. Ready for new simulation.")
        self.export_status.setText("Ready to export...")
        self.progress_bar.setVisible(False)

    def quick_export_json(self):
        """Quick export as encrypted JSON"""
        if not self.app_data.get("results"):
            self.export_status.setText("❌ No data to export. Run simulation first.")
            return

        self.export_status.setText("🚀 Exporting as encrypted JSON...")

        try:
            if EXPORT_AVAILABLE:
                # Use the universal export system
                result = export_app_results(self.app_data, self.app_name, "json", True)

                if result["status"] == "success":
                    self.export_status.setText(f"✅ Exported: {result['export_id']}")
                    self.update_recent_exports(result)
                else:
                    self.export_status.setText(
                        f"❌ Export failed: {result.get('error', 'Unknown error')}"
                    )
            else:
                # Fallback export
                self.fallback_export("json")

        except Exception as e:
            self.export_status.setText(f"❌ Export error: {str(e)}")

    def quick_export_csv(self):
        """Quick export as CSV"""
        if not self.app_data.get("results"):
            self.export_status.setText("❌ No data to export. Run simulation first.")
            return

        self.export_status.setText("📊 Exporting as CSV...")

        try:
            if EXPORT_AVAILABLE:
                result = export_app_results(self.app_data, self.app_name, "csv", False)

                if result["status"] == "success":
                    self.export_status.setText(f"✅ CSV exported: {result['export_id']}")
                    self.update_recent_exports(result)
                else:
                    self.export_status.setText(
                        f"❌ CSV export failed: {result.get('error', 'Unknown error')}"
                    )
            else:
                self.fallback_export("csv")

        except Exception as e:
            self.export_status.setText(f"❌ CSV export error: {str(e)}")

    def show_export_ui(self):
        """Show full export UI"""
        if not self.app_data.get("results"):
            self.export_status.setText("❌ No data to export. Run simulation first.")
            return

        try:
            if EXPORT_AVAILABLE:
                self.export_widget = create_export_widget(self.app_name, self.app_data)
                if self.export_widget:
                    self.export_widget.show()
                    self.export_status.setText("🎛️ Export UI opened")
                else:
                    self.export_status.setText("❌ Could not create export UI")
            else:
                self.export_status.setText("❌ Export UI not available in fallback mode")

        except Exception as e:
            self.export_status.setText(f"❌ UI error: {str(e)}")

    def fallback_export(self, format_type):
        """Fallback export when enhanced system not available"""
        import json
import time
        from pathlib import Path

        try:
            export_dir = Path.home() / "QuantoniumOS_Exports_Basic"
            export_dir.mkdir(exist_ok=True)

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{self.app_name}_export_{timestamp}.json"
            file_path = export_dir / filename

            with open(file_path, "w") as f:
                json.dump(self.app_data, f, indent=2, default=str)

            self.export_status.setText(f"✅ Basic export saved: {filename}")
            self.recent_exports_text.append(
                f"📄 {timestamp} - {format_type.upper()} - {filename}"
            )

        except Exception as e:
            self.export_status.setText(f"❌ Fallback export failed: {str(e)}")

    def update_recent_exports(self, result):
        """Update recent exports display"""
        export_info = f"📄 {result['timestamp']} - {result['format'].upper()}"
        if result.get("encrypted"):
            export_info += " 🔒"
        export_info += f" - {result['export_id']}"

        self.recent_exports_text.append(export_info)

        # Keep only last 5 entries
        content = self.recent_exports_text.toPlainText().split("\n")
        if len(content) > 5:
            content = content[-5:]
            self.recent_exports_text.setPlainText("\n".join(content))


def main():
    """Main function to run the example app"""
    app = QApplication(sys.argv)

    # Create the example app
    example_app = ExampleQuantumApp()
    example_app.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
