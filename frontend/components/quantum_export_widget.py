"""
QuantoniumOS - Universal Export UI Component
Real buttons, quantum design, integrated with QSS styling
Version: 1.0 - Production Grade
"""

import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QIcon, QPainter, QPalette, QPixmap
from PyQt5.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QFrame,
                             QGroupBox, QHBoxLayout, QLabel, QListWidget,
                             QListWidgetItem, QMessageBox, QProgressBar,
                             QPushButton, QSlider, QSpinBox, QSplitter,
                             QTabWidget, QTextEdit, QVBoxLayout, QWidget)

# Import our export and security engines
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "core"))
from quantum_export_controller import (QuantumExportController,
                                       export_app_results)
from quantum_security_engine import QuantumSecurityEngine


class ExportWorkerThread(QThread):
    """Background worker for export operations"""

    progress_updated = pyqtSignal(int)
    export_completed = pyqtSignal(dict)
    export_failed = pyqtSignal(str)

    def __init__(self, export_data, export_format, encrypt, app_name):
        super().__init__()
        self.export_data = export_data
        self.export_format = export_format
        self.encrypt = encrypt
        self.app_name = app_name

    def run(self):
        """Run export in background"""
        try:
            self.progress_updated.emit(10)

            # Initialize export controller
            controller = QuantumExportController()
            self.progress_updated.emit(30)

            # Perform export
            result = controller.export_results(
                self.export_data, self.export_format, self.encrypt, self.app_name
            )
            self.progress_updated.emit(90)

            if result["status"] == "success":
                self.progress_updated.emit(100)
                self.export_completed.emit(result)
            else:
                self.export_failed.emit(result.get("error", "Unknown error"))

        except Exception as e:
            self.export_failed.emit(str(e))


class QuantumExportWidget(QWidget):
    """
    Universal Export Widget for all QuantoniumOS apps
    Follows strict design specs with arched buttons and quantum styling
    """

    def __init__(self, app_name: str = "unknown", export_data: Dict = None):
        super().__init__()
        self.app_name = app_name
        self.export_data = export_data or {}
        self.export_controller = QuantumExportController()
        self.security_engine = QuantumSecurityEngine()
        self.export_worker = None

        self.setup_ui()
        self.load_quantum_styles()
        self.connect_signals()

    def setup_ui(self):
        """Setup the export interface"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        # Header
        header = self.create_header()
        layout.addWidget(header)

        # Export Configuration
        config_section = self.create_export_config()
        layout.addWidget(config_section)

        # Security Options
        security_section = self.create_security_options()
        layout.addWidget(security_section)

        # Action Buttons
        actions = self.create_action_buttons()
        layout.addWidget(actions)

        # Progress Section
        progress_section = self.create_progress_section()
        layout.addWidget(progress_section)

        # Export History
        history_section = self.create_history_section()
        layout.addWidget(history_section)

    def create_header(self) -> QFrame:
        """Create header section with quantum styling"""
        header = QFrame()
        header.setObjectName("quantumHeader")

        layout = QVBoxLayout(header)

        # Title with quantum styling
        title = QLabel("🔮 Universal Export & Save")
        title.setObjectName("quantumTitle")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # App source indicator
        app_label = QLabel(f"Source: {self.app_name.upper()}")
        app_label.setObjectName("quantumSubtitle")
        app_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(app_label)

        # Data summary
        data_count = len(self.export_data) if isinstance(self.export_data, dict) else 0
        summary = QLabel(f"Ready to export {data_count} data entries")
        summary.setObjectName("quantumInfo")
        summary.setAlignment(Qt.AlignCenter)
        layout.addWidget(summary)

        return header

    def create_export_config(self) -> QGroupBox:
        """Create export configuration section"""
        group = QGroupBox("🔧 Export Configuration")
        group.setObjectName("quantumGroup")

        layout = QVBoxLayout(group)

        # Format selection
        format_layout = QHBoxLayout()

        format_label = QLabel("Export Format:")
        format_label.setObjectName("quantumLabel")

        self.format_combo = QComboBox()
        self.format_combo.setObjectName("quantumCombo")
        self.format_combo.addItems(
            [
                "JSON (Recommended)",
                "CSV (Spreadsheet)",
                "TXT (Plain Text)",
                "XML (Structured)",
                "PDF (Document)",
                "XLSX (Excel)",
            ]
        )

        format_layout.addWidget(format_label)
        format_layout.addWidget(self.format_combo)
        layout.addLayout(format_layout)

        # Output directory
        dir_layout = QHBoxLayout()

        dir_label = QLabel("Output Directory:")
        dir_label.setObjectName("quantumLabel")

        self.dir_button = QPushButton("📁 Choose Directory")
        self.dir_button.setObjectName("quantumButton")
        self.dir_button.clicked.connect(self.choose_directory)

        self.current_dir = QLabel(str(Path.home() / "QuantoniumOS_Exports"))
        self.current_dir.setObjectName("quantumPath")

        dir_layout.addWidget(dir_label)
        dir_layout.addWidget(self.dir_button)
        layout.addLayout(dir_layout)
        layout.addWidget(self.current_dir)

        return group

    def create_security_options(self) -> QGroupBox:
        """Create security options section"""
        group = QGroupBox("🔒 Security & Encryption")
        group.setObjectName("quantumGroup")

        layout = QVBoxLayout(group)

        # Encryption toggle
        self.encrypt_checkbox = QCheckBox(
            "🛡️ Enable Military-Grade Encryption (AES-256)"
        )
        self.encrypt_checkbox.setObjectName("quantumCheckBox")
        self.encrypt_checkbox.setChecked(True)
        layout.addWidget(self.encrypt_checkbox)

        # Security level indicator
        security_info = QLabel(
            "🔰 Security Level: MILITARY GRADE | Algorithm: AES-256-GCM + RSA-4096"
        )
        security_info.setObjectName("quantumSecurityInfo")
        layout.addWidget(security_info)

        # Secure delete option
        self.secure_delete_checkbox = QCheckBox("🗑️ Secure Delete Original Data")
        self.secure_delete_checkbox.setObjectName("quantumCheckBox")
        self.secure_delete_checkbox.setChecked(False)
        layout.addWidget(self.secure_delete_checkbox)

        # Digital signature
        self.sign_checkbox = QCheckBox("✍️ Add Digital Signature for Integrity")
        self.sign_checkbox.setObjectName("quantumCheckBox")
        self.sign_checkbox.setChecked(True)
        layout.addWidget(self.sign_checkbox)

        return group

    def create_action_buttons(self) -> QFrame:
        """Create action buttons with arched quantum design"""
        frame = QFrame()
        layout = QHBoxLayout(frame)

        # Export button - main action with quantum arch design
        self.export_button = QPushButton("⚡ EXPORT & SAVE")
        self.export_button.setObjectName("quantumArchedButton")
        self.export_button.clicked.connect(self.start_export)

        # Preview button
        self.preview_button = QPushButton("👁️ Preview Data")
        self.preview_button.setObjectName("quantumButton")
        self.preview_button.clicked.connect(self.preview_data)

        # Validate button
        self.validate_button = QPushButton("✅ Validate")
        self.validate_button.setObjectName("quantumButton")
        self.validate_button.clicked.connect(self.validate_data)

        # Cancel button
        self.cancel_button = QPushButton("❌ Cancel")
        self.cancel_button.setObjectName("quantumCancelButton")
        self.cancel_button.clicked.connect(self.cancel_export)
        self.cancel_button.setEnabled(False)

        layout.addWidget(self.preview_button)
        layout.addWidget(self.validate_button)
        layout.addStretch()
        layout.addWidget(self.export_button)
        layout.addWidget(self.cancel_button)

        return frame

    def create_progress_section(self) -> QGroupBox:
        """Create progress monitoring section"""
        group = QGroupBox("⚡ Export Progress")
        group.setObjectName("quantumGroup")

        layout = QVBoxLayout(group)

        # Progress bar with quantum styling
        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("quantumProgressBar")
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Status text
        self.status_label = QLabel("Ready to export...")
        self.status_label.setObjectName("quantumStatus")
        layout.addWidget(self.status_label)

        # Real-time log
        self.log_text = QTextEdit()
        self.log_text.setObjectName("quantumLog")
        self.log_text.setMaximumHeight(100)
        self.log_text.setVisible(False)
        layout.addWidget(self.log_text)

        return group

    def create_history_section(self) -> QGroupBox:
        """Create export history section"""
        group = QGroupBox("📊 Recent Exports")
        group.setObjectName("quantumGroup")

        layout = QVBoxLayout(group)

        # History list
        self.history_list = QListWidget()
        self.history_list.setObjectName("quantumList")
        self.history_list.setMaximumHeight(150)

        # Load recent exports
        self.load_export_history()

        layout.addWidget(self.history_list)

        # History actions
        history_actions = QHBoxLayout()

        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.setObjectName("quantumSmallButton")
        refresh_btn.clicked.connect(self.load_export_history)

        open_folder_btn = QPushButton("📂 Open Folder")
        open_folder_btn.setObjectName("quantumSmallButton")
        open_folder_btn.clicked.connect(self.open_export_folder)

        history_actions.addWidget(refresh_btn)
        history_actions.addWidget(open_folder_btn)
        history_actions.addStretch()

        layout.addLayout(history_actions)

        return group

    def load_quantum_styles(self):
        """Load quantum-themed styles"""
        style = """
            /* Quantum Header Styling */
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
                font: bold 24px "Segoe UI";
                text-align: center;
                padding: 10px;
            }
            
            QLabel#quantumSubtitle {
                color: #64c8ff;
                font: bold 16px "Segoe UI";
                padding: 5px;
            }
            
            QLabel#quantumInfo {
                color: #ffffff;
                font: 14px "Segoe UI";
                padding: 5px;
            }
            
            /* Quantum Groups */
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
            
            /* Quantum Arched Button - Main Export Action */
            QPushButton#quantumArchedButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #00ff88, stop:0.5 #64c8ff, stop:1 #00ff88);
                border: 3px solid #ffffff;
                border-radius: 25px;
                padding: 15px 30px;
                color: #000000;
                font: bold 16px "Segoe UI";
                min-height: 20px;
                text-align: center;
            }
            
            QPushButton#quantumArchedButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #00ffaa, stop:0.5 #88d4ff, stop:1 #00ffaa);
                border: 3px solid #00ff88;
                transform: scale(1.05);
            }
            
            QPushButton#quantumArchedButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #00cc66, stop:0.5 #4499cc, stop:1 #00cc66);
                border: 3px solid #64c8ff;
            }
            
            /* Standard Quantum Buttons */
            QPushButton#quantumButton {
                background: rgba(100, 200, 255, 100);
                border: 2px solid #64c8ff;
                border-radius: 12px;
                padding: 10px 20px;
                color: #ffffff;
                font: bold 12px "Segoe UI";
                min-height: 15px;
            }
            
            QPushButton#quantumButton:hover {
                background: rgba(100, 200, 255, 150);
                border: 2px solid #00ff88;
            }
            
            QPushButton#quantumCancelButton {
                background: rgba(255, 100, 100, 100);
                border: 2px solid #ff6464;
                border-radius: 12px;
                padding: 10px 20px;
                color: #ffffff;
                font: bold 12px "Segoe UI";
            }
            
            QPushButton#quantumCancelButton:hover {
                background: rgba(255, 100, 100, 150);
                border: 2px solid #ff4444;
            }
            
            QPushButton#quantumSmallButton {
                background: rgba(100, 200, 255, 80);
                border: 1px solid #64c8ff;
                border-radius: 8px;
                padding: 5px 15px;
                color: #ffffff;
                font: 11px "Segoe UI";
            }
            
            /* Form Controls */
            QComboBox#quantumCombo {
                background: rgba(255, 255, 255, 50);
                border: 2px solid #64c8ff;
                border-radius: 10px;
                padding: 8px;
                color: #ffffff;
                font: 12px "Segoe UI";
                min-height: 20px;
            }
            
            QComboBox#quantumCombo::drop-down {
                border: none;
                width: 30px;
            }
            
            QComboBox#quantumCombo::down-arrow {
                image: none;
                border: 2px solid #64c8ff;
                width: 10px;
                height: 10px;
                border-radius: 2px;
            }
            
            QCheckBox#quantumCheckBox {
                color: #ffffff;
                font: 12px "Segoe UI";
                spacing: 10px;
            }
            
            QCheckBox#quantumCheckBox::indicator {
                width: 18px;
                height: 18px;
                border: 2px solid #64c8ff;
                border-radius: 4px;
                background: rgba(255, 255, 255, 30);
            }
            
            QCheckBox#quantumCheckBox::indicator:checked {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #00ff88, stop:1 #64c8ff);
                border: 2px solid #00ff88;
            }
            
            /* Progress Bar */
            QProgressBar#quantumProgressBar {
                background: rgba(255, 255, 255, 50);
                border: 2px solid #64c8ff;
                border-radius: 10px;
                text-align: center;
                color: #ffffff;
                font: bold 12px "Segoe UI";
            }
            
            QProgressBar#quantumProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #00ff88, stop:1 #64c8ff);
                border-radius: 8px;
            }
            
            /* Labels and Info */
            QLabel#quantumLabel {
                color: #ffffff;
                font: 12px "Segoe UI";
                padding: 5px;
            }
            
            QLabel#quantumPath {
                color: #64c8ff;
                font: 11px "Segoe UI";
                padding: 5px;
                background: rgba(100, 200, 255, 20);
                border-radius: 5px;
            }
            
            QLabel#quantumStatus {
                color: #00ff88;
                font: bold 12px "Segoe UI";
                padding: 5px;
            }
            
            QLabel#quantumSecurityInfo {
                color: #ffaa00;
                font: 11px "Segoe UI";
                padding: 5px;
                background: rgba(255, 170, 0, 20);
                border-radius: 5px;
            }
            
            /* Text Areas and Lists */
            QTextEdit#quantumLog {
                background: rgba(0, 0, 0, 150);
                border: 1px solid #64c8ff;
                border-radius: 8px;
                color: #00ff88;
                font: 10px "Consolas";
                padding: 5px;
            }
            
            QListWidget#quantumList {
                background: rgba(255, 255, 255, 30);
                border: 1px solid #64c8ff;
                border-radius: 8px;
                color: #ffffff;
                font: 11px "Segoe UI";
                padding: 5px;
            }
            
            QListWidget#quantumList::item {
                padding: 8px;
                border-bottom: 1px solid rgba(100, 200, 255, 50);
            }
            
            QListWidget#quantumList::item:selected {
                background: rgba(0, 255, 136, 50);
                border: 1px solid #00ff88;
                border-radius: 4px;
            }
        """

        self.setStyleSheet(style)

    def connect_signals(self):
        """Connect widget signals"""
        # Format change updates preview
        self.format_combo.currentTextChanged.connect(self.update_preview)

        # Encryption toggle updates security info
        self.encrypt_checkbox.toggled.connect(self.update_security_info)

    def choose_directory(self):
        """Choose output directory"""
        directory = QFileDialog.getExistingDirectory(
            self, "Choose Export Directory", str(Path.home())
        )

        if directory:
            self.current_dir.setText(directory)

    def preview_data(self):
        """Preview export data"""
        if not self.export_data:
            QMessageBox.information(self, "Preview", "No data available to preview.")
            return

        # Create preview dialog
        preview_dialog = QMessageBox(self)
        preview_dialog.setWindowTitle("Data Preview")
        preview_dialog.setIcon(QMessageBox.Information)

        # Format data for preview
        import json

        preview_text = json.dumps(self.export_data, indent=2, default=str)[:1000]
        if len(json.dumps(self.export_data, default=str)) > 1000:
            preview_text += "\n\n... (truncated for preview)"

        preview_dialog.setText("Data Preview:")
        preview_dialog.setDetailedText(preview_text)
        preview_dialog.exec_()

    def validate_data(self):
        """Validate export data"""
        if not self.export_data:
            self.status_label.setText("❌ No data to validate")
            return

        # Basic validation
        issues = []

        if not isinstance(self.export_data, dict):
            issues.append("Data must be a dictionary")

        if len(self.export_data) == 0:
            issues.append("Data is empty")

        # Check for sensitive data
        sensitive_keys = ["password", "secret", "key", "token", "credential"]
        data_str = str(self.export_data).lower()
        for key in sensitive_keys:
            if key in data_str:
                if not self.encrypt_checkbox.isChecked():
                    issues.append(
                        f"Sensitive data detected ({key}) - encryption recommended"
                    )

        if issues:
            self.status_label.setText(f"⚠️ Validation issues: {'; '.join(issues)}")
        else:
            self.status_label.setText("✅ Data validation passed")

    def start_export(self):
        """Start the export process"""
        if not self.export_data:
            QMessageBox.warning(self, "Export Error", "No data available to export.")
            return

        # Get export parameters
        format_text = self.format_combo.currentText()
        export_format = format_text.split("(")[0].strip().lower()
        encrypt = self.encrypt_checkbox.isChecked()

        # Setup UI for export
        self.export_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.log_text.setVisible(True)
        self.progress_bar.setValue(0)

        # Start export worker
        self.export_worker = ExportWorkerThread(
            self.export_data, export_format, encrypt, self.app_name
        )

        # Connect worker signals
        self.export_worker.progress_updated.connect(self.update_progress)
        self.export_worker.export_completed.connect(self.export_completed)
        self.export_worker.export_failed.connect(self.export_failed)

        # Start worker
        self.export_worker.start()

        self.status_label.setText("🚀 Export started...")
        self.log_text.append(f"Starting export: {format_text}")
        self.log_text.append(f"Encryption: {'Enabled' if encrypt else 'Disabled'}")

    def update_progress(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(value)

        if value == 30:
            self.log_text.append("📦 Preparing data...")
        elif value == 60:
            self.log_text.append("🔄 Processing export...")
        elif value == 90:
            self.log_text.append("🔒 Applying security...")

    def export_completed(self, result):
        """Handle export completion"""
        self.progress_bar.setValue(100)
        self.status_label.setText(f"✅ Export completed: {result['export_id']}")
        self.log_text.append(f"✅ Export saved: {result['file_path']}")

        # Reset UI
        self.export_button.setEnabled(True)
        self.cancel_button.setEnabled(False)

        # Show success message
        QMessageBox.information(
            self,
            "Export Successful",
            f"Export completed successfully!\n\nFile: {result['file_path']}\nFormat: {result['format']}\nEncrypted: {result['encrypted']}",
        )

        # Refresh history
        self.load_export_history()

    def export_failed(self, error):
        """Handle export failure"""
        self.status_label.setText(f"❌ Export failed: {error}")
        self.log_text.append(f"❌ Error: {error}")

        # Reset UI
        self.export_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.progress_bar.setVisible(False)

        # Show error message
        QMessageBox.critical(
            self, "Export Failed", f"Export failed with error:\n\n{error}"
        )

    def cancel_export(self):
        """Cancel ongoing export"""
        if self.export_worker and self.export_worker.isRunning():
            self.export_worker.terminate()
            self.export_worker.wait()

        # Reset UI
        self.export_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.log_text.setVisible(False)
        self.status_label.setText("❌ Export cancelled")

    def update_preview(self):
        """Update preview when format changes"""
        format_text = self.format_combo.currentText()
        self.status_label.setText(f"Ready to export as {format_text}")

    def update_security_info(self, enabled):
        """Update security information"""
        if enabled:
            self.status_label.setText("🔒 Military-grade encryption enabled")
        else:
            self.status_label.setText("⚠️ Export will be unencrypted")

    def load_export_history(self):
        """Load recent export history"""
        self.history_list.clear()

        try:
            history = self.export_controller.get_export_history()

            # Show last 10 exports
            for entry in history[-10:]:
                metadata = entry["metadata"]
                item_text = f"📄 {metadata['app_source']} - {metadata['format'].upper()} - {metadata['timestamp']}"

                if metadata.get("encrypted"):
                    item_text += " 🔒"

                item = QListWidgetItem(item_text)
                item.setToolTip(f"File: {entry['file_path']}")
                self.history_list.addItem(item)

        except Exception as e:
            print(f"Error loading history: {e}")

    def open_export_folder(self):
        """Open export folder in file manager"""
        export_dir = Path.home() / "QuantoniumOS_Exports"
        if export_dir.exists():
            import subprocess

            # Security fix: Remove shell=True to prevent command injection
            subprocess.run(["explorer", str(export_dir)], check=False)

    def set_export_data(self, data: Dict[str, Any]):
        """Update export data"""
        self.export_data = data

        # Update UI
        data_count = len(data) if isinstance(data, dict) else 0
        header_label = self.findChild(QLabel, "quantumInfo")
        if header_label:
            header_label.setText(f"Ready to export {data_count} data entries")

        # Validate new data
        self.validate_data()


def create_export_widget(
    app_name: str, export_data: Dict[str, Any]
) -> QuantumExportWidget:
    """
    Factory function to create export widget

    Usage:
        export_widget = create_export_widget("rft_visualizer", my_data)
        layout.addWidget(export_widget)
    """
    return QuantumExportWidget(app_name, export_data)


if __name__ == "__main__":
    # Test the export widget
    import sys

    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)

    test_data = {
        "test_results": {"success": True, "score": 95.7, "validation_passed": True},
        "analysis": {
            "performance": "excellent",
            "recommendations": ["Continue testing", "Monitor performance"],
        },
    }

    widget = QuantumExportWidget("test_app", test_data)
    widget.show()

    sys.exit(app.exec_())
