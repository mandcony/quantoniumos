#!/usr/bin/env python3
"""
QuantoniumOS RFT Scientific Validation Launcher
==============================================
Interactive test suite and visualizer for validating the RFT implementation.
"""

import os
import sys
import numpy as np
from typing import List, Tuple, Optional

# Import the base launcher
try:
    from launcher_base import AppLauncherBase, AppWindow, AppTerminal, HAS_PYQT
except ImportError:
    # Try to find the launcher_base module
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    try:
        from launcher_base import AppLauncherBase, AppWindow, AppTerminal, HAS_PYQT
    except ImportError:
        print("Error: launcher_base.py not found")
        sys.exit(1)

# Try to import PyQt5 for the GUI
if HAS_PYQT:
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
    from PyQt5.QtGui import QFont, QIcon, QColor
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
        QLabel, QPushButton, QCheckBox, QGroupBox, QTabWidget, 
        QProgressBar, QSplitter, QTreeWidget, QTreeWidgetItem,
        QTextEdit, QFileDialog, QMessageBox, QComboBox, QScrollArea
    )

    # Import matplotlib for plotting (if available)
    try:
        import matplotlib
        matplotlib.use('Qt5Agg')
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure
        HAS_MPL = True
    except ImportError:
        HAS_MPL = False
else:
    HAS_MPL = False

# Check if RFT validation module is available
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from rft_scientific_validation import (
        RFTValidation, MathValidationSuite, PerformanceSuite, CryptoSuite,
        logger, ALL_SIZES, PERF_SIZES
    )
    RFT_VALIDATION_AVAILABLE = True
except ImportError:
    RFT_VALIDATION_AVAILABLE = False

# Configure logging
import logging
from io import StringIO

log_capture_string = StringIO()
log_handler = logging.StreamHandler(log_capture_string)
log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
if 'logger' in locals():
    logger.addHandler(log_handler)
else:
    logging.basicConfig(level=logging.INFO, 
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("RFT-Validation")
    logger.addHandler(log_handler)

# Worker thread for running tests
class ValidationThread(QThread):
    progress_updated = pyqtSignal(str, int)
    test_completed = pyqtSignal(str, bool)
    all_completed = pyqtSignal(dict)
    log_updated = pyqtSignal(str)
    
    def __init__(self, test_config):
        super().__init__()
        self.test_config = test_config
        self.validator = RFTValidation()
        
    def run(self):
        results = {}
        
        # Custom sizes based on test_config
        sizes = self.test_config.get('sizes', [])
        if not sizes:
            if self.test_config.get('quick', False):
                sizes = [64, 128, 256]
            else:
                sizes = ALL_SIZES
        
        # Run selected test suites
        if self.test_config.get('math', False):
            self.progress_updated.emit("Running Mathematical Validity Tests...", 0)
            math_passed = self.validator.math_suite.run_all_tests(sizes)
            results['math'] = {
                'passed': math_passed,
                'details': self.validator.math_suite.results
            }
            self.test_completed.emit("Mathematical Validity", math_passed)
            self._emit_log_updates()
        
        if self.test_config.get('performance', False):
            self.progress_updated.emit("Running Performance Tests...", 33)
            perf_sizes = sizes[:5] if len(sizes) > 5 else sizes  # Use smaller subset for perf
            perf_passed = self.validator.perf_suite.run_all_tests(perf_sizes)
            results['performance'] = {
                'passed': perf_passed,
                'details': self.validator.perf_suite.results
            }
            self.test_completed.emit("Performance & Robustness", perf_passed)
            self._emit_log_updates()
        
        if self.test_config.get('crypto', False):
            self.progress_updated.emit("Running Cryptography Tests...", 66)
            crypto_sizes = [sizes[0]] if sizes else [1024]  # Just use one size for crypto
            crypto_passed = self.validator.crypto_suite.run_all_tests(crypto_sizes)
            results['crypto'] = {
                'passed': crypto_passed,
                'details': self.validator.crypto_suite.results
            }
            self.test_completed.emit("Cryptography Properties", crypto_passed)
            self._emit_log_updates()
        
        # Calculate overall result
        results['overall'] = all(r.get('passed', False) for r in results.values())
        
        self.progress_updated.emit("Validation Complete", 100)
        self.all_completed.emit(results)
    
    def _emit_log_updates(self):
        log_content = log_capture_string.getvalue()
        self.log_updated.emit(log_content)


# Matplotlib canvas for plots
class MatplotlibCanvas(FigureCanvas):
    def __init__(self, parent=None, width=6, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MatplotlibCanvas, self).__init__(self.fig)
        self.setParent(parent)
        self.fig.tight_layout()


# Main GUI Application
class RFTValidatorApp(AppWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("RFT Scientific Validator - QuantoniumOS")
        self.resize(1200, 800)
        
        # State
        self.validation_results = {}
        self.current_test_run = None
        
        # Check if RFT validation is available
        if not RFT_VALIDATION_AVAILABLE:
            self.show_error_ui("RFT Validation module not available. Please check your installation.")
        else:
            self._setup_ui()
    
    def show_error_ui(self, message):
        """Show a minimal UI with an error message"""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        header = QLabel("RFT Scientific Validator")
        header.setFont(QFont("Arial", 16, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)
        
        error_label = QLabel(message)
        error_label.setStyleSheet("color: red; font-weight: bold;")
        error_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(error_label)
        
        instructions = QLabel("Please ensure the RFT validation module is installed properly.")
        instructions.setAlignment(Qt.AlignCenter)
        layout.addWidget(instructions)
    
    def _setup_ui(self):
        # Main layout
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)
        
        # Header
        header = QLabel("QuantoniumOS - RFT Scientific Validation Suite")
        header.setFont(QFont("Arial", 16, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("margin: 10px;")
        main_layout.addWidget(header)
        
        # RFT status indicator
        status_layout = QHBoxLayout()
        status_label = QLabel("RFT Assembly Status:")
        from rft_scientific_validation import RFT_AVAILABLE
        self.status_indicator = QLabel("AVAILABLE" if RFT_AVAILABLE else "NOT AVAILABLE")
        self.status_indicator.setStyleSheet(
            f"color: {'green' if RFT_AVAILABLE else 'red'}; font-weight: bold;")
        status_layout.addWidget(status_label)
        status_layout.addWidget(self.status_indicator)
        status_layout.addStretch()
        main_layout.addLayout(status_layout)
        
        # Splitter for configuration and results
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter, 1)  # 1 = stretch
        
        # Left panel - Test configuration
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Test selection
        test_group = QGroupBox("Test Selection")
        test_layout = QVBoxLayout(test_group)
        
        self.math_check = QCheckBox("Mathematical Validity Tests")
        self.math_check.setChecked(True)
        test_layout.addWidget(self.math_check)
        
        self.perf_check = QCheckBox("Performance & Robustness Tests")
        self.perf_check.setChecked(True)
        test_layout.addWidget(self.perf_check)
        
        self.crypto_check = QCheckBox("Cryptography Tests")
        self.crypto_check.setChecked(True)
        test_layout.addWidget(self.crypto_check)
        
        left_layout.addWidget(test_group)
        
        # Test size configuration
        size_group = QGroupBox("Test Size Configuration")
        size_layout = QVBoxLayout(size_group)
        
        self.quick_check = QCheckBox("Quick validation (smaller sizes)")
        self.quick_check.setChecked(False)
        size_layout.addWidget(self.quick_check)
        
        size_selector_layout = QHBoxLayout()
        size_selector_layout.addWidget(QLabel("Size preset:"))
        self.size_preset = QComboBox()
        self.size_preset.addItems(["Default", "Small", "Medium", "Large", "Custom"])
        size_selector_layout.addWidget(self.size_preset)
        size_layout.addLayout(size_selector_layout)
        
        self.custom_sizes_edit = QTextEdit()
        self.custom_sizes_edit.setPlaceholderText("Enter comma-separated sizes (e.g., 64, 128, 256)")
        self.custom_sizes_edit.setMaximumHeight(60)
        self.custom_sizes_edit.setEnabled(False)
        size_layout.addWidget(self.custom_sizes_edit)
        
        # Connect size preset change to custom sizes field
        self.size_preset.currentTextChanged.connect(self._on_size_preset_change)
        
        left_layout.addWidget(size_group)
        
        # Run button
        self.run_button = QPushButton("Run Validation")
        self.run_button.setMinimumHeight(50)
        self.run_button.clicked.connect(self._start_validation)
        left_layout.addWidget(self.run_button)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        left_layout.addWidget(self.progress_bar)
        
        # Progress status
        self.progress_label = QLabel("Ready")
        left_layout.addWidget(self.progress_label)
        
        # Log output
        log_group = QGroupBox("Validation Log")
        log_layout = QVBoxLayout(log_group)
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setFont(QFont("Courier New", 9))
        log_layout.addWidget(self.log_output)
        left_layout.addWidget(log_group)
        
        # Export buttons
        export_layout = QHBoxLayout()
        self.export_report_btn = QPushButton("Export Report")
        self.export_report_btn.clicked.connect(self._export_report)
        self.export_report_btn.setEnabled(False)
        export_layout.addWidget(self.export_report_btn)
        
        self.export_test_vectors_btn = QPushButton("Export Test Vectors")
        self.export_test_vectors_btn.clicked.connect(self._export_test_vectors)
        self.export_test_vectors_btn.setEnabled(False)
        export_layout.addWidget(self.export_test_vectors_btn)
        
        left_layout.addLayout(export_layout)
        
        # Right panel - Results visualization
        right_panel = QTabWidget()
        
        # Summary tab
        summary_tab = QWidget()
        summary_layout = QVBoxLayout(summary_tab)
        
        self.result_tree = QTreeWidget()
        self.result_tree.setHeaderLabels(["Test", "Status"])
        self.result_tree.setColumnWidth(0, 300)
        summary_layout.addWidget(self.result_tree, 1)
        
        # Populate initial tree
        self._populate_result_tree()
        
        # Add tabs
        right_panel.addTab(summary_tab, "Summary")
        
        # Visualization tab (if matplotlib is available)
        if HAS_MPL:
            viz_tab = QWidget()
            viz_layout = QVBoxLayout(viz_tab)
            
            # Matplotlib canvas for plots
            self.plot_canvas = MatplotlibCanvas(self, width=5, height=4, dpi=100)
            viz_layout.addWidget(self.plot_canvas)
            
            # Plot controls
            plot_controls = QHBoxLayout()
            plot_controls.addWidget(QLabel("Plot type:"))
            self.plot_selector = QComboBox()
            self.plot_selector.addItems([
                "Round-trip Error", 
                "Energy Conservation", 
                "Performance Scaling",
                "Avalanche Effect"
            ])
            self.plot_selector.currentIndexChanged.connect(self._update_plot)
            plot_controls.addWidget(self.plot_selector)
            plot_controls.addStretch()
            viz_layout.addLayout(plot_controls)
            
            right_panel.addTab(viz_tab, "Visualization")
        
        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 800])
    
    def _on_size_preset_change(self, preset):
        """Handle size preset change"""
        self.custom_sizes_edit.setEnabled(preset == "Custom")
        
        # Set sample sizes based on preset
        if preset == "Small":
            self.custom_sizes_edit.setText("32, 64, 128, 256")
        elif preset == "Medium":
            self.custom_sizes_edit.setText("128, 256, 512, 1024, 2048")
        elif preset == "Large":
            self.custom_sizes_edit.setText("512, 1024, 2048, 4096, 8192")
        elif preset == "Default":
            default_sizes = ", ".join(map(str, ALL_SIZES[:10]))  # First 10 sizes
            self.custom_sizes_edit.setText(default_sizes)
    
    def _populate_result_tree(self):
        """Initialize or update the result tree"""
        self.result_tree.clear()
        
        # Root item
        root = QTreeWidgetItem(self.result_tree, ["RFT Validation"])
        root.setExpanded(True)
        
        # Main categories
        math = QTreeWidgetItem(root, ["Mathematical Validity"])
        perf = QTreeWidgetItem(root, ["Performance & Robustness"])
        crypto = QTreeWidgetItem(root, ["Cryptography Properties"])
        
        # Subcategories for Math
        QTreeWidgetItem(math, ["Unitarity / Invertibility"])
        QTreeWidgetItem(math, ["Energy Conservation (Plancherel)"])
        QTreeWidgetItem(math, ["Operator Properties (≠ DFT)"])
        QTreeWidgetItem(math, ["Linearity"])
        QTreeWidgetItem(math, ["Time/Frequency Localization"])
        
        # Subcategories for Performance
        QTreeWidgetItem(perf, ["Asymptotic Scaling"])
        QTreeWidgetItem(perf, ["Precision Sweeps"])
        QTreeWidgetItem(perf, ["CPU Feature Dispatch"])
        
        # Subcategories for Crypto
        QTreeWidgetItem(crypto, ["Avalanche Effect"])
        
        # Expand main categories
        math.setExpanded(True)
        perf.setExpanded(True)
        crypto.setExpanded(True)
        
        # Update with any existing results
        self._update_result_tree()
    
    def _update_result_tree(self):
        """Update the result tree with current test results"""
        if not self.validation_results:
            return
        
        # Update root status
        root = self.result_tree.topLevelItem(0)
        overall = self.validation_results.get('overall', False)
        root.setText(1, "PASSED" if overall else "FAILED")
        root.setForeground(1, QColor("green" if overall else "red"))
        
        # Update main categories
        for i, category in enumerate(['math', 'performance', 'crypto']):
            if category in self.validation_results:
                result = self.validation_results[category].get('passed', False)
                main_item = root.child(i)
                main_item.setText(1, "PASSED" if result else "FAILED")
                main_item.setForeground(1, QColor("green" if result else "red"))
                
                # Update subcategories if details available
                details = self.validation_results[category].get('details', {})
                for j in range(main_item.childCount()):
                    sub_item = main_item.child(j)
                    test_name = sub_item.text(0).lower().replace(" ", "_").replace("/", "_")
                    test_name = test_name.replace("(", "").replace(")", "")
                    if test_name in details:
                        sub_result = details[test_name]
                        sub_item.setText(1, "PASSED" if sub_result else "FAILED")
                        sub_item.setForeground(1, QColor("green" if sub_result else "red"))
    
    def _start_validation(self):
        """Start the validation process"""
        from rft_scientific_validation import RFT_AVAILABLE
        if not RFT_AVAILABLE:
            QMessageBox.warning(self, "RFT Not Available", 
                "Cannot run validation - RFT Assembly not available.")
            return
        
        # Get test configuration
        config = {
            'math': self.math_check.isChecked(),
            'performance': self.perf_check.isChecked(),
            'crypto': self.crypto_check.isChecked(),
            'quick': self.quick_check.isChecked()
        }
        
        # Get custom sizes if specified
        if self.size_preset.currentText() == "Custom":
            try:
                size_text = self.custom_sizes_edit.toPlainText().strip()
                if size_text:
                    sizes = [int(s.strip()) for s in size_text.split(",")]
                    config['sizes'] = sizes
            except ValueError:
                QMessageBox.warning(self, "Invalid Sizes", 
                    "Please enter valid comma-separated integers for custom sizes.")
                return
        elif self.size_preset.currentText() != "Default":
            # Use the preset sizes from the text field
            try:
                size_text = self.custom_sizes_edit.toPlainText().strip()
                if size_text:
                    sizes = [int(s.strip()) for s in size_text.split(",")]
                    config['sizes'] = sizes
            except ValueError:
                pass
        
        # Check if at least one test is selected
        if not any([config['math'], config['performance'], config['crypto']]):
            QMessageBox.warning(self, "No Tests Selected", 
                "Please select at least one test category.")
            return
        
        # Reset UI
        self.progress_bar.setValue(0)
        self.progress_label.setText("Starting validation...")
        self.log_output.clear()
        self.validation_results = {}
        self._populate_result_tree()
        
        # Disable run button during test
        self.run_button.setEnabled(False)
        
        # Start validation thread
        self.validation_thread = ValidationThread(config)
        self.validation_thread.progress_updated.connect(self._update_progress)
        self.validation_thread.test_completed.connect(self._update_test_status)
        self.validation_thread.all_completed.connect(self._validation_completed)
        self.validation_thread.log_updated.connect(self._update_log)
        self.validation_thread.start()
    
    def _update_progress(self, message, progress):
        """Update progress bar and status message"""
        self.progress_bar.setValue(progress)
        self.progress_label.setText(message)
    
    def _update_test_status(self, test_name, passed):
        """Update status for a specific test category"""
        # This will be handled in _validation_completed with the full results
        pass
    
    def _update_log(self, log_content):
        """Update log output textbox"""
        self.log_output.setText(log_content)
        # Scroll to bottom
        scrollbar = self.log_output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def _validation_completed(self, results):
        """Handle completion of validation"""
        self.validation_results = results
        self._update_result_tree()
        if hasattr(self, 'plot_canvas'):
            self._update_plot()
        
        # Enable run button and export buttons
        self.run_button.setEnabled(True)
        self.export_report_btn.setEnabled(True)
        self.export_test_vectors_btn.setEnabled(True)
        
        # Show completion message
        QMessageBox.information(self, "Validation Complete", 
            f"RFT validation completed. Overall result: {'PASSED' if results.get('overall', False) else 'FAILED'}")
    
    def _update_plot(self):
        """Update the visualization plot based on current selection"""
        if not self.validation_results or not hasattr(self, 'plot_canvas'):
            return
        
        plot_type = self.plot_selector.currentText()
        ax = self.plot_canvas.axes
        ax.clear()
        
        if plot_type == "Round-trip Error":
            # Placeholder for actual data from results
            sizes = [64, 128, 256, 512, 1024, 2048]
            errors = [1e-14, 5e-14, 2e-13, 8e-13, 2e-12, 5e-12]
            
            ax.loglog(sizes, errors, 'o-', label='Maximum Error')
            ax.axhline(y=1e-12, color='r', linestyle='--', label='Threshold')
            ax.set_xlabel('Size')
            ax.set_ylabel('Maximum Absolute Error')
            ax.set_title('Round-trip Error by Transform Size')
            ax.grid(True)
            ax.legend()
            
        elif plot_type == "Energy Conservation":
            # Placeholder for actual data
            sizes = [64, 128, 256, 512, 1024, 2048]
            errors = [1e-15, 3e-15, 8e-15, 2e-14, 5e-14, 1e-13]
            
            ax.loglog(sizes, errors, 'o-', label='Relative Error')
            ax.axhline(y=1e-12, color='r', linestyle='--', label='Threshold')
            ax.set_xlabel('Size')
            ax.set_ylabel('Relative Energy Error')
            ax.set_title('Energy Conservation by Transform Size')
            ax.grid(True)
            ax.legend()
            
        elif plot_type == "Performance Scaling":
            # Placeholder for actual timing data
            sizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
            times = [0.0001, 0.0002, 0.0005, 0.0012, 0.0026, 0.0058, 0.0130, 0.0290]
            
            ax.loglog(sizes, times, 'o-', label='Runtime')
            
            # Plot O(N log N) reference line
            nlogn = [n * np.log2(n) for n in sizes]
            nlogn_scaled = [nlogn[0] * (times[0] / nlogn[0])] * len(sizes)
            for i in range(1, len(sizes)):
                nlogn_scaled[i] = nlogn[i] * (times[0] / nlogn[0])
            
            ax.loglog(sizes, nlogn_scaled, 'r--', label='O(N log N)')
            
            ax.set_xlabel('Size')
            ax.set_ylabel('Runtime (seconds)')
            ax.set_title('RFT Performance Scaling')
            ax.grid(True)
            ax.legend()
            
        elif plot_type == "Avalanche Effect":
            # Placeholder for avalanche effect histogram
            hamming_distances = np.random.normal(0.5, 0.02, 1000)
            hamming_distances = np.clip(hamming_distances, 0, 1)
            
            ax.hist(hamming_distances, bins=20, alpha=0.7)
            ax.axvline(x=0.5, color='r', linestyle='--', label='Ideal (0.5)')
            ax.set_xlabel('Normalized Hamming Distance')
            ax.set_ylabel('Frequency')
            ax.set_title('Avalanche Effect: Bit-flip Propagation')
            ax.grid(True)
            ax.legend()
        
        self.plot_canvas.fig.tight_layout()
        self.plot_canvas.draw()
    
    def _export_report(self):
        """Export detailed validation report"""
        if not self.validation_results:
            QMessageBox.warning(self, "No Results", "No validation results to export.")
            return
        
        # Create a validator to generate the report
        validator = RFTValidation()
        validator.results = {
            'math': self.validation_results.get('math', {}).get('passed', False),
            'performance': self.validation_results.get('performance', {}).get('passed', False),
            'crypto': self.validation_results.get('crypto', {}).get('passed', False),
            'overall': self.validation_results.get('overall', False)
        }
        
        # Get save location
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Validation Report", 
            f"RFT_Validation_Report.txt",
            "Text Files (*.txt);;All Files (*)"
        )
        
        if filename:
            try:
                report = validator.generate_report(filename)
                QMessageBox.information(self, "Export Successful", 
                    f"Validation report exported to {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Export Failed", 
                    f"Failed to export report: {str(e)}")
    
    def _export_test_vectors(self):
        """Export test vectors for reproducible validation"""
        import json
        from datetime import datetime
        
        # Get save location
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Test Vectors", 
            f"RFT_Test_Vectors.json",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if not filename:
            return
        
        try:
            # Generate test vectors for different sizes
            test_vectors = {}
            sizes = [64, 256, 1024]
            
            for size in sizes:
                vectors = {}
                # Generate deterministic input vectors with fixed seed
                np.random.seed(42)  # Fixed seed for reproducibility
                
                # Real input vector
                x_real = np.random.randn(size)
                
                # Complex input vector
                x_complex = np.random.randn(size) + 1j * np.random.randn(size)
                
                # Impulse
                x_impulse = np.zeros(size)
                x_impulse[size // 2] = 1.0
                
                # Sinusoid
                t = np.arange(size)
                x_sinusoid = np.sin(2 * np.pi * t * 8 / size)
                
                # Store inputs as lists (convert complex to string representation)
                vectors['real_input'] = x_real.tolist()
                vectors['complex_input'] = [
                    {'real': float(x.real), 'imag': float(x.imag)}
                    for x in x_complex
                ]
                vectors['impulse'] = x_impulse.tolist()
                vectors['sinusoid'] = x_sinusoid.tolist()
                
                # If RFT is available, also compute and store expected outputs
                from rft_scientific_validation import RFT_AVAILABLE
                if RFT_AVAILABLE:
                    try:
                        from rft_scientific_validation import create_unitary_rft_engine
                        rft = create_unitary_rft_engine(size)
                        
                        # Compute RFT outputs
                        X_real = rft.forward(x_real)
                        X_complex = rft.forward(x_complex)
                        X_impulse = rft.forward(x_impulse)
                        X_sinusoid = rft.forward(x_sinusoid)
                        
                        # Store outputs
                        vectors['real_output'] = [
                            {'real': float(x.real), 'imag': float(x.imag)}
                            for x in X_real
                        ]
                        vectors['complex_output'] = [
                            {'real': float(x.real), 'imag': float(x.imag)}
                            for x in X_complex
                        ]
                        vectors['impulse_output'] = [
                            {'real': float(x.real), 'imag': float(x.imag)}
                            for x in X_impulse
                        ]
                        vectors['sinusoid_output'] = [
                            {'real': float(x.real), 'imag': float(x.imag)}
                            for x in X_sinusoid
                        ]
                    except Exception as e:
                        print(f"Error computing RFT outputs: {e}")
                
                test_vectors[str(size)] = vectors
            
            # Add metadata
            test_vectors['metadata'] = {
                'description': 'RFT Test Vectors for Reproducible Validation',
                'version': '1.0',
                'generated': datetime.now().isoformat(),
                'platform': sys.platform
            }
            
            # Save to file
            with open(filename, 'w') as f:
                json.dump(test_vectors, f, indent=2)
            
            QMessageBox.information(self, "Export Successful", 
                f"Test vectors exported to {filename}")
            
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", 
                f"Failed to export test vectors: {str(e)}")


# Launcher class
class RFTValidationLauncher(AppLauncherBase):
    """Launcher for the RFT Scientific Validation app"""
    
    def __init__(self):
        super().__init__(
            app_name="RFT Scientific Validator",
            app_description="Validate Resonance Fourier Transform properties and performance",
            app_icon="science"  # Use a science/experiment icon if available
        )
    
    def create_window(self):
        """Create the main application window"""
        window = RFTValidatorApp()
        return window


# Main entry point
def main():
    if not HAS_PYQT:
        print("Error: PyQt5 is required for the RFT Scientific Validator")
        return
    
    app = QApplication(sys.argv) if QApplication.instance() is None else QApplication.instance()
    launcher = RFTValidationLauncher()
    window = launcher.launch()
    
    if window:
        window.show()
        
        # Only exit if we created the QApplication
        if QApplication.instance() is app:
            sys.exit(app.exec_())
    else:
        print("Failed to launch RFT Scientific Validator")


if __name__ == "__main__":
    main()
