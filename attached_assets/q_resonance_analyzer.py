import sys
import os
import numpy as np
import logging
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, 
    QPushButton, QLabel, QFileDialog, QProgressBar, QTabWidget,
    QSplitter, QFrame, QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont
from PyQt5.QtCore import Qt, QTimer, QSize, pyqtSignal, QPoint
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Add the parent directory to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Import QuantoniumOS Resonance Functions
try:
    from image_resonance_analyzer import ImageResonanceAnalyzer
except ImportError as e:
    logging.error(f"Failed to import image_resonance_analyzer: {e}")
    raise

try:
    from apps.wave_primitives import WaveNumber
except ImportError as e:
    logging.error(f"Failed to import wave_primitives: {e}")
    from wave_primitives import WaveNumber  # Try alternative import path

try:
    from geometric_waveform_hash import geometric_waveform_hash
except ImportError as e:
    logging.error(f"Failed to import geometric_waveform_hash: {e}")
    # Define fallback function for standalone testing
    def geometric_waveform_hash(data):
        import hashlib
        return hashlib.sha256(data).hexdigest()[:16]

# Set up logging
APP_DIR = os.path.dirname(os.path.abspath(__file__))
log_dir = APP_DIR if os.access(APP_DIR, os.W_OK) else os.path.join(os.path.expanduser("~"), "temp")
log_file = os.path.join(log_dir, "q_resonance_analyzer.log")
logging.basicConfig(level=logging.DEBUG, 
                    filename=log_file, 
                    filemode="a",
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
logger.addHandler(console_handler)

# Path to external QSS file
STYLES_QSS = os.path.join(ROOT_DIR, "styles.qss")

def load_stylesheet(qss_path):
    """Load and return the contents of the QSS file if it exists."""
    if os.path.exists(qss_path):
        try:
            with open(qss_path, "r", encoding="utf-8") as f:
                logger.info(f"Stylesheet loaded from {qss_path}")
                return f.read()
        except Exception as e:
            logger.warning(f"Error loading stylesheet: {e}")
            return ""
    else:
        logger.warning(f"QSS file not found: {qss_path}")
        return ""

class ResonanceWaveCanvas(FigureCanvas):
    """Canvas for visualizing resonance waveforms."""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setMinimumHeight(200)
        
        # Configure the figure for a quantum aesthetic
        self.fig.patch.set_facecolor('#121212')
        self.axes.set_facecolor('#1e1e1e')
        self.axes.tick_params(colors='white')
        self.axes.spines['bottom'].set_color('white')
        self.axes.spines['top'].set_color('white') 
        self.axes.spines['right'].set_color('white')
        self.axes.spines['left'].set_color('white')
        self.axes.set_title("Resonance Waveform Analysis", color='white')
        self.axes.set_xlabel("Frequency Domain", color='white')
        self.axes.set_ylabel("Amplitude", color='white')
        
    def plot_resonance_data(self, frequencies, amplitudes, phases=None):
        """Plot resonance data as a waveform."""
        self.axes.clear()
        self.axes.plot(frequencies, amplitudes, 'c-', linewidth=1.5)
        
        # Add phase information if available
        if phases is not None:
            phase_color = np.array([phases]) * 360  # Convert to degrees
            points = self.axes.scatter(frequencies, amplitudes, c=phase_color, 
                                      cmap='hsv', s=30, alpha=0.7)
            
        # Update labels and title
        self.axes.set_title("Resonance Waveform Analysis", color='white')
        self.axes.set_xlabel("Frequency Domain", color='white')
        self.axes.set_ylabel("Amplitude", color='white')
        
        # Make sure ticks are visible
        self.axes.tick_params(colors='white')
        
        # Configure grid lines
        self.axes.grid(True, linestyle='--', alpha=0.3, color='gray')
        
        self.fig.tight_layout()
        self.draw()

class SymmetricalPatternCanvas(FigureCanvas):
    """Canvas for visualizing symmetrical patterns in the image."""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setMinimumHeight(200)
        
        # Configure the figure for a quantum aesthetic
        self.fig.patch.set_facecolor('#121212')
        self.axes.set_facecolor('#1e1e1e')
        self.axes.tick_params(colors='white')
        self.axes.spines['bottom'].set_color('white')
        self.axes.spines['top'].set_color('white') 
        self.axes.spines['right'].set_color('white')
        self.axes.spines['left'].set_color('white')
        self.axes.set_title("Symmetry Analysis", color='white')
        
    def plot_symmetry_data(self, symmetry_matrix, symmetry_score):
        """Plot symmetry heatmap from image analysis."""
        self.axes.clear()
        
        # Plot symmetry matrix as a heatmap
        img = self.axes.imshow(symmetry_matrix, cmap='viridis', interpolation='nearest')
        self.fig.colorbar(img, ax=self.axes, label="Symmetry Strength")
        
        # Add symmetry score
        self.axes.set_title(f"Symmetry Analysis (Score: {symmetry_score:.3f})", color='white')
        self.axes.set_xlabel("X Axis", color='white')
        self.axes.set_ylabel("Y Axis", color='white')
        
        self.fig.tight_layout()
        self.draw()

class QResonanceAnalyzer(QMainWindow):
    """Main window for the Quantum Resonance Analyzer application."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Q-Resonance Analyzer")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize variables
        self.current_image_path = None
        self.analyzer = None
        self.analysis_results = None
        
        # Initialize UI
        self.init_ui()
        
        # Load QSS stylesheet
        self.stylesheet = load_stylesheet(STYLES_QSS)
        if self.stylesheet:
            self.setStyleSheet(self.stylesheet)
        
    def init_ui(self):
        """Initialize the UI components."""
        # Main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Top controls
        self.controls_widget = QWidget()
        self.controls_layout = QHBoxLayout(self.controls_widget)
        
        self.load_image_btn = QPushButton("Load Image")
        self.load_image_btn.clicked.connect(self.load_image)
        self.controls_layout.addWidget(self.load_image_btn)
        
        self.analyze_btn = QPushButton("Analyze Resonance")
        self.analyze_btn.clicked.connect(self.analyze_image)
        self.analyze_btn.setEnabled(False)
        self.controls_layout.addWidget(self.analyze_btn)
        
        self.save_results_btn = QPushButton("Save Results")
        self.save_results_btn.clicked.connect(self.save_results)
        self.save_results_btn.setEnabled(False)
        self.controls_layout.addWidget(self.save_results_btn)
        
        self.status_label = QLabel("Ready")
        self.controls_layout.addWidget(self.status_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.controls_layout.addWidget(self.progress_bar)
        
        self.main_layout.addWidget(self.controls_widget)
        
        # Main content area with splitter
        self.splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - Image display
        self.image_frame = QFrame()
        self.image_frame.setFrameShape(QFrame.StyledPanel)
        self.image_layout = QVBoxLayout(self.image_frame)
        
        self.image_label = QLabel("No image loaded")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 400)
        self.image_layout.addWidget(self.image_label)
        
        self.image_info_label = QLabel("")
        self.image_layout.addWidget(self.image_info_label)
        
        self.splitter.addWidget(self.image_frame)
        
        # Right panel - Analysis tabs
        self.analysis_tabs = QTabWidget()
        
        # Waveform tab
        self.waveform_tab = QWidget()
        self.waveform_layout = QVBoxLayout(self.waveform_tab)
        self.waveform_canvas = ResonanceWaveCanvas(self.waveform_tab)
        self.waveform_layout.addWidget(self.waveform_canvas)
        self.analysis_tabs.addTab(self.waveform_tab, "Resonance Waveform")
        
        # Symmetry tab
        self.symmetry_tab = QWidget()
        self.symmetry_layout = QVBoxLayout(self.symmetry_tab)
        self.symmetry_canvas = SymmetricalPatternCanvas(self.symmetry_tab)
        self.symmetry_layout.addWidget(self.symmetry_canvas)
        self.analysis_tabs.addTab(self.symmetry_tab, "Symmetry Analysis")
        
        # Patterns tab
        self.patterns_tab = QWidget()
        self.patterns_layout = QVBoxLayout(self.patterns_tab)
        
        self.patterns_table = QTableWidget(0, 3)
        self.patterns_table.setHorizontalHeaderLabels(["Pattern Type", "Confidence", "Location"])
        self.patterns_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.patterns_layout.addWidget(self.patterns_table)
        
        self.analysis_tabs.addTab(self.patterns_tab, "Detected Patterns")
        
        # Quantum hash tab
        self.hash_tab = QWidget()
        self.hash_layout = QVBoxLayout(self.hash_tab)
        
        self.hash_info_label = QLabel("Quantum waveform hash will appear here after analysis")
        self.hash_info_label.setAlignment(Qt.AlignCenter)
        self.hash_info_label.setWordWrap(True)
        self.hash_layout.addWidget(self.hash_info_label)
        
        self.hash_code_label = QLabel("")
        self.hash_code_label.setAlignment(Qt.AlignCenter)
        self.hash_code_label.setFont(QFont("Courier New", 12))
        self.hash_layout.addWidget(self.hash_code_label)
        
        self.analysis_tabs.addTab(self.hash_tab, "Quantum Hash")
        
        self.splitter.addWidget(self.analysis_tabs)
        
        # Set initial splitter sizes
        self.splitter.setSizes([400, 800])
        self.main_layout.addWidget(self.splitter)
        
        # Status bar
        self.statusBar().showMessage("Ready")
    
    def load_image(self):
        """Load an image for analysis."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", 
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*)", 
            options=options
        )
        
        if file_path:
            try:
                # Load and display the image
                pixmap = QPixmap(file_path)
                if pixmap.isNull():
                    self.status_label.setText("Invalid image file")
                    return
                
                # Scale if too large
                if pixmap.width() > 800 or pixmap.height() > 600:
                    pixmap = pixmap.scaled(800, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                
                self.image_label.setPixmap(pixmap)
                self.current_image_path = file_path
                
                # Enable analyze button
                self.analyze_btn.setEnabled(True)
                
                # Show image info
                file_info = os.path.getsize(file_path) / 1024  # KB
                image_info = f"File: {os.path.basename(file_path)} | Size: {file_info:.1f} KB | Dimensions: {pixmap.width()}x{pixmap.height()}"
                self.image_info_label.setText(image_info)
                
                self.status_label.setText("Image loaded successfully")
                self.statusBar().showMessage(f"Loaded: {file_path}")
                
                # Reset previous analysis
                self.analysis_results = None
                self.save_results_btn.setEnabled(False)
                
                # Clear all visualizations
                self.clear_visualizations()
                
            except Exception as e:
                logger.error(f"Error loading image: {e}")
                self.status_label.setText(f"Error: {str(e)}")
    
    def analyze_image(self):
        """Analyze the loaded image for resonance patterns."""
        if not self.current_image_path:
            return
        
        try:
            # Update UI
            self.status_label.setText("Analyzing image...")
            self.progress_bar.setValue(10)
            QApplication.processEvents()
            
            # Create analyzer and process image
            self.analyzer = ImageResonanceAnalyzer(self.current_image_path)
            
            # Preprocess
            self.analyzer.preprocess_image()
            self.progress_bar.setValue(30)
            QApplication.processEvents()
            
            # Analyze geometric patterns
            geometric_patterns = self.analyzer.analyze_geometric_patterns()
            self.progress_bar.setValue(50)
            QApplication.processEvents()
            
            # Extract waveforms
            waveforms = self.analyzer.extract_waveforms()
            self.progress_bar.setValue(70)
            QApplication.processEvents()
            
            # Analyze resonance patterns
            resonance_data = self.analyzer.analyze_resonance_patterns()
            self.progress_bar.setValue(90)
            QApplication.processEvents()
            
            # Interpret symbolic meanings
            symbolic_meanings = self.analyzer.interpret_symbolic_meanings()
            
            # Store results
            self.analysis_results = {
                "geometric_patterns": geometric_patterns,
                "waveforms": waveforms,
                "resonance_data": resonance_data,
                "symbolic_meanings": symbolic_meanings,
                "symmetry_score": self.analyzer._calculate_symmetry()
            }
            
            # Update visualizations
            self.update_visualizations()
            
            # Generate and display quantum hash
            image_data = open(self.current_image_path, "rb").read()
            quantum_hash = geometric_waveform_hash(image_data)
            self.hash_code_label.setText(quantum_hash)
            self.hash_info_label.setText(
                f"Quantum Waveform Hash (unique resonance signature):\n"
                f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            
            # Enable save button
            self.save_results_btn.setEnabled(True)
            
            # Update status
            self.status_label.setText("Analysis complete")
            self.progress_bar.setValue(100)
            self.statusBar().showMessage("Image analyzed successfully")
            
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            self.status_label.setText(f"Analysis error: {str(e)}")
            self.progress_bar.setValue(0)
    
    def update_visualizations(self):
        """Update all visualization components with analysis results."""
        if not self.analysis_results:
            return
        
        # Update waveform visualization
        if "resonance_data" in self.analysis_results:
            resonance_data = self.analysis_results["resonance_data"]
            if isinstance(resonance_data, dict) and "frequencies" in resonance_data and "amplitudes" in resonance_data:
                self.waveform_canvas.plot_resonance_data(
                    resonance_data["frequencies"],
                    resonance_data["amplitudes"],
                    resonance_data.get("phases")
                )
        
        # Update symmetry visualization
        if "symmetry_score" in self.analysis_results:
            symmetry_score = self.analysis_results["symmetry_score"]
            # Create a simple symmetry matrix for visualization
            size = 20
            symmetry_matrix = np.zeros((size, size))
            center = size // 2
            for i in range(size):
                for j in range(size):
                    distance = np.sqrt((i - center)**2 + (j - center)**2)
                    symmetry_matrix[i, j] = np.exp(-distance / (size / 4)) * symmetry_score
            
            self.symmetry_canvas.plot_symmetry_data(symmetry_matrix, symmetry_score)
        
        # Update patterns table
        if "geometric_patterns" in self.analysis_results:
            geometric_patterns = self.analysis_results["geometric_patterns"]
            if isinstance(geometric_patterns, list):
                self.patterns_table.setRowCount(len(geometric_patterns))
                for i, pattern in enumerate(geometric_patterns):
                    if isinstance(pattern, dict):
                        self.patterns_table.setItem(i, 0, QTableWidgetItem(pattern.get("type", "Unknown")))
                        self.patterns_table.setItem(i, 1, QTableWidgetItem(f"{pattern.get('confidence', 0):.2f}"))
                        self.patterns_table.setItem(i, 2, QTableWidgetItem(str(pattern.get("location", "N/A"))))
    
    def clear_visualizations(self):
        """Clear all visualization components."""
        # Clear waveform canvas
        self.waveform_canvas.axes.clear()
        self.waveform_canvas.axes.set_title("Resonance Waveform Analysis", color='white')
        self.waveform_canvas.axes.set_xlabel("Frequency Domain", color='white')
        self.waveform_canvas.axes.set_ylabel("Amplitude", color='white')
        self.waveform_canvas.draw()
        
        # Clear symmetry canvas
        self.symmetry_canvas.axes.clear()
        self.symmetry_canvas.axes.set_title("Symmetry Analysis", color='white')
        self.symmetry_canvas.draw()
        
        # Clear patterns table
        self.patterns_table.setRowCount(0)
        
        # Clear hash information
        self.hash_code_label.setText("")
        self.hash_info_label.setText("Quantum waveform hash will appear here after analysis")
    
    def save_results(self):
        """Save analysis results to a file."""
        if not self.analysis_results:
            return
        
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Analysis Results", "", 
            "Text Files (*.txt);;JSON Files (*.json);;All Files (*)", 
            options=options
        )
        
        if file_path:
            try:
                # Format results for output
                output = []
                output.append(f"Q-Resonance Analyzer Results")
                output.append(f"Image: {self.current_image_path}")
                output.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                output.append(f"")
                
                if "symmetry_score" in self.analysis_results:
                    output.append(f"Symmetry Score: {self.analysis_results['symmetry_score']:.3f}")
                
                if "geometric_patterns" in self.analysis_results:
                    output.append(f"")
                    output.append(f"Geometric Patterns:")
                    for pattern in self.analysis_results["geometric_patterns"]:
                        output.append(f"  - Type: {pattern.get('type', 'Unknown')}")
                        output.append(f"    Confidence: {pattern.get('confidence', 0):.2f}")
                        output.append(f"    Location: {pattern.get('location', 'N/A')}")
                
                if "symbolic_meanings" in self.analysis_results:
                    output.append(f"")
                    output.append(f"Symbolic Interpretations:")
                    for meaning in self.analysis_results["symbolic_meanings"]:
                        output.append(f"  - {meaning}")
                
                # Write to file
                with open(file_path, "w") as f:
                    f.write("\n".join(output))
                
                self.statusBar().showMessage(f"Results saved to: {file_path}")
                
            except Exception as e:
                logger.error(f"Error saving results: {e}")
                self.statusBar().showMessage(f"Error saving results: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = QResonanceAnalyzer()
    window.show()
    sys.exit(app.exec_())