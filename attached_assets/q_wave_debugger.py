# File: q_wave_debugger.py

import sys, os, ctypes, platform
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout
)
from PyQt5.QtCore import Qt
import matplotlib
import logging

# Logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(os.path.join(os.path.dirname(__file__), "q_wave_debugger.log")),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)
logger.info("Launching Q-Wave Debugger...")

# Setup paths
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
DLL_DIR = os.path.abspath(os.path.join(ROOT_DIR, "..", "bin"))
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

# Minimal Debugger UI
class QWaveDebugger(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Q-Wave Debugger")
        self.setGeometry(100, 100, 800, 600)

        # Load stylesheet
        self.stylesheet = load_stylesheet(STYLES_QSS)
        if self.stylesheet:
            self.setStyleSheet(self.stylesheet)

        widget = QWidget()
        layout = QVBoxLayout(widget)
        self.setCentralWidget(widget)

        # Import here after headless environment is configured
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        
        self.fig = plt.figure(facecolor="black")
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        self.init_plot()

    def init_plot(self):
        # Use compatible methods
        self.ax.set_xlim([-10, 10])
        self.ax.set_ylim([-5, 5])
        self.ax.set_zlim([-2, 2])
        self.ax.set_facecolor("black")
        self.ax.set_title("Static Debug Canvas", color="white")
        self.ax.xaxis.line.set_color("white")
        self.ax.yaxis.line.set_color("white")
        if hasattr(self.ax, 'zaxis'):
            self.ax.zaxis.line.set_color("white")
        self.ax.grid(True, linestyle="--", color="gray", alpha=0.3)
        self.canvas.draw()

if __name__ == "__main__":
    # Import and use the headless environment setup
    from attached_assets import setup_headless_environment
    env_config = setup_headless_environment()
    logger.info(f"Running on {env_config['platform']} in {'headless' if env_config['headless'] else 'windowed'} mode")
    
    # Set the backend to Agg if in headless mode
    if env_config['headless']:
        matplotlib.use('Agg')
        logger.info("✅ Set matplotlib backend to Agg for headless operation")
    else:
        matplotlib.use('QtAgg')
        logger.info("✅ Set matplotlib backend to QtAgg for windowed operation")
    
    # Check and load DLL
    dll_path = os.path.join(DLL_DIR, "engine_core.dll")
    if os.path.exists(dll_path):
        try:
            if platform.system() == 'Windows':
                os.add_dll_directory(DLL_DIR)
            os.environ["PATH"] = DLL_DIR + os.pathsep + os.environ["PATH"]
            try:
                if platform.system() == 'Windows':
                    ctypes.CDLL(dll_path)
                    logger.info(f"✅ Loaded: {dll_path}")
                else:
                    logger.info(f"⚠️ Skipping DLL load on non-Windows platform: {platform.system()}")
            except Exception as e:
                logger.warning(f"⚠️ DLL exists but failed to load: {e}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to set up DLL directory: {e}")
    else:
        logger.warning(f"⚠️ engine_core.dll not found at {dll_path}")
    
    app = QApplication(sys.argv)
    window = QWaveDebugger()
    window.show()
    sys.exit(app.exec_())
