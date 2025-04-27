# File: q_wave_debugger_min.py

import sys, os, ctypes, platform
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout
)
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import logging

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Launching minimal QWaveDebugger...")

# DLL Setup (loads but does not invoke)
DLL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "bin"))
dll_path = os.path.join(DLL_DIR, "engine_core.dll")
if os.path.exists(dll_path):
    os.add_dll_directory(DLL_DIR)
    os.environ["PATH"] = DLL_DIR + os.pathsep + os.environ["PATH"]
    try:
        ctypes.CDLL(dll_path)
        logger.info(f"✅ Loaded: {dll_path}")
    except Exception as e:
        logger.warning(f"⚠️ DLL exists but failed to load: {e}")
else:
    logger.warning(f"⚠️ engine_core.dll not found at {dll_path}")

# Minimal Debugger UI
class QWaveDebugger(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QWaveDebugger – Canvas Only")
        self.setGeometry(100, 100, 800, 600)

        widget = QWidget()
        layout = QVBoxLayout(widget)
        self.setCentralWidget(widget)

        self.fig = plt.figure(facecolor="black")
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        self.init_plot()

    def init_plot(self):
        self.ax.set_xlim3d([-10, 10])
        self.ax.set_ylim3d([-5, 5])
        self.ax.set_zlim3d([-2, 2])
        self.ax.set_facecolor("black")
        self.ax.set_title("Static Debug Canvas", color="white")
        for axis in [self.ax.xaxis, self.ax.yaxis, self.ax.zaxis]:
            axis.line.set_color("white")
        self.ax.grid(True, linestyle="--", color="gray", alpha=0.3)
        self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = QWaveDebugger()
    window.show()
    sys.exit(app.exec_())
