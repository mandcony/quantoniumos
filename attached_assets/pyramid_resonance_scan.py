import sys
import os
import numpy as np
import re
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QComboBox, QPushButton,
    QVBoxLayout, QTabWidget
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

# === Minimal Symbolic Functions (placeholders for now) ===
def symbolic_phase_alignment(depth): return np.cos(depth / 20)
def geometric_modifier(depth): return 1.0 + 0.1 * np.sin(depth / 10)
def symbolic_signature(depth): return np.exp(-depth / 100)

# === Pyramid Data (from our prior chats) ===
pyramids = {
    "Khufu": {
        "name": "Great Pyramid of Giza (Khufu)",
        "height_m": 137.0,
        "metadata": """
            Descending Passage: entrance to first chamber
            King's Chamber at ~43.2m
            Grand Gallery ~38m–47m
            Voids detected: 15–22m, 43–45m, 55–58m
            Material: Limestone + Granite
        """
    },
    "Khafre": {
        "name": "Pyramid of Khafre",
        "height_m": 136.4,
        "metadata": """
            Entry chamber ~15m
            Subterranean vault at ~30–35m
            EM focus at 40–60m
            Voids: ~17m, 33m
        """
    },
    "Menkaure": {
        "name": "Pyramid of Menkaure",
        "height_m": 62.0,
        "metadata": """
            Small granite chamber at 11.7m
            Interior acoustic shell ~18–25m
            No confirmed voids in scan
        """
    }
}

# === Symbolic Resonance Calculation ===
def symbolic_resonance(height):
    depths = np.linspace(0, height, 200)
    resonance = []
    for d in depths:
        base = np.sin(2 * np.pi * d / (height / 2)) * np.exp(-d / height)
        phase = symbolic_phase_alignment(d)
        geom = geometric_modifier(d)
        sig = symbolic_signature(d)
        combined = base * phase * geom * sig * np.random.normal(1.0, 0.05)
        resonance.append(combined)
    return depths, np.array(resonance)

# === Regex Pocket Extraction (based on metadata) ===
def extract_void_ranges(metadata):
    voids = []
    pattern = r"(\d+(\.\d+)?)[–-](\d+(\.\d+)?)"
    for match in re.findall(pattern, metadata):
        start = float(match[0])
        end = float(match[2])
        voids.append((start, end))
    return voids

# === Match amplitude spikes inside regex voids ===
def detect_resonance_pockets(depths, resonance, voids):
    pockets = []
    for d, amp in zip(depths, resonance):
        for v_start, v_end in voids:
            if v_start <= d <= v_end and abs(amp) > 0.6:
                pockets.append((d, amp))
    return pockets

# === GUI Class ===
class PyramidResonanceGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Symbolic Pyramid Resonance Scanner")
        self.setGeometry(100, 100, 1200, 700)

        # Layout Setup
        widget = QWidget()
        layout = QVBoxLayout(widget)
        self.setCentralWidget(widget)

        self.dropdown = QComboBox()
        for key in pyramids:
            self.dropdown.addItem(key)

        self.scan_button = QPushButton("Run Scan")
        self.scan_button.clicked.connect(self.run_scan)

        self.info_label = QLabel("Select a pyramid and run a symbolic resonance scan.")

        self.tabs = QTabWidget()
        self.fig_2d = Figure()
        self.canvas_2d = FigureCanvas(self.fig_2d)
        self.tabs.addTab(self.canvas_2d, "Resonance Profile")

        self.fig_3d = Figure()
        self.canvas_3d = FigureCanvas(self.fig_3d)
        self.tabs.addTab(self.canvas_3d, "3D Pocket Visualization")

        layout.addWidget(QLabel("Select Pyramid:"))
        layout.addWidget(self.dropdown)
        layout.addWidget(self.scan_button)
        layout.addWidget(self.info_label)
        layout.addWidget(self.tabs)

    def run_scan(self):
        key = self.dropdown.currentText()
        pyramid = pyramids[key]
        height = pyramid["height_m"]
        metadata = pyramid["metadata"]

        self.depths, self.resonance = symbolic_resonance(height)
        voids = extract_void_ranges(metadata)
        self.pockets = detect_resonance_pockets(self.depths, self.resonance, voids)

        self.info_label.setText(f"{pyramid['name']}\nVoid zones (parsed): {voids}")
        self.plot_2d(pyramid["name"])
        self.plot_3d(pyramid["height_m"], pyramid["name"])

    def plot_2d(self, title):
        self.fig_2d.clear()
        ax = self.fig_2d.add_subplot(111)
        ax.plot(self.depths, self.resonance, color='cyan', label="Symbolic Amplitude")
        for d, a in self.pockets:
            ax.plot(d, a, 'ro')
            ax.annotate("Pocket", (d, a), textcoords="offset points", xytext=(0,10), ha='center', color='red')
        ax.set_title(f"Symbolic Resonance Profile – {title}")
        ax.set_xlabel("Depth (m)")
        ax.set_ylabel("Resonance Intensity")
        ax.legend()
        self.canvas_2d.draw()

    def plot_3d(self, height, title):
        self.fig_3d.clear()
        ax = self.fig_3d.add_subplot(111, projection='3d', facecolor='black')
        ax.set_title(f"3D Resonance Map – {title}", color='cyan')
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
        ax.set_zlim(0, height)
        ax.set_xlabel("X", color='white')
        ax.set_ylabel("Y", color='white')
        ax.set_zlabel("Depth (m)", color='white')
        ax.tick_params(colors='white')

        self.draw_wireframe_pyramid(ax, height)

        if self.pockets:
            np.random.seed(42)
            xs = np.random.uniform(-30, 30, len(self.pockets))
            ys = np.random.uniform(-30, 30, len(self.pockets))
            zs = [d for d, _ in self.pockets]
            ax.scatter(xs, ys, zs, c='red', s=60, alpha=0.8)
            for x, y, z in zip(xs, ys, zs):
                ax.text(x, y, z + 1.5, f"{int(z)}m", color='red', fontsize=8)

        self.canvas_3d.draw()
        self.tabs.setCurrentWidget(self.canvas_3d)

    def draw_wireframe_pyramid(self, ax, height=137, base=100):
        h = base / 2
        corners = [[-h, -h, 0], [h, -h, 0], [h, h, 0], [-h, h, 0]]
        apex = [0, 0, height]
        for i in range(4):
            x = [corners[i][0], corners[(i+1)%4][0]]
            y = [corners[i][1], corners[(i+1)%4][1]]
            z = [0, 0]
            ax.plot(x, y, z, color='white')
            ax.plot([corners[i][0], apex[0]], [corners[i][1], apex[1]], [0, apex[2]], color='white')

# === Launch ===
if __name__ == "__main__":
    print("Launching symbolic resonance interface...")
    app = QApplication(sys.argv)
    gui = PyramidResonanceGUI()
    gui.show()
    gui.raise_()
    gui.activateWindow()
    sys.exit(app.exec_())
