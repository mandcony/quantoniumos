#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QuantoniumOS System Monitor (Futura Minimal)
- Clean card layout • Light/Dark toggle • Pause/Resume
- Per-core CPU bars + sparkline trend
- Memory/Disk gauges, Network up/down throughput
- Process table with search + End Task
- RFT (unitary_rft) online indicator
"""

import os, sys, time, psutil, platform, math, signal
from collections import deque
from typing import List, Optional, Tuple

from PyQt5.QtCore import Qt, QTimer, QRectF, QPointF
from PyQt5.QtGui import QFont, QPainter, QPen, QBrush, QColor
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QComboBox, QLineEdit, QTableWidget, QTableWidgetItem,
    QSplitter, QFrame, QMessageBox, QStatusBar, QDialog, QSlider, QTextEdit
)

# --- Optional RFT probe ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "ASSEMBLY", "python_bindings"))
try:
    import unitary_rft  # type: ignore
    RFT_AVAILABLE = True
except Exception:
    RFT_AVAILABLE = False

# --- Optional Matplotlib backend (GUI-safe) -----------------------------------
try:
    import matplotlib
    matplotlib.use("Qt5Agg")
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import numpy as np
    HAS_MPL = True
except Exception:
    HAS_MPL = False


# ---------- Small helpers ----------
def human_bytes(n: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while n >= 1024 and i < len(units) - 1:
        n /= 1024.0; i += 1
    return f"{n:.1f} {units[i]}"

def human_rate(nbytes_per_s: float) -> str:
    # Show in bps-ish but with Bytes base (common in UI)
    return human_bytes(nbytes_per_s) + "/s"


# ---------- Reusable UI primitives ----------
class Card(QFrame):
    """A frosted, rounded card container."""
    def __init__(self, title: str = ""):
        super().__init__()
        self.setObjectName("Card")
        self.setFrameShape(QFrame.NoFrame)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(8)
        self.title_lbl = QLabel(title)
        self.title_lbl.setObjectName("CardTitle")
        lay.addWidget(self.title_lbl)
        self.body = QWidget()
        lay.addWidget(self.body)
        self.setLayout(lay)

class Sparkline(QWidget):
    """Lightweight sparkline for trend (keeps last N values)."""
    def __init__(self, max_points: int = 60):
        super().__init__()
        self.setMinimumHeight(36)
        self.values = deque(maxlen=max_points)

    def add(self, v: float):
        self.values.append(max(0.0, min(100.0, v)))
        self.update()

    def paintEvent(self, _):
        if not self.values:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        w, h = self.width(), self.height()
        # axis baseline
        pen = QPen(QColor(120, 140, 160, 60), 1)
        p.setPen(pen)
        p.drawLine(0, h-1, w, h-1)
        # spark path
        pen = QPen(QColor(50, 140, 255, 200), 2)
        p.setPen(pen)
        step = w / max(1, len(self.values)-1)
        pts = []
        for i, v in enumerate(self.values):
            y = h - (v/100.0) * (h-4) - 2
            pts.append(QPointF(i*step, y))
        for i in range(1, len(pts)):
            p.drawLine(pts[i-1], pts[i])
        # fill under curve (subtle)
        p.setBrush(QBrush(QColor(50, 140, 255, 40)))
        for i in range(len(pts)-1):
            poly = [pts[i], pts[i+1], QPointF((i+1)*step, h-1), QPointF(i*step, h-1)]
            p.drawPolygon(*poly)
        p.end()

class Bar(QWidget):
    """Minimal rounded meter bar (0-100)."""
    def __init__(self):
        super().__init__()
        self.value = 0.0
        self.setMinimumHeight(12)

    def setValue(self, v: float):
        self.value = max(0.0, min(100.0, float(v)))
        self.update()

    def paintEvent(self, _):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing, True)
        r = self.rect().adjusted(0, 0, -1, -1)
        # track
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(120, 140, 160, 40))
        p.drawRoundedRect(r, 6, 6)
        # fill
        w = int(r.width() * (self.value/100.0))
        if w > 0:
            p.setBrush(QColor(80, 180, 120, 220))
            p.drawRoundedRect(QRectF(r.x(), r.y(), w, r.height()), 6, 6)
        p.end()


# ---------- Main Window ----------
class SystemMonitor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QuantoniumOS • System Monitor")
        self.resize(1180, 740)

        # live state
        self.cpu_spark = Sparkline(120)
        self.net_up_spark = Sparkline(120)
        self.net_dn_spark = Sparkline(120)
        self.per_core_bars: List[Bar] = []
        self._running = True
        self._light = True
        self._last_net = psutil.net_io_counters() if hasattr(psutil, "net_io_counters") else None
        self._last_ts = time.time()

        self._build_ui()
        self._apply_style(light=True)
        self._prime_psutil()
        self._start_timer(1000)  # default 1s
        self.statusBar().showMessage(self._status_text())

    # ---------- UI ----------
    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        root = QVBoxLayout(central); root.setContentsMargins(0,0,0,0); root.setSpacing(0)

        # header
        header = QWidget(); header.setFixedHeight(60)
        h = QVBoxLayout(header); h.setContentsMargins(20,8,20,8)
        title = QLabel("System Monitor"); title.setObjectName("Title")
        subtitle = QLabel("Live metrics • minimal visuals • Quantonium aesthetic")
        subtitle.setObjectName("SubTitle")
        h.addWidget(title); h.addWidget(subtitle)
        root.addWidget(header)

        # controls
        ctrl = QWidget(); cl = QHBoxLayout(ctrl); cl.setContentsMargins(16, 0, 16, 8)
        self.theme_btn = QPushButton("Dark/Light"); self.theme_btn.clicked.connect(self.toggle_theme)
        self.pause_btn = QPushButton("Pause"); self.pause_btn.clicked.connect(self.toggle_run)
        self.rate = QComboBox(); self.rate.addItems(["0.5s","1s","2s","5s"])
        self.rate.setCurrentIndex(1); self.rate.currentTextChanged.connect(self._rate_changed)
        self.search = QLineEdit(placeholderText="Search processes…")
        self.search.textChanged.connect(self._filter_processes)
        self.kill_btn = QPushButton("End Task"); self.kill_btn.clicked.connect(self._kill_selected)
        cl.addWidget(self.theme_btn); cl.addWidget(self.pause_btn)
        cl.addWidget(QLabel("Refresh:")); cl.addWidget(self.rate)
        cl.addStretch(1); cl.addWidget(self.search); cl.addWidget(self.kill_btn)
        root.addWidget(ctrl)

        # layout: left cards + right table
        split = QSplitter(Qt.Horizontal); root.addWidget(split, 1)

        # LEFT: metric cards
        left = QWidget(); lg = QGridLayout(left); lg.setContentsMargins(16,16,16,16); lg.setHorizontalSpacing(16); lg.setVerticalSpacing(16)

        # CPU Card
        self.cpu_card = Card("CPU")
        cbl = QVBoxLayout(self.cpu_card.body); cbl.setSpacing(8)
        self.cpu_label = QLabel("0.0%"); self.cpu_label.setObjectName("Big")
        cbl.addWidget(self.cpu_label)
        cbl.addWidget(self.cpu_spark)
        # per-core bars
        cores = psutil.cpu_count(logical=True) or 1
        for _ in range(cores):
            bar = Bar(); self.per_core_bars.append(bar); cbl.addWidget(bar)
        lg.addWidget(self.cpu_card, 0, 0, 2, 1)

        # Memory Card
        self.mem_card = Card("Memory")
        mbl = QVBoxLayout(self.mem_card.body)
        self.mem_label = QLabel("0.0%  (0 / 0)"); self.mem_label.setObjectName("Mid")
        self.mem_bar = Bar()
        mbl.addWidget(self.mem_label); mbl.addWidget(self.mem_bar)
        lg.addWidget(self.mem_card, 0, 1, 1, 1)

        # Disk Card
        self.disk_card = Card("Disk")
        dbl = QVBoxLayout(self.disk_card.body)
        self.disk_label = QLabel("—"); self.disk_label.setObjectName("Mid")
        self.disk_bar = Bar()
        dbl.addWidget(self.disk_label); dbl.addWidget(self.disk_bar)
        lg.addWidget(self.disk_card, 1, 1, 1, 1)

        # Network Card
        self.net_card = Card("Network")
        nbl = QVBoxLayout(self.net_card.body)
        self.net_up = QLabel("↑ 0 B/s"); self.net_dn = QLabel("↓ 0 B/s")
        nbl.addWidget(self.net_up); nbl.addWidget(self.net_dn)
        sparkw = QWidget(); swl = QHBoxLayout(sparkw); swl.setContentsMargins(0,0,0,0); swl.setSpacing(10)
        swl.addWidget(self.net_up_spark); swl.addWidget(self.net_dn_spark)
        nbl.addWidget(sparkw)
        lg.addWidget(self.net_card, 2, 0, 1, 2)

        # RFT Card
        self.rft_card = Card("RFT Assembly")
        rbl = QVBoxLayout(self.rft_card.body)
        self.rft_status = QLabel("🟢 Online" if RFT_AVAILABLE else "🔴 Offline")
        self.rft_status.setAlignment(Qt.AlignLeft)
        rbl.addWidget(self.rft_status)
        self.rft_note = QLabel("unitary_rft bindings detected" if RFT_AVAILABLE else "Python bindings not found")
        rbl.addWidget(self.rft_note)
        
        # Add RFT Visualizer button
        self.rft_viz_btn = QPushButton("🌊 RFT Visualizer")
        self.rft_viz_btn.clicked.connect(self.show_rft_visualizer)
        rbl.addWidget(self.rft_viz_btn)
        
        lg.addWidget(self.rft_card, 3, 0, 1, 2)

        split.addWidget(left)

        # RIGHT: process table
        right = QWidget(); rl = QVBoxLayout(right); rl.setContentsMargins(16,16,16,16)
        self.proc_table = QTableWidget()
        self.proc_table.setColumnCount(5)
        self.proc_table.setHorizontalHeaderLabels(["PID","Name","CPU%","Mem%","User"])
        self.proc_table.verticalHeader().setVisible(False)
        self.proc_table.setSortingEnabled(True)
        self.proc_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.proc_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.proc_table.setAlternatingRowColors(True)
        rl.addWidget(self.proc_table)
        split.addWidget(right)
        split.setSizes([600, 560])

        self.setStatusBar(QStatusBar())

    def _apply_style(self, light=True):
        self._light = light
        if light:
            qss = """
            QMainWindow, QWidget { background:#fafafa; color:#243342; font-family:'Segoe UI',-apple-system,BlinkMacSystemFont,sans-serif; }
            #Title { font-size:20px; font-weight:300; color:#2c3e50; }
            #SubTitle { font-size:11px; color:#8aa0b3; }
            #Card { background:transparent; }
            QFrame#Card { border:1px solid #e9ecef; border-radius:14px; background:#ffffff; }
            QLabel#CardTitle { color:#6c7f90; font-size:12px; letter-spacing:.4px; }
            QLabel#Big { font-size:28px; font-weight:300; color:#2c3e50; }
            QLabel#Mid { font-size:14px; color:#2c3e50; }
            QTableWidget { background:#ffffff; border:1px solid #e9ecef; border-radius:8px; }
            QTableWidget::item:selected { background:#e3f2fd; color:#1976d2; }
            QPushButton { background:#f8f9fa; border:1px solid #dee2e6; border-radius:8px; padding:8px 14px; color:#495057; }
            QPushButton:hover { background:#eef2f6; }
            QLineEdit { background:#ffffff; border:1px solid #e9ecef; border-radius:8px; padding:8px 10px; }
            QComboBox { background:#ffffff; border:1px solid #e9ecef; border-radius:8px; padding:6px 8px; }
            """
        else:
            qss = """
            QMainWindow, QWidget { background:#0f1216; color:#dfe7ef; font-family:'Segoe UI',-apple-system,BlinkMacSystemFont,sans-serif; }
            #Title { font-size:20px; font-weight:300; color:#dfe7ef; }
            #SubTitle { font-size:11px; color:#8aa0b3; }
            QFrame#Card { border:1px solid #1f2a36; border-radius:14px; background:#12161b; }
            QLabel#CardTitle { color:#8aa0b3; font-size:12px; letter-spacing:.4px; }
            QLabel#Big { font-size:28px; font-weight:300; color:#e8eff7; }
            QLabel#Mid { font-size:14px; color:#e8eff7; }
            QTableWidget { background:#12161b; border:1px solid #1f2a36; border-radius:8px; alternate-background-color:#0f141a; }
            QHeaderView::section { background:#12161b; color:#9eb1c5; border:0px; border-bottom:1px solid #1f2a36; padding:6px; }
            QTableWidget::item:selected { background:#1d2b3a; color:#7dc4ff; }
            QPushButton { background:#12161b; border:1px solid #2a3847; border-radius:8px; padding:8px 14px; color:#c8d3de; }
            QPushButton:hover { background:#17202a; }
            QLineEdit { background:#12161b; border:1px solid #1f2a36; border-radius:8px; padding:8px 10px; color:#e8eff7; }
            QComboBox { background:#12161b; border:1px solid #1f2a36; border-radius:8px; padding:6px 8px; color:#e8eff7; }
            """
        self.setStyleSheet(qss)

    # ---------- Engine ----------
    def _prime_psutil(self):
        # Warm up cpu_percent to get instant values
        psutil.cpu_percent(percpu=True)

    def _start_timer(self, ms: int):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(ms)

    def _rate_changed(self, txt: str):
        m = {"0.5s": 500, "1s": 1000, "2s": 2000, "5s": 5000}[txt]
        self.timer.setInterval(m)

    def toggle_theme(self):
        self._apply_style(not self._light)

    def toggle_run(self):
        self._running = not self._running
        self.pause_btn.setText("Resume" if not self._running else "Pause")

    def _status_text(self) -> str:
        osname = f"{platform.system()} {platform.release()}"
        return f"{osname} • RFT {'Online' if RFT_AVAILABLE else 'Offline'}"

    # ---------- Updates ----------
    def _tick(self):
        if not self._running: return
        self._update_cpu()
        self._update_mem()
        self._update_disk()
        self._update_net()
        self._update_processes()
        self.statusBar().showMessage(self._status_text())

    def _update_cpu(self):
        total = psutil.cpu_percent()  # overall
        self.cpu_label.setText(f"{total:.1f}%")
        self.cpu_spark.add(total)
        # per-core
        cores = psutil.cpu_percent(percpu=True)
        # ensure list sizes match
        if len(self.per_core_bars) != len(cores):
            for bar in self.per_core_bars: bar.setParent(None)
            self.per_core_bars.clear()
        # (re)build if mismatch
        if not self.per_core_bars:
            lay: QVBoxLayout = self.cpu_card.body.layout()  # type: ignore
            for _ in range(len(cores)):
                bar = Bar(); self.per_core_bars.append(bar); lay.addWidget(bar)
        for i, v in enumerate(cores):
            self.per_core_bars[i].setValue(v)

    def _update_mem(self):
        vm = psutil.virtual_memory()
        self.mem_bar.setValue(vm.percent)
        self.mem_label.setText(f"{vm.percent:.1f}%   ({human_bytes(vm.used)} / {human_bytes(vm.total)})")

    def _update_disk(self):
        root = os.path.abspath(os.sep)
        du = psutil.disk_usage(root)
        self.disk_bar.setValue(du.percent)
        self.disk_label.setText(f"{du.percent:.1f}%   ({human_bytes(du.used)} / {human_bytes(du.total)})")

    def _update_net(self):
        if not hasattr(psutil, "net_io_counters"): return
        now = time.time()
        cur = psutil.net_io_counters()
        dt = max(1e-6, now - self._last_ts)
        if self._last_net:
            up = (cur.bytes_sent - self._last_net.bytes_sent) / dt
            dn = (cur.bytes_recv - self._last_net.bytes_recv) / dt
            self.net_up.setText(f"↑ {human_rate(up)}")
            self.net_dn.setText(f"↓ {human_rate(dn)}")
            self.net_up_spark.add(min(100.0, 100.0*(up/(1024*1024*10))))  # normalize to 10 MB/s ≈ 100%
            self.net_dn_spark.add(min(100.0, 100.0*(dn/(1024*1024*10))))
        self._last_net = cur; self._last_ts = now

    # ---------- Processes ----------
    def _update_processes(self):
        # collect and filter
        q = self.search.text().lower().strip()
        rows: List[Tuple[int,str,float,float,str]] = []
        for p in psutil.process_iter(["pid","name","cpu_percent","memory_percent","username"]):
            try:
                info = p.info
                name = info.get("name") or "—"
                if q and q not in name.lower(): continue
                rows.append((
                    info.get("pid") or 0,
                    name,
                    float(info.get("cpu_percent") or 0.0),
                    float(info.get("memory_percent") or 0.0),
                    info.get("username") or "—"
                ))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        # sort by CPU desc
        rows.sort(key=lambda r: r[2], reverse=True)
        # limit
        rows = rows[:25]
        # update table
        self.proc_table.setRowCount(len(rows))
        for i, (pid, name, cpu, mem, user) in enumerate(rows):
            self.proc_table.setItem(i, 0, QTableWidgetItem(str(pid)))
            self.proc_table.setItem(i, 1, QTableWidgetItem(name))
            self.proc_table.setItem(i, 2, QTableWidgetItem(f"{cpu:.1f}"))
            self.proc_table.setItem(i, 3, QTableWidgetItem(f"{mem:.1f}"))
            self.proc_table.setItem(i, 4, QTableWidgetItem(user))
        self.proc_table.resizeColumnsToContents()

    def _kill_selected(self):
        r = self.proc_table.currentRow()
        if r < 0: return
        pid_item = self.proc_table.item(r, 0)
        name_item = self.proc_table.item(r, 1)
        if not pid_item: return
        pid = int(pid_item.text())
        name = name_item.text() if name_item else str(pid)
        if QMessageBox.question(self, "End Task",
                                f"Terminate '{name}' (PID {pid})?",
                                QMessageBox.Yes | QMessageBox.No,
                                QMessageBox.No) != QMessageBox.Yes:
            return
        try:
            p = psutil.Process(pid)
            p.terminate()
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))

    # ---------- Theme / search ----------
    def toggle_theme(self):
        self._apply_style(not self._light)

    def _filter_processes(self):
        # handled in update; just trigger immediate refresh
        self._update_processes()
    
    def show_rft_visualizer(self):
        """Show the RFT visualizer as a popup dialog"""
        dialog = RFTVisualizerDialog(self)
        dialog.exec_()


# ---------- RFT Visualizer Dialog ----------
class RFTVisualizerDialog(QDialog):
    """3D RFT Visualizer popup window integrated into System Monitor"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("RFT Visualizer - 3D Recursive Wave Engine")
        self.setGeometry(200, 200, 1200, 700)
        self.setModal(False)  # Allow interaction with parent window
        
        # State
        self.dark_mode = False
        self.time_step = 0.0
        self.recursive_depth = 5
        self.frequency = 1.0
        self.amplitude = 1.0
        self.wave_speed = 0.5
        self.quantum_coupling = 0.618

        # Timer (do NOT start yet—avoid race before canvas exists)
        self.timer = QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.update_visualization)

        # UI
        self.figure = None
        self.ax = None
        self.canvas = None

        self.init_ui()
        self.apply_theme()

        # Start timer only after UI is fully ready and canvas exists
        if HAS_MPL and self.canvas is not None and self.ax is not None:
            self.timer.start()

    # --------------------------- UI ------------------------------------------
    def init_ui(self):
        root = QHBoxLayout(self)
        root.setSpacing(0)
        root.setContentsMargins(0, 0, 0, 0)

        # Left panel
        left = QWidget()
        left.setFixedWidth(280)
        left.setObjectName("LeftPanel")
        lyt = QVBoxLayout(left)
        lyt.setContentsMargins(20, 20, 20, 20)
        lyt.setSpacing(15)

        title = QLabel("RFT Visualizer")
        title.setObjectName("Title")
        lyt.addWidget(title)

        subtitle = QLabel("3D Recursive Wave Engine")
        subtitle.setObjectName("SubTitle")
        lyt.addWidget(subtitle)

        # Depth
        lyt.addWidget(QLabel("Recursive Depth:"))
        self.depth_slider = QSlider(Qt.Horizontal)
        self.depth_slider.setRange(1, 10)
        self.depth_slider.setValue(self.recursive_depth)
        self.depth_slider.valueChanged.connect(self.update_depth)
        lyt.addWidget(self.depth_slider)
        self.depth_label = QLabel(f"Depth: {self.recursive_depth}")
        lyt.addWidget(self.depth_label)

        # Frequency
        lyt.addWidget(QLabel("Frequency:"))
        self.freq_slider = QSlider(Qt.Horizontal)
        self.freq_slider.setRange(1, 50)  # 0.1 .. 5.0 Hz
        self.freq_slider.setValue(int(self.frequency * 10))
        self.freq_slider.valueChanged.connect(self.update_frequency)
        lyt.addWidget(self.freq_slider)
        self.freq_label = QLabel(f"Freq: {self.frequency:.1f} Hz")
        lyt.addWidget(self.freq_label)

        # Buttons
        self.pause_btn = QPushButton("⏸ Pause")
        self.pause_btn.clicked.connect(self.toggle_animation)
        lyt.addWidget(self.pause_btn)

        self.theme_btn = QPushButton("🌙 Dark Mode")
        self.theme_btn.clicked.connect(self.toggle_theme)
        lyt.addWidget(self.theme_btn)

        # Metrics
        metrics_label = QLabel("Wave Metrics")
        metrics_label.setStyleSheet("font-weight: bold; margin-top: 20px; font-size: 14px;")
        lyt.addWidget(metrics_label)

        self.metrics_text = QTextEdit()
        self.metrics_text.setMaximumHeight(250)
        self.metrics_text.setReadOnly(True)
        self.metrics_text.setStyleSheet("""
            QTextEdit {
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 10px;
                line-height: 1.2;
                padding: 8px;
                border-radius: 6px;
            }
        """)
        lyt.addWidget(self.metrics_text)

        lyt.addStretch()
        root.addWidget(left)

        # Right panel (3D)
        if HAS_MPL:
            self.canvas = self.create_3d_canvas()
            root.addWidget(self.canvas, 1)
        else:
            fallback = QLabel("Matplotlib not available — 3D view disabled.")
            fallback.setAlignment(Qt.AlignCenter)
            root.addWidget(fallback, 1)

    def create_3d_canvas(self):
        self.figure = Figure(figsize=(10, 8))
        self.ax = self.figure.add_subplot(111, projection="3d")
        self.ax.set_xlabel("X - Spatial")
        self.ax.set_ylabel("Y - Frequency")
        self.ax.set_zlabel("Z - Amplitude")
        self.ax.set_title("RFT Engine - Recursive Quantum Field")
        canvas = FigureCanvas(self.figure)
        return canvas

    # --------------------------- Theming --------------------------------------
    def apply_theme(self):
        if self.dark_mode:
            qss = """
            QDialog, QWidget { background:#0f1216; color:#dfe7ef; font-family:'Segoe UI'; }
            #Title { font-size:20px; font-weight:300; color:#dfe7ef; }
            #SubTitle { font-size:11px; color:#8aa0b3; }
            #LeftPanel { background:#12161b; border-right:1px solid #1f2a36; }
            QSlider::groove:horizontal { border:1px solid #1f2a36; height:8px; background:#12161b; border-radius:4px; }
            QSlider::handle:horizontal { background:#7dc4ff; width:18px; border-radius:9px; margin:-2px 0; }
            QPushButton { background:#12161b; border:1px solid #2a3847; border-radius:6px; padding:8px 14px; color:#c8d3de; }
            QPushButton:hover { background:#1d2b3a; }
            QTextEdit { background:#0a0a0a; border:1px solid #1f2a36; color:#dfe7ef; }
            """
            self.theme_btn.setText("☀ Light Mode")
        else:
            qss = """
            QDialog, QWidget { background:#fafafa; color:#243342; font-family:'Segoe UI'; }
            #Title { font-size:20px; font-weight:300; color:#2c3e50; }
            #SubTitle { font-size:11px; color:#8aa0b3; }
            #LeftPanel { background:#f8f9fa; border-right:1px solid #dee2e6; }
            QSlider::groove:horizontal { border:1px solid #dee2e6; height:8px; background:#e9ecef; border-radius:4px; }
            QSlider::handle:horizontal { background:#1976d2; width:18px; border-radius:9px; margin:-2px 0; }
            QPushButton { background:#f8f9fa; border:1px solid #dee2e6; border-radius:6px; padding:8px 14px; color:#495057; }
            QPushButton:hover { background:#e9ecef; }
            QTextEdit { background:#ffffff; border:1px solid #dee2e6; color:#243342; }
            """
            self.theme_btn.setText("🌙 Dark Mode")

        self.setStyleSheet(qss)

        if HAS_MPL and self.figure is not None:
            self.figure.patch.set_facecolor("#12161b" if self.dark_mode else "#fafafa")
            if self.ax is not None:
                self.ax.set_facecolor("#0a0a0a" if self.dark_mode else "#ffffff")
            if self.canvas is not None:
                self.canvas.draw_idle()

    # --------------------------- Controls -------------------------------------
    def toggle_theme(self):
        self.dark_mode = not self.dark_mode
        self.apply_theme()

    def update_depth(self, value):
        self.recursive_depth = int(value)
        self.depth_label.setText(f"Depth: {self.recursive_depth}")

    def update_frequency(self, value):
        self.frequency = value / 10.0  # slider 1..50 => 0.1..5.0 Hz
        self.freq_label.setText(f"Freq: {self.frequency:.1f} Hz")

    def toggle_animation(self):
        if not self.timer.isActive():
            if HAS_MPL and self.canvas is not None and self.ax is not None:
                self.timer.start()
                self.pause_btn.setText("⏸ Pause")
        else:
            self.timer.stop()
            self.pause_btn.setText("▶ Resume")

    # --------------------------- Engine ---------------------------------------
    def generate_recursive_wave(self, x, y, t):
        z = np.zeros_like(x)
        for depth in range(self.recursive_depth):
            freq_scale = self.frequency * (1 + depth * self.quantum_coupling)
            wave1 = np.sin(freq_scale * x - self.wave_speed * t + depth * np.pi / 4)
            wave2 = np.cos(freq_scale * y - self.wave_speed * t + depth * np.pi / 3)
            interference = wave1 * wave2
            amp_scale = self.amplitude / (1 + depth * 0.3)
            z += amp_scale * interference * np.exp(-0.1 * depth)
            # φ-coupling modulation
            quantum_mod = np.sin(self.quantum_coupling * (x + y) + t) * 0.2
            z += quantum_mod * amp_scale
        return z

    def update_visualization(self):
        # Guard against race conditions or missing backends
        if not HAS_MPL or self.ax is None or self.canvas is None:
            self.timer.stop()
            return

        self.time_step += 0.1
        self.ax.clear()

        x = np.linspace(-3, 3, 30)
        y = np.linspace(-3, 3, 30)
        X, Y = np.meshgrid(x, y)
        Z = self.generate_recursive_wave(X, Y, self.time_step)

        self.ax.plot_surface(X, Y, Z, cmap="plasma", alpha=0.8, linewidth=0, antialiased=True)
        self.ax.plot_wireframe(X, Y, Z, color="white", alpha=0.3, linewidth=0.5)

        self.ax.set_xlim([-3, 3]); self.ax.set_ylim([-3, 3]); self.ax.set_zlim([-2, 2])
        color = "white" if self.dark_mode else "black"
        self.ax.set_xlabel("X - Spatial", color=color)
        self.ax.set_ylabel("Y - Frequency", color=color)
        self.ax.set_zlabel("Z - Amplitude", color=color)
        self.ax.set_title("RFT Engine - Recursive Quantum Field", color=color)
        self.ax.tick_params(colors=color)
        self.ax.grid(True, alpha=0.3)
        self.ax.set_facecolor("#0a0a0a" if self.dark_mode else "#ffffff")

        self.canvas.draw_idle()
        self.update_metrics(Z)

    # --------------------------- Metrics --------------------------------------
    def update_metrics(self, wave):
        amax = float(np.max(wave))
        amin = float(np.min(wave))
        arange = amax - amin
        mean = float(np.mean(wave))
        std = float(np.std(wave))

        complexity = max(0, min(100, int(std * 100)))
        coherence = max(0, min(100, int(100 - (std * 50))))
        interference = max(0, min(100, int(abs(mean) * 200)))
        coupling_effect = max(0, min(100, int(self.quantum_coupling * 100)))
        phase_sync = max(0, min(100, int(75 + 25 * np.cos(self.time_step))))

        metrics_text = f"""
🌊 QUANTUM FIELD ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 WAVE PROPERTIES:
   • Amplitude Range: {arange:.3f}
   • Peak Intensity: {amax:.3f}
   • Field Average: {mean:.3f}
   • Complexity: {complexity}%

🔄 FIELD DYNAMICS:
   • Coherence Level: {coherence}%
   • Interference: {interference}%
   • Phase Sync: {phase_sync}%
   • Coupling Effect: {coupling_effect}%

⚛️ QUANTUM PARAMETERS:
   • Recursive Depth: {self.recursive_depth} layers
   • Base Frequency: {self.frequency:.1f} Hz
   • Quantum Coupling: {self.quantum_coupling:.3f}
   • Time Evolution: {self.time_step:.1f}s

📈 FIELD EXPLANATION:
   • Higher complexity = more intricate patterns
   • Coherence shows wave synchronization
   • Interference reveals wave interactions
   • Phase sync indicates temporal stability

🔬 OBSERVED EFFECTS:
   • Recursive layers create depth
   • Quantum coupling adds nonlinearity
   • Interference generates complexity
   • Time evolution shows dynamics
"""
        self.metrics_text.setText(metrics_text)


# ---------- Entrypoint ----------
def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app = QApplication(sys.argv)
    app.setApplicationName("QuantoniumOS System Monitor")
    app.setFont(QFont("Segoe UI", 10))
    w = SystemMonitor()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
