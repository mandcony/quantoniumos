#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
# This file is listed in CLAIMS_PRACTICING_FILES.txt and is licensed
# under LICENSE-CLAIMS-NC.md (research/education only). Commercial
# rights require a separate patent license from the author.
"""
QuantSoundDesign - Professional Sound Design Studio with Φ-RFT Native Engine

Design Philosophy:
- FL Studio/Ableton-inspired professional workflow
- Larger track lanes for better visibility
- Refined toggle buttons with clear states
- Smooth, modern gradient aesthetics
- Full-featured arrangement, mixer, and pattern views
"""

import sys
import os
import numpy as np
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

from algorithms.rft.rft_status import get_status as get_kernel_status

try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QPushButton, QSlider, QScrollArea, QFrame, QSplitter,
        QMenu, QAction, QFileDialog, QMessageBox, QSpinBox, QDoubleSpinBox,
        QComboBox, QToolBar, QStatusBar, QDockWidget, QListWidget,
        QListWidgetItem, QGraphicsView, QGraphicsScene, QGraphicsRectItem,
        QGridLayout, QTabWidget, QSizePolicy, QStackedWidget, QProgressBar,
        QToolButton, QTreeView
    )
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QRectF, QPointF, QSize
    from PyQt5.QtGui import (
        QColor, QPainter, QBrush, QPen, QFont, QLinearGradient, 
        QRadialGradient, QPainterPath, QIcon, QPixmap
    )
    from PyQt5.QtWidgets import QFileSystemModel
except ImportError:
    print("PyQt5 required: pip install PyQt5")
    sys.exit(1)

# Import audio backend - support both package and direct execution
try:
    from .audio_backend import get_audio_backend, init_audio, shutdown_audio, TestToneGenerator
    AUDIO_AVAILABLE = True
except ImportError:
    try:
        from audio_backend import get_audio_backend, init_audio, shutdown_audio, TestToneGenerator
        AUDIO_AVAILABLE = True
    except ImportError:
        AUDIO_AVAILABLE = False
        print("Audio backend not available")

# Core session/engine models
try:
    from .engine import Session
    ENGINE_AVAILABLE = True
except ImportError as engine_exc:  # noqa: F841 - surfaced at runtime
    ENGINE_AVAILABLE = False
    print(f"Engine module not available: {engine_exc}")

# Import synth and piano roll
try:
    from .synth_engine import PolySynth, PRESET_LIBRARY
    from .piano_roll import InstrumentEditor, PianoKeyboard, InstrumentBrowser, PianoRollToolbar
    SYNTH_AVAILABLE = True
except ImportError:
    try:
        from synth_engine import PolySynth, PRESET_LIBRARY
        from piano_roll import InstrumentEditor, PianoKeyboard, InstrumentBrowser, PianoRollToolbar
        SYNTH_AVAILABLE = True
    except ImportError as e:
        SYNTH_AVAILABLE = False
        print(f"Synth/Piano roll not available: {e}")

# Import pattern editor
try:
    from .pattern_editor import (
        Pattern, PatternRow, PatternStep, PatternEditorWidget,
        PatternPlayer, DrumSynthesizer, DrumType, InstrumentSelector
    )
    PATTERN_AVAILABLE = True
except ImportError:
    try:
        from pattern_editor import (
            Pattern, PatternRow, PatternStep, PatternEditorWidget,
            PatternPlayer, DrumSynthesizer, DrumType, InstrumentSelector
        )
        PATTERN_AVAILABLE = True
    except ImportError as e:
        PATTERN_AVAILABLE = False
        print(f"Pattern editor not available: {e}")

# Import core infrastructure (KeymapRegistry, SelectionModel, etc.)
try:
    from .core import (
        get_keymap_registry, get_selection_model, get_audition_engine,
        KeymapContext, beats_to_samples, samples_to_beats, snap_to_grid,
        Scale, ScaleType, Chord, ChordType
    )
    CORE_AVAILABLE = True
except ImportError:
    try:
        from core import (
            get_keymap_registry, get_selection_model, get_audition_engine,
            KeymapContext, beats_to_samples, samples_to_beats, snap_to_grid,
            Scale, ScaleType, Chord, ChordType
        )
        CORE_AVAILABLE = True
    except ImportError as e:
        CORE_AVAILABLE = False
        print(f"Core module not available: {e}")

# Import audio settings dialog and meter widget (EPIC 01)
try:
    from .audio_settings_dialog import (
        AudioSettingsDialog, AudioMeterWidget, show_audio_settings_dialog
    )
    AUDIO_SETTINGS_AVAILABLE = True
except ImportError:
    try:
        from audio_settings_dialog import (
            AudioSettingsDialog, AudioMeterWidget, show_audio_settings_dialog
        )
        AUDIO_SETTINGS_AVAILABLE = True
    except ImportError as e:
        AUDIO_SETTINGS_AVAILABLE = False
        print(f"Audio settings dialog not available: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# PROFESSIONAL STYLE THEME - QuantSoundDesign
# ═══════════════════════════════════════════════════════════════════════════════

# Color palette
COLORS = {
    'bg_dark': '#0d0d0d',
    'bg_main': '#151515',
    'bg_panel': '#1a1a1a',
    'bg_track': '#1e1e1e',
    'bg_selected': '#252525',
    'border': '#2a2a2a',
    'border_light': '#3a3a3a',
    'accent': '#00aaff',
    'accent_hover': '#00ccff',
    'accent_green': '#00ffaa',
    'accent_yellow': '#ffaa00',
    'accent_red': '#ff4444',
    'accent_purple': '#aa00ff',
    'text': '#e0e0e0',
    'text_dim': '#888888',
    'text_dark': '#555555',
}

STYLE = """
/* ═══════════════ MAIN WINDOW ═══════════════ */
QMainWindow, QWidget {
    background-color: #0d0d0d;
    color: #e0e0e0;
    font-family: 'Segoe UI', 'SF Pro Display', Arial, sans-serif;
    font-size: 11px;
}

/* ═══════════════ BUTTONS ═══════════════ */
QPushButton {
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #2a2a2a, stop:1 #222222);
    border: 1px solid #3a3a3a;
    border-radius: 4px;
    padding: 8px 16px;
    color: #e0e0e0;
    min-width: 60px;
    font-weight: 500;
}

QPushButton:hover {
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #00bbff, stop:1 #0099dd);
    border-color: #00aaff;
    color: white;
}

QPushButton:pressed {
    background-color: #0077aa;
}

QPushButton:checked {
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #00ffbb, stop:1 #00dd99);
    color: #0d0d0d;
    border-color: #00ffaa;
    font-weight: bold;
}

QPushButton:disabled {
    background-color: #1a1a1a;
    color: #555555;
    border-color: #2a2a2a;
}

/* ═══════════════ SLIDERS ═══════════════ */
QSlider::groove:horizontal {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #1a1a1a, stop:1 #2a2a2a);
    height: 6px;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #00ccff, stop:1 #0099cc);
    width: 16px;
    margin: -5px 0;
    border-radius: 8px;
    border: 2px solid #00aaff;
}

QSlider::handle:horizontal:hover {
    background: #00ffaa;
    border-color: #00ffaa;
}

QSlider::groove:vertical {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #2a2a2a, stop:1 #1a1a1a);
    width: 6px;
    border-radius: 3px;
}

QSlider::handle:vertical {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #00ccff, stop:1 #0099cc);
    height: 16px;
    margin: 0 -5px;
    border-radius: 8px;
    border: 2px solid #00aaff;
}

/* ═══════════════ SCROLL AREAS ═══════════════ */
QScrollArea {
    border: none;
    background-color: transparent;
}

QScrollBar:horizontal {
    background: #0d0d0d;
    height: 12px;
    border-radius: 6px;
}

QScrollBar::handle:horizontal {
    background: #3a3a3a;
    min-width: 40px;
    border-radius: 6px;
}

QScrollBar::handle:horizontal:hover {
    background: #00aaff;
}

QScrollBar:vertical {
    background: #0d0d0d;
    width: 12px;
    border-radius: 6px;
}

QScrollBar::handle:vertical {
    background: #3a3a3a;
    min-height: 40px;
    border-radius: 6px;
}

QScrollBar::handle:vertical:hover {
    background: #00aaff;
}

QScrollBar::add-line, QScrollBar::sub-line {
    height: 0px;
    width: 0px;
}

/* ═══════════════ FRAMES ═══════════════ */
QFrame {
    background-color: transparent;
}

QSplitter::handle {
    background-color: #00aaff;
}

QSplitter::handle:horizontal {
    width: 3px;
}

QSplitter::handle:vertical {
    height: 3px;
}

/* ═══════════════ COMBOS & SPINBOXES ═══════════════ */
QComboBox {
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #2a2a2a, stop:1 #1e1e1e);
    border: 1px solid #3a3a3a;
    border-radius: 4px;
    padding: 6px 10px;
    color: #e0e0e0;
    min-width: 80px;
}

QComboBox:hover {
    border-color: #00aaff;
}

QComboBox::drop-down {
    border: none;
    width: 24px;
    background: transparent;
}

QComboBox::down-arrow {
    width: 12px;
    height: 12px;
}

QComboBox QAbstractItemView {
    background-color: #1a1a1a;
    border: 1px solid #3a3a3a;
    selection-background-color: #00aaff;
    selection-color: white;
}

QSpinBox, QDoubleSpinBox {
    background-color: #1a1a1a;
    border: 1px solid #3a3a3a;
    border-radius: 4px;
    padding: 6px;
    color: #00ffaa;
    font-family: 'Consolas', 'Monaco', monospace;
    font-weight: bold;
}

QSpinBox:hover, QDoubleSpinBox:hover {
    border-color: #00aaff;
}

/* ═══════════════ LISTS ═══════════════ */
QListWidget {
    background-color: #151515;
    border: 1px solid #2a2a2a;
    border-radius: 6px;
    outline: none;
}

QListWidget::item {
    padding: 8px 12px;
    border-radius: 4px;
    margin: 2px 4px;
}

QListWidget::item:hover {
    background-color: #252525;
}

QListWidget::item:selected {
    background-color: #00aaff;
    color: white;
}

/* ═══════════════ TABS ═══════════════ */
QTabWidget::pane {
    border: none;
    background: #151515;
}

QTabBar::tab {
    background: #1a1a1a;
    color: #888888;
    padding: 10px 20px;
    border: none;
    border-bottom: 3px solid transparent;
    font-weight: 500;
    min-width: 100px;
}

QTabBar::tab:selected {
    color: #00aaff;
    border-bottom: 3px solid #00aaff;
    background: #1e1e1e;
}

QTabBar::tab:hover {
    color: #00ffaa;
    background: #222222;
}

/* ═══════════════ STATUS BAR ═══════════════ */
QStatusBar {
    background-color: #0a0a0a;
    color: #888888;
    border-top: 1px solid #2a2a2a;
    font-size: 10px;
}

/* ═══════════════ MENU BAR ═══════════════ */
QMenuBar {
    background-color: #0d0d0d;
    color: #e0e0e0;
    border-bottom: 1px solid #2a2a2a;
    padding: 4px;
}

QMenuBar::item {
    padding: 6px 12px;
    border-radius: 4px;
}

QMenuBar::item:selected {
    background-color: #00aaff;
}

QMenu {
    background-color: #1a1a1a;
    border: 1px solid #3a3a3a;
    border-radius: 6px;
    padding: 4px;
}

QMenu::item {
    padding: 8px 24px;
    border-radius: 4px;
}

QMenu::item:selected {
    background-color: #00aaff;
}

QMenu::separator {
    height: 1px;
    background: #3a3a3a;
    margin: 4px 8px;
}

/* ═══════════════ PROGRESS BAR ═══════════════ */
QProgressBar {
    background-color: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 4px;
    text-align: center;
    color: #888888;
}

QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #00aaff, stop:1 #00ffaa);
    border-radius: 3px;
}

/* ═══════════════ TOOLTIPS ═══════════════ */
QToolTip {
    background-color: #1a1a1a;
    color: #e0e0e0;
    border: 1px solid #00aaff;
    border-radius: 4px;
    padding: 6px;
}
"""

# ═══════════════════════════════════════════════════════════════════════════════
# TRACK HEADER WIDGET - Professional Version
# ═══════════════════════════════════════════════════════════════════════════════

class TrackHeader(QFrame):
    """Professional track header with name, mute, solo, arm, and volume."""
    
    mute_changed = pyqtSignal(int, bool)
    solo_changed = pyqtSignal(int, bool)
    arm_changed = pyqtSignal(int, bool)
    volume_changed = pyqtSignal(int, float)
    
    TRACK_COLORS = [
        '#00aaff', '#00ffaa', '#ffaa00', '#ff6600', 
        '#aa00ff', '#ff00aa', '#00ff66', '#ff4444'
    ]
    
    def __init__(self, track_index: int, track_name: str, track_type: str = "audio", parent=None):
        super().__init__(parent)
        self.track_index = track_index
        self.track_type = track_type
        self.track_color = self.TRACK_COLORS[track_index % len(self.TRACK_COLORS)]
        self.is_selected = False
        
        self.setFixedWidth(200)
        self.setMinimumHeight(100)
        self.setup_ui(track_name)
        self.update_style()
        
    def setup_ui(self, track_name: str):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)
        
        # Top row: Color bar + Track number + Name
        top_row = QHBoxLayout()
        top_row.setSpacing(8)
        
        # Color indicator bar
        self.color_bar = QFrame()
        self.color_bar.setFixedSize(4, 30)
        self.color_bar.setStyleSheet(f"background-color: {self.track_color}; border-radius: 2px;")
        top_row.addWidget(self.color_bar)
        
        # Track number
        self.num_label = QLabel(f"{self.track_index + 1:02d}")
        self.num_label.setStyleSheet(f"""
            color: {self.track_color}; 
            font-size: 14px; 
            font-weight: bold;
            font-family: 'Consolas', monospace;
        """)
        self.num_label.setFixedWidth(24)
        top_row.addWidget(self.num_label)
        
        # Track name (editable style)
        self.name_label = QLabel(track_name)
        self.name_label.setStyleSheet("""
            color: #e0e0e0; 
            font-weight: bold; 
            font-size: 12px;
            padding: 2px 4px;
            border-radius: 2px;
        """)
        self.name_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        top_row.addWidget(self.name_label)
        
        layout.addLayout(top_row)
        
        # Middle row: Mute, Solo, Arm buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)
        
        self.mute_btn = self._create_toggle_btn("M", "Mute", "#ff4444")
        self.mute_btn.clicked.connect(lambda c: self.mute_changed.emit(self.track_index, c))
        
        self.solo_btn = self._create_toggle_btn("S", "Solo", "#ffaa00")
        self.solo_btn.clicked.connect(lambda c: self.solo_changed.emit(self.track_index, c))
        
        self.arm_btn = self._create_toggle_btn("R", "Record", "#ff0000")
        self.arm_btn.clicked.connect(lambda c: self.arm_changed.emit(self.track_index, c))
        
        # Track type indicator
        type_icons = {"drums": "🥁", "synth": "🎹", "audio": "🎵", "fx": "✨"}
        self.type_label = QLabel(type_icons.get(self.track_type, "🎵"))
        self.type_label.setStyleSheet("font-size: 16px;")
        self.type_label.setFixedWidth(24)
        self.type_label.setAlignment(Qt.AlignCenter)
        
        btn_row.addWidget(self.mute_btn)
        btn_row.addWidget(self.solo_btn)
        btn_row.addWidget(self.arm_btn)
        btn_row.addStretch()
        btn_row.addWidget(self.type_label)
        
        layout.addLayout(btn_row)
        
        # Bottom row: Volume slider + dB
        vol_row = QHBoxLayout()
        vol_row.setSpacing(6)
        
        self.vol_slider = QSlider(Qt.Horizontal)
        self.vol_slider.setRange(0, 127)
        self.vol_slider.setValue(100)
        self.vol_slider.setMinimumWidth(100)
        self.vol_slider.valueChanged.connect(self._on_volume_changed)
        vol_row.addWidget(self.vol_slider)
        
        self.db_label = QLabel("0.0")
        self.db_label.setStyleSheet("""
            color: #00ffaa; 
            font-size: 10px; 
            font-family: 'Consolas', monospace;
            min-width: 36px;
        """)
        self.db_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        vol_row.addWidget(self.db_label)
        
        layout.addLayout(vol_row)
        
    def _create_toggle_btn(self, text: str, tooltip: str, active_color: str) -> QPushButton:
        """Create a styled toggle button."""
        btn = QPushButton(text)
        btn.setCheckable(True)
        btn.setFixedSize(28, 24)
        btn.setToolTip(tooltip)
        btn.setStyleSheet(f"""
            QPushButton {{
                background: #252525;
                border: 1px solid #3a3a3a;
                border-radius: 3px;
                font-size: 11px;
                font-weight: bold;
                color: #888;
                min-width: 28px;
                padding: 0;
            }}
            QPushButton:hover {{
                background: #303030;
                border-color: {active_color};
                color: {active_color};
            }}
            QPushButton:checked {{
                background: {active_color};
                border-color: {active_color};
                color: white;
            }}
        """)
        return btn
        
    def _on_volume_changed(self, value):
        if value == 0:
            self.db_label.setText("-inf")
        else:
            db = 20 * np.log10(value / 100)
            self.db_label.setText(f"{db:.1f}")
        self.volume_changed.emit(self.track_index, value / 127.0)
        
    def update_style(self):
        """Update style based on selection state."""
        if self.is_selected:
            self.setStyleSheet(f"""
                TrackHeader {{
                    background-color: #252525;
                    border-right: 3px solid {self.track_color};
                    border-top: 1px solid #3a3a3a;
                    border-bottom: 1px solid #2a2a2a;
                }}
            """)
        else:
            self.setStyleSheet(f"""
                TrackHeader {{
                    background-color: #1a1a1a;
                    border-right: 3px solid {self.track_color};
                    border-top: 1px solid #2a2a2a;
                    border-bottom: 1px solid #1e1e1e;
                }}
                TrackHeader:hover {{
                    background-color: #1e1e1e;
                }}
            """)
            
    def set_selected(self, selected: bool):
        self.is_selected = selected
        self.update_style()


# ═══════════════════════════════════════════════════════════════════════════════
# CLIP DATA CLASS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ClipData:
    """Data for a single clip in the arrangement."""
    start: float
    length: float
    name: str
    color: str = "#00aaff"
    pattern: object = None
    instrument: object = None
    selected: bool = False
    muted: bool = False
    
    def end(self) -> float:
        return self.start + self.length


# ═══════════════════════════════════════════════════════════════════════════════
# TRACK LANE WIDGET - Professional Version with Larger Clips
# ═══════════════════════════════════════════════════════════════════════════════

class TrackLane(QFrame):
    """Professional track lane with larger clips and better visualization."""
    
    clip_selected = pyqtSignal(object)
    clip_double_clicked = pyqtSignal(object)
    clip_moved = pyqtSignal(object, float)  # clip, new_start
    clip_resized = pyqtSignal(object, float)  # clip, new_length
    
    def __init__(self, track_index: int, track_type: str = "audio", 
                 track_color: str = "#00aaff", parent=None):
        super().__init__(parent)
        self.track_index = track_index
        self.track_type = track_type
        self.track_color = track_color
        self.clips: List[ClipData] = []
        self.pixels_per_beat = 30  # LARGER - was 20
        self.selected_clip: Optional[ClipData] = None
        self.hover_clip: Optional[ClipData] = None
        
        # Drag state
        self.dragging = False
        self.drag_clip: Optional[ClipData] = None
        self.drag_start_x = 0
        self.drag_start_beat = 0
        self.resizing = False
        self.resize_edge = None  # 'left' or 'right'
        
        # Create clip by dragging
        self.creating_clip = False
        self.create_start_x = 0
        self.create_start_beat = 0
        self.create_preview_end = 0  # For drawing preview
        
        self.setMinimumHeight(100)  # TALLER - was 60
        self.setMouseTracking(True)
        self.setCursor(Qt.ArrowCursor)
        self.setAcceptDrops(True)
        
    def add_clip(self, start: float, length: float, name: str, 
                 color: str = None) -> ClipData:
        """Add a clip to the track lane."""
        clip = ClipData(
            start=start,
            length=length,
            name=name,
            color=color or self.track_color
        )
        
        # Auto-create pattern for drum tracks
        if PATTERN_AVAILABLE and self.track_type == "drums":
            clip.pattern = Pattern(name=name, is_drum=True, steps=int(length * 4))
        
        self.clips.append(clip)
        self.update()
        return clip
    
    def get_clip_at(self, x: int) -> Optional[ClipData]:
        """Get clip at screen x position."""
        beat = x / self.pixels_per_beat
        for clip in self.clips:
            if clip.start <= beat < clip.end():
                return clip
        return None
    
    def get_resize_edge(self, clip: ClipData, x: int) -> Optional[str]:
        """Check if mouse is over resize edge of clip."""
        if not clip:
            return None
        edge_width = 8  # pixels
        clip_x = int(clip.start * self.pixels_per_beat)
        clip_end_x = int(clip.end() * self.pixels_per_beat)
        
        if abs(x - clip_x) < edge_width:
            return 'left'
        elif abs(x - clip_end_x) < edge_width:
            return 'right'
        return None
    
    def mousePressEvent(self, event):
        clip = self.get_clip_at(event.x())
        if clip:
            # Deselect previous
            if self.selected_clip:
                self.selected_clip.selected = False
            clip.selected = True
            self.selected_clip = clip
            self.clip_selected.emit(clip)
            
            # Check for resize edge
            edge = self.get_resize_edge(clip, event.x())
            if edge:
                self.resizing = True
                self.resize_edge = edge
                self.drag_clip = clip
            else:
                # Start dragging
                self.dragging = True
                self.drag_clip = clip
                self.drag_start_x = event.x()
                self.drag_start_beat = clip.start
            
            self.update()
        else:
            # Click on empty area - start drawing a new clip
            self.creating_clip = True
            self.create_start_x = event.x()
            beat_pos = event.x() / self.pixels_per_beat
            self.create_start_beat = round(beat_pos * 4) / 4  # Snap to quarter
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        if self.dragging and self.drag_clip:
            # Calculate new position
            delta_x = event.x() - self.drag_start_x
            delta_beats = delta_x / self.pixels_per_beat
            new_start = max(0, self.drag_start_beat + delta_beats)
            
            # Snap to grid (quarter notes)
            new_start = round(new_start * 4) / 4
            
            self.drag_clip.start = new_start
            self.update()
            
        elif self.resizing and self.drag_clip:
            beat_pos = event.x() / self.pixels_per_beat
            beat_pos = round(beat_pos * 4) / 4  # Snap
            
            if self.resize_edge == 'left':
                # Resize from left - move start, adjust length
                old_end = self.drag_clip.end()
                new_start = max(0, min(beat_pos, old_end - 1))
                self.drag_clip.length = old_end - new_start
                self.drag_clip.start = new_start
            elif self.resize_edge == 'right':
                # Resize from right - adjust length
                new_length = max(1, beat_pos - self.drag_clip.start)
                self.drag_clip.length = new_length
            
            self.update()
        elif self.creating_clip:
            # Update preview for clip being created
            beat_pos = event.x() / self.pixels_per_beat
            self.create_preview_end = round(beat_pos * 4) / 4
            self.setCursor(Qt.CrossCursor)
            self.update()
        else:
            # Just hovering - update cursor
            clip = self.get_clip_at(event.x())
            if clip != self.hover_clip:
                self.hover_clip = clip
                self.update()
            
            # Set cursor based on position
            if clip:
                edge = self.get_resize_edge(clip, event.x())
                if edge:
                    self.setCursor(Qt.SizeHorCursor)
                else:
                    self.setCursor(Qt.OpenHandCursor)
            else:
                self.setCursor(Qt.ArrowCursor)
        
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        if self.dragging and self.drag_clip:
            self.clip_moved.emit(self.drag_clip, self.drag_clip.start)
        elif self.resizing and self.drag_clip:
            self.clip_resized.emit(self.drag_clip, self.drag_clip.length)
        elif self.creating_clip:
            # Finish creating clip
            end_beat = event.x() / self.pixels_per_beat
            end_beat = round(end_beat * 4) / 4
            
            start = min(self.create_start_beat, end_beat)
            end = max(self.create_start_beat, end_beat)
            length = max(1, end - start)  # Minimum 1 beat
            
            # Create the new clip
            new_clip = self.add_clip(start, length, "New Pattern")
            self.clip_selected.emit(new_clip)
            
            self.creating_clip = False
            self.create_start_beat = 0
            self.create_preview_end = 0
        
        self.dragging = False
        self.resizing = False
        self.drag_clip = None
        self.resize_edge = None
        self.creating_clip = False
        self.setCursor(Qt.ArrowCursor)
        super().mouseReleaseEvent(event)
    
    def mouseDoubleClickEvent(self, event):
        clip = self.get_clip_at(event.x())
        if clip:
            self.clip_double_clicked.emit(clip)
        else:
            # Double-click on empty area - create new clip
            beat_pos = event.x() / self.pixels_per_beat
            beat_pos = round(beat_pos * 4) / 4  # Snap to quarter notes
            new_clip = self.add_clip(beat_pos, 4, "New Clip")  # 1 bar = 4 beats
            self.clip_selected.emit(new_clip)
        super().mouseDoubleClickEvent(event)
        
    def leaveEvent(self, event):
        self.hover_clip = None
        self.setCursor(Qt.ArrowCursor)
        self.update()
        super().leaveEvent(event)
    
    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        height = self.height()
        
        # Background with subtle gradient
        gradient = QLinearGradient(0, 0, 0, height)
        gradient.setColorAt(0, QColor('#181818'))
        gradient.setColorAt(1, QColor('#141414'))
        painter.fillRect(0, 0, self.width(), height, gradient)
        
        # Draw beat grid
        for beat in range(0, 300):
            x = int(beat * self.pixels_per_beat)
            if x > self.width():
                break
                
            if beat % 16 == 0:  # 4 bar marker
                painter.setPen(QPen(QColor('#444444'), 2))
            elif beat % 4 == 0:  # Bar marker
                painter.setPen(QPen(QColor('#333333'), 1))
            else:  # Beat marker
                painter.setPen(QPen(QColor('#222222'), 1))
            painter.drawLine(x, 0, x, height)
        
        # Draw clips with professional styling
        for clip in self.clips:
            self._draw_clip(painter, clip, height)
        
        # Draw clip creation preview
        if self.creating_clip and self.create_preview_end != self.create_start_beat:
            start = min(self.create_start_beat, self.create_preview_end)
            end = max(self.create_start_beat, self.create_preview_end)
            x = int(start * self.pixels_per_beat)
            w = int((end - start) * self.pixels_per_beat)
            y_margin = 6
            
            # Draw semi-transparent preview
            painter.setBrush(QBrush(QColor(0, 170, 255, 80)))
            painter.setPen(QPen(QColor(0, 170, 255), 2, Qt.DashLine))
            painter.drawRoundedRect(x, y_margin, w, height - y_margin * 2, 4, 4)
        
        # Bottom border
        painter.setPen(QPen(QColor('#2a2a2a'), 1))
        painter.drawLine(0, height - 1, self.width(), height - 1)
    
    def _draw_clip(self, painter: QPainter, clip: ClipData, height: int):
        """Draw a single clip with professional styling."""
        x = int(clip.start * self.pixels_per_beat)
        w = int(clip.length * self.pixels_per_beat)
        y_margin = 6
        clip_height = height - y_margin * 2
        
        # Skip if not visible
        if x > self.width() or x + w < 0:
            return
        
        color = QColor(clip.color)
        is_hover = clip == self.hover_clip
        is_selected = clip.selected
        
        # Clip body gradient
        gradient = QLinearGradient(x, y_margin, x, y_margin + clip_height)
        if is_selected:
            gradient.setColorAt(0, color.lighter(140))
            gradient.setColorAt(0.3, color.lighter(120))
            gradient.setColorAt(1, color)
        elif is_hover:
            gradient.setColorAt(0, color.lighter(120))
            gradient.setColorAt(1, color.darker(110))
        else:
            gradient.setColorAt(0, color.lighter(110))
            gradient.setColorAt(0.3, color)
            gradient.setColorAt(1, color.darker(120))
        
        # Draw rounded clip body
        path = QPainterPath()
        path.addRoundedRect(x + 2, y_margin, w - 4, clip_height, 6, 6)
        
        painter.setBrush(QBrush(gradient))
        
        # Border
        if is_selected:
            painter.setPen(QPen(QColor('#ffffff'), 2))
        elif is_hover:
            painter.setPen(QPen(color.lighter(150), 1.5))
        else:
            painter.setPen(QPen(color.lighter(130), 1))
        
        painter.drawPath(path)
        
        # Top highlight
        highlight = QLinearGradient(x, y_margin, x, y_margin + 10)
        highlight.setColorAt(0, QColor(255, 255, 255, 40))
        highlight.setColorAt(1, QColor(255, 255, 255, 0))
        painter.setBrush(QBrush(highlight))
        painter.setPen(Qt.NoPen)
        
        highlight_path = QPainterPath()
        highlight_path.addRoundedRect(x + 3, y_margin + 1, w - 6, 10, 5, 5)
        painter.drawPath(highlight_path)
        
        # Clip header bar
        header_rect = QRectF(x + 2, y_margin, w - 4, 22)
        header_gradient = QLinearGradient(0, y_margin, 0, y_margin + 22)
        header_gradient.setColorAt(0, QColor(0, 0, 0, 60))
        header_gradient.setColorAt(1, QColor(0, 0, 0, 30))
        painter.setBrush(QBrush(header_gradient))
        painter.drawRoundedRect(header_rect, 6, 6)
        
        # Clip name
        painter.setPen(QColor('#ffffff'))
        font = QFont("Segoe UI", 10)
        font.setBold(is_selected)
        painter.setFont(font)
        text_rect = QRectF(x + 8, y_margin + 2, w - 16, 18)
        painter.drawText(text_rect, Qt.AlignLeft | Qt.AlignVCenter, clip.name)
        
        # Draw pattern preview for drum clips
        if clip.pattern and clip.pattern.is_drum and clip_height > 40:
            self._draw_pattern_preview(painter, clip, x, y_margin + 24, w, clip_height - 28)
            
        # Muted overlay
        if clip.muted:
            painter.setBrush(QBrush(QColor(0, 0, 0, 150)))
            painter.drawPath(path)
    
    def _draw_pattern_preview(self, painter: QPainter, clip: ClipData, 
                               x: int, y: int, w: int, h: int):
        """Draw pattern preview inside the clip."""
        if not clip.pattern:
            return
        
        pattern = clip.pattern
        steps = min(pattern.steps, 64)
        rows = min(len(pattern.rows), 8)
        
        step_width = max(3, (w - 12) / steps)
        row_height = max(3, (h - 4) / rows)
        
        painter.setPen(Qt.NoPen)
        
        for row_idx, row in enumerate(pattern.rows[:rows]):
            if row.mute:
                continue
            for step_idx, step in enumerate(row.steps[:steps]):
                if step.active:
                    sx = x + 6 + int(step_idx * step_width)
                    sy = y + 2 + int(row_idx * row_height)
                    
                    # Color based on velocity
                    intensity = int(150 + step.velocity * 105)
                    painter.setBrush(QBrush(QColor(intensity, 255, intensity)))
                    painter.drawRoundedRect(
                        sx, sy, 
                        max(2, int(step_width) - 1), 
                        max(2, int(row_height) - 1),
                        1, 1
                    )


# ═══════════════════════════════════════════════════════════════════════════════
# ARRANGEMENT VIEW - Professional DAW-Style Layout
# ═══════════════════════════════════════════════════════════════════════════════

class TimelineHeader(QFrame):
    """Professional timeline with bar numbers, markers, and start cue."""
    
    start_cue_changed = pyqtSignal(float)  # Signal when start cue is moved
    position_clicked = pyqtSignal(float)   # Signal when user clicks to set position
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pixels_per_beat = 30
        self.setFixedHeight(40)
        self.setMinimumWidth(4000)
        self.start_cue_beat = 0.0  # Start cue position in beats
        self.playhead_beat = 0.0   # Current playhead position
        self.dragging_cue = False
        self.setMouseTracking(True)
        
    def set_playhead(self, beat: float):
        """Set playhead position and repaint"""
        self.playhead_beat = beat
        self.update()
        
    def set_start_cue(self, beat: float):
        """Set start cue position"""
        self.start_cue_beat = max(0, beat)
        self.update()
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            beat_pos = event.x() / self.pixels_per_beat
            # Check if clicking on start cue marker (within 10px)
            cue_x = int(self.start_cue_beat * self.pixels_per_beat)
            if abs(event.x() - cue_x) < 10:
                self.dragging_cue = True
                self.setCursor(Qt.SizeHorCursor)
            else:
                # Click to set playhead position
                self.position_clicked.emit(beat_pos)
        super().mousePressEvent(event)
        
    def mouseMoveEvent(self, event):
        if self.dragging_cue:
            beat_pos = max(0, event.x() / self.pixels_per_beat)
            beat_pos = round(beat_pos * 4) / 4  # Snap to quarter notes
            self.start_cue_beat = beat_pos
            self.start_cue_changed.emit(beat_pos)
            self.update()
        else:
            # Update cursor if near start cue
            cue_x = int(self.start_cue_beat * self.pixels_per_beat)
            if abs(event.x() - cue_x) < 10:
                self.setCursor(Qt.SizeHorCursor)
            else:
                self.setCursor(Qt.ArrowCursor)
        super().mouseMoveEvent(event)
        
    def mouseReleaseEvent(self, event):
        if self.dragging_cue:
            self.dragging_cue = False
            self.setCursor(Qt.ArrowCursor)
        super().mouseReleaseEvent(event)
        
    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        height = self.height()
        
        # Background gradient
        gradient = QLinearGradient(0, 0, 0, height)
        gradient.setColorAt(0, QColor('#1e1e1e'))
        gradient.setColorAt(1, QColor('#181818'))
        painter.fillRect(0, 0, self.width(), height, gradient)
        
        # Draw bar markers
        for beat in range(0, 300):
            x = int(beat * self.pixels_per_beat)
            if x > self.width():
                break
            
            if beat % 16 == 0:  # 4-bar section
                painter.setPen(QPen(QColor('#555555'), 2))
                painter.drawLine(x, height - 20, x, height)
                
                # Section number
                section = beat // 16 + 1
                painter.setPen(QColor('#00aaff'))
                painter.setFont(QFont("Segoe UI", 11, QFont.Bold))
                painter.drawText(x + 4, 16, f"S{section}")
                
            elif beat % 4 == 0:  # Bar marker
                bar = beat // 4 + 1
                painter.setPen(QPen(QColor('#444444'), 1))
                painter.drawLine(x, height - 15, x, height)
                
                # Bar number
                painter.setPen(QColor('#888888'))
                painter.setFont(QFont("Segoe UI", 9))
                painter.drawText(x + 3, 28, str(bar))
                
            else:  # Beat marker
                painter.setPen(QPen(QColor('#333333'), 1))
                painter.drawLine(x, height - 8, x, height)
        
        # Draw start cue marker (orange/yellow triangle)
        cue_x = int(self.start_cue_beat * self.pixels_per_beat)
        painter.setBrush(QBrush(QColor('#ffaa00')))
        painter.setPen(QPen(QColor('#ffcc00'), 1))
        # Draw downward triangle
        triangle = QPainterPath()
        triangle.moveTo(cue_x, 2)
        triangle.lineTo(cue_x - 6, 14)
        triangle.lineTo(cue_x + 6, 14)
        triangle.closeSubpath()
        painter.drawPath(triangle)
        # Draw vertical line from cue
        painter.setPen(QPen(QColor('#ffaa00'), 1, Qt.DashLine))
        painter.drawLine(cue_x, 14, cue_x, height)
        
        # Draw playhead marker (cyan/blue triangle and line)
        if self.playhead_beat >= 0:
            ph_x = int(self.playhead_beat * self.pixels_per_beat)
            painter.setBrush(QBrush(QColor('#00ffff')))
            painter.setPen(QPen(QColor('#00ffff'), 1))
            # Draw downward triangle for playhead
            ph_triangle = QPainterPath()
            ph_triangle.moveTo(ph_x, 2)
            ph_triangle.lineTo(ph_x - 5, 12)
            ph_triangle.lineTo(ph_x + 5, 12)
            ph_triangle.closeSubpath()
            painter.drawPath(ph_triangle)
        
        # Bottom border
        painter.setPen(QPen(QColor('#00aaff'), 2))
        painter.drawLine(0, height - 1, self.width(), height - 1)


class PlayheadOverlay(QWidget):
    """Transparent overlay widget that draws the playhead line across all tracks."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)  # Let clicks pass through
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.playhead_beat = 0.0
        self.pixels_per_beat = 30
        self.is_playing = False
        
    def set_playhead(self, beat: float):
        """Update playhead position"""
        self.playhead_beat = beat
        self.update()
        
    def set_playing(self, playing: bool):
        """Set playing state for visual feedback"""
        self.is_playing = playing
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Calculate playhead x position
        ph_x = int(self.playhead_beat * self.pixels_per_beat)
        
        if ph_x < 0 or ph_x > self.width():
            return
        
        # Draw playhead line - bright cyan when playing, dimmer when stopped
        if self.is_playing:
            color = QColor('#00ffff')
            width = 2
        else:
            color = QColor('#00aaaa')
            width = 1
        
        painter.setPen(QPen(color, width))
        painter.drawLine(ph_x, 0, ph_x, self.height())
        
        # Draw glow effect when playing
        if self.is_playing:
            glow = QColor(0, 255, 255, 30)
            painter.fillRect(ph_x - 3, 0, 6, self.height(), glow)


# ═══════════════════════════════════════════════════════════════════════════════
# AUTOMATION LANE WIDGET
# ═══════════════════════════════════════════════════════════════════════════════

class AutomationLaneWidget(QWidget):
    """Visual editor for automation curves with point editing"""
    
    point_added = pyqtSignal(float, float)  # beat, value
    point_moved = pyqtSignal(int, float, float)  # index, new_beat, new_value
    point_removed = pyqtSignal(int)  # index
    
    def __init__(self, param_name: str = "Volume", parent=None):
        super().__init__(parent)
        self.param_name = param_name
        self.points: List[Tuple[float, float]] = []  # (beat, value) pairs
        
        # Display settings
        self.beat_width = 40
        self.min_value = 0.0
        self.max_value = 1.0
        self.grid_beats = 32
        
        # Editing state
        self.selected_point = -1
        self.dragging = False
        self.drag_start: Optional[QPointF] = None
        
        # Recording
        self.recording = False
        self.record_buffer: List[Tuple[float, float]] = []
        
        self.setMinimumHeight(60)
        self.setMaximumHeight(80)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        
        self.setStyleSheet("background: #151515;")
    
    def set_points(self, points: List[Tuple[float, float]]):
        """Set automation points"""
        self.points = list(points)
        self.update()
    
    def add_point(self, beat: float, value: float):
        """Add a point and keep sorted by beat"""
        self.points.append((beat, value))
        self.points.sort(key=lambda p: p[0])
        self.point_added.emit(beat, value)
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        w = self.width()
        h = self.height()
        
        # Background
        painter.fillRect(0, 0, w, h, QColor("#151515"))
        
        # Grid lines (every beat)
        painter.setPen(QPen(QColor("#222222"), 1))
        for beat in range(self.grid_beats + 1):
            x = beat * self.beat_width
            if x > w:
                break
            if beat % 4 == 0:
                painter.setPen(QPen(QColor("#333333"), 1))
            else:
                painter.setPen(QPen(QColor("#1a1a1a"), 1))
            painter.drawLine(int(x), 0, int(x), h)
        
        # Center line (0.5 value)
        mid_y = h // 2
        painter.setPen(QPen(QColor("#333333"), 1, Qt.DashLine))
        painter.drawLine(0, mid_y, w, mid_y)
        
        # Draw automation curve
        if len(self.points) >= 2:
            path = QPainterPath()
            first_point = self.points[0]
            x0 = first_point[0] * self.beat_width
            y0 = h - (first_point[1] - self.min_value) / (self.max_value - self.min_value) * h
            path.moveTo(x0, y0)
            
            for beat, value in self.points[1:]:
                x = beat * self.beat_width
                y = h - (value - self.min_value) / (self.max_value - self.min_value) * h
                path.lineTo(x, y)
            
            # Draw curve
            painter.setPen(QPen(QColor("#00aaff"), 2))
            painter.drawPath(path)
            
            # Fill under curve
            fill_path = QPainterPath(path)
            fill_path.lineTo(self.points[-1][0] * self.beat_width, h)
            fill_path.lineTo(self.points[0][0] * self.beat_width, h)
            fill_path.closeSubpath()
            painter.fillPath(fill_path, QColor(0, 170, 255, 30))
        
        # Draw points
        for i, (beat, value) in enumerate(self.points):
            x = beat * self.beat_width
            y = h - (value - self.min_value) / (self.max_value - self.min_value) * h
            
            if i == self.selected_point:
                painter.setBrush(QBrush(QColor("#00ffaa")))
                painter.setPen(QPen(QColor("#ffffff"), 2))
                radius = 6
            else:
                painter.setBrush(QBrush(QColor("#00aaff")))
                painter.setPen(QPen(QColor("#ffffff"), 1))
                radius = 4
            
            painter.drawEllipse(QPointF(x, y), radius, radius)
        
        # Param name
        painter.setPen(QColor("#666666"))
        painter.setFont(QFont("Segoe UI", 8))
        painter.drawText(4, 12, self.param_name)
        
        # Recording indicator
        if self.recording:
            painter.fillRect(w - 40, 4, 36, 12, QColor("#ff0000"))
            painter.setPen(QColor("#ffffff"))
            painter.drawText(w - 38, 13, "REC")
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # Check if clicking on a point
            hit = self._hit_test(event.pos())
            if hit >= 0:
                self.selected_point = hit
                self.dragging = True
                self.drag_start = event.pos()
            else:
                # Add new point
                beat = event.x() / self.beat_width
                value = self.max_value - (event.y() / self.height()) * (self.max_value - self.min_value)
                value = max(self.min_value, min(self.max_value, value))
                self.add_point(beat, value)
                
                # Select the new point
                for i, (b, v) in enumerate(self.points):
                    if abs(b - beat) < 0.01:
                        self.selected_point = i
                        break
            
            self.update()
        
        elif event.button() == Qt.RightButton:
            # Remove point
            hit = self._hit_test(event.pos())
            if hit >= 0:
                del self.points[hit]
                self.point_removed.emit(hit)
                self.selected_point = -1
                self.update()
    
    def mouseMoveEvent(self, event):
        if self.dragging and self.selected_point >= 0:
            beat = event.x() / self.beat_width
            value = self.max_value - (event.y() / self.height()) * (self.max_value - self.min_value)
            value = max(self.min_value, min(self.max_value, value))
            beat = max(0, beat)
            
            self.points[self.selected_point] = (beat, value)
            self.points.sort(key=lambda p: p[0])
            
            # Re-find the selected point after sorting
            for i, (b, v) in enumerate(self.points):
                if abs(b - beat) < 0.01 and abs(v - value) < 0.01:
                    self.selected_point = i
                    break
            
            self.update()
    
    def mouseReleaseEvent(self, event):
        if self.dragging and self.selected_point >= 0:
            beat, value = self.points[self.selected_point]
            self.point_moved.emit(self.selected_point, beat, value)
        self.dragging = False
        self.drag_start = None
    
    def _hit_test(self, pos: QPointF) -> int:
        """Check if position hits a point, return index or -1"""
        h = self.height()
        for i, (beat, value) in enumerate(self.points):
            x = beat * self.beat_width
            y = h - (value - self.min_value) / (self.max_value - self.min_value) * h
            
            dist = ((pos.x() - x) ** 2 + (pos.y() - y) ** 2) ** 0.5
            if dist <= 8:
                return i
        return -1
    
    def start_recording(self):
        """Start recording automation"""
        self.recording = True
        self.record_buffer.clear()
        self.update()
    
    def record_value(self, beat: float, value: float):
        """Record a value at beat position"""
        if self.recording:
            self.record_buffer.append((beat, value))
    
    def stop_recording(self):
        """Stop recording and merge buffer"""
        self.recording = False
        
        # Merge recorded points (thin them out)
        if self.record_buffer:
            # Simple thinning: keep every Nth point
            thinned = self.record_buffer[::4]  # Keep every 4th point
            for beat, value in thinned:
                self.add_point(beat, value)
            self.record_buffer.clear()
        
        self.update()


class ArrangementView(QWidget):
    """Professional arrangement view with larger tracks and timeline."""
    
    clip_selected = pyqtSignal(object)
    clip_double_clicked = pyqtSignal(object)
    track_selected = pyqtSignal(int)
    add_track_requested = pyqtSignal(str)  # Signal to main window
    position_changed = pyqtSignal(float)  # Signal when user clicks to change position
    start_cue_changed = pyqtSignal(float)  # Signal when start cue is moved
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.tracks = []
        self.selected_track = -1
        self.playhead_overlay = None
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Top toolbar
        toolbar = QFrame()
        toolbar.setFixedHeight(36)
        toolbar.setStyleSheet("""
            QFrame { 
                background-color: #151515; 
                border-bottom: 1px solid #2a2a2a;
            }
        """)
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(10, 4, 10, 4)
        toolbar_layout.setSpacing(8)
        
        # Zoom controls
        zoom_label = QLabel("Zoom:")
        zoom_label.setStyleSheet("color: #888;")
        toolbar_layout.addWidget(zoom_label)
        
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(10, 60)
        self.zoom_slider.setValue(30)
        self.zoom_slider.setFixedWidth(120)
        self.zoom_slider.valueChanged.connect(self._on_zoom_changed)
        toolbar_layout.addWidget(self.zoom_slider)
        
        toolbar_layout.addSpacing(20)
        
        # Snap controls
        snap_label = QLabel("Snap:")
        snap_label.setStyleSheet("color: #888;")
        toolbar_layout.addWidget(snap_label)
        
        self.snap_combo = QComboBox()
        self.snap_combo.addItems(["Off", "1/4", "1/8", "1/16", "1/32", "Bar"])
        self.snap_combo.setCurrentIndex(1)
        self.snap_combo.setFixedWidth(80)
        toolbar_layout.addWidget(self.snap_combo)
        
        toolbar_layout.addStretch()
        
        # Add track buttons
        add_audio_btn = QPushButton("+ Audio")
        add_audio_btn.setFixedWidth(80)
        add_audio_btn.clicked.connect(lambda: self._request_add_track("audio"))
        toolbar_layout.addWidget(add_audio_btn)
        
        add_inst_btn = QPushButton("+ Instrument")
        add_inst_btn.setFixedWidth(100)
        add_inst_btn.clicked.connect(lambda: self._request_add_track("synth"))
        toolbar_layout.addWidget(add_inst_btn)
        
        add_drums_btn = QPushButton("+ Drums")
        add_drums_btn.setFixedWidth(80)
        add_drums_btn.clicked.connect(lambda: self._request_add_track("drums"))
        toolbar_layout.addWidget(add_drums_btn)
        
        layout.addWidget(toolbar)
        
        # Main content with headers and lanes
        content = QWidget()
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        
        # Left panel: Track headers
        self.header_scroll = QScrollArea()
        self.header_scroll.setWidgetResizable(True)
        self.header_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.header_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.header_scroll.setFixedWidth(204)
        
        self.header_container = QWidget()
        self.header_layout = QVBoxLayout(self.header_container)
        self.header_layout.setContentsMargins(0, 40, 0, 0)  # Space for timeline
        self.header_layout.setSpacing(0)
        self.header_layout.addStretch()
        
        self.header_scroll.setWidget(self.header_container)
        content_layout.addWidget(self.header_scroll)
        
        # Right panel: Timeline + Lanes
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        
        # Timeline header (scrolls horizontally with lanes)
        self.timeline = TimelineHeader()
        
        # Lanes scroll area
        self.lane_scroll = QScrollArea()
        self.lane_scroll.setWidgetResizable(True)
        self.lane_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.lane_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        
        self.lane_container = QWidget()
        self.lane_layout = QVBoxLayout(self.lane_container)
        self.lane_layout.setContentsMargins(0, 0, 0, 0)
        self.lane_layout.setSpacing(0)
        
        # Add timeline to lane container
        self.lane_layout.addWidget(self.timeline)
        self.lane_layout.addStretch()
        
        self.lane_scroll.setWidget(self.lane_container)
        
        # Create playhead overlay on top of lane scroll
        self.playhead_overlay = PlayheadOverlay(self.lane_scroll.viewport())
        self.playhead_overlay.setGeometry(0, 0, 6000, 2000)
        self.playhead_overlay.raise_()  # Ensure it's on top
        
        # Connect timeline signals
        self.timeline.position_clicked.connect(self.position_changed.emit)
        self.timeline.start_cue_changed.connect(self.start_cue_changed.emit)
        
        # Sync vertical scrolling
        self.lane_scroll.verticalScrollBar().valueChanged.connect(
            self.header_scroll.verticalScrollBar().setValue
        )
        self.header_scroll.verticalScrollBar().valueChanged.connect(
            self.lane_scroll.verticalScrollBar().setValue
        )
        
        right_layout.addWidget(self.lane_scroll)
        content_layout.addWidget(right_panel, 1)
        
        layout.addWidget(content, 1)
    
    def set_playhead(self, beat: float):
        """Update playhead position in timeline and overlay."""
        self.timeline.set_playhead(beat)
        if self.playhead_overlay:
            self.playhead_overlay.set_playhead(beat)
            
    def set_playing(self, playing: bool):
        """Update playing state for playhead visualization."""
        if self.playhead_overlay:
            self.playhead_overlay.set_playing(playing)
            
    def set_start_cue(self, beat: float):
        """Set the start cue position."""
        self.timeline.set_start_cue(beat)
    
    def _on_zoom_changed(self, value):
        """Update zoom level for all lanes."""
        for header, lane, widget, track_type in self.tracks:
            lane.pixels_per_beat = value
            lane.update()
        self.timeline.pixels_per_beat = value
        self.timeline.update()
        if self.playhead_overlay:
            self.playhead_overlay.pixels_per_beat = value
            self.playhead_overlay.update()
        
    def _request_add_track(self, track_type: str):
        """Request to add a new track - emit signal to main window."""
        self.add_track_requested.emit(track_type)
        
    def add_track(self, name: str, track_type: str = "audio") -> TrackLane:
        """Add a new track to the arrangement."""
        idx = len(self.tracks)
        
        # Get track color
        track_colors = TrackHeader.TRACK_COLORS
        track_color = track_colors[idx % len(track_colors)]
        
        # Create header
        header = TrackHeader(idx, name, track_type)
        header.mute_changed.connect(self._on_track_mute)
        header.solo_changed.connect(self._on_track_solo)
        header.mousePressEvent = lambda e, h=header: self._select_track(idx)
        
        # Create lane
        lane = TrackLane(idx, track_type, track_color)
        lane.setMinimumWidth(6000)  # Long scrollable area
        lane.pixels_per_beat = self.zoom_slider.value()
        
        # Connect signals
        lane.clip_selected.connect(self.clip_selected.emit)
        lane.clip_double_clicked.connect(self.clip_double_clicked.emit)
        
        # Track widget wrapper for lane
        track_widget = QFrame()
        track_widget.setMinimumHeight(100)
        
        self.tracks.append((header, lane, track_widget, track_type))
        
        # Insert header before stretch
        self.header_layout.insertWidget(self.header_layout.count() - 1, header)
        
        # Insert lane before stretch (after timeline)
        self.lane_layout.insertWidget(self.lane_layout.count() - 1, lane)
        
        return lane
    
    def _on_track_mute(self, track_idx: int, muted: bool):
        """Handle track mute."""
        # Update all clips muted state
        if track_idx < len(self.tracks):
            header, lane, widget, track_type = self.tracks[track_idx]
            for clip in lane.clips:
                clip.muted = muted
            lane.update()
            
    def _on_track_solo(self, track_idx: int, soloed: bool):
        """Handle track solo - mute all others."""
        for idx, (header, lane, widget, track_type) in enumerate(self.tracks):
            if soloed and idx != track_idx:
                header.mute_btn.setChecked(True)
                for clip in lane.clips:
                    clip.muted = True
                lane.update()
                
    def _select_track(self, track_idx: int):
        """Select a track."""
        # Deselect previous
        if 0 <= self.selected_track < len(self.tracks):
            self.tracks[self.selected_track][0].set_selected(False)
        
        # Select new
        self.selected_track = track_idx
        if 0 <= track_idx < len(self.tracks):
            self.tracks[track_idx][0].set_selected(True)
            self.track_selected.emit(track_idx)


# ═══════════════════════════════════════════════════════════════════════════════
# CHANNEL STRIP - Professional Mixer Channel
# ═══════════════════════════════════════════════════════════════════════════════

class MeterWidget(QFrame):
    """VU Meter widget with gradient display."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.level_l = 0.0
        self.level_r = 0.0
        self.peak_l = 0.0
        self.peak_r = 0.0
        self.setFixedWidth(24)
        self.setMinimumHeight(100)
        
    def set_levels(self, left: float, right: float):
        self.level_l = max(0, min(1, left))
        self.level_r = max(0, min(1, right))
        self.peak_l = max(self.peak_l * 0.95, self.level_l)
        self.peak_r = max(self.peak_r * 0.95, self.level_r)
        self.update()
        
    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        w = self.width()
        h = self.height()
        meter_w = 8
        gap = 4
        
        for i, (level, peak) in enumerate([(self.level_l, self.peak_l), 
                                            (self.level_r, self.peak_r)]):
            x = i * (meter_w + gap) + 2
            
            # Background
            painter.fillRect(x, 0, meter_w, h, QColor('#1a1a1a'))
            
            # Level gradient
            level_h = int(level * h)
            if level_h > 0:
                gradient = QLinearGradient(0, h, 0, 0)
                gradient.setColorAt(0, QColor('#00ff00'))
                gradient.setColorAt(0.6, QColor('#ffff00'))
                gradient.setColorAt(0.85, QColor('#ff8800'))
                gradient.setColorAt(1.0, QColor('#ff0000'))
                painter.fillRect(x, h - level_h, meter_w, level_h, gradient)
            
            # Peak indicator with hold
            peak_y = int((1 - peak) * h)
            if peak > 0.9:
                painter.setPen(QPen(QColor('#ff0000'), 3))
            elif peak > 0.7:
                painter.setPen(QPen(QColor('#ffaa00'), 2))
            else:
                painter.setPen(QPen(QColor('#00ff00'), 2))
            painter.drawLine(x, peak_y, x + meter_w, peak_y)
    
    def clear_clip(self):
        """Reset peak hold"""
        self.peak_l = 0.0
        self.peak_r = 0.0
        self.update()


# ═══════════════════════════════════════════════════════════════════════════════
# INSERT SLOT - FX Insert for Channel Strip
# ═══════════════════════════════════════════════════════════════════════════════

class InsertSlot(QFrame):
    """Single insert slot for FX chain"""
    
    fx_changed = pyqtSignal(int, str)  # slot_index, fx_name
    fx_removed = pyqtSignal(int)  # slot_index
    
    FX_LIST = [
        "---",
        "EQ 3-Band",
        "Compressor",
        "Limiter",
        "Reverb",
        "Delay",
        "Chorus",
        "Phaser",
        "Distortion",
        "Filter LP",
        "Filter HP",
        "Gate",
        "Φ-RFT Spatial"
    ]
    
    def __init__(self, slot_index: int, parent=None):
        super().__init__(parent)
        self.slot_index = slot_index
        self.fx_name = "---"
        self.enabled = True
        
        self.setFixedHeight(20)
        self.setStyleSheet("""
            InsertSlot {
                background: #1a1a1a;
                border: 1px solid #333;
                border-radius: 2px;
            }
            InsertSlot:hover {
                border-color: #00aaff;
            }
        """)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 0, 2, 0)
        layout.setSpacing(2)
        
        # Enable button
        self.enable_btn = QPushButton()
        self.enable_btn.setFixedSize(12, 12)
        self.enable_btn.setCheckable(True)
        self.enable_btn.setChecked(True)
        self.enable_btn.setStyleSheet("""
            QPushButton { background: #333; border: none; border-radius: 2px; }
            QPushButton:checked { background: #00aaff; }
        """)
        self.enable_btn.clicked.connect(self._toggle_enable)
        layout.addWidget(self.enable_btn)
        
        # FX label (click to open menu)
        self.fx_label = QLabel("---")
        self.fx_label.setStyleSheet("color: #666; font-size: 8px;")
        self.fx_label.setCursor(Qt.PointingHandCursor)
        self.fx_label.mousePressEvent = self._show_fx_menu
        layout.addWidget(self.fx_label, 1)
    
    def _toggle_enable(self, checked):
        self.enabled = checked
        self.fx_label.setStyleSheet(f"color: {'#aaa' if checked else '#444'}; font-size: 8px;")
    
    def _show_fx_menu(self, event):
        menu = QMenu(self)
        for fx in self.FX_LIST:
            action = menu.addAction(fx)
            action.triggered.connect(lambda c, f=fx: self._set_fx(f))
        menu.exec_(event.globalPos())
    
    def _set_fx(self, fx_name: str):
        self.fx_name = fx_name
        self.fx_label.setText(fx_name if fx_name != "---" else "---")
        if fx_name != "---":
            self.fx_label.setStyleSheet("color: #00ffaa; font-size: 8px;")
            self.fx_changed.emit(self.slot_index, fx_name)
        else:
            self.fx_label.setStyleSheet("color: #666; font-size: 8px;")
            self.fx_removed.emit(self.slot_index)


# ═══════════════════════════════════════════════════════════════════════════════
# SEND KNOB - Aux Send Control
# ═══════════════════════════════════════════════════════════════════════════════

class SendKnob(QFrame):
    """Aux send level control"""
    
    level_changed = pyqtSignal(int, float)  # send_index, level
    
    SEND_COLORS = ["#00aaff", "#00ffaa", "#ffaa00", "#ff00aa"]
    
    def __init__(self, send_index: int, label: str = "A", parent=None):
        super().__init__(parent)
        self.send_index = send_index
        self.level = 0.0
        self.color = self.SEND_COLORS[send_index % len(self.SEND_COLORS)]
        
        self.setFixedSize(32, 40)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)
        
        # Label
        lbl = QLabel(label)
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet(f"color: {self.color}; font-size: 8px; font-weight: bold;")
        layout.addWidget(lbl)
        
        # Knob (slider for now)
        self.slider = QSlider(Qt.Vertical)
        self.slider.setRange(0, 100)
        self.slider.setValue(0)
        self.slider.setFixedHeight(28)
        self.slider.valueChanged.connect(self._on_change)
        self.slider.setStyleSheet(f"""
            QSlider::groove:vertical {{ background: #222; width: 4px; border-radius: 2px; }}
            QSlider::handle:vertical {{ background: {self.color}; height: 8px; margin: -2px; border-radius: 3px; }}
        """)
        layout.addWidget(self.slider, 0, Qt.AlignCenter)
    
    def _on_change(self, value):
        self.level = value / 100.0
        self.level_changed.emit(self.send_index, self.level)


class ChannelStrip(QFrame):
    """Professional mixer channel strip with fader, pan, meters, inserts, and sends."""
    
    volume_changed = pyqtSignal(int, float)
    pan_changed = pyqtSignal(int, float)
    send_changed = pyqtSignal(int, int, float)  # track_idx, send_idx, level
    fx_changed = pyqtSignal(int, int, str)  # track_idx, slot_idx, fx_name
    
    COLORS = TrackHeader.TRACK_COLORS
    
    def __init__(self, track_name: str, index: int, parent=None, show_inserts: bool = True, show_sends: bool = True):
        super().__init__(parent)
        self.index = index
        self.track_color = self.COLORS[index % len(self.COLORS)] if index >= 0 else '#00aaff'
        self.show_inserts = show_inserts
        self.show_sends = show_sends
        self.insert_slots: List[InsertSlot] = []
        self.send_knobs: List[SendKnob] = []
        
        self.setFixedWidth(90 if (show_inserts or show_sends) else 80)
        self.setMinimumHeight(420 if (show_inserts or show_sends) else 360)
        self.setup_ui(track_name)
        self.update_style()
        
    def setup_ui(self, track_name: str):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 6, 4, 6)
        layout.setSpacing(4)
        
        # Color bar at top
        self.color_bar = QFrame()
        self.color_bar.setFixedHeight(4)
        self.color_bar.setStyleSheet(f"background-color: {self.track_color}; border-radius: 2px;")
        layout.addWidget(self.color_bar)
        
        # Track name
        self.name_label = QLabel(track_name)
        self.name_label.setAlignment(Qt.AlignCenter)
        self.name_label.setWordWrap(True)
        self.name_label.setStyleSheet(f"""
            color: {self.track_color}; 
            font-size: 9px; 
            font-weight: bold;
            padding: 2px;
        """)
        layout.addWidget(self.name_label)
        
        # Mute/Solo/Arm buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(2)
        
        self.mute_btn = self._create_btn("M", "#ff4444")
        self.solo_btn = self._create_btn("S", "#ffaa00")
        self.arm_btn = self._create_btn("R", "#ff0000")
        
        btn_row.addWidget(self.mute_btn)
        btn_row.addWidget(self.solo_btn)
        btn_row.addWidget(self.arm_btn)
        layout.addLayout(btn_row)
        
        # Insert slots (if enabled)
        if self.show_inserts and self.index >= 0:
            inserts_frame = QFrame()
            inserts_frame.setStyleSheet("background: #151515; border-radius: 3px;")
            inserts_layout = QVBoxLayout(inserts_frame)
            inserts_layout.setContentsMargins(2, 2, 2, 2)
            inserts_layout.setSpacing(1)
            
            # Label
            ins_label = QLabel("INSERTS")
            ins_label.setStyleSheet("color: #555; font-size: 7px;")
            ins_label.setAlignment(Qt.AlignCenter)
            inserts_layout.addWidget(ins_label)
            
            # 4 insert slots
            for i in range(4):
                slot = InsertSlot(i)
                slot.fx_changed.connect(lambda si, fn, ti=self.index: self.fx_changed.emit(ti, si, fn))
                self.insert_slots.append(slot)
                inserts_layout.addWidget(slot)
            
            layout.addWidget(inserts_frame)
        
        # Pan control
        pan_label = QLabel("PAN")
        pan_label.setStyleSheet("color: #555; font-size: 7px;")
        pan_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(pan_label)
        
        self.pan_slider = QSlider(Qt.Horizontal)
        self.pan_slider.setRange(-100, 100)
        self.pan_slider.setValue(0)
        self.pan_slider.setFixedHeight(14)
        self.pan_slider.valueChanged.connect(
            lambda v: self.pan_changed.emit(self.index, v / 100)
        )
        
        self.pan_value = QLabel("C")
        self.pan_value.setStyleSheet("color: #888; font-size: 8px; min-width: 24px;")
        self.pan_value.setAlignment(Qt.AlignCenter)
        self.pan_slider.valueChanged.connect(self._update_pan_label)
        
        layout.addWidget(self.pan_slider)
        layout.addWidget(self.pan_value)
        
        # Send knobs (if enabled)
        if self.show_sends and self.index >= 0:
            sends_row = QHBoxLayout()
            sends_row.setSpacing(2)
            
            for i, label in enumerate(["A", "B"]):
                knob = SendKnob(i, label)
                knob.level_changed.connect(lambda si, lv, ti=self.index: self.send_changed.emit(ti, si, lv))
                self.send_knobs.append(knob)
                sends_row.addWidget(knob)
            
            layout.addLayout(sends_row)
        
        # Meter + Fader section
        fader_section = QHBoxLayout()
        fader_section.setSpacing(2)
        
        # VU Meter
        self.meter = MeterWidget()
        fader_section.addWidget(self.meter)
        
        # Fader
        fader_container = QVBoxLayout()
        fader_container.setSpacing(2)
        
        self.fader = QSlider(Qt.Vertical)
        self.fader.setRange(0, 127)
        self.fader.setValue(100)
        self.fader.setMinimumHeight(140)
        self.fader.valueChanged.connect(self._on_fader_changed)
        fader_container.addWidget(self.fader, 1)
        
        fader_section.addLayout(fader_container)
        
        layout.addLayout(fader_section, 1)
        
        # dB display with gain staging indicator
        self.db_label = QLabel("0.0 dB")
        self.db_label.setAlignment(Qt.AlignCenter)
        self.db_label.setStyleSheet("""
            color: #00ffaa; 
            font-size: 9px; 
            font-family: 'Consolas', monospace;
            font-weight: bold;
            background: #1a1a1a;
            border-radius: 3px;
            padding: 3px;
        """)
        layout.addWidget(self.db_label)
        
    def _create_btn(self, text: str, active_color: str) -> QPushButton:
        btn = QPushButton(text)
        btn.setCheckable(True)
        btn.setFixedSize(22, 20)
        btn.setStyleSheet(f"""
            QPushButton {{
                background: #252525;
                border: 1px solid #3a3a3a;
                border-radius: 2px;
                font-size: 10px;
                font-weight: bold;
                color: #666;
                min-width: 22px;
                padding: 0;
            }}
            QPushButton:hover {{
                border-color: {active_color};
                color: {active_color};
            }}
            QPushButton:checked {{
                background: {active_color};
                border-color: {active_color};
                color: white;
            }}
        """)
        return btn
        
    def _update_pan_label(self, value):
        if value == 0:
            self.pan_value.setText("C")
        elif value < 0:
            self.pan_value.setText(f"L{abs(value)}")
        else:
            self.pan_value.setText(f"R{value}")
            
    def _on_fader_changed(self, value):
        if value == 0:
            self.db_label.setText("-inf dB")
        else:
            db = 20 * np.log10(value / 100)
            self.db_label.setText(f"{db:.1f} dB")
        self.volume_changed.emit(self.index, value / 127.0)
        
    def update_style(self):
        self.setStyleSheet(f"""
            ChannelStrip {{
                background-color: #1a1a1a;
                border-right: 1px solid #252525;
            }}
        """)
        
    def set_meter_level(self, left: float, right: float):
        self.meter.set_levels(left, right)


# ═══════════════════════════════════════════════════════════════════════════════
# MIXER VIEW - Professional Multi-Channel Mixer
# ═══════════════════════════════════════════════════════════════════════════════

class MixerView(QWidget):
    """Professional mixer view with channel strips, returns, and master."""
    
    track_volume_changed = pyqtSignal(int, float)  # track_idx, volume 0-1
    track_pan_changed = pyqtSignal(int, float)  # track_idx, pan -1 to 1
    track_muted = pyqtSignal(int, bool)  # track_idx, muted
    track_soloed = pyqtSignal(int, bool)  # track_idx, soloed
    send_level_changed = pyqtSignal(int, int, float)  # track_idx, send_idx, level
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.strips = []
        self.return_strips = []
        self.setup_ui()
        
        # Meter animation timer
        self.meter_timer = QTimer(self)
        self.meter_timer.timeout.connect(self._animate_meters)
        self.meter_timer.start(50)
        
    def setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Mixer header/toolbar
        self.setStyleSheet("""
            MixerView {
                background-color: #0d0d0d;
                border-top: 2px solid #00aaff;
            }
        """)
        
        # Scrollable area for channel strips
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("background: transparent; border: none;")
        
        self.strip_container = QWidget()
        self.strip_container.setStyleSheet("background: transparent;")
        self.strip_layout = QHBoxLayout(self.strip_container)
        self.strip_layout.setContentsMargins(8, 8, 8, 8)
        self.strip_layout.setSpacing(2)
        self.strip_layout.addStretch()
        
        scroll.setWidget(self.strip_container)
        layout.addWidget(scroll, 1)
        
        # Divider before returns
        divider1 = QFrame()
        divider1.setFixedWidth(2)
        divider1.setStyleSheet("background-color: #444;")
        layout.addWidget(divider1)
        
        # Return buses section
        returns_container = QWidget()
        returns_container.setStyleSheet("background: #0a0a0a;")
        returns_layout = QVBoxLayout(returns_container)
        returns_layout.setContentsMargins(4, 4, 4, 4)
        returns_layout.setSpacing(0)
        
        # Returns label
        returns_label = QLabel("RETURNS")
        returns_label.setStyleSheet("color: #666; font-size: 8px; font-weight: bold;")
        returns_label.setAlignment(Qt.AlignCenter)
        returns_layout.addWidget(returns_label)
        
        # Return strips
        returns_strips = QHBoxLayout()
        returns_strips.setSpacing(2)
        
        # Return A (Reverb)
        self.return_a = ChannelStrip("Return A", -2, show_inserts=False, show_sends=False)
        self.return_a.setFixedWidth(70)
        self.return_a.color_bar.setStyleSheet("background: #00aaff; border-radius: 2px;")
        self.return_a.name_label.setStyleSheet("color: #00aaff; font-size: 9px; font-weight: bold;")
        returns_strips.addWidget(self.return_a)
        self.return_strips.append(self.return_a)
        
        # Return B (Delay)
        self.return_b = ChannelStrip("Return B", -3, show_inserts=False, show_sends=False)
        self.return_b.setFixedWidth(70)
        self.return_b.color_bar.setStyleSheet("background: #00ffaa; border-radius: 2px;")
        self.return_b.name_label.setStyleSheet("color: #00ffaa; font-size: 9px; font-weight: bold;")
        returns_strips.addWidget(self.return_b)
        self.return_strips.append(self.return_b)
        
        returns_layout.addLayout(returns_strips)
        layout.addWidget(returns_container)
        
        # Divider before master
        divider2 = QFrame()
        divider2.setFixedWidth(2)
        divider2.setStyleSheet("background-color: #00aaff;")
        layout.addWidget(divider2)
        
        # Master channel (always visible)
        self.master = ChannelStrip("MASTER", -1, show_inserts=True, show_sends=False)
        self.master.setFixedWidth(95)
        self.master.color_bar.setStyleSheet("background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00aaff, stop:1 #00ffaa); border-radius: 2px;")
        self.master.name_label.setStyleSheet("color: #00ffaa; font-size: 10px; font-weight: bold;")
        layout.addWidget(self.master)
        
    def add_strip(self, name: str) -> ChannelStrip:
        """Add a channel strip."""
        idx = len(self.strips)
        strip = ChannelStrip(name, idx)
        
        # Connect strip signals to mixer signals
        strip.volume_changed.connect(self.track_volume_changed.emit)
        strip.pan_changed.connect(self.track_pan_changed.emit)
        strip.mute_btn.clicked.connect(lambda c, i=idx: self.track_muted.emit(i, c))
        strip.solo_btn.clicked.connect(lambda c, i=idx: self.track_soloed.emit(i, c))
        strip.send_changed.connect(self.send_level_changed.emit)
        
        self.strips.append(strip)
        self.strip_layout.insertWidget(self.strip_layout.count() - 1, strip)
        return strip
        
    def _animate_meters(self):
        """Animate VU meters with simulated levels."""
        import random
        for strip in self.strips:
            # Simulate some activity
            base = 0.3 + random.random() * 0.4
            strip.set_meter_level(base + random.random() * 0.1, 
                                  base + random.random() * 0.1)
        
        # Master gets sum
        if self.strips:
            avg = sum(s.meter.level_l for s in self.strips) / len(self.strips)
            self.master.set_meter_level(avg * 1.1, avg * 1.1)


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSPORT BAR - Professional Transport Controls
# ═══════════════════════════════════════════════════════════════════════════════

class TransportBar(QFrame):
    """Professional transport controls: play, stop, record, tempo, time display."""
    
    play_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()
    record_clicked = pyqtSignal()
    tempo_changed = pyqtSignal(float)
    loop_toggled = pyqtSignal(bool)
    metronome_toggled = pyqtSignal(bool)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_playing = False
        self.is_recording = False
        self.loop_enabled = False
        self.metronome_enabled = True
        
        # Tap tempo state
        self.tap_times = []
        self.tap_timeout = 2.0  # Reset after 2 seconds of no taps
        
        self.setFixedHeight(60)
        self.setup_ui()
        
    def setup_ui(self):
        self.setStyleSheet("""
            TransportBar {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #151515, stop:1 #0a0a0a);
                border-top: 2px solid #00aaff;
            }
        """)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 6, 16, 6)
        layout.setSpacing(12)
        
        # Position display (large, prominent)
        self.position_display = QFrame()
        self.position_display.setStyleSheet("""
            QFrame {
                background-color: #000000;
                border: 2px solid #2a2a2a;
                border-radius: 6px;
            }
        """)
        pos_layout = QHBoxLayout(self.position_display)
        pos_layout.setContentsMargins(12, 4, 12, 4)
        
        self.position_label = QLabel("001 : 01 : 000")
        self.position_label.setStyleSheet("""
            color: #00ffaa;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 20px;
            font-weight: bold;
        """)
        pos_layout.addWidget(self.position_label)
        
        layout.addWidget(self.position_display)
        
        layout.addSpacing(20)
        
        # Transport buttons - large and prominent
        transport_frame = QFrame()
        transport_layout = QHBoxLayout(transport_frame)
        transport_layout.setContentsMargins(0, 0, 0, 0)
        transport_layout.setSpacing(6)
        
        # Go to start
        self.start_btn = self._create_transport_btn("⏮", "Go to Start", "#888888")
        self.start_btn.clicked.connect(self._on_go_start)
        transport_layout.addWidget(self.start_btn)
        
        # Stop
        self.stop_btn = self._create_transport_btn("⏹", "Stop", "#ffffff")
        self.stop_btn.clicked.connect(self.on_stop)
        transport_layout.addWidget(self.stop_btn)
        
        # Play/Pause
        self.play_btn = self._create_transport_btn("▶", "Play", "#00ffaa", size=48)
        self.play_btn.clicked.connect(self.on_play)
        transport_layout.addWidget(self.play_btn)
        
        # Record
        self.record_btn = self._create_transport_btn("⏺", "Record", "#ff4444")
        self.record_btn.setCheckable(True)
        self.record_btn.clicked.connect(self._on_record)
        transport_layout.addWidget(self.record_btn)
        
        # Loop
        self.loop_btn = self._create_transport_btn("🔁", "Loop", "#ffaa00")
        self.loop_btn.setCheckable(True)
        self.loop_btn.clicked.connect(self._on_loop_toggle)
        transport_layout.addWidget(self.loop_btn)
        
        layout.addWidget(transport_frame)
        
        layout.addSpacing(20)
        
        # Tempo section
        tempo_frame = QFrame()
        tempo_frame.setStyleSheet("""
            QFrame {
                background-color: #1a1a1a;
                border: 1px solid #2a2a2a;
                border-radius: 6px;
                padding: 4px;
            }
        """)
        tempo_layout = QHBoxLayout(tempo_frame)
        tempo_layout.setContentsMargins(8, 4, 8, 4)
        tempo_layout.setSpacing(8)
        
        tempo_label = QLabel("BPM")
        tempo_label.setStyleSheet("color: #888; font-size: 10px;")
        tempo_layout.addWidget(tempo_label)
        
        self.tempo_spin = QDoubleSpinBox()
        self.tempo_spin.setRange(20, 300)
        self.tempo_spin.setValue(120)
        self.tempo_spin.setSingleStep(0.5)
        self.tempo_spin.setDecimals(1)
        self.tempo_spin.setFixedWidth(80)
        self.tempo_spin.setStyleSheet("""
            QDoubleSpinBox {
                background: #0d0d0d;
                color: #00ffaa;
                font-size: 14px;
                font-weight: bold;
                border: none;
            }
        """)
        self.tempo_spin.valueChanged.connect(lambda v: self.tempo_changed.emit(v))
        tempo_layout.addWidget(self.tempo_spin)
        
        # Tap tempo button
        self.tap_btn = QPushButton("TAP")
        self.tap_btn.setFixedSize(40, 28)
        self.tap_btn.setStyleSheet("""
            QPushButton {
                background: #252525;
                border: 1px solid #3a3a3a;
                border-radius: 4px;
                color: #888;
                font-size: 9px;
                font-weight: bold;
            }
            QPushButton:hover {
                border-color: #00aaff;
                color: #00aaff;
            }
            QPushButton:pressed {
                background: #00aaff;
                color: white;
            }
        """)
        self.tap_btn.clicked.connect(self._on_tap_tempo)
        tempo_layout.addWidget(self.tap_btn)
        
        layout.addWidget(tempo_frame)
        
        # Time signature
        sig_frame = QFrame()
        sig_frame.setStyleSheet("""
            QFrame {
                background-color: #1a1a1a;
                border: 1px solid #2a2a2a;
                border-radius: 6px;
            }
        """)
        sig_layout = QHBoxLayout(sig_frame)
        sig_layout.setContentsMargins(8, 4, 8, 4)
        
        sig_label = QLabel("Time")
        sig_label.setStyleSheet("color: #888; font-size: 10px;")
        sig_layout.addWidget(sig_label)
        
        self.time_sig = QComboBox()
        self.time_sig.addItems(["4/4", "3/4", "6/8", "5/4", "7/8", "2/4"])
        self.time_sig.setFixedWidth(60)
        sig_layout.addWidget(self.time_sig)
        
        layout.addWidget(sig_frame)
        
        # Metronome toggle
        self.metro_btn = QPushButton("🔔")
        self.metro_btn.setCheckable(True)
        self.metro_btn.setChecked(True)
        self.metro_btn.setFixedSize(36, 36)
        self.metro_btn.setToolTip("Metronome")
        self.metro_btn.setStyleSheet("""
            QPushButton {
                background: #252525;
                border: 1px solid #3a3a3a;
                border-radius: 4px;
                font-size: 16px;
            }
            QPushButton:hover {
                border-color: #00aaff;
            }
            QPushButton:checked {
                background: #00aaff;
                border-color: #00aaff;
            }
        """)
        self.metro_btn.clicked.connect(self._on_metro_toggle)
        layout.addWidget(self.metro_btn)
        
        layout.addStretch()
        
        # Performance info
        info_frame = QFrame()
        info_layout = QVBoxLayout(info_frame)
        info_layout.setContentsMargins(0, 0, 0, 0)
        info_layout.setSpacing(2)
        
        self.cpu_label = QLabel("CPU: 2.1%")
        self.cpu_label.setStyleSheet("color: #00ffaa; font-size: 10px;")
        info_layout.addWidget(self.cpu_label)
        
        self.latency_label = QLabel("Latency: 5.8ms")
        self.latency_label.setStyleSheet("color: #888; font-size: 10px;")
        info_layout.addWidget(self.latency_label)
        
        self.engine_label = QLabel("Φ-RFT Native")
        self.engine_label.setStyleSheet("color: #00aaff; font-size: 10px; font-weight: bold;")
        info_layout.addWidget(self.engine_label)
        
        layout.addWidget(info_frame)
        
    def _create_transport_btn(self, icon: str, tooltip: str, color: str, size: int = 40) -> QPushButton:
        btn = QPushButton(icon)
        btn.setFixedSize(size, size)
        btn.setToolTip(tooltip)
        btn.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2a2a2a, stop:1 #1e1e1e);
                border: 2px solid #3a3a3a;
                border-radius: {size // 2}px;
                font-size: {size // 2}px;
                color: {color};
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3a3a3a, stop:1 #2a2a2a);
                border-color: {color};
            }}
            QPushButton:pressed {{
                background: {color};
                color: #0d0d0d;
            }}
            QPushButton:checked {{
                background: {color};
                border-color: {color};
                color: white;
            }}
        """)
        return btn
        
    def _on_go_start(self):
        self.position_label.setText("001 : 01 : 000")
        if self.is_playing:
            self.on_stop()
            
    def _on_record(self, checked):
        self.is_recording = checked
        self.record_clicked.emit()
        
    def _on_loop_toggle(self, checked):
        self.loop_enabled = checked
        self.loop_toggled.emit(checked)
        
    def _on_metro_toggle(self, checked):
        self.metronome_enabled = checked
        self.metronome_toggled.emit(checked)
    
    def _on_tap_tempo(self):
        """Handle tap tempo - calculate BPM from tap intervals."""
        import time
        current_time = time.time()
        
        # Clear old taps if timeout
        if self.tap_times and (current_time - self.tap_times[-1]) > self.tap_timeout:
            self.tap_times.clear()
        
        self.tap_times.append(current_time)
        
        # Need at least 2 taps to calculate tempo
        if len(self.tap_times) >= 2:
            # Calculate average interval from last 4 taps
            recent_taps = self.tap_times[-4:]
            intervals = [recent_taps[i+1] - recent_taps[i] for i in range(len(recent_taps)-1)]
            avg_interval = sum(intervals) / len(intervals)
            
            # Convert to BPM
            bpm = 60.0 / avg_interval
            bpm = max(20, min(300, bpm))  # Clamp to valid range
            
            # Update tempo
            self.tempo_spin.setValue(round(bpm, 1))
        
        # Visual feedback - flash button
        self.tap_btn.setStyleSheet("""
            QPushButton {
                background: #00aaff;
                border: 1px solid #00aaff;
                border-radius: 4px;
                color: white;
                font-size: 9px;
                font-weight: bold;
            }
        """)
        QTimer.singleShot(100, self._reset_tap_btn_style)
    
    def _reset_tap_btn_style(self):
        """Reset tap button style after flash."""
        self.tap_btn.setStyleSheet("""
            QPushButton {
                background: #252525;
                border: 1px solid #3a3a3a;
                border-radius: 4px;
                color: #888;
                font-size: 9px;
                font-weight: bold;
            }
            QPushButton:hover {
                border-color: #00aaff;
                color: #00aaff;
            }
            QPushButton:pressed {
                background: #00aaff;
                color: white;
            }
        """)
        
    def on_play(self):
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.play_btn.setText("⏸")
            self.play_btn.setStyleSheet("""
                QPushButton {
                    background: #00ffaa;
                    border: 2px solid #00ffaa;
                    border-radius: 24px;
                    font-size: 24px;
                    color: #0d0d0d;
                }
                QPushButton:hover {
                    background: #00dd88;
                }
            """)
        else:
            self.play_btn.setText("▶")
            self.play_btn.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #2a2a2a, stop:1 #1e1e1e);
                    border: 2px solid #3a3a3a;
                    border-radius: 24px;
                    font-size: 24px;
                    color: #00ffaa;
                }
                QPushButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #3a3a3a, stop:1 #2a2a2a);
                    border-color: #00ffaa;
                }
            """)
        self.play_clicked.emit()
        
    def on_stop(self):
        self.is_playing = False
        self.play_btn.setText("▶")
        self.play_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2a2a2a, stop:1 #1e1e1e);
                border: 2px solid #3a3a3a;
                border-radius: 24px;
                font-size: 24px;
                color: #00ffaa;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3a3a3a, stop:1 #2a2a2a);
                border-color: #00ffaa;
            }
        """)
        self.position_label.setText("001 : 01 : 000")
        self.stop_clicked.emit()
        
    def update_position(self, bar: int, beat: int, tick: int):
        self.position_label.setText(f"{bar:03d} : {beat:02d} : {tick:03d}")

    def update_engine_status(self, is_native: bool, status_text: Optional[str] = None):
        """Visual badge showing whether the Φ-RFT engine is native or fallback."""
        if not hasattr(self, "engine_label"):
            return
        label = status_text or ("Φ-RFT Native" if is_native else "Φ-RFT Fallback")
        color = "#00ffaa" if is_native else "#ffcc66"
        self.engine_label.setText(label)
        self.engine_label.setStyleSheet(
            f"color: {color}; font-size: 10px; font-weight: bold;"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# DEVICE PANEL - Professional Effects Chain
# ═══════════════════════════════════════════════════════════════════════════════

class DevicePanel(QFrame):
    """Professional device panel showing effects chain on selected track."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.devices = []
        self.setMinimumHeight(140)
        self.setup_ui()
        
    def setup_ui(self):
        self.setStyleSheet("""
            DevicePanel {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #181818, stop:1 #121212);
                border-top: 2px solid #00aaff;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(6)
        
        # Header bar
        header = QHBoxLayout()
        header.setSpacing(12)
        
        title = QLabel("🎛️ DEVICE CHAIN")
        title.setStyleSheet("color: #00aaff; font-size: 11px; font-weight: bold;")
        header.addWidget(title)
        
        self.track_label = QLabel("No track selected")
        self.track_label.setStyleSheet("color: #888; font-size: 10px;")
        header.addWidget(self.track_label)
        
        header.addStretch()
        
        # Quick add buttons
        for dev_name, icon in [("EQ", "📊"), ("Comp", "🔊"), ("Reverb", "🌊"), ("Delay", "⏱️")]:
            btn = QPushButton(f"{icon} {dev_name}")
            btn.setFixedHeight(24)
            btn.setStyleSheet("""
                QPushButton {
                    background: #252525;
                    border: 1px solid #3a3a3a;
                    border-radius: 4px;
                    color: #888;
                    font-size: 10px;
                    padding: 0 8px;
                    min-width: 60px;
                }
                QPushButton:hover {
                    border-color: #00aaff;
                    color: #00aaff;
                }
            """)
            btn.clicked.connect(lambda checked, n=dev_name.lower(): self.add_device_widget(n, dev_name))
            header.addWidget(btn)
        
        layout.addLayout(header)
        
        # Device chain scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("background: transparent; border: none;")
        scroll.setFixedHeight(90)
        
        self.device_container = QWidget()
        self.device_container.setStyleSheet("background: transparent;")
        self.device_layout = QHBoxLayout(self.device_container)
        self.device_layout.setContentsMargins(0, 4, 0, 4)
        self.device_layout.setSpacing(8)
        
        # Add device button (always first)
        self.add_btn = self._create_add_device_btn()
        self.device_layout.addWidget(self.add_btn)
        self.device_layout.addStretch()
        
        scroll.setWidget(self.device_container)
        layout.addWidget(scroll)
        
    def _create_add_device_btn(self) -> QPushButton:
        btn = QPushButton("+ Add\nDevice")
        btn.setFixedSize(80, 70)
        btn.setStyleSheet("""
            QPushButton {
                background: #1a1a1a;
                border: 2px dashed #3a3a3a;
                border-radius: 8px;
                color: #666;
                font-size: 10px;
            }
            QPushButton:hover {
                border-color: #00aaff;
                color: #00aaff;
                background: #1e1e1e;
            }
        """)
        btn.clicked.connect(self._show_add_device_menu)
        return btn
    
    def _show_add_device_menu(self):
        """Show popup menu to add devices."""
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background: #1a1a1a;
                border: 1px solid #3a3a3a;
                border-radius: 6px;
                padding: 4px;
            }
            QMenu::item {
                padding: 8px 20px;
                border-radius: 4px;
                color: #ccc;
            }
            QMenu::item:selected {
                background: #00aaff;
                color: white;
            }
            QMenu::separator {
                height: 1px;
                background: #3a3a3a;
                margin: 4px 8px;
            }
        """)
        
        # EQ submenu
        eq_menu = menu.addMenu("📊 EQ")
        native_available = getattr(self.window(), "rft_native_active", False)
        rft_eq_action = eq_menu.addAction("Φ-RFT EQ", lambda: self.add_device_widget("eq", "EQ"))
        if not native_available:
            rft_eq_action.setEnabled(False)
            rft_eq_action.setText("⚠ Φ-RFT EQ (Requires Native)")
        eq_menu.addAction("Parametric EQ", lambda: self.add_device_widget("eq", "Parametric"))
        eq_menu.addAction("Graphic EQ", lambda: self.add_device_widget("eq", "Graphic"))
        
        # Dynamics submenu
        dyn_menu = menu.addMenu("🔊 Dynamics")
        dyn_menu.addAction("Compressor", lambda: self.add_device_widget("comp", "Compressor"))
        dyn_menu.addAction("Limiter", lambda: self.add_device_widget("comp", "Limiter"))
        dyn_menu.addAction("Gate", lambda: self.add_device_widget("comp", "Gate"))
        
        # Effects submenu
        fx_menu = menu.addMenu("✨ Effects")
        rft_reverb = fx_menu.addAction("Φ-RFT Reverb", lambda: self.add_device_widget("reverb", "Reverb"))
        if not native_available:
            rft_reverb.setEnabled(False)
            rft_reverb.setText("⚠ Φ-RFT Reverb (Requires Native)")
        fx_menu.addAction("Delay", lambda: self.add_device_widget("delay", "Delay"))
        fx_menu.addAction("Chorus", lambda: self.add_device_widget("chorus", "Chorus"))
        fx_menu.addAction("Phaser", lambda: self.add_device_widget("phaser", "Phaser"))
        fx_menu.addAction("Flanger", lambda: self.add_device_widget("flanger", "Flanger"))
        
        # Utility
        util_menu = menu.addMenu("🔧 Utility")
        util_menu.addAction("Gain", lambda: self.add_device_widget("utility", "Gain"))
        util_menu.addAction("Stereo Width", lambda: self.add_device_widget("utility", "Width"))
        util_menu.addAction("Analyzer", lambda: self.add_device_widget("meter", "Analyzer"))
        
        # Show menu at button position
        menu.exec_(self.add_btn.mapToGlobal(self.add_btn.rect().topRight()))
        
    def set_track(self, name: str):
        """Set the current track being edited."""
        self.track_label.setText(f"Track: {name}")
        
    def add_device_widget(self, device_type: str, name: str):
        """Add a device widget to the chain."""
        device = QFrame()
        device.setFixedSize(160, 70)
        device.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2a2a2a, stop:1 #1e1e1e);
                border: 1px solid #3a3a3a;
                border-radius: 8px;
            }
            QFrame:hover {
                border-color: #00aaff;
            }
        """)
        
        device_layout = QVBoxLayout(device)
        device_layout.setContentsMargins(8, 4, 8, 4)
        device_layout.setSpacing(2)
        
        # Device header
        header_row = QHBoxLayout()
        header = QLabel(f"🎛️ Φ-{name}")
        header.setStyleSheet("color: #00aaff; font-weight: bold; font-size: 10px;")
        header_row.addWidget(header)
        
        # Power button
        power_btn = QPushButton("⚡")
        power_btn.setCheckable(True)
        power_btn.setChecked(True)
        power_btn.setFixedSize(18, 18)
        power_btn.setStyleSheet("""
            QPushButton {
                background: #00ffaa;
                border: none;
                border-radius: 9px;
                font-size: 10px;
            }
            QPushButton:checked {
                background: #00ffaa;
            }
            QPushButton:!checked {
                background: #444;
            }
        """)
        header_row.addWidget(power_btn)
        device_layout.addLayout(header_row)
        
        # Controls based on type
        ctrl_layout = QHBoxLayout()
        ctrl_layout.setSpacing(4)
        
        if device_type == "eq":
            for band in ["Lo", "Mid", "Hi"]:
                self._add_mini_knob(ctrl_layout, band)
        elif device_type == "comp":
            for param in ["Thr", "Rat", "Atk"]:
                self._add_mini_knob(ctrl_layout, param)
        elif device_type == "reverb":
            for param in ["Size", "Dcy", "Mix"]:
                self._add_mini_knob(ctrl_layout, param)
        elif device_type == "delay":
            for param in ["Time", "Fbk", "Mix"]:
                self._add_mini_knob(ctrl_layout, param)
        else:
            self._add_mini_knob(ctrl_layout, "Mix")
            
        device_layout.addLayout(ctrl_layout)
        
        self.devices.append(device)
        
        # Insert before the add button
        self.device_layout.insertWidget(len(self.devices) - 1, device)
        
    def _add_mini_knob(self, layout: QHBoxLayout, label: str):
        """Add a mini knob control."""
        container = QVBoxLayout()
        container.setSpacing(1)
        
        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, 100)
        slider.setValue(50)
        slider.setFixedSize(40, 12)
        
        lbl = QLabel(label)
        lbl.setStyleSheet("color: #666; font-size: 8px;")
        lbl.setAlignment(Qt.AlignCenter)
        
        container.addWidget(slider)
        container.addWidget(lbl)
        layout.addLayout(container)


# ═══════════════════════════════════════════════════════════════════════════════
# MEDIA POOL / FILE BROWSER - EPIC 07
# ═══════════════════════════════════════════════════════════════════════════════

class MediaPoolWidget(QFrame):
    """File browser and media pool for audio samples and project assets."""
    
    # Signal emitted when user wants to import a file to arrangement
    file_import_requested = pyqtSignal(str)  # path
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_path = os.path.expanduser("~")
        self.preview_playing = False
        self.audio_preview = None
        self.setup_ui()
        
    def setup_ui(self):
        self.setStyleSheet("""
            MediaPoolWidget {
                background: #0d0d0d;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Navigation bar
        nav_bar = QFrame()
        nav_bar.setFixedHeight(32)
        nav_bar.setStyleSheet("background: #151515; border-bottom: 1px solid #2a2a2a;")
        nav_layout = QHBoxLayout(nav_bar)
        nav_layout.setContentsMargins(4, 0, 4, 0)
        nav_layout.setSpacing(4)
        
        # Back/Forward/Up buttons
        self.back_btn = QPushButton("◀")
        self.back_btn.setFixedSize(24, 24)
        self.back_btn.setToolTip("Back")
        self.back_btn.clicked.connect(self.go_back)
        nav_layout.addWidget(self.back_btn)
        
        self.up_btn = QPushButton("▲")
        self.up_btn.setFixedSize(24, 24)
        self.up_btn.setToolTip("Parent folder")
        self.up_btn.clicked.connect(self.go_up)
        nav_layout.addWidget(self.up_btn)
        
        self.home_btn = QPushButton("🏠")
        self.home_btn.setFixedSize(24, 24)
        self.home_btn.setToolTip("Home folder")
        self.home_btn.clicked.connect(self.go_home)
        nav_layout.addWidget(self.home_btn)
        
        # Path display
        self.path_label = QLabel(self.current_path)
        self.path_label.setStyleSheet("color: #888; font-size: 9px; padding-left: 4px;")
        self.path_label.setMinimumWidth(50)
        nav_layout.addWidget(self.path_label, 1)
        
        layout.addWidget(nav_bar)
        
        # Quick access buttons
        quick_bar = QFrame()
        quick_bar.setFixedHeight(28)
        quick_bar.setStyleSheet("background: #121212;")
        quick_layout = QHBoxLayout(quick_bar)
        quick_layout.setContentsMargins(4, 2, 4, 2)
        quick_layout.setSpacing(2)
        
        for name, icon in [("Desktop", "🖥️"), ("Music", "🎵"), ("Project", "📁")]:
            btn = QPushButton(f"{icon}")
            btn.setToolTip(name)
            btn.setFixedSize(28, 22)
            btn.setStyleSheet("""
                QPushButton {
                    background: #1a1a1a;
                    border: 1px solid #333;
                    border-radius: 3px;
                    font-size: 12px;
                }
                QPushButton:hover {
                    border-color: #00aaff;
                    background: #252525;
                }
            """)
            btn.clicked.connect(lambda checked, n=name: self.go_to_special(n))
            quick_layout.addWidget(btn)
        quick_layout.addStretch()
        
        layout.addWidget(quick_bar)
        
        # File system view
        self.file_model = QFileSystemModel()
        self.file_model.setRootPath("")
        self.file_model.setNameFilters([
            "*.wav", "*.mp3", "*.flac", "*.ogg", "*.aiff", "*.aif",
            "*.mid", "*.midi", "*.qsd"
        ])
        self.file_model.setNameFilterDisables(False)
        
        self.file_tree = QTreeView()
        self.file_tree.setModel(self.file_model)
        self.file_tree.setRootIndex(self.file_model.index(self.current_path))
        self.file_tree.setColumnWidth(0, 150)
        self.file_tree.hideColumn(1)  # Size
        self.file_tree.hideColumn(2)  # Type
        self.file_tree.hideColumn(3)  # Date
        self.file_tree.setHeaderHidden(True)
        self.file_tree.setAnimated(True)
        self.file_tree.setIndentation(12)
        self.file_tree.setDragEnabled(True)
        self.file_tree.setStyleSheet("""
            QTreeView {
                background: #0d0d0d;
                color: #ccc;
                border: none;
                font-size: 11px;
            }
            QTreeView::item {
                padding: 4px 2px;
                border-radius: 3px;
            }
            QTreeView::item:hover {
                background: #1a1a1a;
            }
            QTreeView::item:selected {
                background: #00557f;
                color: white;
            }
            QTreeView::branch:has-children:!has-siblings:closed,
            QTreeView::branch:closed:has-children:has-siblings {
                border-image: none;
                image: url(none);
            }
            QTreeView::branch:open:has-children:!has-siblings,
            QTreeView::branch:open:has-children:has-siblings {
                border-image: none;
                image: url(none);
            }
        """)
        self.file_tree.clicked.connect(self.on_file_clicked)
        self.file_tree.doubleClicked.connect(self.on_file_double_clicked)
        layout.addWidget(self.file_tree, 1)
        
        # Preview bar
        preview_bar = QFrame()
        preview_bar.setFixedHeight(36)
        preview_bar.setStyleSheet("background: #151515; border-top: 1px solid #2a2a2a;")
        preview_layout = QHBoxLayout(preview_bar)
        preview_layout.setContentsMargins(8, 0, 8, 0)
        preview_layout.setSpacing(6)
        
        self.preview_btn = QPushButton("▶")
        self.preview_btn.setFixedSize(28, 24)
        self.preview_btn.setToolTip("Preview audio file")
        self.preview_btn.setStyleSheet("""
            QPushButton {
                background: #00aaff;
                border: none;
                border-radius: 4px;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #00ccff;
            }
            QPushButton:pressed {
                background: #0088cc;
            }
        """)
        self.preview_btn.clicked.connect(self.toggle_preview)
        preview_layout.addWidget(self.preview_btn)
        
        self.file_info_label = QLabel("Select an audio file")
        self.file_info_label.setStyleSheet("color: #888; font-size: 10px;")
        preview_layout.addWidget(self.file_info_label, 1)
        
        self.import_btn = QPushButton("+ Import")
        self.import_btn.setFixedSize(60, 24)
        self.import_btn.setToolTip("Import to arrangement")
        self.import_btn.setStyleSheet("""
            QPushButton {
                background: #2a2a2a;
                border: 1px solid #00ffaa;
                border-radius: 4px;
                color: #00ffaa;
                font-size: 10px;
            }
            QPushButton:hover {
                background: #00ffaa;
                color: black;
            }
        """)
        self.import_btn.clicked.connect(self.import_selected_file)
        preview_layout.addWidget(self.import_btn)
        
        layout.addWidget(preview_bar)
        
        # History for back navigation
        self.history = []
        self.history_index = -1
        
        self._apply_button_style()
        
    def _apply_button_style(self):
        """Apply consistent button styling."""
        btn_style = """
            QPushButton {
                background: #1a1a1a;
                border: 1px solid #333;
                border-radius: 4px;
                color: #aaa;
            }
            QPushButton:hover {
                border-color: #00aaff;
                color: #00aaff;
            }
            QPushButton:pressed {
                background: #252525;
            }
        """
        self.back_btn.setStyleSheet(btn_style)
        self.up_btn.setStyleSheet(btn_style)
        self.home_btn.setStyleSheet(btn_style)
        
    def navigate_to(self, path: str):
        """Navigate to a directory."""
        if os.path.isdir(path):
            self.current_path = path
            self.file_tree.setRootIndex(self.file_model.index(path))
            self.path_label.setText(self._truncate_path(path))
            self.path_label.setToolTip(path)
            # Add to history
            if self.history_index < len(self.history) - 1:
                self.history = self.history[:self.history_index + 1]
            self.history.append(path)
            self.history_index = len(self.history) - 1
            
    def _truncate_path(self, path: str, max_len: int = 25) -> str:
        """Truncate path for display."""
        if len(path) <= max_len:
            return path
        parts = path.split(os.sep)
        if len(parts) <= 2:
            return "..." + path[-max_len:]
        return parts[0] + os.sep + "..." + os.sep + parts[-1]
        
    def go_back(self):
        """Go to previous directory in history."""
        if self.history_index > 0:
            self.history_index -= 1
            path = self.history[self.history_index]
            self.current_path = path
            self.file_tree.setRootIndex(self.file_model.index(path))
            self.path_label.setText(self._truncate_path(path))
            
    def go_up(self):
        """Go to parent directory."""
        parent = os.path.dirname(self.current_path)
        if parent and parent != self.current_path:
            self.navigate_to(parent)
            
    def go_home(self):
        """Go to home directory."""
        self.navigate_to(os.path.expanduser("~"))
        
    def go_to_special(self, name: str):
        """Go to a special folder."""
        home = os.path.expanduser("~")
        if name == "Desktop":
            path = os.path.join(home, "Desktop")
        elif name == "Music":
            path = os.path.join(home, "Music")
        elif name == "Project":
            # Stay in current working directory
            path = os.getcwd()
        else:
            path = home
        if os.path.isdir(path):
            self.navigate_to(path)
            
    def on_file_clicked(self, index):
        """Handle file selection."""
        path = self.file_model.filePath(index)
        if os.path.isfile(path):
            ext = os.path.splitext(path)[1].lower()
            if ext in ['.wav', '.mp3', '.flac', '.ogg', '.aiff', '.aif']:
                size = os.path.getsize(path)
                size_str = f"{size / 1024:.1f} KB" if size < 1024*1024 else f"{size / (1024*1024):.1f} MB"
                self.file_info_label.setText(f"{os.path.basename(path)} ({size_str})")
            else:
                self.file_info_label.setText(os.path.basename(path))
                
    def on_file_double_clicked(self, index):
        """Handle double-click - navigate to folder or import file."""
        path = self.file_model.filePath(index)
        if os.path.isdir(path):
            self.navigate_to(path)
        elif os.path.isfile(path):
            ext = os.path.splitext(path)[1].lower()
            if ext in ['.wav', '.mp3', '.flac', '.ogg', '.aiff', '.aif']:
                self.file_import_requested.emit(path)
                
    def toggle_preview(self):
        """Toggle audio preview playback."""
        if self.preview_playing:
            self.stop_preview()
        else:
            self.start_preview()
            
    def start_preview(self):
        """Start previewing the selected audio file."""
        index = self.file_tree.currentIndex()
        if index.isValid():
            path = self.file_model.filePath(index)
            if os.path.isfile(path):
                ext = os.path.splitext(path)[1].lower()
                if ext in ['.wav', '.mp3', '.flac', '.ogg', '.aiff', '.aif']:
                    self.preview_playing = True
                    self.preview_btn.setText("⏹")
                    # Actual audio preview would go here
                    # For now just update UI
                    
    def stop_preview(self):
        """Stop audio preview."""
        self.preview_playing = False
        self.preview_btn.setText("▶")
        
    def import_selected_file(self):
        """Import the selected file to the arrangement."""
        index = self.file_tree.currentIndex()
        if index.isValid():
            path = self.file_model.filePath(index)
            if os.path.isfile(path):
                self.file_import_requested.emit(path)


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORT DIALOG - EPIC 08
# ═══════════════════════════════════════════════════════════════════════════════

class ExportDialog(QWidget):
    """Export dialog for rendering project to audio files."""
    
    export_started = pyqtSignal(dict)  # Export settings
    
    def __init__(self, session=None, parent=None):
        super().__init__(parent)
        self.session = session
        self.setWindowTitle("Export Audio")
        self.setWindowFlags(Qt.Window | Qt.WindowCloseButtonHint)
        self.setFixedSize(480, 520)
        self.setStyleSheet("""
            QWidget {
                background: #1a1a1a;
                color: #e0e0e0;
            }
            QGroupBox {
                border: 1px solid #3a3a3a;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 8px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                color: #00aaff;
            }
        """)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)
        
        # Header
        header = QLabel("📤 EXPORT AUDIO")
        header.setStyleSheet("font-size: 16px; font-weight: bold; color: #00aaff;")
        layout.addWidget(header)
        
        # Format section
        format_group = QFrame()
        format_group.setStyleSheet("QFrame { background: #222; border-radius: 8px; padding: 12px; }")
        format_layout = QVBoxLayout(format_group)
        
        format_label = QLabel("Format")
        format_label.setStyleSheet("color: #888; font-size: 10px;")
        format_layout.addWidget(format_label)
        
        format_row = QHBoxLayout()
        self.format_combo = QComboBox()
        self.format_combo.addItems(["WAV (Lossless)", "FLAC (Lossless)", "MP3 (320kbps)", "OGG (High Quality)"])
        self.format_combo.setStyleSheet("""
            QComboBox {
                background: #2a2a2a;
                border: 1px solid #3a3a3a;
                border-radius: 4px;
                padding: 6px 12px;
                min-width: 200px;
            }
            QComboBox:hover {
                border-color: #00aaff;
            }
            QComboBox::drop-down {
                border: none;
                width: 24px;
            }
        """)
        format_row.addWidget(self.format_combo)
        format_row.addStretch()
        format_layout.addLayout(format_row)
        
        # Bit depth
        depth_row = QHBoxLayout()
        depth_label = QLabel("Bit Depth:")
        depth_label.setStyleSheet("color: #aaa;")
        depth_row.addWidget(depth_label)
        self.depth_combo = QComboBox()
        self.depth_combo.addItems(["16-bit", "24-bit", "32-bit float"])
        self.depth_combo.setCurrentIndex(1)  # Default 24-bit
        self.depth_combo.setStyleSheet(self.format_combo.styleSheet())
        depth_row.addWidget(self.depth_combo)
        depth_row.addStretch()
        format_layout.addLayout(depth_row)
        
        # Sample rate
        sr_row = QHBoxLayout()
        sr_label = QLabel("Sample Rate:")
        sr_label.setStyleSheet("color: #aaa;")
        sr_row.addWidget(sr_label)
        self.sr_combo = QComboBox()
        self.sr_combo.addItems(["44100 Hz", "48000 Hz", "88200 Hz", "96000 Hz"])
        self.sr_combo.setCurrentIndex(1)  # Default 48kHz
        self.sr_combo.setStyleSheet(self.format_combo.styleSheet())
        sr_row.addWidget(self.sr_combo)
        sr_row.addStretch()
        format_layout.addLayout(sr_row)
        
        layout.addWidget(format_group)
        
        # Range section
        range_group = QFrame()
        range_group.setStyleSheet("QFrame { background: #222; border-radius: 8px; padding: 12px; }")
        range_layout = QVBoxLayout(range_group)
        
        range_label = QLabel("Export Range")
        range_label.setStyleSheet("color: #888; font-size: 10px;")
        range_layout.addWidget(range_label)
        
        self.range_full = QPushButton("Full Project")
        self.range_full.setCheckable(True)
        self.range_full.setChecked(True)
        self.range_loop = QPushButton("Loop Region")
        self.range_loop.setCheckable(True)
        self.range_selection = QPushButton("Selection")
        self.range_selection.setCheckable(True)
        
        btn_style = """
            QPushButton {
                background: #2a2a2a;
                border: 1px solid #3a3a3a;
                border-radius: 4px;
                padding: 8px 16px;
                color: #aaa;
            }
            QPushButton:hover {
                border-color: #00aaff;
            }
            QPushButton:checked {
                background: #00557f;
                border-color: #00aaff;
                color: white;
            }
        """
        self.range_full.setStyleSheet(btn_style)
        self.range_loop.setStyleSheet(btn_style)
        self.range_selection.setStyleSheet(btn_style)
        
        # Make them mutually exclusive
        self.range_full.clicked.connect(lambda: self._set_range("full"))
        self.range_loop.clicked.connect(lambda: self._set_range("loop"))
        self.range_selection.clicked.connect(lambda: self._set_range("selection"))
        
        range_btn_row = QHBoxLayout()
        range_btn_row.addWidget(self.range_full)
        range_btn_row.addWidget(self.range_loop)
        range_btn_row.addWidget(self.range_selection)
        range_layout.addLayout(range_btn_row)
        
        layout.addWidget(range_group)
        
        # Stem export section
        stem_group = QFrame()
        stem_group.setStyleSheet("QFrame { background: #222; border-radius: 8px; padding: 12px; }")
        stem_layout = QVBoxLayout(stem_group)
        
        stem_header = QHBoxLayout()
        stem_label = QLabel("Stem Export")
        stem_label.setStyleSheet("color: #888; font-size: 10px;")
        stem_header.addWidget(stem_label)
        
        self.stem_checkbox = QPushButton("Export Stems")
        self.stem_checkbox.setCheckable(True)
        self.stem_checkbox.setStyleSheet(btn_style)
        stem_header.addStretch()
        stem_header.addWidget(self.stem_checkbox)
        stem_layout.addLayout(stem_header)
        
        stem_note = QLabel("Creates separate file for each track")
        stem_note.setStyleSheet("color: #666; font-size: 9px; font-style: italic;")
        stem_layout.addWidget(stem_note)
        
        layout.addWidget(stem_group)
        
        # Normalize / Dither section
        options_group = QFrame()
        options_group.setStyleSheet("QFrame { background: #222; border-radius: 8px; padding: 12px; }")
        options_layout = QVBoxLayout(options_group)
        
        options_label = QLabel("Processing")
        options_label.setStyleSheet("color: #888; font-size: 10px;")
        options_layout.addWidget(options_label)
        
        opts_row = QHBoxLayout()
        self.normalize_btn = QPushButton("Normalize")
        self.normalize_btn.setCheckable(True)
        self.normalize_btn.setStyleSheet(btn_style)
        opts_row.addWidget(self.normalize_btn)
        
        self.dither_btn = QPushButton("Dither")
        self.dither_btn.setCheckable(True)
        self.dither_btn.setChecked(True)
        self.dither_btn.setStyleSheet(btn_style)
        opts_row.addWidget(self.dither_btn)
        
        opts_row.addStretch()
        options_layout.addLayout(opts_row)
        
        layout.addWidget(options_group)
        
        # Destination
        dest_group = QFrame()
        dest_group.setStyleSheet("QFrame { background: #222; border-radius: 8px; padding: 12px; }")
        dest_layout = QVBoxLayout(dest_group)
        
        dest_label = QLabel("Destination")
        dest_label.setStyleSheet("color: #888; font-size: 10px;")
        dest_layout.addWidget(dest_label)
        
        dest_row = QHBoxLayout()
        self.dest_path = QLabel("No file selected")
        self.dest_path.setStyleSheet("color: #aaa; padding: 6px;")
        dest_row.addWidget(self.dest_path, 1)
        
        browse_btn = QPushButton("Browse...")
        browse_btn.setStyleSheet("""
            QPushButton {
                background: #00aaff;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #00ccff;
            }
        """)
        browse_btn.clicked.connect(self.browse_destination)
        dest_row.addWidget(browse_btn)
        dest_layout.addLayout(dest_row)
        
        layout.addWidget(dest_group)
        
        layout.addStretch()
        
        # Export button
        export_row = QHBoxLayout()
        export_row.addStretch()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet("""
            QPushButton {
                background: #2a2a2a;
                border: 1px solid #3a3a3a;
                border-radius: 4px;
                padding: 10px 24px;
                color: #aaa;
            }
            QPushButton:hover {
                border-color: #ff5555;
                color: #ff5555;
            }
        """)
        cancel_btn.clicked.connect(self.close)
        export_row.addWidget(cancel_btn)
        
        self.export_btn = QPushButton("🚀 Export")
        self.export_btn.setStyleSheet("""
            QPushButton {
                background: #00ffaa;
                border: none;
                border-radius: 4px;
                padding: 10px 32px;
                color: black;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover {
                background: #00ff88;
            }
            QPushButton:disabled {
                background: #444;
                color: #666;
            }
        """)
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.start_export)
        export_row.addWidget(self.export_btn)
        
        layout.addLayout(export_row)
        
        self.export_path = None
        
    def _set_range(self, range_type: str):
        """Set export range - mutually exclusive buttons."""
        self.range_full.setChecked(range_type == "full")
        self.range_loop.setChecked(range_type == "loop")
        self.range_selection.setChecked(range_type == "selection")
        
    def browse_destination(self):
        """Open file dialog to select export destination."""
        format_map = {
            "WAV (Lossless)": "WAV Audio (*.wav)",
            "FLAC (Lossless)": "FLAC Audio (*.flac)",
            "MP3 (320kbps)": "MP3 Audio (*.mp3)",
            "OGG (High Quality)": "OGG Audio (*.ogg)"
        }
        ext_map = {
            "WAV (Lossless)": ".wav",
            "FLAC (Lossless)": ".flac",
            "MP3 (320kbps)": ".mp3",
            "OGG (High Quality)": ".ogg"
        }
        
        current_format = self.format_combo.currentText()
        filter_str = format_map.get(current_format, "All Files (*)")
        default_ext = ext_map.get(current_format, ".wav")
        
        filename, _ = QFileDialog.getSaveFileName(
            self, 'Export To', '', filter_str
        )
        if filename:
            if not filename.endswith(default_ext):
                filename += default_ext
            self.export_path = filename
            # Truncate display
            display_name = os.path.basename(filename)
            self.dest_path.setText(display_name)
            self.dest_path.setToolTip(filename)
            self.export_btn.setEnabled(True)
            
    def start_export(self):
        """Start the export process."""
        if not self.export_path:
            return
            
        # Collect settings
        settings = {
            'path': self.export_path,
            'format': self.format_combo.currentText(),
            'bit_depth': self.depth_combo.currentText(),
            'sample_rate': int(self.sr_combo.currentText().replace(' Hz', '')),
            'range': 'full' if self.range_full.isChecked() else ('loop' if self.range_loop.isChecked() else 'selection'),
            'stems': self.stem_checkbox.isChecked(),
            'normalize': self.normalize_btn.isChecked(),
            'dither': self.dither_btn.isChecked()
        }
        
        self.export_started.emit(settings)
        self.close()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN QUANTSOUNDDESIGN WINDOW
# ═══════════════════════════════════════════════════════════════════════════════

class QuantSoundDesign(QMainWindow):
    """QuantSoundDesign - Professional Sound Design Studio with Φ-RFT Native Engine."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QuantSoundDesign - Φ-RFT Sound Design Studio")
        self.setGeometry(50, 50, 1600, 1000)
        self.setMinimumSize(1200, 800)
        self.setStyleSheet(STYLE)
        
        # Initialize audio backend
        self.audio_backend = None
        self.tone_gen = None
        self.synth = None
        self.pattern_player = None
        self.drum_synth = None
        self.current_clip = None
        self.selected_track_idx = -1
        self.left_panel = None
        self.h_splitter = None
        self.v_splitter = None
        self.bottom_panel = None
        self.current_tool = "draw"  # Default editing tool
        self.editor_tabs_container = None
        self._last_split_sizes = None
        self._last_h_split_sizes = None
        self.toggle_editors_action = None
        self.toggle_browser_action = None
        self.editor_toggle_btn = None
        self._start_cue_beat = 0.0  # Start cue position for playback
        self.current_project_path = None  # Path to current project file
        self.project_modified = False  # Track unsaved changes
        self.require_native_rft = False
        self.require_native_action = None
        self.rft_status_info: Dict[str, Any] = {}
        self.rft_native_active = False
        self.rft_indicator = None
        self.session: Optional[Session] = None
        self.audio_engine = None
        
        if AUDIO_AVAILABLE:
            self.audio_backend = get_audio_backend()
            self.tone_gen = TestToneGenerator()
            if self.audio_backend.start():
                print("[OK] Audio backend initialized")
            else:
                print("[!!] Audio backend failed to start")
        
        # Initialize synth and connect to audio backend
        if SYNTH_AVAILABLE:
            self.synth = PolySynth()
            if self.audio_backend:
                self.audio_backend.set_synth(self.synth)
            print("[OK] Synthesizer initialized and connected")
        
        # Initialize pattern player and drum synthesizer
        if PATTERN_AVAILABLE:
            self.pattern_player = PatternPlayer()
            self.drum_synth = DrumSynthesizer()
            # Connect pattern player to audio backend for playback
            if self.audio_backend:
                self.audio_backend.set_pattern_player(self.pattern_player, self.drum_synth)
            print("[OK] Pattern editor and drum synth initialized")
        
        # Initialize core infrastructure
        if CORE_AVAILABLE:
            self.keymap = get_keymap_registry()
            self.selection = get_selection_model()
            self.audition = get_audition_engine()
            if self.audio_backend:
                self.audition.set_audio_backend(self.audio_backend)
            print("[OK] Core infrastructure initialized (KeymapRegistry, SelectionModel)")
        
        self.setup_ui()
        self.setup_shortcuts()  # Set up keyboard shortcuts
        self.create_blank_session()
        self.setup_timer()
        self.connect_transport()
        
        # Show startup message
        print("\n" + "="*60)
        print("  QUANTSOUNDDESIGN - Φ-RFT Sound Design Studio")
        print("="*60)
        print("  Press A-K keys to play notes")
        print("  Click on clips to edit patterns")
        print("  Use the mixer to adjust levels")
        print("="*60 + "\n")
        
    def setup_ui(self):
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Main horizontal splitter (left panel + main content)
        h_splitter = QSplitter(Qt.Horizontal)
        h_splitter.setHandleWidth(8)
        h_splitter.setChildrenCollapsible(False)
        h_splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #1c1c1c;
            }
            QSplitter::handle:hover {
                background-color: #00aaff;
            }
        """)
        self.h_splitter = h_splitter
        
        # Left sidebar - Browser Panel with tabs
        left_panel = None
        if SYNTH_AVAILABLE:
            left_panel = QFrame()
            left_panel.setMaximumWidth(280)
            left_panel.setMinimumWidth(200)
            left_panel.setStyleSheet("""
                QFrame {
                    background-color: #0d0d0d;
                    border-right: 2px solid #00aaff;
                }
            """)
            left_layout = QVBoxLayout(left_panel)
            left_layout.setContentsMargins(0, 0, 0, 0)
            left_layout.setSpacing(0)
            
            # Browser tabs
            self.browser_tabs = QTabWidget()
            self.browser_tabs.setTabPosition(QTabWidget.North)
            self.browser_tabs.setStyleSheet("""
                QTabWidget::pane {
                    border: none;
                    background: #0d0d0d;
                }
                QTabBar::tab {
                    background: #151515;
                    color: #888;
                    padding: 8px 16px;
                    border: none;
                    border-bottom: 2px solid transparent;
                    font-size: 10px;
                    font-weight: bold;
                }
                QTabBar::tab:selected {
                    background: #1a1a1a;
                    color: #00aaff;
                    border-bottom: 2px solid #00aaff;
                }
                QTabBar::tab:hover:!selected {
                    background: #1a1a1a;
                    color: #aaa;
                }
            """)
            
            # Instruments tab
            self.instrument_browser = InstrumentBrowser()
            self.instrument_browser.preset_selected.connect(self.on_preset_selected)
            self.browser_tabs.addTab(self.instrument_browser, "🎹 Instruments")
            
            # Files/Media Pool tab
            self.media_pool = MediaPoolWidget()
            self.media_pool.file_import_requested.connect(self.on_file_import)
            self.browser_tabs.addTab(self.media_pool, "📁 Files")
            
            left_layout.addWidget(self.browser_tabs)
            
            h_splitter.addWidget(left_panel)
            self.left_panel = left_panel
        
        # Main content area
        content_area = QWidget()
        content_layout = QVBoxLayout(content_area)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        
        # Main vertical splitter
        v_splitter = QSplitter(Qt.Vertical)
        v_splitter.setHandleWidth(10)
        v_splitter.setChildrenCollapsible(False)
        v_splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #1c1c1c;
            }
            QSplitter::handle:hover {
                background-color: #00aaff;
            }
        """)
        self.v_splitter = v_splitter
        
        # Arrangement view (takes most space)
        self.arrangement = ArrangementView()
        self.arrangement.clip_selected.connect(self.on_clip_selected)
        self.arrangement.clip_double_clicked.connect(self.on_clip_double_clicked)
        self.arrangement.track_selected.connect(self.on_track_selected)
        self.arrangement.add_track_requested.connect(self.add_track)  # Connect add track buttons
        self.arrangement.position_changed.connect(self.on_position_changed)  # Timeline click-to-set
        self.arrangement.start_cue_changed.connect(self.on_start_cue_changed)  # Start cue drag
        v_splitter.addWidget(self.arrangement)
        
        # Bottom section with tabs (Pattern / Piano Roll / Devices / Mixer)
        bottom_panel = QFrame()
        bottom_layout = QVBoxLayout(bottom_panel)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(0)
        bottom_header = QFrame()
        bottom_header.setStyleSheet("background-color: #101010; border-top: 1px solid #2a2a2a;")
        bh_layout = QHBoxLayout(bottom_header)
        bh_layout.setContentsMargins(12, 4, 12, 4)
        bh_layout.setSpacing(6)
        header_label = QLabel("Editors")
        header_label.setStyleSheet("color: #888; font-weight: bold;")
        bh_layout.addWidget(header_label)
        bh_layout.addStretch()
        self.editor_toggle_btn = QToolButton()
        self.editor_toggle_btn.setCheckable(True)
        self.editor_toggle_btn.setChecked(True)
        self.editor_toggle_btn.setArrowType(Qt.DownArrow)
        self.editor_toggle_btn.setToolTip("Show/Hide editor panel")
        self.editor_toggle_btn.setStyleSheet("QToolButton { color: #00aaff; }")
        self.editor_toggle_btn.toggled.connect(self._toggle_editors_panel)
        bh_layout.addWidget(self.editor_toggle_btn)
        bottom_layout.addWidget(bottom_header)
        
        # Editor tabs
        self.editor_tabs = QTabWidget()
        self.editor_tabs.setStyleSheet("""
            QTabWidget::pane {
                border: none;
                background: #0d0d0d;
            }
            QTabBar::tab {
                background: #1a1a1a;
                color: #888;
                padding: 10px 20px;
                border: none;
                border-bottom: 3px solid transparent;
                font-weight: 500;
                min-width: 100px;
            }
            QTabBar::tab:selected {
                color: #00aaff;
                border-bottom: 3px solid #00aaff;
                background: #151515;
            }
            QTabBar::tab:hover {
                color: #00ffaa;
                background: #1e1e1e;
            }
        """)
        
        # Pattern Editor (Step Sequencer)
        if PATTERN_AVAILABLE:
            self.pattern_editor = PatternEditorWidget()
            self.pattern_editor.pattern_changed.connect(self.on_pattern_changed)
            self.pattern_editor.step_preview.connect(self.on_step_preview)
            self.editor_tabs.addTab(self.pattern_editor, "🥁 Pattern Editor")
        
        # Piano Roll / Instrument Editor
        if SYNTH_AVAILABLE:
            self.instrument_editor = InstrumentEditor(self.synth)
            self.editor_tabs.addTab(self.instrument_editor, "🎹 Piano Roll")
        
        # Device panel
        self.devices = DevicePanel()
        self.editor_tabs.addTab(self.devices, "🎛️ Device Chain")
        
        # Mixer view as tab
        self.mixer = MixerView()
        self.mixer.track_volume_changed.connect(self.on_track_volume_changed)
        self.mixer.track_pan_changed.connect(self.on_track_pan_changed)
        self.mixer.track_muted.connect(self.on_track_muted)
        self.mixer.track_soloed.connect(self.on_track_soloed)
        self.editor_tabs.addTab(self.mixer, "🎚️ Mixer")
        
        self.editor_tabs_container = QWidget()
        tabs_container_layout = QVBoxLayout(self.editor_tabs_container)
        tabs_container_layout.setContentsMargins(0, 0, 0, 0)
        tabs_container_layout.setSpacing(0)
        tabs_container_layout.addWidget(self.editor_tabs)
        bottom_layout.addWidget(self.editor_tabs_container)
        
        v_splitter.addWidget(bottom_panel)
        self.bottom_panel = bottom_panel
        
        # Set initial splitter sizes (arrangement gets more space)
        v_splitter.setSizes([550, 350])
        
        content_layout.addWidget(v_splitter)
        
        h_splitter.addWidget(content_area)
        h_splitter.setSizes([220, 1200])
        
        main_layout.addWidget(h_splitter, 1)
        
        # Transport bar at bottom
        self.transport = TransportBar()
        main_layout.addWidget(self.transport)
        
        # Status bar
        self.status = QStatusBar()
        self.status.setStyleSheet("""
            QStatusBar {
                background: #0a0a0a;
                color: #888;
                border-top: 1px solid #2a2a2a;
                font-size: 10px;
                padding: 4px;
            }
        """)
        self.setStatusBar(self.status)
        self.status.showMessage("QuantSoundDesign Ready | Press A-K to play!")

        # Φ-RFT status indicator (global truth source)
        self.rft_indicator = QLabel("Φ-RFT: unknown")
        self.rft_indicator.setStyleSheet("""
            QLabel {
                padding: 2px 10px;
                border-radius: 6px;
                border: 1px solid #444;
                color: #ddd;
                background: #222;
                font-weight: bold;
            }
        """)
        self.status.addPermanentWidget(self.rft_indicator)
        
        # Add audio performance meter to status bar (EPIC 01)
        if AUDIO_SETTINGS_AVAILABLE:
            self.audio_meter = AudioMeterWidget()
            self.audio_meter.clicked.connect(self.show_audio_settings)
            self.status.addPermanentWidget(self.audio_meter)
        else:
            self.audio_meter = None

        # Reflect current Φ-RFT availability for indicator + transport badge
        self._refresh_rft_status_ui()
        
        # Menu bar
        self.setup_menus()
        # Ensure splitter defaults stored for future toggles
        if self.v_splitter:
            self._last_split_sizes = self.v_splitter.sizes()
        if self.h_splitter:
            self._last_h_split_sizes = self.h_splitter.sizes()
        if self.editor_tabs_container is not None:
            self.editor_tabs_container.setVisible(True)

    def _toggle_editors_panel(self, visible: bool):
        self._set_editors_panel_visible(visible)

    def _set_editors_panel_visible(self, visible: bool):
        if not self.editor_tabs_container or not self.v_splitter:
            return
        self.editor_tabs_container.setVisible(visible)
        if visible:
            if self._last_split_sizes:
                self.v_splitter.setSizes(self._last_split_sizes)
            else:
                self.v_splitter.setSizes([550, 350])
        else:
            self._last_split_sizes = self.v_splitter.sizes()
            self.v_splitter.setSizes([self.v_splitter.height(), 0])
        if self.editor_toggle_btn and self.editor_toggle_btn.isChecked() != visible:
            self.editor_toggle_btn.blockSignals(True)
            self.editor_toggle_btn.setChecked(visible)
            self.editor_toggle_btn.blockSignals(False)
        if self.toggle_editors_action and self.toggle_editors_action.isChecked() != visible:
            self.toggle_editors_action.blockSignals(True)
            self.toggle_editors_action.setChecked(visible)
            self.toggle_editors_action.blockSignals(False)
        if self.editor_toggle_btn:
            self.editor_toggle_btn.setArrowType(Qt.DownArrow if visible else Qt.RightArrow)

    def _toggle_instrument_browser(self, visible: bool):
        if not self.left_panel or not self.h_splitter:
            return
        self.left_panel.setVisible(visible)
        if visible:
            if self._last_h_split_sizes:
                self.h_splitter.setSizes(self._last_h_split_sizes)
            else:
                self.h_splitter.setSizes([220, max(800, self.width() - 220)])
        else:
            self._last_h_split_sizes = self.h_splitter.sizes()
            self.h_splitter.setSizes([0, self.width()])
        if self.toggle_browser_action and self.toggle_browser_action.isChecked() != visible:
            self.toggle_browser_action.blockSignals(True)
            self.toggle_browser_action.setChecked(visible)
            self.toggle_browser_action.blockSignals(False)

    def _refresh_rft_status_ui(self, force: bool = False):
        """Update all surfaces (status bar, transport, menus) with Φ-RFT status."""
        try:
            status = get_kernel_status(use_cache=not force)
        except Exception as exc:  # noqa: BLE001 - show unexpected issues to the user
            status = {"unitary": False, "error": str(exc)}

        self.rft_status_info = status
        self.rft_native_active = bool(status.get("unitary") and not status.get("is_mock", False))

        indicator_text = (
            "Φ-RFT: NATIVE (unitary)"
            if self.rft_native_active
            else "Φ-RFT: FALLBACK (FFT/φ only)"
        )
        tooltip = "Native Φ-RFT kernel loaded" if self.rft_native_active else (
            status.get("error") or "Golden-ratio DSP running in FFT fallback"
        )

        if self.rft_indicator:
            bg = "#0f2f1a" if self.rft_native_active else "#3a2500"
            border = "#1f6c32" if self.rft_native_active else "#d39e00"
            fg = "#7dffb2" if self.rft_native_active else "#ffd37a"
            self.rft_indicator.setText(indicator_text)
            self.rft_indicator.setToolTip(tooltip)
            self.rft_indicator.setStyleSheet(
                f"QLabel {{ padding: 2px 10px; border-radius: 6px; border: 1px solid {border};"
                f" background: {bg}; color: {fg}; font-weight: bold; }}"
            )

        if getattr(self, "transport", None):
            self.transport.update_engine_status(self.rft_native_active)

        self._enforce_native_requirement_if_needed()

    def _enforce_native_requirement_if_needed(self):
        if not getattr(self, "require_native_rft", False):
            return
        if self.rft_native_active:
            return
        QMessageBox.critical(
            self,
            "Φ-RFT Kernel Required",
            "Native Φ-RFT kernel not loaded. Can't start engine while 'Require Native Φ-RFT' is enabled.",
        )
        QTimer.singleShot(0, self.close)

    def _reset_session_model(self, session: Optional[Session] = None):
        """Reset or attach a session model + engine, wiring audio backend"""
        if not ENGINE_AVAILABLE:
            print("Engine module unavailable; session model cannot be initialized")
            return

        self.session = session or Session()
        try:
            self.audio_engine = self.session.ensure_engine()
        except Exception as exc:  # noqa: BLE001 - log for debugging
            self.audio_engine = None
            print(f"Failed to initialize audio engine: {exc}")

        if self.audio_backend and self.session:
            self.audio_backend.attach_session(self.session)

    def setup_shortcuts(self):
        """Set up global keyboard shortcuts using the centralized KeymapRegistry."""
        if not CORE_AVAILABLE:
            return
            
        from PyQt5.QtWidgets import QShortcut
        from PyQt5.QtGui import QKeySequence
        
        # Transport shortcuts
        space_shortcut = QShortcut(QKeySequence(Qt.Key_Space), self)
        space_shortcut.activated.connect(self.toggle_playback)
        
        enter_shortcut = QShortcut(QKeySequence(Qt.Key_Return), self)
        enter_shortcut.activated.connect(self.stop_playback)
        
        # Navigation shortcuts
        home_shortcut = QShortcut(QKeySequence(Qt.Key_Home), self)
        home_shortcut.activated.connect(self.goto_start)
        
        end_shortcut = QShortcut(QKeySequence(Qt.Key_End), self)
        end_shortcut.activated.connect(self.goto_end)
        
        # Loop toggle
        l_shortcut = QShortcut(QKeySequence(Qt.Key_L), self)
        l_shortcut.activated.connect(self.toggle_loop)
        
        # Record toggle
        r_shortcut = QShortcut(QKeySequence(Qt.Key_R), self)
        r_shortcut.activated.connect(self.toggle_record)
        
        # View shortcuts (F5-F9)
        f5_shortcut = QShortcut(QKeySequence(Qt.Key_F5), self)
        f5_shortcut.activated.connect(lambda: self.switch_view("arrangement"))
        
        f6_shortcut = QShortcut(QKeySequence(Qt.Key_F6), self)
        f6_shortcut.activated.connect(lambda: self.switch_view("pattern"))
        
        f7_shortcut = QShortcut(QKeySequence(Qt.Key_F7), self)
        f7_shortcut.activated.connect(lambda: self.switch_view("piano"))
        
        f8_shortcut = QShortcut(QKeySequence(Qt.Key_F8), self)
        f8_shortcut.activated.connect(lambda: self.switch_view("devices"))
        
        f9_shortcut = QShortcut(QKeySequence(Qt.Key_F9), self)
        f9_shortcut.activated.connect(lambda: self.switch_view("mixer"))
        
        # Additional view shortcuts (F10-F12)
        f10_shortcut = QShortcut(QKeySequence(Qt.Key_F10), self)
        f10_shortcut.activated.connect(self.show_audio_settings)  # EPIC 01: Audio Settings
        
        f11_shortcut = QShortcut(QKeySequence(Qt.Key_F11), self)
        f11_shortcut.activated.connect(self.toggle_fullscreen)
        
        f12_shortcut = QShortcut(QKeySequence(Qt.Key_F12), self)
        f12_shortcut.activated.connect(self.show_script_console)
        
        # F1 Help
        f1_shortcut = QShortcut(QKeySequence(Qt.Key_F1), self)
        f1_shortcut.activated.connect(self.show_shortcuts)
        
        # Tool shortcuts
        b_shortcut = QShortcut(QKeySequence(Qt.Key_B), self)
        b_shortcut.activated.connect(lambda: self.set_tool("draw"))
        
        e_shortcut = QShortcut(QKeySequence(Qt.Key_E), self)
        e_shortcut.activated.connect(lambda: self.set_tool("erase"))
        
        m_shortcut = QShortcut(QKeySequence(Qt.Key_M), self)
        m_shortcut.activated.connect(lambda: self.set_tool("mute"))
        
        # Use 'S' for slice instead of 'C' (C is often used for copy)
        s_shortcut = QShortcut(QKeySequence(Qt.Key_S), self)
        s_shortcut.activated.connect(lambda: self.set_tool("slice"))
        
        p_shortcut = QShortcut(QKeySequence(Qt.Key_P), self)
        p_shortcut.activated.connect(lambda: self.set_tool("paint"))
        
        # Zoom shortcuts
        plus_shortcut = QShortcut(QKeySequence(Qt.Key_Plus), self)
        plus_shortcut.activated.connect(self.zoom_in)
        
        minus_shortcut = QShortcut(QKeySequence(Qt.Key_Minus), self)
        minus_shortcut.activated.connect(self.zoom_out)
        
        print("[OK] Keyboard shortcuts initialized")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SHORTCUT ACTION HANDLERS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def toggle_loop(self):
        """Toggle loop mode."""
        if hasattr(self, 'transport'):
            self.transport.toggle_loop()
            loop_state = "ON" if self.transport.loop_btn.isChecked() else "OFF"
            self.status.showMessage(f"Loop: {loop_state}")
    
    def toggle_record(self):
        """Toggle record arm."""
        if hasattr(self, 'transport'):
            self.transport.toggle_record()
            rec_state = "ARMED" if self.transport.rec_btn.isChecked() else "OFF"
            self.status.showMessage(f"Record: {rec_state}")
    
    def switch_view(self, view_name: str):
        """Switch to a specific view/tab."""
        if hasattr(self, 'editor_tabs'):
            tab_map = {
                "arrangement": 0,
                "pattern": 0,  # Pattern is usually first tab
                "piano": 1,
                "devices": 2,
                "mixer": 3
            }
            idx = tab_map.get(view_name, 0)
            if idx < self.editor_tabs.count():
                self.editor_tabs.setCurrentIndex(idx)
                self.status.showMessage(f"View: {view_name.capitalize()}")
    
    def zoom_in(self):
        """Zoom in on arrangement."""
        if hasattr(self, 'arrangement'):
            # Increase pixels per beat
            current = getattr(self.arrangement, 'pixels_per_beat', 50)
            self.arrangement.pixels_per_beat = min(current * 1.25, 200)
            self.arrangement.update()
            self.status.showMessage(f"Zoom: {int(self.arrangement.pixels_per_beat)}%")
    
    def zoom_out(self):
        """Zoom out on arrangement."""
        if hasattr(self, 'arrangement'):
            current = getattr(self.arrangement, 'pixels_per_beat', 50)
            self.arrangement.pixels_per_beat = max(current / 1.25, 10)
            self.arrangement.update()
            self.status.showMessage(f"Zoom: {int(self.arrangement.pixels_per_beat)}%")

    def toggle_playback(self):
        """Toggle play/pause."""
        if self.is_playing:
            self.stop()
        else:
            self.play()
    
    def stop_playback(self):
        """Stop and reset to start."""
        self.stop()
        if hasattr(self, 'arrangement') and self.arrangement:
            self.arrangement.set_playhead(0.0)
    
    def goto_start(self):
        """Go to start of arrangement."""
        if hasattr(self, 'arrangement') and self.arrangement:
            self.arrangement.set_playhead(0.0)
            self.status.showMessage("Returned to start")
    
    def goto_end(self):
        """Go to end of arrangement."""
        self.status.showMessage("Go to end (coming soon)")
    
    def show_step_sequencer(self):
        """Show step sequencer (F10)."""
        self.show_pattern_editor()
        self.status.showMessage("Step Sequencer")
    
    def toggle_fullscreen(self):
        """Toggle fullscreen mode (F11)."""
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()
    
    def show_script_console(self):
        """Show script console (F12)."""
        self.status.showMessage("Script Console (coming soon)")
    
    def show_audio_settings(self):
        """Show audio settings dialog (F10)."""
        if AUDIO_SETTINGS_AVAILABLE:
            dialog = AudioSettingsDialog(self)
            dialog.settings_changed.connect(self._on_audio_settings_changed)
            dialog.exec_()
        else:
            self.status.showMessage("Audio settings not available")
    
    def _on_audio_settings_changed(self, settings):
        """Handle audio settings changes."""
        self.status.showMessage(
            f"Audio: {settings.sample_rate}Hz, {settings.buffer_size} samples "
            f"({settings.get_latency_ms():.1f}ms latency)")
    
    def set_tool(self, tool_name: str):
        """Set the current editing tool."""
        self.current_tool = tool_name
        tool_display = {
            "draw": "✏️ Draw Tool (B)",
            "erase": "🗑️ Erase Tool (E)", 
            "mute": "🔇 Mute Tool (M)",
            "slice": "✂️ Slice Tool (C)",
            "paint": "🖌️ Paint Tool (P)",
            "zoom": "🔍 Zoom Tool (Z)",
        }
        self.status.showMessage(tool_display.get(tool_name, f"Tool: {tool_name}"))

    def setup_menus(self):
        menubar = self.menuBar()
        menubar.setStyleSheet("background-color: #1a1a1a; color: #e0e0e0;")
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        new_action = QAction("New Project", self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self.on_new_project)
        file_menu.addAction(new_action)

        demo_action = QAction("Load Demo Session", self)
        demo_action.triggered.connect(self.setup_demo_session)
        file_menu.addAction(demo_action)
        
        open_action = QAction("Open...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.on_open_project)
        file_menu.addAction(open_action)
        
        save_action = QAction("Save", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.on_save_project)
        file_menu.addAction(save_action)
        
        save_as_action = QAction("Save As...", self)
        save_as_action.setShortcut("Ctrl+Shift+S")
        save_as_action.triggered.connect(self.on_save_project_as)
        file_menu.addAction(save_as_action)
        
        file_menu.addSeparator()
        
        export_action = QAction("Export Audio...", self)
        export_action.setShortcut("Ctrl+Shift+E")
        export_action.triggered.connect(self.on_export_audio)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Alt+F4")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Edit menu
        edit_menu = menubar.addMenu("Edit")
        
        undo_action = QAction("Undo", self)
        undo_action.setShortcut("Ctrl+Z")
        undo_action.triggered.connect(self.on_undo)
        edit_menu.addAction(undo_action)
        
        redo_action = QAction("Redo", self)
        redo_action.setShortcut("Ctrl+Y")
        redo_action.triggered.connect(self.on_redo)
        edit_menu.addAction(redo_action)
        
        edit_menu.addSeparator()
        
        cut_action = QAction("Cut", self)
        cut_action.setShortcut("Ctrl+X")
        cut_action.triggered.connect(self.on_cut)
        edit_menu.addAction(cut_action)
        
        copy_action = QAction("Copy", self)
        copy_action.setShortcut("Ctrl+C")
        copy_action.triggered.connect(self.on_copy)
        edit_menu.addAction(copy_action)
        
        paste_action = QAction("Paste", self)
        paste_action.setShortcut("Ctrl+V")
        paste_action.triggered.connect(self.on_paste)
        edit_menu.addAction(paste_action)
        
        delete_action = QAction("Delete", self)
        delete_action.setShortcut("Delete")
        delete_action.triggered.connect(self.on_delete)
        edit_menu.addAction(delete_action)
        
        edit_menu.addSeparator()
        
        select_all_action = QAction("Select All", self)
        select_all_action.setShortcut("Ctrl+A")
        select_all_action.triggered.connect(self.on_select_all)
        edit_menu.addAction(select_all_action)
        
        # Track menu
        track_menu = menubar.addMenu("Track")
        
        add_audio = QAction("Add Audio Track", self)
        add_audio.setShortcut("Ctrl+Shift+A")
        add_audio.triggered.connect(lambda: self.add_track("audio"))
        track_menu.addAction(add_audio)
        
        add_inst = QAction("Add Instrument Track", self)
        add_inst.setShortcut("Ctrl+Shift+I")
        add_inst.triggered.connect(lambda: self.add_track("synth"))
        track_menu.addAction(add_inst)
        
        add_drums = QAction("Add Drum Track", self)
        add_drums.setShortcut("Ctrl+Shift+D")
        add_drums.triggered.connect(lambda: self.add_track("drums"))
        track_menu.addAction(add_drums)
        
        track_menu.addSeparator()
        
        delete_track = QAction("Delete Selected Track", self)
        delete_track.triggered.connect(self.on_delete_track)
        track_menu.addAction(delete_track)
        
        duplicate_track = QAction("Duplicate Track", self)
        duplicate_track.setShortcut("Ctrl+D")
        duplicate_track.triggered.connect(self.on_duplicate_track)
        track_menu.addAction(duplicate_track)
        
        # View menu
        view_menu = menubar.addMenu("View")
        
        arr_action = QAction("Arrangement", self)
        arr_action.setShortcut("F5")
        arr_action.triggered.connect(lambda: self.editor_tabs.setCurrentIndex(0) if hasattr(self, 'editor_tabs') else None)
        view_menu.addAction(arr_action)
        
        pattern_action = QAction("Pattern Editor", self)
        pattern_action.setShortcut("F6")
        pattern_action.triggered.connect(self.show_pattern_editor)
        view_menu.addAction(pattern_action)
        
        piano_action = QAction("Piano Roll", self)
        piano_action.setShortcut("F7")
        piano_action.triggered.connect(self.show_piano_roll)
        view_menu.addAction(piano_action)
        
        mix_action = QAction("Mixer", self)
        mix_action.setShortcut("F9")
        mix_action.triggered.connect(self.show_mixer)
        view_menu.addAction(mix_action)
        
        devices_action = QAction("Device Chain", self)
        devices_action.setShortcut("F8")
        devices_action.triggered.connect(self.show_devices)
        view_menu.addAction(devices_action)

        view_menu.addSeparator()
        self.toggle_editors_action = QAction("Show Editors Panel", self)
        self.toggle_editors_action.setCheckable(True)
        self.toggle_editors_action.setChecked(True)
        self.toggle_editors_action.toggled.connect(self._toggle_editors_panel)
        view_menu.addAction(self.toggle_editors_action)

        self.toggle_browser_action = QAction("Show Instrument Browser", self)
        self.toggle_browser_action.setCheckable(True)
        browser_available = SYNTH_AVAILABLE and self.left_panel is not None
        self.toggle_browser_action.setChecked(browser_available)
        self.toggle_browser_action.setEnabled(browser_available)
        if browser_available:
            self.toggle_browser_action.toggled.connect(self._toggle_instrument_browser)
        view_menu.addAction(self.toggle_browser_action)
        
        view_menu.addSeparator()
        
        # Audio Settings (EPIC 01)
        audio_settings_action = QAction("Audio Settings...", self)
        audio_settings_action.setShortcut("F10")
        audio_settings_action.triggered.connect(self.show_audio_settings)
        view_menu.addAction(audio_settings_action)
        
        # Settings / Preferences menu
        settings_menu = menubar.addMenu("Settings")
        self.require_native_action = QAction("Require Native Φ-RFT", self)
        self.require_native_action.setCheckable(True)
        self.require_native_action.setChecked(self.require_native_rft)
        self.require_native_action.toggled.connect(self._on_require_native_changed)
        settings_menu.addAction(self.require_native_action)

        # Help menu
        help_menu = menubar.addMenu("Help")
        
        about_action = QAction("About QuantSoundDesign", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
        shortcuts_action = QAction("Keyboard Shortcuts", self)
        shortcuts_action.setShortcut("F1")
        shortcuts_action.triggered.connect(self.show_shortcuts)
        help_menu.addAction(shortcuts_action)
    
    def _on_require_native_changed(self, checked: bool):
        self.require_native_rft = checked
        if checked:
            self.status.showMessage("Native Φ-RFT enforcement enabled", 4000)
            self._enforce_native_requirement_if_needed()
        else:
            self.status.showMessage("Native Φ-RFT enforcement disabled", 3000)

    # ═══════════════════════════════════════════════════════════════════════════
    # MENU ACTION HANDLERS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def on_new_project(self):
        """Create a new empty project."""
        if self.project_modified:
            reply = QMessageBox.question(self, 'New Project',
                'Create a new project? Unsaved changes will be lost.',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply != QMessageBox.Yes:
                return
        self.create_blank_session()
        self.current_project_path = None
        self.project_modified = False
        self.update_window_title()
        self.status.showMessage("New blank project created")
    
    def on_open_project(self):
        """Open a project file."""
        if self.project_modified:
            reply = QMessageBox.question(self, 'Open Project',
                'Open a project? Unsaved changes will be lost.',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply != QMessageBox.Yes:
                return
        
        filename, _ = QFileDialog.getOpenFileName(
            self, 'Open Project', '', 
            'QuantSoundDesign Projects (*.qsd);;All Files (*)'
        )
        if filename:
            try:
                from .engine import Session
                self.session = Session.load_from_file(filename)
                self._reset_session_model(self.session)
                self.current_project_path = filename
                self.project_modified = False
                self.update_window_title()
                self.refresh_ui_from_session()
                self.status.showMessage(f"Opened: {filename}")
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'Failed to open project:\n{str(e)}')
    
    def on_save_project(self):
        """Save the current project."""
        if self.current_project_path:
            try:
                self.session.save_to_file(self.current_project_path)
                self.project_modified = False
                self.update_window_title()
                self.status.showMessage(f"Saved: {self.current_project_path}")
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'Failed to save project:\n{str(e)}')
        else:
            self.on_save_project_as()
    
    def on_save_project_as(self):
        """Save project with a new name."""
        filename, _ = QFileDialog.getSaveFileName(
            self, 'Save Project As', '',
            'QuantSoundDesign Projects (*.qsd);;All Files (*)'
        )
        if filename:
            if not filename.endswith('.qsd'):
                filename += '.qsd'
            try:
                self.session.save_to_file(filename)
                self.current_project_path = filename
                self.project_modified = False
                self.update_window_title()
                self.status.showMessage(f"Saved as: {filename}")
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'Failed to save project:\n{str(e)}')
    
    def update_window_title(self):
        """Update window title with project name and modified state."""
        base_title = "QuantSoundDesign - Φ-RFT Sound Design Studio"
        if self.current_project_path:
            import os
            project_name = os.path.basename(self.current_project_path)
            modified = " *" if self.project_modified else ""
            self.setWindowTitle(f"{project_name}{modified} - {base_title}")
        else:
            modified = " *" if self.project_modified else ""
            self.setWindowTitle(f"Untitled{modified} - {base_title}")
    
    def mark_project_modified(self):
        """Mark the project as having unsaved changes."""
        if not self.project_modified:
            self.project_modified = True
            self.update_window_title()
    
    def refresh_ui_from_session(self):
        """Refresh all UI components from current session data."""
        # Update transport display
        if hasattr(self, 'transport'):
            self.transport.set_bpm(self.session.tempo_bpm)
        # Update arrangement view
        if hasattr(self, 'arrangement'):
            self.arrangement.update()
        # Update mixer
        if hasattr(self, 'mixer_view'):
            self.mixer_view.update()
        self.status.showMessage("Session loaded")
    
    def on_export_audio(self):
        """Export audio to file."""
        self.export_dialog = ExportDialog(session=self.session, parent=self)
        self.export_dialog.export_started.connect(self.perform_export)
        self.export_dialog.show()

    def _estimate_project_length_beats(self) -> float:
        """Estimate total project length based on arrangement clips"""
        max_end = 0.0
        if hasattr(self, 'arrangement'):
            for header, lane, widget, track_type in getattr(self.arrangement, 'tracks', []):
                for clip in getattr(lane, 'clips', []):
                    if hasattr(clip, 'end'):
                        max_end = max(max_end, clip.end())
                    else:
                        max_end = max(max_end, clip.start + getattr(clip, 'length', 4.0))
        return max_end if max_end > 0 else 16.0

    def _determine_export_range(self, settings: dict) -> Tuple[float, float]:
        """Compute start/end beats for export respecting loop/selection"""
        start_beat = 0.0
        end_beat = self._estimate_project_length_beats()
        export_range = settings.get('range', 'full')

        if export_range == 'loop' and self.session and self.session.transport.loop_enabled:
            start_beat = self.session.transport.loop_start_beats
            end_beat = self.session.transport.loop_end_beats
        elif export_range == 'selection' and self.current_clip:
            start_beat = getattr(self.current_clip, 'start', start_beat)
            if hasattr(self.current_clip, 'end'):
                end_beat = max(start_beat + 1.0, self.current_clip.end())
            else:
                end_beat = max(start_beat + 1.0, start_beat + getattr(self.current_clip, 'length', 4.0))
        end_beat = max(end_beat, start_beat + 1.0)
        return start_beat, end_beat
        
    def perform_export(self, settings: dict):
        """Perform the actual audio export."""
        path = settings['path']
        export_format = settings['format']
        progress = QProgressBar(self)
        progress.setRange(0, 100)
        progress.setValue(0)
        self.status.addPermanentWidget(progress)

        try:
            if not self.session:
                raise RuntimeError("Session not initialized; cannot export")

            if not export_format.startswith("WAV"):
                raise ValueError("Only WAV exports are supported in this build")

            engine = self.session.ensure_engine()
            sample_rate = settings.get('sample_rate', int(self.session.sample_rate))
            bit_depth = settings.get('bit_depth', '24-bit')
            start_beat, end_beat = self._determine_export_range(settings)

            if settings.get('stems'):
                # Stem export: render each track separately
                stem_dir = os.path.splitext(path)[0] + "_stems"
                os.makedirs(stem_dir, exist_ok=True)
                stem_count = 0
                
                if hasattr(self, 'arrangement'):
                    tracks = getattr(self.arrangement, 'tracks', [])
                    for i, (header, lane, widget, track_type) in enumerate(tracks):
                        track_name = getattr(header, 'track_name', f'Track_{i+1}')
                        # Render this track solo
                        try:
                            stem_buffer = engine.render_track_offline(
                                i, start_beat, end_beat,
                                sample_rate=sample_rate,
                                progress_callback=lambda pct: progress.setValue(5 + int(pct * 80 / max(1, len(tracks))))
                            )
                            if stem_buffer.size > 0:
                                stem_audio = stem_buffer.T.astype(np.float32, copy=False)
                                stem_audio = np.clip(stem_audio, -1.0, 1.0)
                                stem_path = os.path.join(stem_dir, f"{track_name}.wav")
                                self._write_wav_file(stem_path, stem_audio, sample_rate, bit_depth)
                                stem_count += 1
                        except Exception as stem_err:
                            self.status.showMessage(f"Stem '{track_name}' failed: {stem_err}", 4000)
                
                self.status.showMessage(f"Exported {stem_count} stems to {stem_dir}", 6000)

            progress.setValue(5)
            buffer = engine.render_offline(
                start_beat,
                end_beat,
                sample_rate=sample_rate,
                progress_callback=lambda pct: progress.setValue(5 + int(pct * 80))
            )

            if buffer.size == 0:
                raise RuntimeError("Nothing to render in the selected range")

            audio = buffer.T.astype(np.float32, copy=False)
            audio = np.clip(audio, -1.0, 1.0)

            if settings.get('normalize'):
                peak = np.max(np.abs(audio))
                if peak > 0:
                    audio *= 0.999 / peak

            if settings.get('dither') and bit_depth in ('16-bit', '24-bit'):
                rng = np.random.default_rng()
                lsb = 1.0 / (2 ** (15 if bit_depth == '16-bit' else 23))
                noise = (rng.random(audio.shape, dtype=np.float32) - rng.random(audio.shape, dtype=np.float32)) * lsb
                audio = np.clip(audio + noise, -1.0, 1.0)

            self._write_wav_file(path, audio, sample_rate, bit_depth)
            progress.setValue(100)
            self.status.showMessage(f"✓ Exported: {os.path.basename(path)}")

        except Exception as e:
            QMessageBox.critical(self, 'Export Error', f'Failed to export:\n{str(e)}')
        finally:
            self.status.removeWidget(progress)
            progress.deleteLater()

    def _write_wav_file(self, path: str, audio: np.ndarray, sample_rate: int, bit_depth: str) -> None:
        """Write numpy audio buffer to WAV on disk"""
        import wave

        if audio.ndim == 1:
            audio = audio[:, None]

        frames, channels = audio.shape
        flat = np.ascontiguousarray(audio.reshape(-1))

        if bit_depth == '16-bit':
            pcm = np.clip(flat * 32767.0, -32768, 32767).astype('<i2')
            sampwidth = 2
            payload = pcm.tobytes()
        elif bit_depth == '24-bit':
            pcm = np.clip(flat * 8388607.0, -8388608, 8388607).astype('<i4')
            byte_view = pcm.view(np.uint8).reshape(-1, 4)
            payload = byte_view[:, :3].reshape(-1).tobytes()
            sampwidth = 3
        else:
            raise ValueError(f"Unsupported bit depth: {bit_depth}")

        with wave.open(path, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sampwidth)
            wf.setframerate(sample_rate)
            wf.writeframes(payload)
    
    def on_undo(self):
        """Undo last action."""
        self.status.showMessage("Undo (coming soon)")
    
    def on_redo(self):
        """Redo last undone action."""
        self.status.showMessage("Redo (coming soon)")
    
    def on_cut(self):
        """Cut selected clip."""
        if self.current_clip:
            self.clipboard_clip = self.current_clip
            self.status.showMessage(f"Cut: {self.current_clip.name}")
    
    def on_copy(self):
        """Copy selected clip."""
        if self.current_clip:
            self.clipboard_clip = self.current_clip
            self.status.showMessage(f"Copied: {self.current_clip.name}")
    
    def on_paste(self):
        """Paste clip from clipboard."""
        if hasattr(self, 'clipboard_clip') and self.clipboard_clip:
            self.status.showMessage(f"Paste: {self.clipboard_clip.name} (paste coming soon)")
    
    def on_delete(self):
        """Delete selected clip."""
        if self.current_clip:
            clip_name = self.current_clip.name
            # Find and remove from track lane
            for header, lane, widget, track_type in self.arrangement.tracks:
                if self.current_clip in lane.clips:
                    lane.clips.remove(self.current_clip)
                    lane.update()
                    break
            self.current_clip = None
            self.status.showMessage(f"Deleted: {clip_name}")
    
    def on_select_all(self):
        """Select all clips in arrangement."""
        self.status.showMessage("Select all (coming soon)")
    
    def on_delete_track(self):
        """Delete the selected track."""
        if 0 <= self.selected_track_idx < len(self.arrangement.tracks):
            header, lane, widget, track_type = self.arrangement.tracks[self.selected_track_idx]
            track_name = header.name_label.text()
            
            # Remove from arrangement
            header.deleteLater()
            lane.deleteLater()
            self.arrangement.tracks.pop(self.selected_track_idx)
            
            # Remove from mixer
            if self.selected_track_idx < len(self.mixer.strips):
                self.mixer.strips[self.selected_track_idx].deleteLater()
                self.mixer.strips.pop(self.selected_track_idx)
            
            self.selected_track_idx = -1
            self.status.showMessage(f"Deleted track: {track_name}")
    
    def on_duplicate_track(self):
        """Duplicate the selected track."""
        if 0 <= self.selected_track_idx < len(self.arrangement.tracks):
            header, lane, widget, track_type = self.arrangement.tracks[self.selected_track_idx]
            track_name = header.name_label.text()
            new_name = f"{track_name} (Copy)"
            self.add_track(track_type)
            self.status.showMessage(f"Duplicated: {track_name}")
    
    def show_pattern_editor(self):
        """Show pattern editor tab."""
        if hasattr(self, 'editor_tabs'):
            for i in range(self.editor_tabs.count()):
                if "Pattern" in self.editor_tabs.tabText(i):
                    self.editor_tabs.setCurrentIndex(i)
                    break
    
    def show_piano_roll(self):
        """Show piano roll tab."""
        if hasattr(self, 'editor_tabs'):
            for i in range(self.editor_tabs.count()):
                if "Piano" in self.editor_tabs.tabText(i):
                    self.editor_tabs.setCurrentIndex(i)
                    break
    
    def show_mixer(self):
        """Show mixer tab."""
        if hasattr(self, 'editor_tabs'):
            for i in range(self.editor_tabs.count()):
                if "Mixer" in self.editor_tabs.tabText(i):
                    self.editor_tabs.setCurrentIndex(i)
                    break
    
    def show_devices(self):
        """Show device chain tab."""
        if hasattr(self, 'editor_tabs'):
            for i in range(self.editor_tabs.count()):
                if "Device" in self.editor_tabs.tabText(i):
                    self.editor_tabs.setCurrentIndex(i)
                    break
    
    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(self, "About QuantSoundDesign",
            "<h2>QuantSoundDesign</h2>"
            "<p><b>Φ-RFT Sound Design Studio</b></p>"
            "<p>Version 1.0.0</p>"
            "<p>A professional sound design studio built with UnitaryRFT transforms.</p>"
            "<p>© 2025 QuantoniumOS</p>"
        )
    
    def show_shortcuts(self):
        """Show keyboard shortcuts."""
        shortcuts = """
        <style>
            h3 { color: #00aaff; }
            table { border-collapse: collapse; width: 100%; }
            th { background: #333; color: #00aaff; padding: 8px; text-align: left; }
            td { padding: 6px 12px; border-bottom: 1px solid #333; }
            .category { color: #00ffaa; font-weight: bold; padding-top: 12px; }
        </style>
        <h3>⌨️ Keyboard Shortcuts</h3>
        
        <p class="category">🎬 Transport</p>
        <table>
        <tr><td><b>Space</b></td><td>Play / Pause</td></tr>
        <tr><td><b>Enter</b></td><td>Stop &amp; return to start</td></tr>
        <tr><td><b>Home</b></td><td>Go to start</td></tr>
        <tr><td><b>End</b></td><td>Go to end</td></tr>
        <tr><td><b>L</b></td><td>Toggle loop</td></tr>
        <tr><td><b>R</b></td><td>Toggle record</td></tr>
        </table>
        
        <p class="category">📁 Project</p>
        <table>
        <tr><td><b>Ctrl+N</b></td><td>New project</td></tr>
        <tr><td><b>Ctrl+O</b></td><td>Open project</td></tr>
        <tr><td><b>Ctrl+S</b></td><td>Save project</td></tr>
        <tr><td><b>Ctrl+Shift+S</b></td><td>Save As...</td></tr>
        <tr><td><b>Ctrl+Shift+E</b></td><td>Export audio</td></tr>
        </table>
        
        <p class="category">✂️ Editing</p>
        <table>
        <tr><td><b>Ctrl+Z</b></td><td>Undo</td></tr>
        <tr><td><b>Ctrl+Y</b></td><td>Redo</td></tr>
        <tr><td><b>Ctrl+X</b></td><td>Cut</td></tr>
        <tr><td><b>Ctrl+C</b></td><td>Copy</td></tr>
        <tr><td><b>Ctrl+V</b></td><td>Paste</td></tr>
        <tr><td><b>Ctrl+A</b></td><td>Select all</td></tr>
        <tr><td><b>Delete</b></td><td>Delete selected</td></tr>
        <tr><td><b>Ctrl+D</b></td><td>Duplicate</td></tr>
        </table>
        
        <p class="category">🎹 Tracks</p>
        <table>
        <tr><td><b>Ctrl+Shift+A</b></td><td>Add audio track</td></tr>
        <tr><td><b>Ctrl+Shift+I</b></td><td>Add instrument track</td></tr>
        <tr><td><b>Ctrl+Shift+D</b></td><td>Add drum track</td></tr>
        </table>
        
        <p class="category">🔧 Tools</p>
        <table>
        <tr><td><b>B</b></td><td>Draw/Brush tool</td></tr>
        <tr><td><b>E</b></td><td>Erase tool</td></tr>
        <tr><td><b>M</b></td><td>Mute tool</td></tr>
        <tr><td><b>C</b></td><td>Slice/Cut tool</td></tr>
        <tr><td><b>P</b></td><td>Paint tool</td></tr>
        <tr><td><b>Z</b></td><td>Zoom tool</td></tr>
        </table>
        
        <p class="category">👁️ Views</p>
        <table>
        <tr><td><b>F5</b></td><td>Arrangement view</td></tr>
        <tr><td><b>F6</b></td><td>Pattern editor</td></tr>
        <tr><td><b>F7</b></td><td>Piano roll</td></tr>
        <tr><td><b>F8</b></td><td>Device chain</td></tr>
        <tr><td><b>F9</b></td><td>Mixer</td></tr>
        <tr><td><b>F10</b></td><td>Audio settings</td></tr>
        <tr><td><b>F11</b></td><td>Fullscreen</td></tr>
        <tr><td><b>F1</b></td><td>Show this help</td></tr>
        </table>
        
        <p class="category">🎵 Pattern Editor</p>
        <table>
        <tr><td><b>Ctrl+C</b></td><td>Copy step</td></tr>
        <tr><td><b>Ctrl+V</b></td><td>Paste step</td></tr>
        <tr><td><b>Ctrl+D</b></td><td>Duplicate step</td></tr>
        <tr><td><b>←/→</b></td><td>Nudge step</td></tr>
        <tr><td><b>Shift+D</b></td><td>Double pattern</td></tr>
        <tr><td><b>Shift+H</b></td><td>Halve pattern</td></tr>
        </table>
        
        <p class="category">🎹 Piano (Live Play)</p>
        <table>
        <tr><td><b>A-K</b></td><td>Play notes (C-B)</td></tr>
        <tr><td><b>W,E,T,Y,U</b></td><td>Sharp notes</td></tr>
        </table>
        """
        msg = QMessageBox(self)
        msg.setWindowTitle("Keyboard Shortcuts")
        msg.setTextFormat(Qt.RichText)
        msg.setText(shortcuts)
        msg.setStyleSheet("""
            QMessageBox {
                background: #1a1a1a;
            }
            QLabel {
                color: #e0e0e0;
                min-width: 400px;
            }
            QPushButton {
                background: #00aaff;
                border: none;
                padding: 8px 20px;
                border-radius: 4px;
                color: white;
            }
        """)
        msg.exec_()
        
    def add_track(self, track_type: str):
        """Add a new track."""
        count = len(self.arrangement.tracks) + 1
        name = f"{track_type} {count}"
        
        # Add to arrangement
        lane = self.arrangement.add_track(name)
        
        # Add to mixer
        self.mixer.add_strip(name)
        
        self.status.showMessage(f"Added {name}")
        
    def setup_demo_session(self):
        """Set up a demo session with some tracks."""
        self._reset_session_model()
        self.clear_session()
        # Add some demo tracks with different types
        track_configs = [
            ("Drums", "drums", "#00aaff"),
            ("Bass", "synth", "#00ffaa"),
            ("Synth Lead", "synth", "#ffaa00"),
            ("Pad", "synth", "#ff6600"),
            ("FX", "audio", "#aa00ff"),
        ]
        
        for i, (name, track_type, color) in enumerate(track_configs):
            lane = self.arrangement.add_track(name, track_type)
            self.mixer.add_strip(name)
            
            # Add demo clips
            if name == "Drums":
                # Create drum clips with patterns
                for bar in range(0, 32, 4):
                    clip = lane.add_clip(bar * 4, 16, "Beat", color)
                    
                    # Set up a basic drum pattern if available
                    if PATTERN_AVAILABLE and clip.pattern:
                        # Four on the floor kick
                        for step in [0, 4, 8, 12]:
                            if step < len(clip.pattern.rows[0].steps):
                                clip.pattern.rows[0].steps[step].active = True
                        # Snare on 2 and 4
                        for step in [4, 12]:
                            if step < len(clip.pattern.rows[1].steps):
                                clip.pattern.rows[1].steps[step].active = True
                        # Hi-hats on 8ths
                        for step in range(0, 16, 2):
                            if step < len(clip.pattern.rows[3].steps):
                                clip.pattern.rows[3].steps[step].active = True
                                
            elif name == "Bass":
                for bar in range(0, 32, 8):
                    lane.add_clip(bar * 4, 32, "Bass Loop", color)
            elif name == "Synth Lead":
                lane.add_clip(16, 32, "Lead A", color)
                lane.add_clip(80, 24, "Lead B", color)
            elif name == "Pad":
                lane.add_clip(0, 64, "Ambient", color)
            else:  # FX
                lane.add_clip(28, 8, "Riser", color)
                lane.add_clip(60, 8, "Drop FX", color)
        
        # Add demo devices (respect native kernel availability)
        if self.rft_native_active:
            self.devices.add_device_widget("eq", "Φ-RFT EQ")
        else:
            self.devices.add_device_widget("eq", "Parametric EQ")
        
        # Load first drum pattern into editor
        if PATTERN_AVAILABLE and hasattr(self, 'pattern_editor'):
            first_drums = self.arrangement.tracks[0][1]  # First track lane
            if first_drums.clips and first_drums.clips[0].pattern:
                self.pattern_editor.set_pattern(first_drums.clips[0].pattern)
                self.current_clip = first_drums.clips[0]
        
        # Add some demo audio clips to the backend
        if self.audio_backend and self.tone_gen:
            # Create a simple drum pattern
            drum = self.tone_gen.drum_hit(0.2)
            for beat in range(0, 64, 4):  # Kick on each bar
                self.audio_backend.add_clip(beat, drum * 0.8)
            for beat in range(2, 64, 4):  # Snare on 3
                self.audio_backend.add_clip(beat, drum * 0.5)
        self.status.showMessage("Demo session loaded")

    def clear_session(self):
        """Remove all arrangement tracks, mixer strips, and reset editors."""
        for header, lane, widget, track_type in list(self.arrangement.tracks):
            header.deleteLater()
            lane.deleteLater()
            if widget is not None:
                widget.deleteLater()
        self.arrangement.tracks.clear()
        for strip in list(self.mixer.strips):
            strip.deleteLater()
        self.mixer.strips.clear()
        self.current_clip = None
        self.selected_track_idx = -1
        if PATTERN_AVAILABLE and hasattr(self, 'pattern_editor'):
            empty_pattern = Pattern(name="Empty Pattern", is_drum=True)
            self.pattern_editor.set_pattern(empty_pattern)

    def create_blank_session(self):
        """Create a minimal blank song with empty drum pattern ready to edit."""
        self._reset_session_model()
        self.clear_session()
        drum_lane = self.arrangement.add_track("Drums 1", "drums")
        self.mixer.add_strip("Drums 1")
        blank_clip = None
        if PATTERN_AVAILABLE:
            blank_clip = drum_lane.add_clip(0, 16, "Pattern 1", drum_lane.track_color)
            if blank_clip.pattern and hasattr(self, 'pattern_editor'):
                self.pattern_editor.set_pattern(blank_clip.pattern)
                # Register pattern with pattern player for playback
                if self.pattern_player:
                    self.pattern_player.set_pattern("drums_1", blank_clip.pattern)
        self.current_clip = blank_clip
        self.status.showMessage("Blank session loaded. Double-click lanes to add clips.")
    
    def connect_transport(self):
        """Connect transport controls to audio backend"""
        self.transport.play_clicked.connect(self.on_play_clicked)
        self.transport.stop_clicked.connect(self.on_stop_clicked)
        self.transport.tempo_changed.connect(self.on_tempo_changed)
        self.transport.metronome_toggled.connect(self.on_metronome_toggled)
    
    def on_metronome_toggled(self, enabled: bool):
        """Handle metronome toggle"""
        if self.audio_backend:
            self.audio_backend.metronome_enabled = enabled
            status = "ON 🔔" if enabled else "OFF 🔕"
            self.status.showMessage(f"Metronome {status}")
    
    def on_play_clicked(self):
        """Handle play/pause button"""
        if self.audio_backend:
            if self.transport.is_playing:
                self.audio_backend.play()
                metro_status = "🔔" if self.audio_backend.metronome_enabled else ""
                self.status.showMessage(f"▶ Playing {metro_status}")
            else:
                self.audio_backend.pause()
                self.status.showMessage("⏸ Paused")
    
    def on_stop_clicked(self):
        """Handle stop button"""
        if self.audio_backend:
            self.audio_backend.stop_playback()
            self.playback_position = 0
        # Update playhead to start position
        self.arrangement.set_playhead(0)
        self.arrangement.set_playing(False)
        self.status.showMessage("⏹ Stopped")
    
    def on_position_changed(self, beat: float):
        """Handle click on timeline to set playback position."""
        self.playback_position = beat
        if self.audio_backend:
            self.audio_backend.set_position(beat)
        # Update displays
        bar = int(beat / 4) + 1
        beat_num = int(beat % 4) + 1
        tick = int((beat % 1) * 960)
        self.transport.update_position(bar, beat_num, tick)
        self.arrangement.set_playhead(beat)
        self.status.showMessage(f"Position: Bar {bar}, Beat {beat_num}")
    
    def on_start_cue_changed(self, beat: float):
        """Handle start cue marker drag."""
        # Store start cue position for playback
        self._start_cue_beat = beat
        bar = int(beat / 4) + 1
        beat_num = int(beat % 4) + 1
        self.status.showMessage(f"Start Cue: Bar {bar}, Beat {beat_num}")
    
    def on_tempo_changed(self, bpm: float):
        """Handle tempo change"""
        if self.audio_backend:
            self.audio_backend.set_tempo(bpm)
        self.status.showMessage(f"Tempo: {bpm:.1f} BPM")
        
    def setup_timer(self):
        """Timer for playback position updates."""
        self.playback_timer = QTimer(self)
        self.playback_timer.timeout.connect(self.update_playback)
        self.playback_timer.start(50)  # 20 FPS
        
        self.playback_position = 0  # In beats
        
    def update_playback(self):
        """Update playback position display and playhead."""
        if self.audio_backend and self.transport.is_playing:
            # Get actual position from audio backend
            self.playback_position = self.audio_backend.get_position_beats()
            
            bar = int(self.playback_position / 4) + 1
            beat = int(self.playback_position % 4) + 1
            tick = int((self.playback_position % 1) * 960)
            
            self.transport.update_position(bar, beat, tick)
            
            # Update arrangement playhead
            self.arrangement.set_playhead(self.playback_position)
            self.arrangement.set_playing(True)
        elif self.transport.is_playing:
            # Fallback if no audio backend
            tempo = self.transport.tempo_spin.value()
            beats_per_second = tempo / 60
            self.playback_position += beats_per_second * 0.05
            
            bar = int(self.playback_position / 4) + 1
            beat = int(self.playback_position % 4) + 1
            tick = int((self.playback_position % 1) * 960)
            
            self.transport.update_position(bar, beat, tick)
            
            # Update arrangement playhead
            self.arrangement.set_playhead(self.playback_position)
            self.arrangement.set_playing(True)
        else:
            # Not playing - update playhead to show current position but dim
            self.arrangement.set_playhead(self.playback_position)
            self.arrangement.set_playing(False)
    
    def on_preset_selected(self, preset):
        """Handle preset selection from browser"""
        if self.synth:
            self.synth.set_preset(preset)
            self.status.showMessage(f"🎹 Loaded: {preset.name} | Press A-K to play!")
    
    def on_file_import(self, file_path: str):
        """Handle file import from media pool."""
        import os
        filename = os.path.basename(file_path)
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext in ['.wav', '.mp3', '.flac', '.ogg', '.aiff', '.aif']:
            # Import audio file to first available audio track
            target_track_idx = -1
            for idx, (header, lane, widget, track_type) in enumerate(self.arrangement.tracks):
                if track_type == "audio":
                    target_track_idx = idx
                    break
            
            if target_track_idx == -1:
                # No audio track - add one
                self.add_track("audio")
                target_track_idx = len(self.arrangement.tracks) - 1
                
            # Create audio clip at playhead position
            if target_track_idx >= 0 and target_track_idx < len(self.arrangement.tracks):
                header, lane, widget, track_type = self.arrangement.tracks[target_track_idx]
                
                # Import with default 4-bar length (adjust based on file analysis later)
                from .engine import Clip, ClipKind
                clip = Clip(
                    kind=ClipKind.AUDIO,
                    name=filename,
                    start_beat=self.playback_position,
                    length_beats=4.0,
                    color="#00ffaa"
                )
                clip.audio_file = file_path  # Store file path for later loading
                lane.clips.append(clip)
                lane.update()
                
                self.mark_project_modified()
                self.status.showMessage(f"📁 Imported: {filename} to track {target_track_idx + 1}")
        elif ext in ['.mid', '.midi']:
            # Import MIDI file
            self.status.showMessage(f"🎹 MIDI import: {filename} (coming soon)")
        elif ext == '.qsd':
            # Open project file
            try:
                from .engine import Session
                self.session = Session.load_from_file(file_path)
                self._reset_session_model(self.session)
                self.current_project_path = file_path
                self.project_modified = False
                self.update_window_title()
                self.refresh_ui_from_session()
                self.status.showMessage(f"Opened project: {filename}")
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'Failed to open project:\n{str(e)}')
        else:
            self.status.showMessage(f"Unknown file type: {ext}")
    
    def on_clip_selected(self, clip):
        """Handle clip selection in arrangement - route to pattern editor"""
        self.current_clip = clip
        
        # If it has a pattern, load into pattern editor
        if PATTERN_AVAILABLE and hasattr(self, 'pattern_editor') and clip.pattern:
            self.pattern_editor.set_pattern(clip.pattern)
            
            # Register pattern with pattern player for playback
            if self.pattern_player:
                self.pattern_player.set_pattern("active", clip.pattern)
            
            # Switch to pattern tab
            if hasattr(self, 'editor_tabs'):
                for i in range(self.editor_tabs.count()):
                    if "Pattern" in self.editor_tabs.tabText(i):
                        self.editor_tabs.setCurrentIndex(i)
                        break
            
            # Update device panel with track info
            track_name = clip.name.split()[0] if ' ' in clip.name else clip.name
            self.devices.set_track(track_name)
            
            self.status.showMessage(f"🥁 Editing pattern: {clip.name} | Click cells to add drums, press PLAY to hear")
        else:
            self.status.showMessage(f"Selected: {clip.name}")
    
    def on_clip_double_clicked(self, clip):
        """Handle double-click on clip to open editor in focus"""
        self.current_clip = clip
        
        if clip.pattern:
            # Open pattern editor with this pattern
            if PATTERN_AVAILABLE and hasattr(self, 'pattern_editor'):
                self.pattern_editor.set_pattern(clip.pattern)
                if hasattr(self, 'editor_tabs'):
                    for i in range(self.editor_tabs.count()):
                        if "Pattern" in self.editor_tabs.tabText(i):
                            self.editor_tabs.setCurrentIndex(i)
                            break
            
            # Maximize the bottom panel for editing
            self.status.showMessage(f"🥁 Editing: {clip.name} | Draw steps, hold for velocity")
        else:
            # For audio clips, we could show waveform editor
            self.status.showMessage(f"📁 Audio clip: {clip.name} (waveform editor coming soon)")
    
    def on_pattern_changed(self):
        """Handle pattern changes - update audio preview and clip display"""
        if not self.current_clip:
            return
        
        # The current_clip.pattern is already updated because PatternEditorWidget
        # directly modifies the Pattern object (the steps are shared references)
        
        # Update the clip display in the arrangement view
        for header, lane, widget, track_type in self.arrangement.tracks:
            lane.update()  # Refresh all lanes to show pattern changes
        
        # If playing, trigger audio preview
        if self.transport.is_playing and self.pattern_player and self.current_clip.pattern:
            # Pattern player already has a reference to the pattern
            pass
        
        self.status.showMessage(f"Pattern updated: {self.current_clip.name}")
    
    def on_track_selected(self, track_idx: int):
        """Handle track selection."""
        self.selected_track_idx = track_idx
        if 0 <= track_idx < len(self.arrangement.tracks):
            header, lane, widget, track_type = self.arrangement.tracks[track_idx]
            track_name = header.name_label.text()
            self.devices.set_track(track_name)
            self.status.showMessage(f"Track selected: {track_name}")
    
    def on_track_volume_changed(self, track_idx: int, volume: float):
        """Handle mixer fader change - route to audio backend."""
        if self.audio_backend:
            self.audio_backend.set_track_volume(track_idx, volume)
        
        # Also update track header
        if 0 <= track_idx < len(self.arrangement.tracks):
            header, lane, widget, track_type = self.arrangement.tracks[track_idx]
            header.vol_slider.blockSignals(True)
            header.vol_slider.setValue(int(volume * 127))
            header.vol_slider.blockSignals(False)
    
    def on_track_pan_changed(self, track_idx: int, pan: float):
        """Handle mixer pan change - route to audio backend."""
        if self.audio_backend:
            self.audio_backend.set_track_pan(track_idx, pan)
    
    def on_track_muted(self, track_idx: int, muted: bool):
        """Handle mixer mute button - sync with track header."""
        if 0 <= track_idx < len(self.arrangement.tracks):
            header, lane, widget, track_type = self.arrangement.tracks[track_idx]
            header.mute_btn.blockSignals(True)
            header.mute_btn.setChecked(muted)
            header.mute_btn.blockSignals(False)
            
            # Mute all clips in the lane
            for clip in lane.clips:
                clip.muted = muted
            lane.update()
            
            if self.audio_backend:
                self.audio_backend.set_track_mute(track_idx, muted)
    
    def on_track_soloed(self, track_idx: int, soloed: bool):
        """Handle mixer solo button - mute others."""
        if soloed:
            # Solo this track - mute all others
            for idx, (header, lane, widget, track_type) in enumerate(self.arrangement.tracks):
                if idx != track_idx:
                    header.mute_btn.setChecked(True)
                    for clip in lane.clips:
                        clip.muted = True
                    lane.update()
        else:
            # Unsolo - restore previous mute states (simplified: unmute all)
            for idx, (header, lane, widget, track_type) in enumerate(self.arrangement.tracks):
                if idx != track_idx:
                    header.mute_btn.setChecked(False)
                    for clip in lane.clips:
                        clip.muted = False
                    lane.update()
    
    def trigger_drum_preview(self, drum_type, velocity: float = 0.8):
        """Play a drum sound for preview"""
        if self.drum_synth and self.audio_backend:
            audio = self.drum_synth.synthesize(drum_type, velocity=velocity, duration=0.3)
            self.audio_backend.play_preview(audio)
            self.status.showMessage(f"🥁 {drum_type.value}")

    def on_step_preview(self, row_index: int, velocity: float):
        """Handle step toggled on in pattern editor - play the drum sound for that row"""
        if not self.current_clip or not self.current_clip.pattern:
            return
        pattern = self.current_clip.pattern
        if row_index < 0 or row_index >= len(pattern.rows):
            return
        row = pattern.rows[row_index]
        if row.drum_type:
            self.trigger_drum_preview(row.drum_type, velocity)

    def closeEvent(self, event):
        """Handle application close with unsaved changes check and cleanup."""
        # Check for unsaved changes
        if self.project_modified:
            reply = QMessageBox.question(
                self, 'Unsaved Changes',
                'You have unsaved changes. Do you want to save before closing?',
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Save
            )
            
            if reply == QMessageBox.Save:
                self.on_save_project()
                if self.project_modified:  # Save was cancelled or failed
                    event.ignore()
                    return
            elif reply == QMessageBox.Cancel:
                event.ignore()
                return
        
        # Stop audio playback
        try:
            if self.audio_backend:
                self.audio_backend.stop()
        except Exception as e:
            print(f"Warning: Error stopping audio backend: {e}")
        
        # Stop any running timers
        try:
            if hasattr(self, 'timer') and self.timer:
                self.timer.stop()
        except Exception as e:
            print(f"Warning: Error stopping timer: {e}")
        
        # Accept close event
        event.accept()
    
    def keyPressEvent(self, event):
        """Handle keyboard input for synth"""
        # Forward to instrument editor if available
        if SYNTH_AVAILABLE and hasattr(self, 'instrument_editor'):
            self.instrument_editor.keyboard.keyPressEvent(event)
        super().keyPressEvent(event)
    
    def keyReleaseEvent(self, event):
        """Handle keyboard release for synth"""
        if SYNTH_AVAILABLE and hasattr(self, 'instrument_editor'):
            self.instrument_editor.keyboard.keyReleaseEvent(event)
        super().keyReleaseEvent(event)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Launch QuantSoundDesign with error handling."""
    import traceback
    import logging
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(),
        ]
    )
    logger = logging.getLogger('QuantSoundDesign')
    
    try:
        app = QApplication(sys.argv)
        app.setStyle("Fusion")
        app.setApplicationName("QuantSoundDesign")
        app.setApplicationVersion("1.0.0")
        app.setOrganizationName("QuantoniumOS")
        
        # Set exception hook for Qt
        def exception_hook(exc_type, exc_value, exc_tb):
            logger.error("Unhandled exception:", exc_info=(exc_type, exc_value, exc_tb))
            # Show error dialog
            error_msg = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("Error")
            msg.setText("An unexpected error occurred.")
            msg.setDetailedText(error_msg)
            msg.exec_()
        
        sys.excepthook = exception_hook
        
        window = QuantSoundDesign()
        window.show()
        
        logger.info("QuantSoundDesign started successfully")
        sys.exit(app.exec_())
        
    except Exception as e:
        logger.error(f"Failed to start QuantSoundDesign: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
