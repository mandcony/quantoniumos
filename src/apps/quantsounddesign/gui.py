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
from typing import Optional, List, Dict
from dataclasses import dataclass, field
from enum import Enum

try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QPushButton, QSlider, QScrollArea, QFrame, QSplitter,
        QMenu, QAction, QFileDialog, QMessageBox, QSpinBox, QDoubleSpinBox,
        QComboBox, QToolBar, QStatusBar, QDockWidget, QListWidget,
        QListWidgetItem, QGraphicsView, QGraphicsScene, QGraphicsRectItem,
        QGridLayout, QTabWidget, QSizePolicy, QStackedWidget, QProgressBar,
        QToolButton
    )
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QRectF, QPointF, QSize
    from PyQt5.QtGui import (
        QColor, QPainter, QBrush, QPen, QFont, QLinearGradient, 
        QRadialGradient, QPainterPath, QIcon, QPixmap
    )
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

# Import synth and piano roll
try:
    from .synth_engine import PolySynth, PRESET_LIBRARY
    from .piano_roll import InstrumentEditor, PianoKeyboard, InstrumentBrowser
    SYNTH_AVAILABLE = True
except ImportError:
    try:
        from synth_engine import PolySynth, PRESET_LIBRARY
        from piano_roll import InstrumentEditor, PianoKeyboard, InstrumentBrowser
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
            
            # Peak indicator
            peak_y = int((1 - peak) * h)
            if peak > 0.9:
                painter.setPen(QPen(QColor('#ff0000'), 2))
            else:
                painter.setPen(QPen(QColor('#ffffff'), 2))
            painter.drawLine(x, peak_y, x + meter_w, peak_y)


class ChannelStrip(QFrame):
    """Professional mixer channel strip with fader, pan, meters."""
    
    volume_changed = pyqtSignal(int, float)
    pan_changed = pyqtSignal(int, float)
    
    COLORS = TrackHeader.TRACK_COLORS
    
    def __init__(self, track_name: str, index: int, parent=None):
        super().__init__(parent)
        self.index = index
        self.track_color = self.COLORS[index % len(self.COLORS)] if index >= 0 else '#00aaff'
        
        self.setFixedWidth(80)
        self.setMinimumHeight(360)
        self.setup_ui(track_name)
        self.update_style()
        
    def setup_ui(self, track_name: str):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 8, 6, 8)
        layout.setSpacing(6)
        
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
            font-size: 10px; 
            font-weight: bold;
            padding: 4px;
        """)
        layout.addWidget(self.name_label)
        
        # Mute/Solo/Arm buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(3)
        
        self.mute_btn = self._create_btn("M", "#ff4444")
        self.solo_btn = self._create_btn("S", "#ffaa00")
        self.arm_btn = self._create_btn("R", "#ff0000")
        
        btn_row.addWidget(self.mute_btn)
        btn_row.addWidget(self.solo_btn)
        btn_row.addWidget(self.arm_btn)
        layout.addLayout(btn_row)
        
        # Pan control
        pan_row = QHBoxLayout()
        pan_row.setSpacing(4)
        
        pan_label = QLabel("PAN")
        pan_label.setStyleSheet("color: #666; font-size: 8px;")
        pan_label.setAlignment(Qt.AlignCenter)
        
        self.pan_slider = QSlider(Qt.Horizontal)
        self.pan_slider.setRange(-100, 100)
        self.pan_slider.setValue(0)
        self.pan_slider.setFixedHeight(16)
        self.pan_slider.valueChanged.connect(
            lambda v: self.pan_changed.emit(self.index, v / 100)
        )
        
        self.pan_value = QLabel("C")
        self.pan_value.setStyleSheet("color: #888; font-size: 9px; min-width: 24px;")
        self.pan_value.setAlignment(Qt.AlignCenter)
        self.pan_slider.valueChanged.connect(self._update_pan_label)
        
        layout.addWidget(pan_label)
        layout.addWidget(self.pan_slider)
        layout.addWidget(self.pan_value)
        
        # Meter + Fader section
        fader_section = QHBoxLayout()
        fader_section.setSpacing(4)
        
        # VU Meter
        self.meter = MeterWidget()
        fader_section.addWidget(self.meter)
        
        # Fader
        fader_container = QVBoxLayout()
        fader_container.setSpacing(2)
        
        self.fader = QSlider(Qt.Vertical)
        self.fader.setRange(0, 127)
        self.fader.setValue(100)
        self.fader.setMinimumHeight(160)
        self.fader.valueChanged.connect(self._on_fader_changed)
        fader_container.addWidget(self.fader, 1)
        
        fader_section.addLayout(fader_container)
        
        layout.addLayout(fader_section, 1)
        
        # dB display
        self.db_label = QLabel("0.0 dB")
        self.db_label.setAlignment(Qt.AlignCenter)
        self.db_label.setStyleSheet("""
            color: #00ffaa; 
            font-size: 10px; 
            font-family: 'Consolas', monospace;
            font-weight: bold;
            background: #1a1a1a;
            border-radius: 3px;
            padding: 4px;
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
    """Professional mixer view with channel strips and master."""
    
    track_volume_changed = pyqtSignal(int, float)  # track_idx, volume 0-1
    track_pan_changed = pyqtSignal(int, float)  # track_idx, pan -1 to 1
    track_muted = pyqtSignal(int, bool)  # track_idx, muted
    track_soloed = pyqtSignal(int, bool)  # track_idx, soloed
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.strips = []
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
        
        # Scrollable area for strips
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
        
        # Divider
        divider = QFrame()
        divider.setFixedWidth(2)
        divider.setStyleSheet("background-color: #00aaff;")
        layout.addWidget(divider)
        
        # Master channel (always visible)
        self.master = ChannelStrip("MASTER", -1)
        self.master.setFixedWidth(90)
        self.master.color_bar.setStyleSheet("background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00aaff, stop:1 #00ffaa); border-radius: 2px;")
        self.master.name_label.setStyleSheet("color: #00ffaa; font-size: 11px; font-weight: bold;")
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
        eq_menu.addAction("Φ-RFT EQ", lambda: self.add_device_widget("eq", "EQ"))
        eq_menu.addAction("Parametric EQ", lambda: self.add_device_widget("eq", "Parametric"))
        eq_menu.addAction("Graphic EQ", lambda: self.add_device_widget("eq", "Graphic"))
        
        # Dynamics submenu
        dyn_menu = menu.addMenu("🔊 Dynamics")
        dyn_menu.addAction("Compressor", lambda: self.add_device_widget("comp", "Compressor"))
        dyn_menu.addAction("Limiter", lambda: self.add_device_widget("comp", "Limiter"))
        dyn_menu.addAction("Gate", lambda: self.add_device_widget("comp", "Gate"))
        
        # Effects submenu
        fx_menu = menu.addMenu("✨ Effects")
        fx_menu.addAction("Φ-RFT Reverb", lambda: self.add_device_widget("reverb", "Reverb"))
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
        
        # Left sidebar - Instrument Browser
        left_panel = None
        if SYNTH_AVAILABLE:
            left_panel = QFrame()
            left_panel.setMaximumWidth(240)
            left_panel.setStyleSheet("""
                QFrame {
                    background-color: #0d0d0d;
                    border-right: 2px solid #00aaff;
                }
            """)
            left_layout = QVBoxLayout(left_panel)
            left_layout.setContentsMargins(0, 0, 0, 0)
            left_layout.setSpacing(0)
            
            # Browser header
            browser_header = QFrame()
            browser_header.setFixedHeight(36)
            browser_header.setStyleSheet("background: #151515; border-bottom: 1px solid #2a2a2a;")
            bh_layout = QHBoxLayout(browser_header)
            bh_layout.setContentsMargins(12, 0, 12, 0)
            
            browser_title = QLabel("🎹 INSTRUMENTS")
            browser_title.setStyleSheet("color: #00aaff; font-weight: bold; font-size: 11px;")
            bh_layout.addWidget(browser_title)
            bh_layout.addStretch()
            
            left_layout.addWidget(browser_header)
            
            self.instrument_browser = InstrumentBrowser()
            self.instrument_browser.preset_selected.connect(self.on_preset_selected)
            left_layout.addWidget(self.instrument_browser)
            
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
        self.status.showMessage("QuantSoundDesign Ready | Φ-RFT Engine Active | Press A-K to play!")
        
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
        
        # Additional view shortcuts (F10-F12)
        f10_shortcut = QShortcut(QKeySequence(Qt.Key_F10), self)
        f10_shortcut.activated.connect(self.show_step_sequencer)
        
        f11_shortcut = QShortcut(QKeySequence(Qt.Key_F11), self)
        f11_shortcut.activated.connect(self.toggle_fullscreen)
        
        f12_shortcut = QShortcut(QKeySequence(Qt.Key_F12), self)
        f12_shortcut.activated.connect(self.show_script_console)
        
        # Tool shortcuts
        b_shortcut = QShortcut(QKeySequence(Qt.Key_B), self)
        b_shortcut.activated.connect(lambda: self.set_tool("draw"))
        
        e_shortcut = QShortcut(QKeySequence(Qt.Key_E), self)
        e_shortcut.activated.connect(lambda: self.set_tool("erase"))
        
        m_shortcut = QShortcut(QKeySequence(Qt.Key_M), self)
        m_shortcut.activated.connect(lambda: self.set_tool("mute"))
        
        c_shortcut = QShortcut(QKeySequence(Qt.Key_C), self)
        c_shortcut.activated.connect(lambda: self.set_tool("slice"))
        
        p_shortcut = QShortcut(QKeySequence(Qt.Key_P), self)
        p_shortcut.activated.connect(lambda: self.set_tool("paint"))
        
        z_shortcut = QShortcut(QKeySequence(Qt.Key_Z), self)
        z_shortcut.activated.connect(lambda: self.set_tool("zoom"))
        
        print("[OK] Keyboard shortcuts initialized")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SHORTCUT ACTION HANDLERS
    # ═══════════════════════════════════════════════════════════════════════════
    
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
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        about_action = QAction("About QuantSoundDesign", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
        shortcuts_action = QAction("Keyboard Shortcuts", self)
        shortcuts_action.setShortcut("F1")
        shortcuts_action.triggered.connect(self.show_shortcuts)
        help_menu.addAction(shortcuts_action)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # MENU ACTION HANDLERS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def on_new_project(self):
        """Create a new empty project."""
        reply = QMessageBox.question(self, 'New Project',
            'Create a new project? Unsaved changes will be lost.',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.create_blank_session()
            self.status.showMessage("New blank project created")
    
    def on_open_project(self):
        """Open a project file."""
        filename, _ = QFileDialog.getOpenFileName(
            self, 'Open Project', '', 
            'QuantSoundDesign Projects (*.qsd);;All Files (*)'
        )
        if filename:
            self.status.showMessage(f"Opening: {filename} (project loading coming soon)")
    
    def on_save_project(self):
        """Save the current project."""
        self.status.showMessage("Project saved (save functionality coming soon)")
    
    def on_save_project_as(self):
        """Save project with a new name."""
        filename, _ = QFileDialog.getSaveFileName(
            self, 'Save Project As', '',
            'QuantSoundDesign Projects (*.qsd);;All Files (*)'
        )
        if filename:
            self.status.showMessage(f"Saved as: {filename} (save functionality coming soon)")
    
    def on_export_audio(self):
        """Export audio to file."""
        filename, _ = QFileDialog.getSaveFileName(
            self, 'Export Audio', '',
            'WAV Audio (*.wav);;FLAC Audio (*.flac);;MP3 Audio (*.mp3)'
        )
        if filename:
            self.status.showMessage(f"Exporting to: {filename} (export coming soon)")
    
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
        <h3>Keyboard Shortcuts</h3>
        <table>
        <tr><td><b>Space</b></td><td>Play/Pause</td></tr>
        <tr><td><b>Enter</b></td><td>Stop</td></tr>
        <tr><td><b>A-K</b></td><td>Play notes</td></tr>
        <tr><td><b>Ctrl+N</b></td><td>New Project</td></tr>
        <tr><td><b>Ctrl+S</b></td><td>Save</td></tr>
        <tr><td><b>Ctrl+Z</b></td><td>Undo</td></tr>
        <tr><td><b>Ctrl+Shift+A</b></td><td>Add Audio Track</td></tr>
        <tr><td><b>Ctrl+Shift+I</b></td><td>Add Instrument Track</td></tr>
        <tr><td><b>Ctrl+Shift+D</b></td><td>Add Drum Track</td></tr>
        <tr><td><b>F5</b></td><td>Arrangement View</td></tr>
        <tr><td><b>F6</b></td><td>Pattern Editor</td></tr>
        <tr><td><b>F7</b></td><td>Piano Roll</td></tr>
        <tr><td><b>F8</b></td><td>Device Chain</td></tr>
        <tr><td><b>F9</b></td><td>Mixer</td></tr>
        <tr><td><b>Delete</b></td><td>Delete Selected</td></tr>
        </table>
        """
        QMessageBox.information(self, "Keyboard Shortcuts", shortcuts)
        
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
        
        # Add demo devices
        self.devices.add_device_widget("eq", "Φ-RFT EQ")
        
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
        """Clean up audio on close"""
        if self.audio_backend:
            self.audio_backend.stop()
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
    """Launch QuantSoundDesign."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    window = QuantSoundDesign()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
