#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
# This file is listed in CLAIMS_PRACTICING_FILES.txt and is licensed
# under LICENSE-CLAIMS-NC.md (research/education only). Commercial
# rights require a separate patent license from the author.
"""
QuantSoundDesign Piano Roll & Instrument Browser Widgets

EPIC 02: Pro-Grade Piano Roll with:
- Full SelectionModel integration
- Note drag/resize with snap
- Velocity lane editing
- Scale/key highlighting and snap-to-scale
- Quantize tools
- Audition rules

Provides:
- Interactive piano roll for MIDI editing
- Keyboard piano with visual feedback
- Instrument browser with drag-drop
"""

import sys
import numpy as np
from typing import Optional, List, Dict, Callable, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto

try:
    from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
        QSlider, QScrollArea, QFrame, QTreeWidget, QTreeWidgetItem,
        QSplitter, QLineEdit, QComboBox, QGraphicsView, QGraphicsScene,
        QGraphicsRectItem, QTabWidget, QListWidget, QListWidgetItem,
        QMenu, QAction, QDialog, QSpinBox, QCheckBox, QDialogButtonBox,
        QFormLayout, QGroupBox, QShortcut, QRubberBand, QApplication
    )
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QRectF, QPointF, QSize, QRect
    from PyQt5.QtGui import (
        QColor, QPainter, QBrush, QPen, QFont, QLinearGradient,
        QKeyEvent, QPainterPath, QKeySequence, QCursor
    )
except ImportError:
    print("PyQt5 required")
    sys.exit(1)

# Support both package and direct execution
try:
    from .synth_engine import (
        PRESET_LIBRARY, InstrumentPreset, PolySynth,
        key_to_note, note_to_name, SIMPLE_KEYBOARD_MAP
    )
except ImportError:
    from synth_engine import (
        PRESET_LIBRARY, InstrumentPreset, PolySynth,
        key_to_note, note_to_name, SIMPLE_KEYBOARD_MAP
    )

# Import core module for SelectionModel, Scale, etc.
try:
    from .core import (
        get_selection_model, get_audition_engine, Scale, ScaleType, 
        snap_to_grid, NOTE_NAMES
    )
    CORE_AVAILABLE = True
except ImportError:
    try:
        from core import (
            get_selection_model, get_audition_engine, Scale, ScaleType,
            snap_to_grid, NOTE_NAMES
        )
        CORE_AVAILABLE = True
    except ImportError:
        CORE_AVAILABLE = False
        NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        def snap_to_grid(beats, grid_size):
            if grid_size <= 0:
                return beats
            return round(beats / grid_size) * grid_size


# ═══════════════════════════════════════════════════════════════════════════════
# STYLE
# ═══════════════════════════════════════════════════════════════════════════════

PIANO_WHITE = "#e8e8e8"
PIANO_BLACK = "#1a1a1a"
PIANO_PRESSED = "#00aaff"
PIANO_ACTIVE = "#00ffaa"
NOTE_COLOR = "#00aaff"
NOTE_SELECTED = "#00ffaa"
SCALE_HIGHLIGHT = "#1a2a3a"  # Subtle highlight for scale notes
VELOCITY_BAR_COLOR = "#00aaff"


# ═══════════════════════════════════════════════════════════════════════════════
# NOTE DATA CLASS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class NoteData:
    """Represents a single MIDI note in the piano roll"""
    note: int            # MIDI note number (0-127)
    start: float         # Start time in beats
    duration: float      # Duration in beats
    velocity: float = 0.8  # 0.0 - 1.0
    channel: int = 0
    muted: bool = False
    
    # Unique ID for selection tracking
    _id: int = field(default_factory=lambda: id(object()))
    
    def __hash__(self):
        return self._id
    
    def __eq__(self, other):
        if isinstance(other, NoteData):
            return self._id == other._id
        return False
    
    @property
    def end(self) -> float:
        return self.start + self.duration
    
    def to_dict(self) -> Dict:
        return {
            'note': self.note,
            'start': self.start,
            'duration': self.duration,
            'velocity': self.velocity,
            'channel': self.channel,
            'muted': self.muted
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'NoteData':
        return cls(
            note=data['note'],
            start=data['start'],
            duration=data['duration'],
            velocity=data.get('velocity', 0.8),
            channel=data.get('channel', 0),
            muted=data.get('muted', False)
        )


# ═══════════════════════════════════════════════════════════════════════════════
# EDIT TOOL ENUM
# ═══════════════════════════════════════════════════════════════════════════════

class PianoRollTool(Enum):
    """Editing tools for piano roll"""
    POINTER = auto()   # Select, move, resize
    DRAW = auto()      # Draw new notes
    ERASE = auto()     # Erase notes
    PAINT = auto()     # Paint mode (continuous drawing)
    SLICE = auto()     # Slice notes
    MUTE = auto()      # Toggle mute
    VELOCITY = auto()  # Edit velocity


# ═══════════════════════════════════════════════════════════════════════════════
# PIANO KEYBOARD WIDGET
# ═══════════════════════════════════════════════════════════════════════════════

class PianoKeyboard(QWidget):
    """
    Visual piano keyboard with:
    - Mouse click to play
    - Computer keyboard input (ASDFGHJK)
    - Visual feedback for active notes
    """
    
    note_on = pyqtSignal(int, float)   # note, velocity
    note_off = pyqtSignal(int)          # note
    
    def __init__(self, octaves: int = 3, start_octave: int = 3, parent=None):
        super().__init__(parent)
        self.octaves = octaves
        self.start_octave = start_octave
        self.white_key_width = 28
        self.black_key_width = 18
        self.white_key_height = 100
        self.black_key_height = 60
        
        self.active_notes: set = set()
        self.pressed_keys: set = set()  # Computer keyboard keys
        
        self.setMinimumHeight(self.white_key_height + 10)
        self.setFocusPolicy(Qt.StrongFocus)
        
        # Calculate width based on octaves
        white_keys_per_octave = 7
        total_white_keys = self.octaves * white_keys_per_octave + 1  # +1 for final C
        self.setMinimumWidth(total_white_keys * self.white_key_width)
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw white keys first
        x = 0
        white_key_notes = []
        
        for octave in range(self.octaves + 1):
            note_in_octave = [0, 2, 4, 5, 7, 9, 11]  # C D E F G A B
            for i, note_offset in enumerate(note_in_octave):
                if octave == self.octaves and i > 0:
                    break
                    
                note = (self.start_octave + octave) * 12 + note_offset
                is_active = note in self.active_notes
                
                # Draw white key
                color = QColor(PIANO_ACTIVE if is_active else PIANO_WHITE)
                painter.setBrush(QBrush(color))
                painter.setPen(QPen(QColor("#888888"), 1))
                painter.drawRect(x, 0, self.white_key_width - 1, self.white_key_height)
                
                # Draw key label (for computer keyboard mapping)
                key_label = self._get_key_label(note)
                if key_label:
                    painter.setPen(QColor("#888888"))
                    painter.setFont(QFont("Segoe UI", 8))
                    painter.drawText(x + 8, self.white_key_height - 8, key_label.upper())
                
                white_key_notes.append((x, note))
                x += self.white_key_width
        
        # Draw black keys on top
        x = 0
        for octave in range(self.octaves):
            black_positions = [1, 2, 4, 5, 6]  # C# D# F# G# A#
            black_offsets = [1, 3, 6, 8, 10]
            
            for i, (pos, offset) in enumerate(zip(black_positions, black_offsets)):
                note = (self.start_octave + octave) * 12 + offset
                is_active = note in self.active_notes
                
                bx = x + pos * self.white_key_width - self.black_key_width // 2
                
                color = QColor(PIANO_ACTIVE if is_active else PIANO_BLACK)
                painter.setBrush(QBrush(color))
                painter.setPen(QPen(QColor("#000000"), 1))
                painter.drawRect(bx, 0, self.black_key_width, self.black_key_height)
                
                # Key label
                key_label = self._get_key_label(note)
                if key_label:
                    painter.setPen(QColor("#888888"))
                    painter.setFont(QFont("Segoe UI", 7))
                    painter.drawText(bx + 4, self.black_key_height - 6, key_label.upper())
            
            x += 7 * self.white_key_width
    
    def _get_key_label(self, note: int) -> Optional[str]:
        """Get computer keyboard key for this note"""
        for key, midi in SIMPLE_KEYBOARD_MAP.items():
            if midi == note:
                return key
        return None
    
    def _note_at_pos(self, x: int, y: int) -> Optional[int]:
        """Get MIDI note at mouse position"""
        # Check black keys first (they're on top)
        white_x = 0
        for octave in range(self.octaves):
            black_positions = [1, 2, 4, 5, 6]
            black_offsets = [1, 3, 6, 8, 10]
            
            for pos, offset in zip(black_positions, black_offsets):
                bx = white_x + pos * self.white_key_width - self.black_key_width // 2
                if bx <= x <= bx + self.black_key_width and y <= self.black_key_height:
                    return (self.start_octave + octave) * 12 + offset
            
            white_x += 7 * self.white_key_width
        
        # Check white keys
        white_index = x // self.white_key_width
        octave = white_index // 7
        key_in_octave = white_index % 7
        white_notes = [0, 2, 4, 5, 7, 9, 11]
        
        if octave <= self.octaves and key_in_octave < len(white_notes):
            return (self.start_octave + octave) * 12 + white_notes[key_in_octave]
        
        return None
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            note = self._note_at_pos(event.x(), event.y())
            if note is not None:
                self.active_notes.add(note)
                self.note_on.emit(note, 0.8)
                self.update()
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            note = self._note_at_pos(event.x(), event.y())
            if note is not None and note in self.active_notes:
                self.active_notes.discard(note)
                self.note_off.emit(note)
                self.update()
    
    def keyPressEvent(self, event: QKeyEvent):
        if event.isAutoRepeat():
            return
        
        key = event.text().lower()
        if key in self.pressed_keys:
            return
        
        note = key_to_note(key, use_simple=True)
        if note is not None:
            self.pressed_keys.add(key)
            self.active_notes.add(note)
            self.note_on.emit(note, 0.8)
            self.update()
    
    def keyReleaseEvent(self, event: QKeyEvent):
        if event.isAutoRepeat():
            return
        
        key = event.text().lower()
        if key not in self.pressed_keys:
            return
        
        note = key_to_note(key, use_simple=True)
        if note is not None:
            self.pressed_keys.discard(key)
            self.active_notes.discard(note)
            self.note_off.emit(note)
            self.update()
    
    def set_active_notes(self, notes: List[int]):
        """Set which notes appear pressed (for external control)"""
        self.active_notes = set(notes)
        self.update()


# ═══════════════════════════════════════════════════════════════════════════════
# PIANO ROLL (MIDI Editor) - EPIC 02 Pro-Grade Implementation
# ═══════════════════════════════════════════════════════════════════════════════

class PianoRollView(QWidget):
    """
    Pro-grade MIDI note editor with:
    - Full selection model (single, multi, marquee, range)
    - Note drag and resize with snap
    - Velocity lane editing
    - Scale/key highlighting
    - Quantize tools
    - Full keyboard shortcuts
    """
    
    note_added = pyqtSignal(object)    # NoteData
    note_removed = pyqtSignal(object)  # NoteData
    notes_changed = pyqtSignal()       # General change signal
    selection_changed = pyqtSignal()   # Selection changed
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Note storage
        self.notes: List[NoteData] = []
        self.selected_notes: Set[NoteData] = set()
        
        # Display settings
        self.note_height = 14
        self.beat_width = 50
        self.key_width = 50
        self.num_octaves = 5
        self.start_note = 36  # C2
        self.velocity_lane_height = 60
        self.show_velocity_lane = True
        
        # Grid settings
        self.grid_beats = 64  # 16 bars
        self.snap_size = 0.25  # 1/16 note
        self.snap_enabled = True
        
        # Scale/key settings
        self.scale: Optional[Scale] = None
        self.show_scale_highlighting = True
        self.snap_to_scale = False
        self.fold_to_scale = False
        
        # Tool state
        self.current_tool = PianoRollTool.POINTER
        self.draw_velocity = 0.8
        self.draw_duration = 0.25  # Default note length
        
        # Interaction state
        self._drag_mode: Optional[str] = None  # 'move', 'resize_start', 'resize_end', 'velocity'
        self._drag_start_pos: Optional[QPointF] = None
        self._drag_start_notes: Dict[NoteData, Tuple[float, float]] = {}  # note -> (start, dur)
        self._marquee_rect: Optional[QRect] = None
        self._marquee_start: Optional[QPointF] = None
        
        # Clipboard
        self._clipboard: List[NoteData] = []
        
        # Audition
        self._last_audition_note = -1
        
        # UI
        self.setMinimumSize(900, 500)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        
        # Setup shortcuts
        self._setup_shortcuts()
    
    def _setup_shortcuts(self):
        """Setup keyboard shortcuts for piano roll operations"""
        # Select all
        select_all = QShortcut(QKeySequence("Ctrl+A"), self)
        select_all.activated.connect(self.select_all)
        
        # Delete
        delete = QShortcut(QKeySequence(Qt.Key_Delete), self)
        delete.activated.connect(self.delete_selected)
        
        backspace = QShortcut(QKeySequence(Qt.Key_Backspace), self)
        backspace.activated.connect(self.delete_selected)
        
        # Copy/Cut/Paste
        copy = QShortcut(QKeySequence("Ctrl+C"), self)
        copy.activated.connect(self.copy_selected)
        
        cut = QShortcut(QKeySequence("Ctrl+X"), self)
        cut.activated.connect(self.cut_selected)
        
        paste = QShortcut(QKeySequence("Ctrl+V"), self)
        paste.activated.connect(self.paste)
        
        # Duplicate
        duplicate = QShortcut(QKeySequence("Ctrl+D"), self)
        duplicate.activated.connect(self.duplicate_selected)
        
        # Quantize
        quantize = QShortcut(QKeySequence("Q"), self)
        quantize.activated.connect(self.quantize_selected)
        
        quantize_ends = QShortcut(QKeySequence("Shift+Q"), self)
        quantize_ends.activated.connect(lambda: self.quantize_selected(ends_only=True))
        
        # Arrow key nudge
        left = QShortcut(QKeySequence(Qt.Key_Left), self)
        left.activated.connect(lambda: self.nudge_selected(-self.snap_size, 0))
        
        right = QShortcut(QKeySequence(Qt.Key_Right), self)
        right.activated.connect(lambda: self.nudge_selected(self.snap_size, 0))
        
        up = QShortcut(QKeySequence(Qt.Key_Up), self)
        up.activated.connect(lambda: self.nudge_selected(0, 1))
        
        down = QShortcut(QKeySequence(Qt.Key_Down), self)
        down.activated.connect(lambda: self.nudge_selected(0, -1))
        
        # Octave transpose
        oct_up = QShortcut(QKeySequence("Shift+Up"), self)
        oct_up.activated.connect(lambda: self.nudge_selected(0, 12))
        
        oct_down = QShortcut(QKeySequence("Shift+Down"), self)
        oct_down.activated.connect(lambda: self.nudge_selected(0, -12))
        
        # Length adjustment
        lengthen = QShortcut(QKeySequence("Shift+Right"), self)
        lengthen.activated.connect(lambda: self.adjust_length(self.snap_size))
        
        shorten = QShortcut(QKeySequence("Shift+Left"), self)
        shorten.activated.connect(lambda: self.adjust_length(-self.snap_size))
        
        # Escape to deselect
        escape = QShortcut(QKeySequence(Qt.Key_Escape), self)
        escape.activated.connect(self.deselect_all)
        
        # Scale toggle
        scale_toggle = QShortcut(QKeySequence("K"), self)
        scale_toggle.activated.connect(self.toggle_scale_highlighting)
    
    @property
    def total_notes(self) -> int:
        return self.num_octaves * 12
    
    @property
    def content_height(self) -> int:
        h = self.total_notes * self.note_height
        if self.show_velocity_lane:
            h += self.velocity_lane_height
        return h
    
    def set_scale(self, scale: Optional[Scale]):
        """Set the scale for highlighting and snap"""
        self.scale = scale
        self.update()
    
    def toggle_scale_highlighting(self):
        """Toggle scale highlighting on/off"""
        self.show_scale_highlighting = not self.show_scale_highlighting
        self.update()
    
    # ─────────────────────────────────────────────────────────────────────────
    # PAINTING
    # ─────────────────────────────────────────────────────────────────────────
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Background
        painter.fillRect(self.rect(), QColor("#1a1a1a"))
        
        # Calculate visible area
        velocity_top = self.total_notes * self.note_height
        
        # Draw piano keys and grid rows
        self._draw_piano_keys(painter)
        self._draw_grid(painter)
        
        # Draw scale highlighting
        if self.show_scale_highlighting and self.scale:
            self._draw_scale_highlighting(painter)
        
        # Draw notes
        for note in self.notes:
            self._draw_note(painter, note)
        
        # Draw velocity lane
        if self.show_velocity_lane:
            self._draw_velocity_lane(painter, velocity_top)
        
        # Draw marquee selection
        if self._marquee_rect:
            painter.setPen(QPen(QColor("#00aaff"), 1, Qt.DashLine))
            painter.setBrush(QBrush(QColor(0, 170, 255, 30)))
            painter.drawRect(self._marquee_rect)
    
    def _draw_piano_keys(self, painter: QPainter):
        """Draw piano keyboard on the left"""
        for i in range(self.total_notes):
            note = self.start_note + (self.total_notes - 1 - i)
            y = i * self.note_height
            
            is_black = (note % 12) in [1, 3, 6, 8, 10]
            
            # Key background
            if is_black:
                color = QColor("#2a2a2a")
            else:
                color = QColor("#3a3a3a")
            
            # Highlight if in scale
            if self.show_scale_highlighting and self.scale and self.scale.is_in_scale(note):
                if not is_black:
                    color = QColor("#2a3a4a")
            
            painter.fillRect(0, y, self.key_width, self.note_height - 1, color)
            
            # Note name on C notes
            if note % 12 == 0:
                painter.setPen(QColor("#888888"))
                painter.setFont(QFont("Segoe UI", 7))
                octave = note // 12 - 1
                painter.drawText(4, y + self.note_height - 3, f"C{octave}")
    
    def _draw_grid(self, painter: QPainter):
        """Draw the grid lines"""
        # Vertical beat lines
        for beat in range(self.grid_beats + 1):
            x = self.key_width + beat * self.beat_width
            
            # Bar lines thicker
            if beat % 4 == 0:
                painter.setPen(QPen(QColor("#444444"), 1))
            elif beat % 1 == 0:
                painter.setPen(QPen(QColor("#2a2a2a"), 1))
            else:
                painter.setPen(QPen(QColor("#222222"), 1))
            
            painter.drawLine(x, 0, x, self.total_notes * self.note_height)
        
        # Horizontal note lines
        for i in range(self.total_notes + 1):
            y = i * self.note_height
            note = self.start_note + (self.total_notes - 1 - i)
            
            if note % 12 == 0:  # C notes
                painter.setPen(QPen(QColor("#444444"), 1))
            else:
                painter.setPen(QPen(QColor("#252525"), 1))
            
            painter.drawLine(self.key_width, y, self.width(), y)
    
    def _draw_scale_highlighting(self, painter: QPainter):
        """Draw subtle highlighting for scale notes"""
        if not self.scale:
            return
        
        for i in range(self.total_notes):
            note = self.start_note + (self.total_notes - 1 - i)
            if self.scale.is_in_scale(note):
                y = i * self.note_height
                painter.fillRect(
                    self.key_width, y, 
                    self.width() - self.key_width, self.note_height,
                    QColor(SCALE_HIGHLIGHT)
                )
    
    def _draw_note(self, painter: QPainter, note: NoteData):
        """Draw a single note"""
        if note.note < self.start_note or note.note >= self.start_note + self.total_notes:
            return
        
        row = self.total_notes - 1 - (note.note - self.start_note)
        x = self.key_width + note.start * self.beat_width
        y = row * self.note_height
        w = max(4, note.duration * self.beat_width - 2)
        h = self.note_height - 2
        
        # Color based on velocity and selection
        is_selected = note in self.selected_notes
        
        if note.muted:
            color = QColor(80, 80, 80, 150)
        elif is_selected:
            alpha = int(180 + note.velocity * 75)
            color = QColor(0, 255, 170, alpha)
        else:
            alpha = int(150 + note.velocity * 105)
            color = QColor(0, 170, 255, alpha)
        
        # Draw note rectangle
        painter.fillRect(int(x), int(y + 1), int(w), int(h), color)
        
        # Border
        border_color = QColor("#ffffff") if is_selected else QColor("#88ccff")
        painter.setPen(QPen(border_color, 1 if is_selected else 0.5))
        painter.drawRect(int(x), int(y + 1), int(w), int(h))
        
        # Velocity bar at bottom of note
        vel_height = max(2, int(h * 0.2))
        vel_color = QColor(255, int(255 * note.velocity), 0)
        painter.fillRect(int(x + 1), int(y + h - vel_height + 1), int(w - 2), vel_height, vel_color)
        
        # Resize handles for selected notes
        if is_selected and w > 10:
            handle_w = 4
            # Left handle
            painter.fillRect(int(x), int(y + 1), handle_w, int(h), QColor(255, 255, 255, 100))
            # Right handle
            painter.fillRect(int(x + w - handle_w), int(y + 1), handle_w, int(h), QColor(255, 255, 255, 100))
    
    def _draw_velocity_lane(self, painter: QPainter, top_y: int):
        """Draw the velocity editing lane"""
        # Background
        painter.fillRect(0, top_y, self.width(), self.velocity_lane_height, QColor("#151515"))
        
        # Border
        painter.setPen(QPen(QColor("#333333"), 1))
        painter.drawLine(0, top_y, self.width(), top_y)
        
        # Label
        painter.setPen(QColor("#666666"))
        painter.setFont(QFont("Segoe UI", 8))
        painter.drawText(4, top_y + 15, "Velocity")
        
        # Draw velocity bars for each note
        for note in self.notes:
            if note.note < self.start_note or note.note >= self.start_note + self.total_notes:
                continue
            
            x = self.key_width + note.start * self.beat_width
            w = max(3, note.duration * self.beat_width - 2)
            
            bar_height = int(note.velocity * (self.velocity_lane_height - 10))
            bar_y = top_y + self.velocity_lane_height - bar_height - 5
            
            is_selected = note in self.selected_notes
            color = QColor(0, 255, 170) if is_selected else QColor(0, 170, 255)
            
            painter.fillRect(int(x), bar_y, int(w), bar_height, color)
    
    # ─────────────────────────────────────────────────────────────────────────
    # MOUSE INTERACTION
    # ─────────────────────────────────────────────────────────────────────────
    
    def mousePressEvent(self, event):
        pos = event.pos()
        
        # Check if in velocity lane
        velocity_top = self.total_notes * self.note_height
        if self.show_velocity_lane and pos.y() > velocity_top:
            self._handle_velocity_click(event, velocity_top)
            return
        
        # Piano key click - audition only
        if pos.x() < self.key_width:
            note = self._note_at_y(pos.y())
            if note is not None:
                self._audition_note(note)
                # Select all notes of this pitch (Ctrl+click on key)
                if event.modifiers() & Qt.ControlModifier:
                    pitch_notes = [n for n in self.notes if n.note == note]
                    self.selected_notes = set(pitch_notes)
                    self.selection_changed.emit()
                    self.update()
            return
        
        # Grid area click
        if event.button() == Qt.LeftButton:
            self._handle_left_click(event)
        elif event.button() == Qt.RightButton:
            self._handle_right_click(event)
    
    def _handle_left_click(self, event):
        """Handle left click in grid area"""
        pos = event.pos()
        beat = self._beat_at_x(pos.x())
        note_pitch = self._note_at_y(pos.y())
        
        if note_pitch is None:
            return
        
        # Check for note under cursor
        hit_note, hit_region = self._note_hit_test(pos)
        
        if self.current_tool == PianoRollTool.DRAW or self.current_tool == PianoRollTool.PAINT:
            if hit_note:
                # Click on existing note - delete it
                if event.modifiers() & Qt.ShiftModifier:
                    self._delete_note(hit_note)
                else:
                    # Select it
                    self._select_note(hit_note, event.modifiers() & Qt.ControlModifier)
            else:
                # Draw new note
                self._draw_note_at(beat, note_pitch)
        
        elif self.current_tool == PianoRollTool.POINTER:
            if hit_note:
                # Check for resize handles
                if hit_region == 'left':
                    self._start_resize(hit_note, 'resize_start', pos)
                elif hit_region == 'right':
                    self._start_resize(hit_note, 'resize_end', pos)
                else:
                    # Move
                    if not (event.modifiers() & Qt.ControlModifier) and hit_note not in self.selected_notes:
                        self.selected_notes.clear()
                    
                    if event.modifiers() & Qt.ControlModifier:
                        if hit_note in self.selected_notes:
                            self.selected_notes.discard(hit_note)
                        else:
                            self.selected_notes.add(hit_note)
                    else:
                        self.selected_notes.add(hit_note)
                    
                    self._start_move(pos)
                
                self.selection_changed.emit()
            else:
                # Start marquee selection
                if not (event.modifiers() & Qt.ControlModifier):
                    self.selected_notes.clear()
                self._marquee_start = pos
                self._marquee_rect = QRect(pos, pos)
                self.selection_changed.emit()
        
        elif self.current_tool == PianoRollTool.MUTE:
            if hit_note:
                hit_note.muted = not hit_note.muted
                self.notes_changed.emit()
        
        self.update()
    
    def _handle_right_click(self, event):
        """Handle right click - context menu or erase"""
        pos = event.pos()
        hit_note, _ = self._note_hit_test(pos)
        
        if hit_note:
            # Show context menu
            menu = QMenu(self)
            
            delete_action = menu.addAction("Delete")
            delete_action.triggered.connect(lambda: self._delete_note(hit_note))
            
            mute_action = menu.addAction("Mute" if not hit_note.muted else "Unmute")
            mute_action.triggered.connect(lambda: self._toggle_mute(hit_note))
            
            menu.addSeparator()
            
            quantize_action = menu.addAction("Quantize")
            quantize_action.triggered.connect(self.quantize_selected)
            
            menu.exec_(event.globalPos())
    
    def _handle_velocity_click(self, event, velocity_top: int):
        """Handle click in velocity lane"""
        pos = event.pos()
        beat = self._beat_at_x(pos.x())
        
        # Find note at this beat
        for note in self.notes:
            if note.start <= beat < note.end:
                # Calculate new velocity from Y position
                rel_y = pos.y() - velocity_top
                new_vel = 1.0 - (rel_y / self.velocity_lane_height)
                new_vel = max(0.0, min(1.0, new_vel))
                note.velocity = new_vel
                
                self._drag_mode = 'velocity'
                self._drag_start_pos = pos
                
                self.notes_changed.emit()
                self.update()
                return
    
    def mouseMoveEvent(self, event):
        pos = event.pos()
        
        if self._drag_mode == 'move':
            self._do_move(pos)
        elif self._drag_mode in ('resize_start', 'resize_end'):
            self._do_resize(pos)
        elif self._drag_mode == 'velocity':
            self._do_velocity_drag(pos)
        elif self._marquee_start:
            self._marquee_rect = QRect(self._marquee_start, pos).normalized()
            self._update_marquee_selection()
            self.update()
        else:
            # Update cursor based on position
            self._update_cursor(pos)
    
    def mouseReleaseEvent(self, event):
        if self._drag_mode:
            self._drag_mode = None
            self._drag_start_pos = None
            self._drag_start_notes.clear()
            self.notes_changed.emit()
        
        if self._marquee_start:
            self._marquee_start = None
            self._marquee_rect = None
            self.selection_changed.emit()
        
        self.update()
    
    def _update_cursor(self, pos: QPointF):
        """Update cursor based on hover position"""
        hit_note, region = self._note_hit_test(pos)
        
        if hit_note and self.current_tool == PianoRollTool.POINTER:
            if region in ('left', 'right'):
                self.setCursor(Qt.SizeHorCursor)
            else:
                self.setCursor(Qt.SizeAllCursor)
        else:
            self.setCursor(Qt.ArrowCursor)
    
    # ─────────────────────────────────────────────────────────────────────────
    # DRAG OPERATIONS
    # ─────────────────────────────────────────────────────────────────────────
    
    def _start_move(self, pos: QPointF):
        """Start moving selected notes"""
        self._drag_mode = 'move'
        self._drag_start_pos = pos
        self._drag_start_notes = {n: (n.start, n.note) for n in self.selected_notes}
    
    def _do_move(self, pos: QPointF):
        """Perform move operation"""
        if not self._drag_start_pos or not self._drag_start_notes:
            return
        
        delta_x = pos.x() - self._drag_start_pos.x()
        delta_y = pos.y() - self._drag_start_pos.y()
        
        delta_beats = delta_x / self.beat_width
        delta_notes = -int(delta_y / self.note_height)
        
        # Apply snap if enabled (unless Alt is held)
        no_snap = QApplication.keyboardModifiers() & Qt.AltModifier
        
        for note, (orig_start, orig_pitch) in self._drag_start_notes.items():
            new_start = orig_start + delta_beats
            new_pitch = orig_pitch + delta_notes
            
            if self.snap_enabled and not no_snap:
                new_start = snap_to_grid(new_start, self.snap_size)
            
            # Snap to scale if enabled
            if self.snap_to_scale and self.scale:
                new_pitch = self.scale.snap_to_scale(new_pitch)
            
            note.start = max(0, new_start)
            note.note = max(0, min(127, new_pitch))
        
        self.update()
    
    def _start_resize(self, note: NoteData, mode: str, pos: QPointF):
        """Start resizing a note"""
        self._drag_mode = mode
        self._drag_start_pos = pos
        
        if note not in self.selected_notes:
            self.selected_notes = {note}
        
        self._drag_start_notes = {n: (n.start, n.duration) for n in self.selected_notes}
    
    def _do_resize(self, pos: QPointF):
        """Perform resize operation"""
        if not self._drag_start_pos:
            return
        
        delta_x = pos.x() - self._drag_start_pos.x()
        delta_beats = delta_x / self.beat_width
        
        no_snap = QApplication.keyboardModifiers() & Qt.AltModifier
        
        for note, (orig_start, orig_dur) in self._drag_start_notes.items():
            if self._drag_mode == 'resize_start':
                new_start = orig_start + delta_beats
                new_dur = orig_dur - delta_beats
                
                if self.snap_enabled and not no_snap:
                    new_start = snap_to_grid(new_start, self.snap_size)
                    new_dur = (orig_start + orig_dur) - new_start
                
                if new_dur >= self.snap_size:
                    note.start = max(0, new_start)
                    note.duration = new_dur
            
            elif self._drag_mode == 'resize_end':
                new_dur = orig_dur + delta_beats
                
                if self.snap_enabled and not no_snap:
                    new_end = snap_to_grid(orig_start + new_dur, self.snap_size)
                    new_dur = new_end - orig_start
                
                note.duration = max(self.snap_size, new_dur)
        
        self.update()
    
    def _do_velocity_drag(self, pos: QPointF):
        """Adjust velocity by dragging"""
        velocity_top = self.total_notes * self.note_height
        rel_y = pos.y() - velocity_top
        new_vel = 1.0 - (rel_y / self.velocity_lane_height)
        new_vel = max(0.0, min(1.0, new_vel))
        
        beat = self._beat_at_x(pos.x())
        for note in self.notes:
            if note.start <= beat < note.end:
                note.velocity = new_vel
        
        self.update()
    
    # ─────────────────────────────────────────────────────────────────────────
    # SELECTION
    # ─────────────────────────────────────────────────────────────────────────
    
    def _select_note(self, note: NoteData, additive: bool = False):
        """Select a single note"""
        if not additive:
            self.selected_notes.clear()
        
        if note in self.selected_notes:
            self.selected_notes.discard(note)
        else:
            self.selected_notes.add(note)
        
        # Audition on select
        self._audition_note(note.note)
        
        self.selection_changed.emit()
        self.update()
    
    def _update_marquee_selection(self):
        """Update selection based on marquee rectangle"""
        if not self._marquee_rect:
            return
        
        rect = self._marquee_rect
        
        for note in self.notes:
            if note.note < self.start_note or note.note >= self.start_note + self.total_notes:
                continue
            
            row = self.total_notes - 1 - (note.note - self.start_note)
            x = self.key_width + note.start * self.beat_width
            y = row * self.note_height
            w = note.duration * self.beat_width
            h = self.note_height
            
            note_rect = QRect(int(x), int(y), int(w), int(h))
            
            if rect.intersects(note_rect):
                self.selected_notes.add(note)
            elif not (QApplication.keyboardModifiers() & Qt.ControlModifier):
                self.selected_notes.discard(note)
    
    def select_all(self):
        """Select all notes"""
        self.selected_notes = set(self.notes)
        self.selection_changed.emit()
        self.update()
    
    def deselect_all(self):
        """Deselect all notes"""
        self.selected_notes.clear()
        self.selection_changed.emit()
        self.update()
    
    # ─────────────────────────────────────────────────────────────────────────
    # NOTE OPERATIONS
    # ─────────────────────────────────────────────────────────────────────────
    
    def _draw_note_at(self, beat: float, pitch: int):
        """Draw a new note at position"""
        if self.snap_enabled:
            beat = snap_to_grid(beat, self.snap_size)
        
        if self.snap_to_scale and self.scale:
            pitch = self.scale.snap_to_scale(pitch)
        
        note = NoteData(
            note=pitch,
            start=beat,
            duration=self.draw_duration,
            velocity=self.draw_velocity
        )
        self.notes.append(note)
        self.selected_notes = {note}
        
        # Audition
        self._audition_note(pitch)
        
        self.note_added.emit(note)
        self.notes_changed.emit()
        self.update()
    
    def _delete_note(self, note: NoteData):
        """Delete a single note"""
        if note in self.notes:
            self.notes.remove(note)
            self.selected_notes.discard(note)
            self.note_removed.emit(note)
            self.notes_changed.emit()
            self.update()
    
    def _toggle_mute(self, note: NoteData):
        """Toggle mute state of a note"""
        note.muted = not note.muted
        self.notes_changed.emit()
        self.update()
    
    def delete_selected(self):
        """Delete all selected notes"""
        for note in list(self.selected_notes):
            if note in self.notes:
                self.notes.remove(note)
                self.note_removed.emit(note)
        self.selected_notes.clear()
        self.notes_changed.emit()
        self.selection_changed.emit()
        self.update()
    
    def copy_selected(self):
        """Copy selected notes to clipboard"""
        self._clipboard = [NoteData.from_dict(n.to_dict()) for n in self.selected_notes]
    
    def cut_selected(self):
        """Cut selected notes"""
        self.copy_selected()
        self.delete_selected()
    
    def paste(self):
        """Paste notes from clipboard"""
        if not self._clipboard:
            return
        
        # Find the earliest note in clipboard
        min_start = min(n.start for n in self._clipboard)
        
        # Paste at playhead position (or beat 0 if no playhead)
        playhead_beat = getattr(self, 'playhead_beat', 0.0)
        paste_offset = playhead_beat
        
        new_notes = []
        for clip_note in self._clipboard:
            new_note = NoteData.from_dict(clip_note.to_dict())
            new_note.start = clip_note.start - min_start + paste_offset
            new_notes.append(new_note)
            self.notes.append(new_note)
        
        self.selected_notes = set(new_notes)
        self.notes_changed.emit()
        self.selection_changed.emit()
        self.update()
    
    def duplicate_selected(self):
        """Duplicate selected notes (Ctrl+D)"""
        if not self.selected_notes:
            return
        
        # Find the span of selected notes
        min_start = min(n.start for n in self.selected_notes)
        max_end = max(n.end for n in self.selected_notes)
        span = max_end - min_start
        
        new_notes = []
        for note in self.selected_notes:
            new_note = NoteData.from_dict(note.to_dict())
            new_note.start = note.start + span
            new_notes.append(new_note)
            self.notes.append(new_note)
        
        self.selected_notes = set(new_notes)
        self.notes_changed.emit()
        self.selection_changed.emit()
        self.update()
    
    def nudge_selected(self, delta_beats: float, delta_pitch: int):
        """Nudge selected notes by amount"""
        for note in self.selected_notes:
            note.start = max(0, note.start + delta_beats)
            note.note = max(0, min(127, note.note + delta_pitch))
        
        self.notes_changed.emit()
        self.update()
    
    def adjust_length(self, delta_beats: float):
        """Adjust length of selected notes"""
        for note in self.selected_notes:
            new_dur = note.duration + delta_beats
            note.duration = max(self.snap_size, new_dur)
        
        self.notes_changed.emit()
        self.update()
    
    def quantize_selected(self, ends_only: bool = False):
        """Quantize selected notes to grid"""
        for note in self.selected_notes:
            if not ends_only:
                note.start = snap_to_grid(note.start, self.snap_size)
            
            end = note.start + note.duration
            quantized_end = snap_to_grid(end, self.snap_size)
            note.duration = max(self.snap_size, quantized_end - note.start)
        
        self.notes_changed.emit()
        self.update()
    
    # ─────────────────────────────────────────────────────────────────────────
    # HIT TESTING
    # ─────────────────────────────────────────────────────────────────────────
    
    def _note_hit_test(self, pos: QPointF) -> Tuple[Optional[NoteData], Optional[str]]:
        """Test if position hits a note and which region"""
        for note in reversed(self.notes):  # Test in reverse draw order
            if note.note < self.start_note or note.note >= self.start_note + self.total_notes:
                continue
            
            row = self.total_notes - 1 - (note.note - self.start_note)
            x = self.key_width + note.start * self.beat_width
            y = row * self.note_height
            w = note.duration * self.beat_width
            h = self.note_height
            
            if y <= pos.y() <= y + h and x <= pos.x() <= x + w:
                # Determine region
                handle_width = min(8, w * 0.25)
                
                if pos.x() <= x + handle_width:
                    return note, 'left'
                elif pos.x() >= x + w - handle_width:
                    return note, 'right'
                else:
                    return note, 'body'
        
        return None, None
    
    def _beat_at_x(self, x: float) -> float:
        """Get beat position at x coordinate"""
        return (x - self.key_width) / self.beat_width
    
    def _note_at_y(self, y: float) -> Optional[int]:
        """Get MIDI note at y coordinate"""
        row = int(y / self.note_height)
        if 0 <= row < self.total_notes:
            return self.start_note + (self.total_notes - 1 - row)
        return None
    
    # ─────────────────────────────────────────────────────────────────────────
    # AUDITION
    # ─────────────────────────────────────────────────────────────────────────
    
    def _audition_note(self, pitch: int):
        """Play a note preview"""
        if pitch == self._last_audition_note:
            return
        
        self._last_audition_note = pitch
        
        if CORE_AVAILABLE:
            audition = get_audition_engine()
            audition.audition_note(pitch, velocity=0.8)
    
    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────────────────────
    
    def add_note(self, note: int, start_beat: float, duration: float, velocity: float = 0.8):
        """Programmatically add a note"""
        note_data = NoteData(note=note, start=start_beat, duration=duration, velocity=velocity)
        self.notes.append(note_data)
        self.note_added.emit(note_data)
        self.update()
    
    def clear(self):
        """Clear all notes"""
        self.notes.clear()
        self.selected_notes.clear()
        self.notes_changed.emit()
        self.update()
    
    def get_notes(self) -> List[Dict]:
        """Get all notes as dicts"""
        return [n.to_dict() for n in self.notes]
    
    def set_tool(self, tool: PianoRollTool):
        """Set the current editing tool"""
        self.current_tool = tool
        self.update()


# ═══════════════════════════════════════════════════════════════════════════════
# PIANO ROLL TOOLBAR
# ═══════════════════════════════════════════════════════════════════════════════

class PianoRollToolbar(QWidget):
    """
    Toolbar for piano roll with:
    - Tool selection (pointer, draw, paint, erase, mute)
    - Snap settings
    - Scale/key selection
    - Quantize buttons
    """
    
    tool_changed = pyqtSignal(object)  # PianoRollTool
    snap_changed = pyqtSignal(float)
    scale_changed = pyqtSignal(object)  # Scale or None
    
    def __init__(self, piano_roll: PianoRollView, parent=None):
        super().__init__(parent)
        self.piano_roll = piano_roll
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)
        
        # Tool buttons
        tool_group = QHBoxLayout()
        tool_group.setSpacing(2)
        
        self.tool_buttons: Dict[PianoRollTool, QPushButton] = {}
        
        tools = [
            (PianoRollTool.POINTER, "▲", "Pointer (V) - Select and move notes"),
            (PianoRollTool.DRAW, "✏", "Draw (D) - Add notes"),
            (PianoRollTool.PAINT, "🖌", "Paint (P) - Paint notes"),
            (PianoRollTool.ERASE, "⌫", "Erase (E) - Delete notes"),
            (PianoRollTool.MUTE, "🔇", "Mute (M) - Toggle note mute"),
        ]
        
        for tool, icon, tooltip in tools:
            btn = QPushButton(icon)
            btn.setFixedSize(28, 28)
            btn.setCheckable(True)
            btn.setToolTip(tooltip)
            btn.clicked.connect(lambda checked, t=tool: self._set_tool(t))
            tool_group.addWidget(btn)
            self.tool_buttons[tool] = btn
        
        # Set initial tool
        self.tool_buttons[PianoRollTool.POINTER].setChecked(True)
        
        layout.addLayout(tool_group)
        
        # Separator
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.VLine)
        sep1.setStyleSheet("color: #444;")
        layout.addWidget(sep1)
        
        # Snap settings
        snap_layout = QHBoxLayout()
        snap_layout.setSpacing(4)
        
        self.snap_checkbox = QCheckBox("Snap")
        self.snap_checkbox.setChecked(True)
        self.snap_checkbox.toggled.connect(self._snap_toggled)
        snap_layout.addWidget(self.snap_checkbox)
        
        self.snap_combo = QComboBox()
        self.snap_combo.addItems(["1/1", "1/2", "1/4", "1/8", "1/16", "1/32"])
        self.snap_combo.setCurrentText("1/16")
        self.snap_combo.currentTextChanged.connect(self._snap_size_changed)
        self.snap_combo.setFixedWidth(60)
        snap_layout.addWidget(self.snap_combo)
        
        layout.addLayout(snap_layout)
        
        # Separator
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.VLine)
        sep2.setStyleSheet("color: #444;")
        layout.addWidget(sep2)
        
        # Scale/Key selection
        scale_layout = QHBoxLayout()
        scale_layout.setSpacing(4)
        
        scale_layout.addWidget(QLabel("Key:"))
        
        self.root_combo = QComboBox()
        self.root_combo.addItems(NOTE_NAMES)
        self.root_combo.setCurrentText("C")
        self.root_combo.currentTextChanged.connect(self._scale_updated)
        self.root_combo.setFixedWidth(50)
        scale_layout.addWidget(self.root_combo)
        
        self.scale_combo = QComboBox()
        scale_items = ["None"] + [st.value for st in ScaleType] if CORE_AVAILABLE else ["None"]
        self.scale_combo.addItems(scale_items)
        self.scale_combo.currentTextChanged.connect(self._scale_updated)
        self.scale_combo.setFixedWidth(100)
        scale_layout.addWidget(self.scale_combo)
        
        self.scale_highlight_btn = QPushButton("🎹")
        self.scale_highlight_btn.setFixedSize(28, 28)
        self.scale_highlight_btn.setCheckable(True)
        self.scale_highlight_btn.setChecked(True)
        self.scale_highlight_btn.setToolTip("Show scale highlighting (K)")
        self.scale_highlight_btn.toggled.connect(self._scale_highlight_toggled)
        scale_layout.addWidget(self.scale_highlight_btn)
        
        self.snap_to_scale_btn = QPushButton("⟷")
        self.snap_to_scale_btn.setFixedSize(28, 28)
        self.snap_to_scale_btn.setCheckable(True)
        self.snap_to_scale_btn.setToolTip("Snap to scale notes")
        self.snap_to_scale_btn.toggled.connect(self._snap_to_scale_toggled)
        scale_layout.addWidget(self.snap_to_scale_btn)
        
        layout.addLayout(scale_layout)
        
        # Separator
        sep3 = QFrame()
        sep3.setFrameShape(QFrame.VLine)
        sep3.setStyleSheet("color: #444;")
        layout.addWidget(sep3)
        
        # Quantize buttons
        quantize_layout = QHBoxLayout()
        quantize_layout.setSpacing(4)
        
        quantize_btn = QPushButton("Q")
        quantize_btn.setFixedSize(28, 28)
        quantize_btn.setToolTip("Quantize selected (Q)")
        quantize_btn.clicked.connect(self.piano_roll.quantize_selected)
        quantize_layout.addWidget(quantize_btn)
        
        quantize_ends_btn = QPushButton("Q⟶")
        quantize_ends_btn.setFixedWidth(36)
        quantize_ends_btn.setToolTip("Quantize ends only (Shift+Q)")
        quantize_ends_btn.clicked.connect(lambda: self.piano_roll.quantize_selected(ends_only=True))
        quantize_layout.addWidget(quantize_ends_btn)
        
        layout.addLayout(quantize_layout)
        
        # Stretch
        layout.addStretch()
        
        # Selection info
        self.selection_label = QLabel("0 notes")
        self.selection_label.setStyleSheet("color: #888;")
        layout.addWidget(self.selection_label)
        
        # Connect to piano roll
        self.piano_roll.selection_changed.connect(self._update_selection_info)
        
        # Styling
        self.setStyleSheet("""
            PianoRollToolbar {
                background: #252525;
                border-bottom: 1px solid #333;
            }
            QPushButton {
                background: #333;
                border: 1px solid #444;
                border-radius: 3px;
                color: #ccc;
            }
            QPushButton:checked {
                background: #00aaff;
                color: #fff;
            }
            QPushButton:hover {
                background: #444;
            }
            QComboBox {
                background: #333;
                border: 1px solid #444;
                color: #ccc;
                padding: 2px 4px;
            }
            QCheckBox {
                color: #aaa;
            }
            QLabel {
                color: #aaa;
            }
        """)
        
        # Setup shortcuts
        self._setup_shortcuts()
    
    def _setup_shortcuts(self):
        """Tool shortcuts"""
        shortcuts = [
            ("V", PianoRollTool.POINTER),
            ("D", PianoRollTool.DRAW),
            ("P", PianoRollTool.PAINT),
            ("E", PianoRollTool.ERASE),
            ("M", PianoRollTool.MUTE),
        ]
        
        for key, tool in shortcuts:
            shortcut = QShortcut(QKeySequence(key), self.piano_roll)
            shortcut.activated.connect(lambda t=tool: self._set_tool(t))
    
    def _set_tool(self, tool: PianoRollTool):
        """Set the active tool"""
        for t, btn in self.tool_buttons.items():
            btn.setChecked(t == tool)
        
        self.piano_roll.set_tool(tool)
        self.tool_changed.emit(tool)
    
    def _snap_toggled(self, enabled: bool):
        """Handle snap toggle"""
        self.piano_roll.snap_enabled = enabled
        self.snap_combo.setEnabled(enabled)
    
    def _snap_size_changed(self, text: str):
        """Handle snap size change"""
        snap_map = {
            "1/1": 4.0,
            "1/2": 2.0,
            "1/4": 1.0,
            "1/8": 0.5,
            "1/16": 0.25,
            "1/32": 0.125,
        }
        snap_size = snap_map.get(text, 0.25)
        self.piano_roll.snap_size = snap_size
        self.snap_changed.emit(snap_size)
    
    def _scale_updated(self):
        """Handle scale selection change"""
        if not CORE_AVAILABLE:
            return
        
        scale_name = self.scale_combo.currentText()
        
        if scale_name == "None":
            self.piano_roll.set_scale(None)
            self.scale_changed.emit(None)
            return
        
        root_name = self.root_combo.currentText()
        root = NOTE_NAMES.index(root_name)
        
        try:
            scale_type = ScaleType(scale_name)
            scale = Scale(root, scale_type)
            self.piano_roll.set_scale(scale)
            self.scale_changed.emit(scale)
        except ValueError:
            pass
    
    def _scale_highlight_toggled(self, enabled: bool):
        """Toggle scale highlighting"""
        self.piano_roll.show_scale_highlighting = enabled
        self.piano_roll.update()
    
    def _snap_to_scale_toggled(self, enabled: bool):
        """Toggle snap to scale"""
        self.piano_roll.snap_to_scale = enabled
    
    def _update_selection_info(self):
        """Update selection count label"""
        count = len(self.piano_roll.selected_notes)
        if count == 0:
            self.selection_label.setText("0 notes")
        elif count == 1:
            note = list(self.piano_roll.selected_notes)[0]
            self.selection_label.setText(f"1 note: {note_to_name(note.note)}")
        else:
            self.selection_label.setText(f"{count} notes")


# ═══════════════════════════════════════════════════════════════════════════════
# INSTRUMENT BROWSER
# ═══════════════════════════════════════════════════════════════════════════════

class InstrumentBrowser(QWidget):
    """
    Browse and select instruments/presets:
    - Category tree (Lead, Bass, Pad, etc.)
    - Preset list with preview
    - Drag-drop to tracks
    """
    
    preset_selected = pyqtSignal(object)  # InstrumentPreset
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_preset: Optional[InstrumentPreset] = None
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        # Header
        header = QLabel("🎹 Instruments")
        header.setStyleSheet("color: #00aaff; font-weight: bold; font-size: 12px;")
        layout.addWidget(header)
        
        # Search
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search presets...")
        self.search.setStyleSheet("""
            QLineEdit {
                background: #2a2a2a;
                border: 1px solid #3a3a3a;
                border-radius: 4px;
                padding: 6px;
                color: #e0e0e0;
            }
            QLineEdit:focus {
                border-color: #00aaff;
            }
        """)
        self.search.textChanged.connect(self.filter_presets)
        layout.addWidget(self.search)
        
        # Category tabs
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #3a3a3a;
                background: #1a1a1a;
            }
            QTabBar::tab {
                background: #2a2a2a;
                color: #888;
                padding: 6px 12px;
                border: 1px solid #3a3a3a;
                border-bottom: none;
            }
            QTabBar::tab:selected {
                background: #1a1a1a;
                color: #00aaff;
            }
        """)
        
        # Add category tabs
        for category in PRESET_LIBRARY.keys():
            list_widget = QListWidget()
            list_widget.setStyleSheet("""
                QListWidget {
                    background: #1a1a1a;
                    border: none;
                    color: #e0e0e0;
                }
                QListWidget::item {
                    padding: 8px;
                    border-bottom: 1px solid #2a2a2a;
                }
                QListWidget::item:selected {
                    background: #00aaff;
                    color: #1a1a1a;
                }
                QListWidget::item:hover {
                    background: #2a2a2a;
                }
            """)
            
            # Add presets
            for preset_name, preset in PRESET_LIBRARY[category].items():
                item = QListWidgetItem(preset_name)
                item.setData(Qt.UserRole, preset)
                list_widget.addItem(item)
            
            list_widget.itemClicked.connect(self.on_preset_clicked)
            list_widget.itemDoubleClicked.connect(self.on_preset_double_clicked)
            
            self.tabs.addTab(list_widget, category)
        
        layout.addWidget(self.tabs)
        
        # Preview section
        preview_frame = QFrame()
        preview_frame.setStyleSheet("background: #222222; border-radius: 8px;")
        preview_layout = QVBoxLayout(preview_frame)
        preview_layout.setContentsMargins(8, 8, 8, 8)
        
        self.preview_name = QLabel("Select a preset")
        self.preview_name.setStyleSheet("color: #00ffaa; font-weight: bold;")
        preview_layout.addWidget(self.preview_name)
        
        self.preview_info = QLabel("")
        self.preview_info.setStyleSheet("color: #888; font-size: 10px;")
        self.preview_info.setWordWrap(True)
        preview_layout.addWidget(self.preview_info)
        
        self.load_btn = QPushButton("Load Preset")
        self.load_btn.setStyleSheet("""
            QPushButton {
                background: #00aaff;
                color: #1a1a1a;
                border: none;
                border-radius: 4px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #00ccff;
            }
        """)
        self.load_btn.clicked.connect(self.load_selected_preset)
        self.load_btn.setEnabled(False)
        preview_layout.addWidget(self.load_btn)
        
        layout.addWidget(preview_frame)
    
    def on_preset_clicked(self, item: QListWidgetItem):
        preset = item.data(Qt.UserRole)
        self.current_preset = preset
        
        self.preview_name.setText(preset.name)
        
        # Build info string
        osc_info = f"OSC1: {preset.osc1.shape.value}"
        if preset.osc2.level > 0:
            osc_info += f" | OSC2: {preset.osc2.shape.value}"
        
        filter_info = f"Filter: {preset.filter_type.value} @ {preset.filter_cutoff}Hz"
        env_info = f"ADSR: {preset.amp_env.attack:.2f}s / {preset.amp_env.decay:.2f}s / {preset.amp_env.sustain:.0%} / {preset.amp_env.release:.2f}s"
        
        self.preview_info.setText(f"{osc_info}\n{filter_info}\n{env_info}")
        self.load_btn.setEnabled(True)
    
    def on_preset_double_clicked(self, item: QListWidgetItem):
        self.on_preset_clicked(item)
        self.load_selected_preset()
    
    def load_selected_preset(self):
        if self.current_preset:
            self.preset_selected.emit(self.current_preset)
    
    def filter_presets(self, text: str):
        """Filter presets by search text"""
        text = text.lower()
        
        for i in range(self.tabs.count()):
            list_widget = self.tabs.widget(i)
            for j in range(list_widget.count()):
                item = list_widget.item(j)
                matches = text in item.text().lower()
                item.setHidden(not matches)


# ═══════════════════════════════════════════════════════════════════════════════
# SYNTH PARAMETER PANEL
# ═══════════════════════════════════════════════════════════════════════════════

class SynthPanel(QWidget):
    """
    Synth parameter editor:
    - Oscillator controls
    - Filter controls  
    - Envelope controls
    - Real-time parameter changes
    """
    
    parameter_changed = pyqtSignal(str, float)  # param_name, value
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        
        # Header
        header = QLabel("🎛️ Synthesizer")
        header.setStyleSheet("color: #00aaff; font-weight: bold; font-size: 12px;")
        layout.addWidget(header)
        
        # Oscillator section
        osc_frame = self._create_section("Oscillators")
        osc_layout = QHBoxLayout()
        
        for i, name in enumerate(["OSC 1", "OSC 2", "OSC 3"]):
            osc_widget = self._create_osc_panel(name, i)
            osc_layout.addWidget(osc_widget)
        
        osc_frame.layout().addLayout(osc_layout)
        layout.addWidget(osc_frame)
        
        # Filter section
        filter_frame = self._create_section("Filter")
        filter_layout = QHBoxLayout()
        
        # Cutoff
        cutoff_layout = QVBoxLayout()
        cutoff_label = QLabel("Cutoff")
        cutoff_label.setStyleSheet("color: #888; font-size: 9px;")
        cutoff_label.setAlignment(Qt.AlignCenter)
        self.cutoff_slider = QSlider(Qt.Vertical)
        self.cutoff_slider.setRange(20, 20000)
        self.cutoff_slider.setValue(5000)
        self.cutoff_slider.setMinimumHeight(80)
        cutoff_layout.addWidget(cutoff_label)
        cutoff_layout.addWidget(self.cutoff_slider, 0, Qt.AlignHCenter)
        filter_layout.addLayout(cutoff_layout)
        
        # Resonance
        res_layout = QVBoxLayout()
        res_label = QLabel("Res")
        res_label.setStyleSheet("color: #888; font-size: 9px;")
        res_label.setAlignment(Qt.AlignCenter)
        self.res_slider = QSlider(Qt.Vertical)
        self.res_slider.setRange(0, 100)
        self.res_slider.setValue(30)
        self.res_slider.setMinimumHeight(80)
        res_layout.addWidget(res_label)
        res_layout.addWidget(self.res_slider, 0, Qt.AlignHCenter)
        filter_layout.addLayout(res_layout)
        
        # Filter type
        type_layout = QVBoxLayout()
        type_label = QLabel("Type")
        type_label.setStyleSheet("color: #888; font-size: 9px;")
        self.filter_type = QComboBox()
        self.filter_type.addItems(["LP", "HP", "BP", "Φ-Res"])
        self.filter_type.setStyleSheet("""
            QComboBox {
                background: #2a2a2a;
                border: 1px solid #3a3a3a;
                border-radius: 4px;
                padding: 4px;
                color: #e0e0e0;
            }
        """)
        type_layout.addWidget(type_label)
        type_layout.addWidget(self.filter_type)
        type_layout.addStretch()
        filter_layout.addLayout(type_layout)
        
        filter_frame.layout().addLayout(filter_layout)
        layout.addWidget(filter_frame)
        
        # Envelope section
        env_frame = self._create_section("Amp Envelope")
        env_layout = QHBoxLayout()
        
        for name in ["A", "D", "S", "R"]:
            env_widget = self._create_env_slider(name)
            env_layout.addLayout(env_widget)
        
        env_frame.layout().addLayout(env_layout)
        layout.addWidget(env_frame)
        
        layout.addStretch()
    
    def _create_section(self, title: str) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet("""
            QFrame {
                background: #222222;
                border-radius: 8px;
                margin: 2px;
            }
        """)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(8, 4, 8, 8)
        
        label = QLabel(title)
        label.setStyleSheet("color: #888; font-size: 10px; font-weight: bold;")
        layout.addWidget(label)
        
        return frame
    
    def _create_osc_panel(self, name: str, index: int) -> QFrame:
        frame = QFrame()
        frame.setFixedWidth(80)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        
        # Name
        label = QLabel(name)
        label.setStyleSheet("color: #00aaff; font-size: 9px; font-weight: bold;")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
        
        # Waveform selector
        wave_combo = QComboBox()
        wave_combo.addItems(["Sin", "Saw", "Sqr", "Tri", "Φ"])
        wave_combo.setStyleSheet("""
            QComboBox {
                background: #2a2a2a;
                border: 1px solid #3a3a3a;
                border-radius: 4px;
                padding: 2px;
                color: #e0e0e0;
                font-size: 9px;
            }
        """)
        layout.addWidget(wave_combo)
        
        # Level slider
        level_slider = QSlider(Qt.Vertical)
        level_slider.setRange(0, 100)
        level_slider.setValue(100 if index == 0 else 50)
        level_slider.setMinimumHeight(50)
        layout.addWidget(level_slider, 0, Qt.AlignHCenter)
        
        level_label = QLabel("Level")
        level_label.setStyleSheet("color: #666; font-size: 8px;")
        level_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(level_label)
        
        return frame
    
    def _create_env_slider(self, name: str) -> QVBoxLayout:
        layout = QVBoxLayout()
        
        label = QLabel(name)
        label.setStyleSheet("color: #888; font-size: 10px; font-weight: bold;")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
        
        slider = QSlider(Qt.Vertical)
        slider.setRange(0, 100)
        
        # Default values
        defaults = {"A": 5, "D": 30, "S": 70, "R": 40}
        slider.setValue(defaults.get(name, 50))
        slider.setMinimumHeight(60)
        layout.addWidget(slider, 0, Qt.AlignHCenter)
        
        return layout


# ═══════════════════════════════════════════════════════════════════════════════
# COMBINED INSTRUMENT EDITOR
# ═══════════════════════════════════════════════════════════════════════════════

class InstrumentEditor(QWidget):
    """
    Combined view with:
    - Piano keyboard at bottom
    - Piano roll in center with toolbar
    - Instrument browser on right
    - Synth controls on left
    """
    
    def __init__(self, synth: Optional[PolySynth] = None, parent=None):
        super().__init__(parent)
        self.synth = synth or PolySynth()
        self.setup_ui()
        self.connect_signals()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Main splitter
        splitter = QSplitter(Qt.Horizontal)
        
        # Left: Synth panel
        self.synth_panel = SynthPanel()
        self.synth_panel.setMaximumWidth(280)
        splitter.addWidget(self.synth_panel)
        
        # Center: Piano roll with toolbar
        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)
        
        # Piano roll in scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        scroll.setStyleSheet("background: #1a1a1a; border: none;")
        
        self.piano_roll = PianoRollView()
        scroll.setWidget(self.piano_roll)
        
        # Add toolbar ABOVE piano roll
        self.piano_roll_toolbar = PianoRollToolbar(self.piano_roll)
        center_layout.addWidget(self.piano_roll_toolbar)
        center_layout.addWidget(scroll, 1)
        
        # Piano keyboard at bottom
        self.keyboard = PianoKeyboard(octaves=4, start_octave=3)
        center_layout.addWidget(self.keyboard)
        
        splitter.addWidget(center)
        
        # Right: Instrument browser
        self.browser = InstrumentBrowser()
        self.browser.setMaximumWidth(250)
        splitter.addWidget(self.browser)
        
        splitter.setSizes([250, 600, 250])
        layout.addWidget(splitter)
    
    def connect_signals(self):
        # Keyboard -> synth
        self.keyboard.note_on.connect(self.on_note_on)
        self.keyboard.note_off.connect(self.on_note_off)
        
        # Browser -> synth
        self.browser.preset_selected.connect(self.on_preset_selected)
        
        # Piano roll -> keyboard (highlight active notes)
        self.piano_roll.notes_changed.connect(self._on_notes_changed)
    
    def on_note_on(self, note: int, velocity: float):
        if self.synth:
            self.synth.note_on(note, velocity)
    
    def on_note_off(self, note: int):
        if self.synth:
            self.synth.note_off(note)
    
    def on_preset_selected(self, preset: InstrumentPreset):
        if self.synth:
            self.synth.set_preset(preset)
            print(f"Loaded preset: {preset.name}")
    
    def _on_notes_changed(self):
        """Handle piano roll note changes"""
        # Could update keyboard display or trigger playback preview
        pass
    
    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────────────────────
    
    def add_note(self, note: int, start: float, duration: float, velocity: float = 0.8):
        """Add a note to the piano roll"""
        self.piano_roll.add_note(note, start, duration, velocity)
    
    def clear_notes(self):
        """Clear all notes"""
        self.piano_roll.clear()
    
    def get_notes(self) -> List[Dict]:
        """Get all notes as dicts"""
        return self.piano_roll.get_notes()
    
    def set_scale(self, scale: Optional['Scale']):
        """Set the active scale for highlighting/snapping"""
        self.piano_roll.set_scale(scale)
        if hasattr(self, 'piano_roll_toolbar'):
            # Sync toolbar state
            pass
