"""
Wave DAW Piano Roll & Instrument Browser Widgets

Provides:
- Interactive piano roll for MIDI editing
- Keyboard piano with visual feedback
- Instrument browser with drag-drop
"""

import sys
import numpy as np
from typing import Optional, List, Dict, Callable

try:
    from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
        QSlider, QScrollArea, QFrame, QTreeWidget, QTreeWidgetItem,
        QSplitter, QLineEdit, QComboBox, QGraphicsView, QGraphicsScene,
        QGraphicsRectItem, QTabWidget, QListWidget, QListWidgetItem
    )
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QRectF, QPointF, QSize
    from PyQt5.QtGui import (
        QColor, QPainter, QBrush, QPen, QFont, QLinearGradient,
        QKeyEvent, QPainterPath
    )
except ImportError:
    print("PyQt5 required")
    sys.exit(1)

from .synth_engine import (
    PRESET_LIBRARY, InstrumentPreset, PolySynth,
    key_to_note, note_to_name, SIMPLE_KEYBOARD_MAP
)


# ═══════════════════════════════════════════════════════════════════════════════
# STYLE
# ═══════════════════════════════════════════════════════════════════════════════

PIANO_WHITE = "#e8e8e8"
PIANO_BLACK = "#1a1a1a"
PIANO_PRESSED = "#00aaff"
PIANO_ACTIVE = "#00ffaa"
NOTE_COLOR = "#00aaff"
NOTE_SELECTED = "#00ffaa"


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
# PIANO ROLL (MIDI Editor)
# ═══════════════════════════════════════════════════════════════════════════════

class PianoRollView(QWidget):
    """
    MIDI note editor with:
    - Note grid (click to add/remove notes)
    - Velocity editing
    - Piano keyboard on left
    - Time ruler on top
    """
    
    note_added = pyqtSignal(int, float, float)    # note, start_beat, duration
    note_removed = pyqtSignal(int, float)         # note, start_beat
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.notes: List[Dict] = []  # {note, start, duration, velocity}
        
        self.note_height = 12
        self.beat_width = 40
        self.num_octaves = 4
        self.start_note = 36  # C2
        
        self.grid_beats = 64  # 16 bars of 4/4
        self.snap = 0.25  # Snap to 16th notes
        
        self.selected_note = None
        self.draw_mode = True
        
        self.setMinimumSize(800, 400)
        self.setFocusPolicy(Qt.StrongFocus)
    
    @property
    def total_notes(self):
        return self.num_octaves * 12
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Background
        painter.fillRect(self.rect(), QColor("#1a1a1a"))
        
        # Draw piano keys on left
        key_width = 40
        for i in range(self.total_notes):
            note = self.start_note + (self.total_notes - 1 - i)
            y = i * self.note_height
            
            is_black = (note % 12) in [1, 3, 6, 8, 10]
            color = QColor("#2a2a2a") if is_black else QColor("#3a3a3a")
            
            painter.fillRect(0, y, key_width, self.note_height - 1, color)
            
            # Note name on C notes
            if note % 12 == 0:
                painter.setPen(QColor("#888888"))
                painter.setFont(QFont("Segoe UI", 7))
                painter.drawText(4, y + self.note_height - 3, note_to_name(note))
        
        # Draw grid
        painter.setPen(QPen(QColor("#2a2a2a"), 1))
        
        for beat in range(self.grid_beats + 1):
            x = key_width + beat * self.beat_width
            
            # Stronger line on bar boundaries
            if beat % 4 == 0:
                painter.setPen(QPen(QColor("#444444"), 1))
            else:
                painter.setPen(QPen(QColor("#2a2a2a"), 1))
            
            painter.drawLine(x, 0, x, self.total_notes * self.note_height)
        
        # Horizontal lines
        for i in range(self.total_notes + 1):
            y = i * self.note_height
            note = self.start_note + (self.total_notes - 1 - i)
            
            if note % 12 == 0:
                painter.setPen(QPen(QColor("#444444"), 1))
            else:
                painter.setPen(QPen(QColor("#252525"), 1))
            
            painter.drawLine(key_width, y, self.width(), y)
        
        # Draw notes
        for note_data in self.notes:
            self._draw_note(painter, note_data, key_width)
    
    def _draw_note(self, painter: QPainter, note_data: Dict, key_offset: int):
        note = note_data['note']
        start = note_data['start']
        duration = note_data['duration']
        velocity = note_data.get('velocity', 0.8)
        
        if note < self.start_note or note >= self.start_note + self.total_notes:
            return
        
        row = self.total_notes - 1 - (note - self.start_note)
        x = key_offset + start * self.beat_width
        y = row * self.note_height
        w = duration * self.beat_width - 2
        h = self.note_height - 2
        
        # Color based on velocity
        alpha = int(150 + velocity * 105)
        color = QColor(0, 170, 255, alpha)
        
        if note_data == self.selected_note:
            color = QColor(0, 255, 170, alpha)
        
        painter.fillRect(int(x), int(y), int(w), int(h), color)
        painter.setPen(QPen(QColor("#ffffff"), 1))
        painter.drawRect(int(x), int(y), int(w), int(h))
    
    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        
        key_width = 40
        if event.x() < key_width:
            return
        
        # Calculate position
        beat = (event.x() - key_width) / self.beat_width
        beat = round(beat / self.snap) * self.snap  # Snap
        
        row = event.y() // self.note_height
        note = self.start_note + (self.total_notes - 1 - row)
        
        # Check if clicking existing note
        for note_data in self.notes:
            if (note_data['note'] == note and 
                note_data['start'] <= beat < note_data['start'] + note_data['duration']):
                if event.modifiers() & Qt.ShiftModifier:
                    # Delete note
                    self.notes.remove(note_data)
                    self.note_removed.emit(note, note_data['start'])
                else:
                    self.selected_note = note_data
                self.update()
                return
        
        # Add new note
        if self.draw_mode:
            new_note = {
                'note': note,
                'start': beat,
                'duration': self.snap * 4,  # Quarter note default
                'velocity': 0.8
            }
            self.notes.append(new_note)
            self.selected_note = new_note
            self.note_added.emit(note, beat, new_note['duration'])
            self.update()
    
    def add_note(self, note: int, start_beat: float, duration: float, velocity: float = 0.8):
        """Programmatically add a note"""
        self.notes.append({
            'note': note,
            'start': start_beat,
            'duration': duration,
            'velocity': velocity
        })
        self.update()
    
    def clear(self):
        """Clear all notes"""
        self.notes.clear()
        self.selected_note = None
        self.update()
    
    def get_notes(self) -> List[Dict]:
        """Get all notes"""
        return self.notes.copy()


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
    - Piano roll in center
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
        
        # Center: Piano roll
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
