#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
# This file is listed in CLAIMS_PRACTICING_FILES.txt and is licensed
# under LICENSE-CLAIMS-NC.md (research/education only). Commercial
# rights require a separate patent license from the author.
"""
QuantSoundDesign Core - Shared infrastructure for DAW components.

This module provides:
- KeymapRegistry: Centralized keyboard shortcut management
- SelectionModel: Unified selection handling across all views
- TimeUtils: Accurate beat/sample conversions
- AuditionEngine: Sound preview with debouncing

Per spec: Single source of truth, no ad-hoc implementations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Callable, Optional, Any, Tuple
from enum import Enum, auto
import time

try:
    from PyQt5.QtCore import Qt, QObject, pyqtSignal
    from PyQt5.QtWidgets import QShortcut, QWidget
    from PyQt5.QtGui import QKeySequence
    HAS_QT = True
except ImportError:
    HAS_QT = False
    QObject = object


# ═══════════════════════════════════════════════════════════════════════════════
# KEYMAP REGISTRY - Centralized shortcut management
# ═══════════════════════════════════════════════════════════════════════════════

class KeymapContext(Enum):
    """Contexts for keyboard shortcuts"""
    GLOBAL = auto()
    ARRANGEMENT = auto()
    CHANNEL_RACK = auto()
    PIANOROLL = auto()
    MIXER = auto()


@dataclass
class KeyBinding:
    """A single key binding definition"""
    key: str                          # e.g., "Ctrl+C", "F5", "Space"
    action_id: str                    # Unique action identifier
    description: str                  # Human-readable description
    context: KeymapContext            # Which context this applies to
    callback: Optional[Callable] = None
    enabled: bool = True


class KeymapRegistry:
    """
    Centralized keyboard shortcut registry.
    
    All shortcuts MUST be registered here - no hard-coding in components.
    Supports context-based activation and user customization.
    """
    
    # Default keymaps per spec
    DEFAULT_BINDINGS = {
        # Window toggles (GLOBAL)
        "view.playlist": ("F5", "Toggle Playlist/Arrangement", KeymapContext.GLOBAL),
        "view.channel_rack": ("F6", "Toggle Channel Rack", KeymapContext.GLOBAL),
        "view.piano_roll": ("F7", "Toggle Piano Roll", KeymapContext.GLOBAL),
        "view.plugin_picker": ("F8", "Open Plugin Picker", KeymapContext.GLOBAL),
        "view.mixer": ("F9", "Toggle Mixer", KeymapContext.GLOBAL),
        "view.settings": ("F10", "Open Settings", KeymapContext.GLOBAL),
        "view.project_info": ("F11", "Project Info", KeymapContext.GLOBAL),
        "view.close_all": ("F12", "Close All Floating Windows", KeymapContext.GLOBAL),
        
        # Transport (GLOBAL)
        "transport.play_pause": ("Space", "Play/Pause", KeymapContext.GLOBAL),
        "transport.stop": ("Return", "Stop & Return to Start", KeymapContext.GLOBAL),
        "transport.play_selection": ("Ctrl+Space", "Play from Selection", KeymapContext.GLOBAL),
        "transport.loop_toggle": ("Num0", "Toggle Loop", KeymapContext.GLOBAL),
        "transport.panic": ("Ctrl+.", "Panic - All Notes Off", KeymapContext.GLOBAL),
        
        # Navigation (GLOBAL)
        "nav.zoom_in": ("Ctrl++", "Zoom In", KeymapContext.GLOBAL),
        "nav.zoom_out": ("Ctrl+-", "Zoom Out", KeymapContext.GLOBAL),
        "nav.zoom_fit": ("Ctrl+0", "Zoom to Fit", KeymapContext.GLOBAL),
        "nav.zoom_selection": ("Shift+Z", "Zoom to Selection", KeymapContext.GLOBAL),
        
        # Edit (GLOBAL)
        "edit.undo": ("Ctrl+Z", "Undo", KeymapContext.GLOBAL),
        "edit.redo": ("Ctrl+Shift+Z", "Redo", KeymapContext.GLOBAL),
        "edit.cut": ("Ctrl+X", "Cut", KeymapContext.GLOBAL),
        "edit.copy": ("Ctrl+C", "Copy", KeymapContext.GLOBAL),
        "edit.paste": ("Ctrl+V", "Paste", KeymapContext.GLOBAL),
        "edit.delete": ("Delete", "Delete", KeymapContext.GLOBAL),
        "edit.select_all": ("Ctrl+A", "Select All", KeymapContext.GLOBAL),
        "edit.deselect": ("Escape", "Deselect All", KeymapContext.GLOBAL),
        
        # Piano Roll Tools
        "tool.pointer": ("1", "Pointer Tool", KeymapContext.PIANOROLL),
        "tool.draw": ("2", "Draw Tool", KeymapContext.PIANOROLL),
        "tool.paint": ("3", "Paint Tool", KeymapContext.PIANOROLL),
        "tool.slice": ("4", "Slice Tool", KeymapContext.PIANOROLL),
        "tool.mute": ("5", "Mute Tool", KeymapContext.PIANOROLL),
        "tool.zoom": ("Z", "Zoom Tool (Hold)", KeymapContext.PIANOROLL),
        
        # Piano Roll Operations
        "pianoroll.quantize": ("Q", "Quantize", KeymapContext.PIANOROLL),
        "pianoroll.quantize_ends": ("Shift+Q", "Quantize Ends", KeymapContext.PIANOROLL),
        "pianoroll.quantize_dialog": ("Alt+Q", "Quantize Options", KeymapContext.PIANOROLL),
        "pianoroll.scale_toggle": ("K", "Toggle Scale Highlighting", KeymapContext.PIANOROLL),
        "pianoroll.duplicate": ("Ctrl+D", "Duplicate Selection", KeymapContext.PIANOROLL),
        "pianoroll.repeat": ("Ctrl+B", "Repeat Selection", KeymapContext.PIANOROLL),
        
        # Note Movement
        "note.nudge_left": ("Left", "Nudge Left", KeymapContext.PIANOROLL),
        "note.nudge_right": ("Right", "Nudge Right", KeymapContext.PIANOROLL),
        "note.nudge_left_fine": ("Ctrl+Left", "Nudge Left (Fine)", KeymapContext.PIANOROLL),
        "note.nudge_right_fine": ("Ctrl+Right", "Nudge Right (Fine)", KeymapContext.PIANOROLL),
        "note.transpose_up": ("Up", "Transpose Up", KeymapContext.PIANOROLL),
        "note.transpose_down": ("Down", "Transpose Down", KeymapContext.PIANOROLL),
        "note.transpose_octave_up": ("Shift+Up", "Transpose Octave Up", KeymapContext.PIANOROLL),
        "note.transpose_octave_down": ("Shift+Down", "Transpose Octave Down", KeymapContext.PIANOROLL),
        "note.lengthen": ("Shift+Right", "Lengthen Note", KeymapContext.PIANOROLL),
        "note.shorten": ("Shift+Left", "Shorten Note", KeymapContext.PIANOROLL),
        
        # Arrangement
        "arrangement.select_all": ("Ctrl+A", "Select All Clips", KeymapContext.ARRANGEMENT),
        "arrangement.repeat": ("Ctrl+B", "Repeat Selection", KeymapContext.ARRANGEMENT),
        
        # Channel Rack
        "channel.move_up": ("Ctrl+Up", "Move Channel Up", KeymapContext.CHANNEL_RACK),
        "channel.move_down": ("Ctrl+Down", "Move Channel Down", KeymapContext.CHANNEL_RACK),
        "channel.rename": ("F2", "Rename Channel", KeymapContext.CHANNEL_RACK),
    }
    
    def __init__(self):
        self.bindings: Dict[str, KeyBinding] = {}
        self.active_context: KeymapContext = KeymapContext.GLOBAL
        self.shortcuts: Dict[str, QShortcut] = {} if HAS_QT else {}
        self._load_defaults()
    
    def _load_defaults(self):
        """Load default keybindings"""
        for action_id, (key, desc, context) in self.DEFAULT_BINDINGS.items():
            self.register(action_id, key, desc, context)
    
    def register(self, action_id: str, key: str, description: str, 
                 context: KeymapContext, callback: Callable = None):
        """Register a keyboard shortcut"""
        self.bindings[action_id] = KeyBinding(
            key=key,
            action_id=action_id,
            description=description,
            context=context,
            callback=callback
        )
    
    def bind(self, action_id: str, callback: Callable):
        """Bind a callback to an existing action"""
        if action_id in self.bindings:
            self.bindings[action_id].callback = callback
    
    def set_context(self, context: KeymapContext):
        """Set the active context for shortcut handling"""
        self.active_context = context
    
    def get_binding(self, action_id: str) -> Optional[KeyBinding]:
        """Get a binding by action ID"""
        return self.bindings.get(action_id)
    
    def get_key(self, action_id: str) -> str:
        """Get the key for an action"""
        binding = self.bindings.get(action_id)
        return binding.key if binding else ""
    
    def execute(self, action_id: str) -> bool:
        """Execute an action by ID"""
        binding = self.bindings.get(action_id)
        if binding and binding.callback and binding.enabled:
            binding.callback()
            return True
        return False
    
    def get_actions_for_context(self, context: KeymapContext) -> List[KeyBinding]:
        """Get all bindings for a context"""
        return [b for b in self.bindings.values() if b.context == context]
    
    def setup_shortcuts(self, widget: 'QWidget'):
        """Set up Qt shortcuts for a widget"""
        if not HAS_QT:
            return
        
        for action_id, binding in self.bindings.items():
            if binding.callback:
                shortcut = QShortcut(QKeySequence(binding.key), widget)
                shortcut.activated.connect(binding.callback)
                self.shortcuts[action_id] = shortcut


# ═══════════════════════════════════════════════════════════════════════════════
# SELECTION MODEL - Unified selection handling
# ═══════════════════════════════════════════════════════════════════════════════

class SelectionModel(QObject if HAS_QT else object):
    """
    Unified selection model for all views.
    
    All selection operations MUST go through this class.
    No ad-hoc selection in components.
    """
    
    if HAS_QT:
        selection_changed = pyqtSignal()
        item_selected = pyqtSignal(object)
        items_selected = pyqtSignal(list)
    
    def __init__(self):
        if HAS_QT:
            super().__init__()
        self._selected: Set[Any] = set()
        self._anchor: Optional[Any] = None  # For range selection
        self._last_selected: Optional[Any] = None
    
    @property
    def selected(self) -> Set[Any]:
        """Get the current selection set"""
        return self._selected.copy()
    
    @property
    def count(self) -> int:
        """Number of selected items"""
        return len(self._selected)
    
    @property
    def is_empty(self) -> bool:
        """Check if selection is empty"""
        return len(self._selected) == 0
    
    def select_single(self, item: Any):
        """Select a single item, clearing previous selection"""
        self._selected.clear()
        self._selected.add(item)
        self._anchor = item
        self._last_selected = item
        self._emit_changed()
    
    def toggle(self, item: Any):
        """Toggle an item in the selection (Ctrl+click behavior)"""
        if item in self._selected:
            self._selected.discard(item)
        else:
            self._selected.add(item)
        self._last_selected = item
        self._emit_changed()
    
    def add(self, item: Any):
        """Add an item to selection without clearing"""
        self._selected.add(item)
        self._last_selected = item
        self._emit_changed()
    
    def remove(self, item: Any):
        """Remove an item from selection"""
        self._selected.discard(item)
        self._emit_changed()
    
    def select_multiple(self, items: List[Any]):
        """Select multiple items, clearing previous"""
        self._selected.clear()
        self._selected.update(items)
        if items:
            self._last_selected = items[-1]
        self._emit_changed()
    
    def add_multiple(self, items: List[Any]):
        """Add multiple items to selection"""
        self._selected.update(items)
        if items:
            self._last_selected = items[-1]
        self._emit_changed()
    
    def select_rect(self, items: List[Any], additive: bool = False):
        """
        Select items within a rectangle marquee.
        
        Args:
            items: Items within the rectangle
            additive: If True, add to existing selection (Shift+drag)
        """
        if not additive:
            self._selected.clear()
        self._selected.update(items)
        self._emit_changed()
    
    def select_range(self, item: Any, all_items: List[Any]):
        """
        Select a range from anchor to item (Shift+click behavior).
        
        Args:
            item: The end of the range
            all_items: Ordered list of all items to determine range
        """
        if self._anchor is None:
            self.select_single(item)
            return
        
        try:
            start_idx = all_items.index(self._anchor)
            end_idx = all_items.index(item)
            if start_idx > end_idx:
                start_idx, end_idx = end_idx, start_idx
            range_items = all_items[start_idx:end_idx + 1]
            self._selected.clear()
            self._selected.update(range_items)
            self._last_selected = item
            self._emit_changed()
        except ValueError:
            self.select_single(item)
    
    def clear(self):
        """Clear all selection"""
        self._selected.clear()
        self._anchor = None
        self._last_selected = None
        self._emit_changed()
    
    def is_selected(self, item: Any) -> bool:
        """Check if an item is selected"""
        return item in self._selected
    
    def _emit_changed(self):
        """Emit selection changed signal"""
        if HAS_QT and hasattr(self, 'selection_changed'):
            self.selection_changed.emit()


# ═══════════════════════════════════════════════════════════════════════════════
# TIME UTILITIES - Accurate beat/sample conversions
# ═══════════════════════════════════════════════════════════════════════════════

# Ticks per beat (standard MIDI resolution)
TICKS_PER_BEAT = 960


def beats_to_samples(beats: float, tempo_bpm: float, sample_rate: int) -> int:
    """
    Convert beats to samples.
    
    Args:
        beats: Time in beats
        tempo_bpm: Tempo in BPM
        sample_rate: Audio sample rate
    
    Returns:
        Sample position (integer)
    """
    seconds_per_beat = 60.0 / tempo_bpm
    seconds = beats * seconds_per_beat
    return int(seconds * sample_rate)


def samples_to_beats(samples: int, tempo_bpm: float, sample_rate: int) -> float:
    """
    Convert samples to beats.
    
    Args:
        samples: Sample position
        tempo_bpm: Tempo in BPM
        sample_rate: Audio sample rate
    
    Returns:
        Time in beats
    """
    seconds = samples / sample_rate
    seconds_per_beat = 60.0 / tempo_bpm
    return seconds / seconds_per_beat


def beats_to_ticks(beats: float) -> int:
    """Convert beats to MIDI ticks"""
    return int(beats * TICKS_PER_BEAT)


def ticks_to_beats(ticks: int) -> float:
    """Convert MIDI ticks to beats"""
    return ticks / TICKS_PER_BEAT


def beats_to_bar_beat_tick(beats: float, beats_per_bar: int = 4) -> Tuple[int, int, int]:
    """
    Convert beats to bar:beat:tick format.
    
    Returns:
        (bar, beat, tick) - all 1-indexed for display
    """
    total_ticks = beats_to_ticks(beats)
    ticks_per_bar = TICKS_PER_BEAT * beats_per_bar
    
    bar = int(total_ticks // ticks_per_bar) + 1
    remaining = total_ticks % ticks_per_bar
    beat = int(remaining // TICKS_PER_BEAT) + 1
    tick = int(remaining % TICKS_PER_BEAT)
    
    return (bar, beat, tick)


def bar_beat_tick_to_beats(bar: int, beat: int, tick: int, beats_per_bar: int = 4) -> float:
    """
    Convert bar:beat:tick to beats.
    
    Args:
        bar, beat, tick: 1-indexed position
        beats_per_bar: Time signature numerator
    """
    return (bar - 1) * beats_per_bar + (beat - 1) + tick / TICKS_PER_BEAT


def snap_to_grid(beats: float, grid_size: float) -> float:
    """
    Snap a beat position to the nearest grid line.
    
    Args:
        beats: Beat position
        grid_size: Grid size in beats (e.g., 0.25 for 1/16th notes)
    """
    if grid_size <= 0:
        return beats
    return round(beats / grid_size) * grid_size


def quantize_length(length: float, grid_size: float, min_length: float = None) -> float:
    """
    Quantize a note length to grid.
    
    Args:
        length: Note length in beats
        grid_size: Grid size in beats
        min_length: Minimum length (defaults to grid_size)
    """
    if grid_size <= 0:
        return length
    
    min_length = min_length or grid_size
    quantized = round(length / grid_size) * grid_size
    return max(min_length, quantized)


# ═══════════════════════════════════════════════════════════════════════════════
# AUDITION ENGINE - Sound preview with debouncing
# ═══════════════════════════════════════════════════════════════════════════════

class AuditionEngine:
    """
    Handles sound preview/audition with proper debouncing.
    
    Per spec: Every UI action implying sound must call defined audition rules.
    Audition must be non-blocking and debounced.
    """
    
    def __init__(self, audio_backend=None, debounce_ms: float = 50.0):
        self.audio_backend = audio_backend
        self.debounce_ms = debounce_ms
        self.enabled = True
        self.volume = 0.8
        
        self._last_audition_time: float = 0.0
        self._active_notes: Dict[int, float] = {}  # pitch -> start_time
    
    def set_audio_backend(self, backend):
        """Set the audio backend for preview playback"""
        self.audio_backend = backend
    
    def _should_debounce(self) -> bool:
        """Check if we should skip this audition due to debouncing"""
        now = time.time() * 1000
        if now - self._last_audition_time < self.debounce_ms:
            return True
        self._last_audition_time = now
        return False
    
    def audition_note(self, pitch: int, velocity: float = 0.8, 
                      duration: float = 0.3, channel: int = 0):
        """
        Play a single note preview (piano roll click, etc.)
        
        Args:
            pitch: MIDI note number
            velocity: Velocity 0-1
            duration: Duration in seconds
            channel: MIDI channel
        """
        if not self.enabled or not self.audio_backend:
            return
        
        if self._should_debounce():
            return
        
        # Trigger note preview through audio backend
        try:
            if hasattr(self.audio_backend, 'synth') and self.audio_backend.synth:
                self.audio_backend.synth.note_on(pitch, int(velocity * 127))
                # Schedule note off (simplified - in production use proper scheduling)
        except Exception:
            pass
    
    def audition_drum(self, drum_type, velocity: float = 0.8):
        """
        Play a drum sound preview (step sequencer toggle).
        
        Args:
            drum_type: DrumType enum value
            velocity: Velocity 0-1
        """
        if not self.enabled or not self.audio_backend:
            return
        
        if self._should_debounce():
            return
        
        try:
            if hasattr(self.audio_backend, 'play_preview'):
                # Import here to avoid circular imports
                from pattern_editor import DrumSynthesizer, DrumType
                synth = DrumSynthesizer()
                audio = synth.synthesize(drum_type, velocity)
                self.audio_backend.play_preview(audio * self.volume)
        except Exception:
            pass
    
    def audition_audio(self, audio_data, velocity: float = 1.0):
        """
        Play arbitrary audio data for preview.
        
        Args:
            audio_data: numpy array of audio samples
            velocity: Volume multiplier
        """
        if not self.enabled or not self.audio_backend:
            return
        
        if self._should_debounce():
            return
        
        try:
            if hasattr(self.audio_backend, 'play_preview'):
                self.audio_backend.play_preview(audio_data * velocity * self.volume)
        except Exception:
            pass
    
    def start_sustained_note(self, pitch: int, velocity: float = 0.8):
        """
        Start a sustained note (piano keyboard mouse down).
        Call stop_sustained_note on mouse up.
        """
        if not self.enabled or not self.audio_backend:
            return
        
        self._active_notes[pitch] = time.time()
        
        try:
            if hasattr(self.audio_backend, 'synth') and self.audio_backend.synth:
                self.audio_backend.synth.note_on(pitch, int(velocity * 127))
        except Exception:
            pass
    
    def stop_sustained_note(self, pitch: int):
        """Stop a sustained note (piano keyboard mouse up)."""
        if pitch in self._active_notes:
            del self._active_notes[pitch]
        
        try:
            if hasattr(self.audio_backend, 'synth') and self.audio_backend.synth:
                self.audio_backend.synth.note_off(pitch)
        except Exception:
            pass
    
    def stop_all(self):
        """Stop all active audition notes (panic)"""
        for pitch in list(self._active_notes.keys()):
            self.stop_sustained_note(pitch)
        self._active_notes.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# SCALE & KEY UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

class ScaleType(Enum):
    """Common scale types"""
    MAJOR = "major"
    MINOR = "minor"
    HARMONIC_MINOR = "harmonic_minor"
    MELODIC_MINOR = "melodic_minor"
    DORIAN = "dorian"
    PHRYGIAN = "phrygian"
    LYDIAN = "lydian"
    MIXOLYDIAN = "mixolydian"
    LOCRIAN = "locrian"
    PENTATONIC_MAJOR = "pentatonic_major"
    PENTATONIC_MINOR = "pentatonic_minor"
    BLUES = "blues"
    CHROMATIC = "chromatic"


# Scale intervals from root (semitones)
SCALE_INTERVALS = {
    ScaleType.MAJOR: [0, 2, 4, 5, 7, 9, 11],
    ScaleType.MINOR: [0, 2, 3, 5, 7, 8, 10],
    ScaleType.HARMONIC_MINOR: [0, 2, 3, 5, 7, 8, 11],
    ScaleType.MELODIC_MINOR: [0, 2, 3, 5, 7, 9, 11],
    ScaleType.DORIAN: [0, 2, 3, 5, 7, 9, 10],
    ScaleType.PHRYGIAN: [0, 1, 3, 5, 7, 8, 10],
    ScaleType.LYDIAN: [0, 2, 4, 6, 7, 9, 11],
    ScaleType.MIXOLYDIAN: [0, 2, 4, 5, 7, 9, 10],
    ScaleType.LOCRIAN: [0, 1, 3, 5, 6, 8, 10],
    ScaleType.PENTATONIC_MAJOR: [0, 2, 4, 7, 9],
    ScaleType.PENTATONIC_MINOR: [0, 3, 5, 7, 10],
    ScaleType.BLUES: [0, 3, 5, 6, 7, 10],
    ScaleType.CHROMATIC: list(range(12)),
}

# Note names
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


@dataclass
class Scale:
    """Represents a musical scale"""
    root: int  # 0-11 (C=0, C#=1, etc.)
    scale_type: ScaleType
    
    @property
    def name(self) -> str:
        return f"{NOTE_NAMES[self.root]} {self.scale_type.value.replace('_', ' ').title()}"
    
    @property
    def intervals(self) -> List[int]:
        return SCALE_INTERVALS[self.scale_type]
    
    def get_scale_notes(self, octave_start: int = 0, octave_end: int = 10) -> List[int]:
        """Get all MIDI notes in this scale within octave range"""
        notes = []
        for octave in range(octave_start, octave_end + 1):
            for interval in self.intervals:
                note = self.root + interval + (octave * 12)
                if 0 <= note <= 127:
                    notes.append(note)
        return notes
    
    def is_in_scale(self, pitch: int) -> bool:
        """Check if a MIDI pitch is in this scale"""
        pitch_class = pitch % 12
        return (pitch_class - self.root) % 12 in self.intervals
    
    def snap_to_scale(self, pitch: int) -> int:
        """Snap a pitch to the nearest note in scale"""
        if self.is_in_scale(pitch):
            return pitch
        
        pitch_class = pitch % 12
        octave = pitch // 12
        
        # Find nearest scale degree
        min_dist = 12
        nearest = pitch_class
        for interval in self.intervals:
            scale_pitch_class = (self.root + interval) % 12
            dist = min(abs(pitch_class - scale_pitch_class), 
                      12 - abs(pitch_class - scale_pitch_class))
            if dist < min_dist:
                min_dist = dist
                nearest = scale_pitch_class
        
        return octave * 12 + nearest
    
    def get_degree(self, pitch: int) -> Optional[int]:
        """Get the scale degree (1-7) of a pitch, or None if not in scale"""
        if not self.is_in_scale(pitch):
            return None
        pitch_class = (pitch - self.root) % 12
        try:
            return self.intervals.index(pitch_class) + 1
        except ValueError:
            return None


# ═══════════════════════════════════════════════════════════════════════════════
# CHORD HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

class ChordType(Enum):
    """Common chord types"""
    MAJOR = "major"
    MINOR = "minor"
    DIMINISHED = "dim"
    AUGMENTED = "aug"
    MAJOR_7 = "maj7"
    MINOR_7 = "min7"
    DOMINANT_7 = "7"
    DIMINISHED_7 = "dim7"
    HALF_DIMINISHED = "m7b5"
    SUS2 = "sus2"
    SUS4 = "sus4"
    ADD9 = "add9"
    POWER = "5"


CHORD_INTERVALS = {
    ChordType.MAJOR: [0, 4, 7],
    ChordType.MINOR: [0, 3, 7],
    ChordType.DIMINISHED: [0, 3, 6],
    ChordType.AUGMENTED: [0, 4, 8],
    ChordType.MAJOR_7: [0, 4, 7, 11],
    ChordType.MINOR_7: [0, 3, 7, 10],
    ChordType.DOMINANT_7: [0, 4, 7, 10],
    ChordType.DIMINISHED_7: [0, 3, 6, 9],
    ChordType.HALF_DIMINISHED: [0, 3, 6, 10],
    ChordType.SUS2: [0, 2, 7],
    ChordType.SUS4: [0, 5, 7],
    ChordType.ADD9: [0, 4, 7, 14],
    ChordType.POWER: [0, 7],
}


@dataclass
class Chord:
    """Represents a chord"""
    root: int  # MIDI note number
    chord_type: ChordType
    inversion: int = 0  # 0 = root, 1 = first, 2 = second, etc.
    
    @property
    def name(self) -> str:
        note_name = NOTE_NAMES[self.root % 12]
        return f"{note_name}{self.chord_type.value}"
    
    def get_notes(self) -> List[int]:
        """Get MIDI notes for this chord"""
        intervals = CHORD_INTERVALS[self.chord_type].copy()
        
        # Apply inversion
        for i in range(self.inversion):
            if i < len(intervals):
                intervals[i] += 12
        intervals.sort()
        
        return [self.root + interval for interval in intervals]


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL INSTANCES
# ═══════════════════════════════════════════════════════════════════════════════

# Singleton instances
_keymap_registry: Optional[KeymapRegistry] = None
_selection_model: Optional[SelectionModel] = None
_audition_engine: Optional[AuditionEngine] = None


def get_keymap_registry() -> KeymapRegistry:
    """Get the global keymap registry"""
    global _keymap_registry
    if _keymap_registry is None:
        _keymap_registry = KeymapRegistry()
    return _keymap_registry


def get_selection_model() -> SelectionModel:
    """Get the global selection model"""
    global _selection_model
    if _selection_model is None:
        _selection_model = SelectionModel()
    return _selection_model


def get_audition_engine() -> AuditionEngine:
    """Get the global audition engine"""
    global _audition_engine
    if _audition_engine is None:
        _audition_engine = AuditionEngine()
    return _audition_engine
