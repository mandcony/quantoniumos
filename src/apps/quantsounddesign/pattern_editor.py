#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
# This file is listed in CLAIMS_PRACTICING_FILES.txt and is licensed
# under LICENSE-CLAIMS-NC.md (research/education only). Commercial
# rights require a separate patent license from the author.
"""
QuantSoundDesign Pattern Editor - Step sequencer and pattern editing.

Provides:
- Step sequencer for drums
- Pattern editor for melodic instruments
- Drum kit with multiple sounds
- Per-track pattern assignment
- Φ-RFT enhanced synthesis via RFTMW engine
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Callable
from enum import Enum
import uuid

try:
    from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton,
        QLabel, QComboBox, QScrollArea, QFrame, QSlider, QSpinBox, QDoubleSpinBox,
        QTabWidget, QListWidget, QListWidgetItem, QGroupBox, QSplitter,
        QMenu, QAction, QCheckBox, QShortcut
    )
    from PyQt5.QtCore import Qt, pyqtSignal, QTimer
    from PyQt5.QtGui import QPainter, QColor, QBrush, QPen, QFont, QLinearGradient, QKeySequence
    HAS_PYQT5 = True
except ImportError:
    HAS_PYQT5 = False
    # Dummy classes for when PyQt5 is not available
    class QWidget: pass
    class QPushButton: pass
    class QLabel: pass
    class QComboBox: pass
    class QVBoxLayout: pass
    class QHBoxLayout: pass
    class QGridLayout: pass
    class QScrollArea: pass
    class QFrame: pass
    class QSlider: pass
    class QSpinBox: pass
    class QDoubleSpinBox: pass
    class QTabWidget: pass
    class QListWidget: pass
    class QListWidgetItem: pass
    class QGroupBox: pass
    class QSplitter: pass
    class QMenu: pass
    class QAction: pass
    class QCheckBox: pass
    class QShortcut: pass
    class Qt: 
        AlignCenter = 0
        AlignLeft = 0
        AlignRight = 0
        Horizontal = 0
        Vertical = 0
        LeftButton = 0
        RightButton = 0
        Key_Space = 0
    def pyqtSignal(*args, **kwargs):
        """Dummy pyqtSignal that accepts any arguments"""
        return None
    class QTimer: pass
    class QPainter: pass
    class QColor: 
        def __init__(self, *args): pass
    class QBrush: pass
    class QPen: pass
    class QFont: pass
    class QLinearGradient: pass
    class QKeySequence: pass

# Golden ratio
PHI = (1 + np.sqrt(5)) / 2

# Optional RFTMW engine for Φ-enhanced synthesis
try:
    from quantonium_os_src.engine.RFTMW import MiddlewareTransformEngine, TransformProfile
    _rft_engine = MiddlewareTransformEngine()
    HAS_RFTMW = True
except ImportError:
    _rft_engine = None
    HAS_RFTMW = False


# ═══════════════════════════════════════════════════════════════════════════════
# DRUM SYNTHESIS
# ═══════════════════════════════════════════════════════════════════════════════

class DrumType(Enum):
    KICK = "kick"
    SNARE = "snare"
    CLAP = "clap"
    HIHAT_CLOSED = "hihat_closed"
    HIHAT_OPEN = "hihat_open"
    TOM_HIGH = "tom_high"
    TOM_MID = "tom_mid"
    TOM_LOW = "tom_low"
    CRASH = "crash"
    RIDE = "ride"
    RIMSHOT = "rimshot"
    COWBELL = "cowbell"
    SHAKER = "shaker"
    CLAV = "clav"
    PERC_1 = "perc_1"
    PERC_2 = "perc_2"


@dataclass
class DrumSound:
    """Parameters for a drum sound"""
    name: str
    drum_type: DrumType
    
    # Synthesis parameters
    pitch: float = 60.0          # Base frequency
    pitch_decay: float = 50.0    # Pitch envelope decay rate
    pitch_amount: float = 200.0  # Pitch envelope amount
    
    amp_attack: float = 0.001    # Amplitude attack
    amp_decay: float = 0.1       # Amplitude decay
    amp_sustain: float = 0.0     # Amplitude sustain
    amp_release: float = 0.05    # Amplitude release
    
    noise_amount: float = 0.0    # White noise mix
    noise_decay: float = 0.05    # Noise decay
    
    filter_freq: float = 8000.0  # Filter cutoff
    filter_reso: float = 0.0     # Filter resonance
    
    distortion: float = 0.0      # Distortion amount
    
    gain: float = 0.8            # Output gain


class DrumSynthesizer:
    """Synthesize drum sounds"""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.sounds = self._create_default_kit()
    
    def _create_default_kit(self) -> Dict[DrumType, DrumSound]:
        """Create default 808-style drum kit"""
        return {
            DrumType.KICK: DrumSound(
                name="808 Kick",
                drum_type=DrumType.KICK,
                pitch=50.0,
                pitch_decay=25.0,
                pitch_amount=150.0,
                amp_decay=0.4,
                noise_amount=0.05,
                noise_decay=0.01,
                distortion=0.1,
                gain=1.0
            ),
            DrumType.SNARE: DrumSound(
                name="808 Snare",
                drum_type=DrumType.SNARE,
                pitch=180.0,
                pitch_decay=40.0,
                pitch_amount=80.0,
                amp_decay=0.15,
                noise_amount=0.6,
                noise_decay=0.12,
                filter_freq=6000.0,
                gain=0.8
            ),
            DrumType.CLAP: DrumSound(
                name="808 Clap",
                drum_type=DrumType.CLAP,
                pitch=400.0,
                amp_attack=0.005,
                amp_decay=0.2,
                noise_amount=0.9,
                noise_decay=0.15,
                filter_freq=3000.0,
                gain=0.7
            ),
            DrumType.HIHAT_CLOSED: DrumSound(
                name="Closed Hat",
                drum_type=DrumType.HIHAT_CLOSED,
                pitch=8000.0,
                amp_decay=0.05,
                noise_amount=0.95,
                noise_decay=0.04,
                filter_freq=10000.0,
                filter_reso=0.3,
                gain=0.5
            ),
            DrumType.HIHAT_OPEN: DrumSound(
                name="Open Hat",
                drum_type=DrumType.HIHAT_OPEN,
                pitch=8000.0,
                amp_decay=0.3,
                noise_amount=0.95,
                noise_decay=0.25,
                filter_freq=9000.0,
                filter_reso=0.2,
                gain=0.5
            ),
            DrumType.TOM_HIGH: DrumSound(
                name="High Tom",
                drum_type=DrumType.TOM_HIGH,
                pitch=200.0,
                pitch_decay=30.0,
                pitch_amount=100.0,
                amp_decay=0.2,
                noise_amount=0.1,
                gain=0.7
            ),
            DrumType.TOM_MID: DrumSound(
                name="Mid Tom",
                drum_type=DrumType.TOM_MID,
                pitch=140.0,
                pitch_decay=25.0,
                pitch_amount=80.0,
                amp_decay=0.25,
                noise_amount=0.1,
                gain=0.7
            ),
            DrumType.TOM_LOW: DrumSound(
                name="Low Tom",
                drum_type=DrumType.TOM_LOW,
                pitch=90.0,
                pitch_decay=20.0,
                pitch_amount=60.0,
                amp_decay=0.3,
                noise_amount=0.1,
                gain=0.7
            ),
            DrumType.CRASH: DrumSound(
                name="Crash",
                drum_type=DrumType.CRASH,
                pitch=6000.0,
                amp_decay=0.8,
                noise_amount=0.85,
                noise_decay=0.7,
                filter_freq=12000.0,
                gain=0.6
            ),
            DrumType.RIDE: DrumSound(
                name="Ride",
                drum_type=DrumType.RIDE,
                pitch=5000.0,
                amp_decay=0.5,
                noise_amount=0.7,
                noise_decay=0.4,
                filter_freq=10000.0,
                filter_reso=0.2,
                gain=0.5
            ),
            DrumType.RIMSHOT: DrumSound(
                name="Rimshot",
                drum_type=DrumType.RIMSHOT,
                pitch=600.0,
                pitch_decay=80.0,
                amp_decay=0.08,
                noise_amount=0.4,
                noise_decay=0.03,
                filter_freq=4000.0,
                gain=0.7
            ),
            DrumType.COWBELL: DrumSound(
                name="Cowbell",
                drum_type=DrumType.COWBELL,
                pitch=560.0,
                amp_decay=0.2,
                noise_amount=0.0,
                gain=0.6
            ),
            DrumType.SHAKER: DrumSound(
                name="Shaker",
                drum_type=DrumType.SHAKER,
                pitch=6000.0,
                amp_decay=0.08,
                noise_amount=1.0,
                noise_decay=0.06,
                filter_freq=8000.0,
                gain=0.4
            ),
            DrumType.CLAV: DrumSound(
                name="Clave",
                drum_type=DrumType.CLAV,
                pitch=2500.0,
                pitch_decay=100.0,
                amp_decay=0.06,
                noise_amount=0.1,
                gain=0.6
            ),
            DrumType.PERC_1: DrumSound(
                name="Φ-Perc 1",
                drum_type=DrumType.PERC_1,
                pitch=440.0 * PHI,
                pitch_decay=60.0,
                pitch_amount=100.0,
                amp_decay=0.12,
                noise_amount=0.2,
                gain=0.6
            ),
            DrumType.PERC_2: DrumSound(
                name="Φ-Perc 2",
                drum_type=DrumType.PERC_2,
                pitch=440.0 / PHI,
                pitch_decay=40.0,
                pitch_amount=50.0,
                amp_decay=0.18,
                noise_amount=0.15,
                gain=0.6
            ),
        }
    
    def synthesize(self, drum_type: DrumType, velocity: float = 1.0, 
                   duration: float = 0.5) -> np.ndarray:
        """Synthesize a drum hit"""
        sound = self.sounds.get(drum_type)
        if sound is None:
            return np.zeros(int(duration * self.sample_rate), dtype=np.float32)
        
        samples = int(duration * self.sample_rate)
        t = np.arange(samples) / self.sample_rate
        
        # Pitch envelope
        pitch_env = sound.pitch + sound.pitch_amount * np.exp(-t * sound.pitch_decay)
        
        # Generate tone with pitch envelope
        phase = np.cumsum(2 * np.pi * pitch_env / self.sample_rate)
        tone = np.sin(phase)
        
        # Generate noise
        noise = np.random.randn(samples) * sound.noise_amount
        noise *= np.exp(-t / max(0.001, sound.noise_decay))
        
        # Combine tone and noise
        signal = tone * (1 - sound.noise_amount) + noise
        
        # Simple lowpass filter
        if sound.filter_freq < self.sample_rate / 2:
            alpha = sound.filter_freq / (self.sample_rate / 2)
            alpha = min(1.0, max(0.01, alpha))
            filtered = np.zeros_like(signal)
            filtered[0] = signal[0]
            for i in range(1, len(signal)):
                filtered[i] = alpha * signal[i] + (1 - alpha) * filtered[i-1]
            signal = filtered
        
        # Amplitude envelope (ADSR simplified)
        attack_samples = int(sound.amp_attack * self.sample_rate)
        decay_samples = int(sound.amp_decay * self.sample_rate)
        
        envelope = np.ones(samples)
        
        # Attack
        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        
        # Decay to sustain
        if decay_samples > 0 and attack_samples < samples:
            decay_end = min(attack_samples + decay_samples, samples)
            decay_len = decay_end - attack_samples
            envelope[attack_samples:decay_end] = np.linspace(1, sound.amp_sustain, decay_len)
            envelope[decay_end:] = sound.amp_sustain
        
        # Release (exponential)
        envelope *= np.exp(-t / max(0.001, sound.amp_decay))
        
        # Apply envelope
        signal *= envelope
        
        # Distortion
        if sound.distortion > 0:
            signal = np.tanh(signal * (1 + sound.distortion * 10))
        
        # Apply gain and velocity
        signal *= sound.gain * velocity
        
        # Clip
        signal = np.clip(signal, -1.0, 1.0)
        
        return signal.astype(np.float32)

    def synthesize_rft(self, drum_type: DrumType, velocity: float = 1.0,
                       duration: float = 0.5) -> np.ndarray:
        """Synthesize a drum hit with Φ-RFT wave-space enhancement.

        Uses the RFTMW engine to process the drum sound through wave-space,
        adding harmonic richness based on golden ratio transforms.
        """
        # First generate the base drum sound
        base = self.synthesize(drum_type, velocity, duration)

        if not HAS_RFTMW or _rft_engine is None:
            return base

        try:
            # Convert to bytes for RFTMW processing
            audio_bytes = base.tobytes()
            profile = TransformProfile(data_type='audio', priority='accuracy', size=len(audio_bytes))
            result = _rft_engine.compute_in_wavespace(audio_bytes, operation='identity', profile=profile)

            # Convert back to float32
            enhanced = np.frombuffer(result.output_binary, dtype=np.float32)
            # Ensure same length
            if len(enhanced) < len(base):
                enhanced = np.pad(enhanced, (0, len(base) - len(enhanced)))
            elif len(enhanced) > len(base):
                enhanced = enhanced[:len(base)]

            # Blend original with RFT-enhanced (subtle effect)
            blend = 0.15  # 15% RFT enhancement
            output = base * (1 - blend) + enhanced * blend
            return np.clip(output, -1.0, 1.0).astype(np.float32)
        except Exception:
            # Fallback to base sound on any error
            return base


# ═══════════════════════════════════════════════════════════════════════════════
# PATTERN DATA
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PatternStep:
    """Single step in a pattern"""
    active: bool = False
    velocity: float = 0.8       # 0.0 - 1.0
    note: int = 60              # MIDI note (for melodic patterns)
    probability: float = 1.0    # Chance to trigger
    
    # Micro-timing
    offset: float = 0.0         # -0.5 to +0.5 step offset
    
    # Additional parameters
    decay: float = 1.0          # Note length multiplier
    slide: bool = False         # Portamento to next note


@dataclass
class PatternRow:
    """One row in a pattern (one drum sound or one pitch)"""
    name: str = "Row"
    drum_type: Optional[DrumType] = None    # For drum patterns
    note: int = 60                           # For melodic patterns (base note)
    steps: List[PatternStep] = field(default_factory=list)
    mute: bool = False
    solo: bool = False
    volume: float = 1.0
    
    def __post_init__(self):
        if not self.steps:
            self.steps = [PatternStep() for _ in range(16)]


@dataclass
class Pattern:
    """Complete pattern with multiple rows"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = "Pattern 1"
    color: str = "#00aaff"
    
    # Timing
    steps: int = 16             # Steps per pattern
    step_division: int = 4      # Steps per beat (4 = 16th notes)
    swing: float = 0.0          # -1.0 to 1.0 swing amount
    
    # Pattern type
    is_drum: bool = True        # Drum or melodic pattern
    
    # Rows
    rows: List[PatternRow] = field(default_factory=list)
    probability_lane: List[float] = field(default_factory=list)
    humanize_velocity: float = 0.0   # 0-1 multiplier for velocity variation
    humanize_probability: float = 0.0  # 0-1 additive probability jitter
    humanize_timing: float = 0.0    # reserved for future micro-timing use
    
    def __post_init__(self):
        if not self.rows and self.is_drum:
            # Create default drum rows
            drum_order = [
                DrumType.KICK, DrumType.SNARE, DrumType.CLAP,
                DrumType.HIHAT_CLOSED, DrumType.HIHAT_OPEN,
                DrumType.TOM_HIGH, DrumType.TOM_MID, DrumType.TOM_LOW,
                DrumType.CRASH, DrumType.RIDE, DrumType.RIMSHOT,
                DrumType.COWBELL, DrumType.SHAKER, DrumType.CLAV,
                DrumType.PERC_1, DrumType.PERC_2
            ]
            self.rows = [
                PatternRow(
                    name=dt.value.replace("_", " ").title(),
                    drum_type=dt,
                    steps=[PatternStep() for _ in range(self.steps)]
                )
                for dt in drum_order
            ]
        if not self.probability_lane or len(self.probability_lane) != self.steps:
            self.probability_lane = [1.0 for _ in range(self.steps)]
    
    @property
    def length_beats(self) -> float:
        """Pattern length in beats"""
        return self.steps / self.step_division
    
    def resize(self, new_steps: int):
        """Resize pattern to new step count"""
        for row in self.rows:
            if len(row.steps) < new_steps:
                row.steps.extend([PatternStep() for _ in range(new_steps - len(row.steps))])
            elif len(row.steps) > new_steps:
                row.steps = row.steps[:new_steps]
        if len(self.probability_lane) < new_steps:
            self.probability_lane.extend([1.0 for _ in range(new_steps - len(self.probability_lane))])
        elif len(self.probability_lane) > new_steps:
            self.probability_lane = self.probability_lane[:new_steps]
        self.steps = new_steps
    
    def get_active_steps_at(self, step: int) -> List[tuple]:
        """Get all active steps at given step index"""
        result = []
        for row in self.rows:
            if row.mute:
                continue
            if step < len(row.steps) and row.steps[step].active:
                result.append((row, row.steps[step]))
        return result

    def probability_at(self, step: int) -> float:
        if 0 <= step < len(self.probability_lane):
            return self.probability_lane[step]
        return 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# PATTERN PLAYER
# ═══════════════════════════════════════════════════════════════════════════════

class PatternPlayer:
    """Plays patterns with real-time audio synthesis"""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.drum_synth = DrumSynthesizer(sample_rate)
        
        # Active patterns per track
        self.patterns: Dict[str, Pattern] = {}  # track_id -> pattern
        
        # Playback state
        self.current_step = 0
        self.playing = False
        
        # Pending sounds to mix
        self._pending_sounds: List[tuple] = []  # (sample_offset, audio_data)
        self._active_sounds: List[tuple] = []   # Currently playing sounds
    
    def set_pattern(self, track_id: str, pattern: Pattern):
        """Set pattern for a track"""
        self.patterns[track_id] = pattern
    
    def trigger_step(self, step: int, synth=None):
        """Trigger all sounds at a step"""
        for track_id, pattern in self.patterns.items():
            if step >= pattern.steps:
                continue
            
            for row, step_data in pattern.get_active_steps_at(step):
                if np.random.random() > step_data.probability:
                    continue
                
                velocity = step_data.velocity * row.volume
                
                if pattern.is_drum and row.drum_type:
                    # Synthesize drum sound
                    audio = self.drum_synth.synthesize(
                        row.drum_type, 
                        velocity=velocity,
                        duration=0.5
                    )
                    self._pending_sounds.append((0, audio))
                elif synth is not None:
                    # Trigger synth note
                    synth.note_on(step_data.note, int(velocity * 127))
    
    def render(self, frames: int, tempo: float = 120.0) -> np.ndarray:
        """Render audio for a block of frames"""
        output = np.zeros(frames, dtype=np.float32)
        
        # Mix active sounds
        new_active = []
        for offset, audio in self._active_sounds:
            if offset >= len(audio):
                continue
            
            end = min(offset + frames, len(audio))
            samples = end - offset
            output[:samples] += audio[offset:end]
            
            if end < len(audio):
                new_active.append((end, audio))
        
        # Add new pending sounds
        for offset, audio in self._pending_sounds:
            samples = min(frames, len(audio))
            output[:samples] += audio[:samples]
            if samples < len(audio):
                new_active.append((samples, audio))
        
        self._active_sounds = new_active
        self._pending_sounds.clear()
        
        return output


# ═══════════════════════════════════════════════════════════════════════════════
# PATTERN EDITOR WIDGET
# ═══════════════════════════════════════════════════════════════════════════════

class StepButton(QWidget):
    """FL Studio-style step button with velocity-based coloring and visual feedback"""
    
    step_changed = pyqtSignal(int, int, bool, float)  # row, step, active, velocity
    step_selected = pyqtSignal(int, int)  # row, step - for selection
    velocity_edit_requested = pyqtSignal(int, int)  # row, step - for velocity popup
    
    # FL Studio color palette for velocity
    VEL_COLORS = [
        QColor(60, 60, 60),      # 0% - dark gray (off)
        QColor(120, 80, 40),     # 10% - brown
        QColor(180, 100, 30),    # 20% - burnt orange
        QColor(220, 120, 20),    # 30% - orange
        QColor(245, 140, 10),    # 40% - bright orange
        QColor(255, 160, 0),     # 50% - gold
        QColor(255, 180, 30),    # 60% - yellow-orange
        QColor(255, 200, 60),    # 70% - light orange
        QColor(255, 220, 100),   # 80% - pale yellow
        QColor(255, 240, 150),   # 90% - cream
        QColor(255, 255, 200),   # 100% - almost white
    ]
    
    def __init__(self, row: int, step: int, parent=None):
        super().__init__(parent)
        self.row = row
        self.step = step
        self.active = False
        self.velocity = 0.8
        self.probability = 1.0
        self.offset = 0.0
        self.decay = 1.0
        self.slide = False
        self.selected = False
        self.playing = False  # Playhead is on this step
        self.hover = False
        self._drag_start_y = 0
        self._drag_velocity = 0.8
        
        self.setFixedSize(36, 28)  # FL Studio proportions
        self.setMouseTracking(True)
        self.setCursor(Qt.PointingHandCursor)
        self._update_tooltip()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        w, h = self.width(), self.height()
        
        # Determine bar grouping colors (FL Studio style - every 4 steps grouped)
        bar_group = self.step // 4
        is_odd_bar = (bar_group % 2) == 1
        is_downbeat = (self.step % 4) == 0
        
        # Background based on position in bar
        if self.active:
            # Velocity-based color (FL Studio orange gradient)
            vel_idx = min(10, max(0, int(self.velocity * 10)))
            base_color = self.VEL_COLORS[vel_idx]
            
            # Draw gradient fill
            gradient = QLinearGradient(0, 0, 0, h)
            gradient.setColorAt(0, base_color.lighter(120))
            gradient.setColorAt(0.5, base_color)
            gradient.setColorAt(1, base_color.darker(130))
            painter.setBrush(QBrush(gradient))
        else:
            # Inactive - subtle grid pattern like FL
            if is_odd_bar:
                bg = QColor(45, 42, 38) if is_downbeat else QColor(38, 36, 33)
            else:
                bg = QColor(55, 52, 48) if is_downbeat else QColor(48, 45, 42)
            painter.setBrush(QBrush(bg))
        
        # Border
        if self.playing:
            painter.setPen(QPen(QColor(0, 255, 100), 2))
        elif self.selected:
            painter.setPen(QPen(QColor(255, 255, 255), 2))
        elif self.hover:
            painter.setPen(QPen(QColor(100, 160, 220), 1))
        elif self.active:
            painter.setPen(QPen(QColor(80, 60, 40), 1))
        else:
            painter.setPen(QPen(QColor(60, 55, 50), 1))
        
        # Draw rounded rectangle (FL style)
        painter.drawRoundedRect(1, 1, w - 2, h - 2, 3, 3)
        
        # Draw velocity bar at bottom (FL Studio feature)
        if self.active:
            bar_h = 4
            bar_w = int((w - 6) * self.velocity)
            vel_bar_color = QColor(255, 100, 50) if self.velocity > 0.8 else QColor(255, 180, 80)
            painter.fillRect(3, h - bar_h - 2, bar_w, bar_h, vel_bar_color)
        
        # Probability indicator (dimmed if < 100%)
        if self.active and self.probability < 1.0:
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(0, 0, 0, int(150 * (1 - self.probability)))))
            painter.drawRoundedRect(1, 1, w - 2, h - 2, 3, 3)
            # Draw probability percentage
            painter.setPen(QColor(255, 255, 255, 180))
            painter.setFont(QFont("Segoe UI", 7))
            painter.drawText(self.rect(), Qt.AlignCenter, f"{int(self.probability * 100)}%")
        
        # Slide indicator
        if self.active and self.slide:
            painter.setPen(QColor(0, 255, 255))
            painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
            painter.drawText(self.rect(), Qt.AlignCenter, "→")
    
    def enterEvent(self, event):
        self.hover = True
        self.update()
    
    def leaveEvent(self, event):
        self.hover = False
        self.update()
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_start_y = event.y()
            self._drag_velocity = self.velocity
            # Toggle on click
            self.active = not self.active
            self.update()
            self.step_changed.emit(self.row, self.step, self.active, self.velocity)
            self.step_selected.emit(self.row, self.step)
        elif event.button() == Qt.RightButton:
            # Right-click opens velocity editor
            self.velocity_edit_requested.emit(self.row, self.step)
    
    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton and self.active:
            # Drag up/down to change velocity (FL Studio behavior)
            delta = self._drag_start_y - event.y()
            new_vel = self._drag_velocity + delta * 0.01
            new_vel = max(0.1, min(1.0, new_vel))
            if abs(new_vel - self.velocity) > 0.01:
                self.velocity = new_vel
                self.update()
                self._update_tooltip()
                self.step_changed.emit(self.row, self.step, self.active, self.velocity)
    
    def mouseReleaseEvent(self, event):
        pass  # Nothing special needed
    
    def set_state(
        self,
        active: bool,
        velocity: float = 0.8,
        probability: float = 1.0,
        offset: float = 0.0,
        decay: float = 1.0,
        slide: bool = False,
    ):
        self.active = active
        self.velocity = velocity
        self.probability = max(0.0, min(1.0, probability))
        self.offset = max(-0.5, min(0.5, offset))
        self.decay = max(0.1, min(4.0, decay))
        self.slide = slide
        self._update_tooltip()
        self.update()
    
    def set_selected(self, selected: bool):
        """Set selected state (white border)"""
        self.selected = selected
        self.update()
    
    def set_playing(self, playing: bool):
        """Set playing state (green border - playhead is here)"""
        self.playing = playing
        self.update()
    
    def set_velocity(self, velocity: float):
        """Update velocity and refresh display"""
        self.velocity = max(0.0, min(1.0, velocity))
        self._update_tooltip()
        self.update()
        self.step_changed.emit(self.row, self.step, self.active, self.velocity)

    def set_probability(self, probability: float):
        """Update probability for this step."""
        self.probability = max(0.0, min(1.0, probability))
        self._update_tooltip()
        self.update()
    
    def set_offset(self, offset: float):
        self.offset = max(-0.5, min(0.5, offset))
        self._update_tooltip()

    def set_decay(self, decay: float):
        self.decay = max(0.1, min(4.0, decay))
        self._update_tooltip()

    def set_slide(self, slide: bool):
        self.slide = slide
        self.update()

    def _update_tooltip(self):
        self.setToolTip(
            f"Step {self.step + 1} | Vel: {int(self.velocity * 100)}% | "
            f"Prob: {int(self.probability * 100)}% | Offset: {self.offset * 100:+.0f}%"
        )
    
    def set_highlight(self, highlighted: bool):
        """Highlight for playhead (deprecated - use set_playing)"""
        self.set_playing(highlighted)


class DrumRowWidget(QWidget):
    """FL Studio-style drum row with LED-style label and step pads"""
    
    row_changed = pyqtSignal(int)  # row index
    row_selected = pyqtSignal(int)  # row selected for editing
    step_selected = pyqtSignal(int, int)  # row, step selected
    velocity_edit = pyqtSignal(int, int)  # row, step for velocity edit popup
    step_activated = pyqtSignal(int, float)  # row_index, velocity - for triggering preview sound
    
    # FL Studio row colors
    ROW_COLORS = [
        QColor(255, 120, 80),   # Kick - red-orange
        QColor(255, 180, 60),   # Snare - orange
        QColor(200, 120, 255),  # Clap - purple
        QColor(80, 200, 255),   # Hihat C - cyan
        QColor(100, 255, 200),  # Hihat O - teal
        QColor(255, 100, 150),  # Tom Hi - pink
        QColor(255, 140, 100),  # Tom Mid - salmon
        QColor(200, 100, 80),   # Tom Lo - brown
        QColor(255, 220, 100),  # Crash - yellow
        QColor(180, 180, 255),  # Ride - light blue
        QColor(255, 160, 200),  # Rim - light pink
        QColor(200, 255, 150),  # Cowbell - lime
    ]
    
    def __init__(self, row_index: int, row_data: PatternRow, parent=None):
        super().__init__(parent)
        self.row_index = row_index
        self.row_data = row_data
        self.step_buttons: List[StepButton] = []
        self.is_selected = False
        self.row_color = self.ROW_COLORS[row_index % len(self.ROW_COLORS)]
        
        self.setFixedHeight(34)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 1, 4, 1)
        layout.setSpacing(1)
        
        # FL Studio style - colored indicator LED + name
        self.indicator = QFrame()
        self.indicator.setFixedSize(6, 28)
        self.indicator.setStyleSheet(f"""
            QFrame {{
                background-color: {self.row_color.name()};
                border-radius: 2px;
            }}
        """)
        layout.addWidget(self.indicator)
        
        # Row label with mute/solo - FL Studio compact style
        label_frame = QFrame()
        label_frame.setFixedWidth(100)
        label_frame.setCursor(Qt.PointingHandCursor)
        label_frame.mousePressEvent = lambda e: self._on_row_click()
        label_layout = QHBoxLayout(label_frame)
        label_layout.setContentsMargins(4, 0, 2, 0)
        label_layout.setSpacing(2)
        
        # Compact mute button (FL style - just colored when active)
        self.mute_btn = QPushButton()
        self.mute_btn.setFixedSize(18, 18)
        self.mute_btn.setCheckable(True)
        self.mute_btn.setToolTip("Mute")
        self.mute_btn.setStyleSheet("""
            QPushButton { 
                background: #2a2a2a; 
                border: none; 
                border-radius: 2px;
                image: none;
            }
            QPushButton:hover { background: #3a3a3a; }
            QPushButton:checked { background: #ff4444; }
        """)
        self.mute_btn.clicked.connect(self._on_mute)
        
        # Compact solo button
        self.solo_btn = QPushButton()
        self.solo_btn.setFixedSize(18, 18)
        self.solo_btn.setCheckable(True)
        self.solo_btn.setToolTip("Solo")
        self.solo_btn.setStyleSheet("""
            QPushButton { 
                background: #2a2a2a; 
                border: none; 
                border-radius: 2px;
            }
            QPushButton:hover { background: #3a3a3a; }
            QPushButton:checked { background: #00cc66; }
        """)
        self.solo_btn.clicked.connect(self._on_solo)
        
        # Name label - FL style truncated
        self.name_label = QLabel(row_data.name[:8])  # Truncate long names
        self.name_label.setStyleSheet(f"""
            color: {self.row_color.name()}; 
            font-size: 10px; 
            font-weight: bold;
            font-family: 'Segoe UI', sans-serif;
        """)
        self.name_label.setFixedWidth(55)
        
        label_layout.addWidget(self.mute_btn)
        label_layout.addWidget(self.solo_btn)
        label_layout.addWidget(self.name_label)
        
        self.label_frame = label_frame
        layout.addWidget(label_frame)
        
        # Separator between label and steps
        sep = QFrame()
        sep.setFixedWidth(1)
        sep.setStyleSheet("background: #3a3a3a;")
        layout.addWidget(sep)
        
        # Step buttons - FL Studio grouping with bar separators
        steps_container = QWidget()
        steps_layout = QHBoxLayout(steps_container)
        steps_layout.setContentsMargins(2, 0, 0, 0)
        steps_layout.setSpacing(1)
        
        for step in range(len(row_data.steps)):
            # Add bar separator every 4 steps (FL Studio style)
            if step > 0 and step % 4 == 0:
                bar_sep = QFrame()
                bar_sep.setFixedWidth(2)
                bar_sep.setStyleSheet("background: #4a4a4a;")
                steps_layout.addWidget(bar_sep)
            
            btn = StepButton(row_index, step)
            btn.set_state(
                row_data.steps[step].active,
                row_data.steps[step].velocity,
                row_data.steps[step].probability,
                row_data.steps[step].offset,
                row_data.steps[step].decay,
                row_data.steps[step].slide,
            )
            btn.step_changed.connect(self._on_step_changed)
            btn.step_selected.connect(self._on_step_selected)
            btn.velocity_edit_requested.connect(self._on_velocity_edit)
            self.step_buttons.append(btn)
            steps_layout.addWidget(btn)
        
        steps_layout.addStretch()
        layout.addWidget(steps_container, 1)
        
        self._update_selection_style()
    
    def _on_row_click(self):
        """Row header clicked - select this row"""
        self.row_selected.emit(self.row_index)
    
    def set_selected(self, selected: bool):
        """Set row selection state"""
        self.is_selected = selected
        self._update_selection_style()
    
    def _update_selection_style(self):
        """Update visual style based on selection"""
        if self.is_selected:
            self.label_frame.setStyleSheet("""
                QFrame {
                    background: #2d2d2d;
                    border: 1px solid #00aaff;
                    border-radius: 3px;
                }
            """)
        else:
            self.label_frame.setStyleSheet("""
                QFrame {
                    background: #1e1e1e;
                    border: 1px solid transparent;
                    border-radius: 3px;
                }
                QFrame:hover {
                    background: #252525;
                    border: 1px solid #3a3a3a;
                }
            """)
    
    def _on_mute(self):
        self.row_data.mute = self.mute_btn.isChecked()
        self.row_changed.emit(self.row_index)
    
    def _on_solo(self):
        self.row_data.solo = self.solo_btn.isChecked()
        self.row_changed.emit(self.row_index)
    
    def _on_step_changed(self, row: int, step: int, active: bool, velocity: float):
        self.row_data.steps[step].active = active
        self.row_data.steps[step].velocity = velocity
        self.row_changed.emit(self.row_index)
        # Emit preview signal when step is activated (turned on)
        if active:
            self.step_activated.emit(self.row_index, velocity)
    
    def _on_step_selected(self, row: int, step: int):
        """Step was clicked - emit selection"""
        self.step_selected.emit(row, step)
    
    def _on_velocity_edit(self, row: int, step: int):
        """Step was held - request velocity edit"""
        self.velocity_edit.emit(row, step)
    
    def highlight_step(self, step: int):
        """Highlight the current playhead step"""
        for i, btn in enumerate(self.step_buttons):
            btn.set_playing(i == step)
    
    def select_step(self, step: int):
        """Select a specific step (deselect others)"""
        for i, btn in enumerate(self.step_buttons):
            btn.set_selected(i == step)

    def refresh(self):
        """Refresh all step buttons from row data."""
        for idx in range(len(self.step_buttons)):
            self.refresh_step(idx)

    def refresh_step(self, step_idx: int):
        if 0 <= step_idx < len(self.step_buttons) and step_idx < len(self.row_data.steps):
            step_data = self.row_data.steps[step_idx]
            self.step_buttons[step_idx].set_state(
                step_data.active,
                step_data.velocity,
                step_data.probability,
                step_data.offset,
                step_data.decay,
                step_data.slide,
            )


class PatternEditorWidget(QWidget):
    """Full pattern editor with step sequencer grid - Maschine/Push style"""
    
    pattern_changed = pyqtSignal()
    cell_selected = pyqtSignal(int, int)  # row, step - for routing to arrangement clip
    step_preview = pyqtSignal(int, float)  # row_index, velocity - for sound preview on toggle

    SCALE_PRESETS = {
        "Major": [0, 2, 4, 5, 7, 9, 11],
        "Natural Minor": [0, 2, 3, 5, 7, 8, 10],
        "Harmonic Minor": [0, 2, 3, 5, 7, 8, 11],
        "Dorian": [0, 2, 3, 5, 7, 9, 10],
        "Mixolydian": [0, 2, 4, 5, 7, 9, 10],
        "Phrygian": [0, 1, 3, 5, 7, 8, 10],
        "Lydian": [0, 2, 4, 6, 7, 9, 11],
        "Whole Tone": [0, 2, 4, 6, 8, 10],
        "Pentatonic Major": [0, 2, 4, 7, 9],
        "Pentatonic Minor": [0, 3, 5, 7, 10],
    }
    NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pattern: Optional[Pattern] = None
        self.row_widgets: List[DrumRowWidget] = []
        self.current_step = 0
        self.selected_row = -1
        self.selected_step = -1
        
        # Step clipboard for copy/paste
        self._step_clipboard: Optional[PatternStep] = None
        self._row_clipboard: Optional[List[PatternStep]] = None
        
        self._setup_ui()
        self._setup_shortcuts()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        
        # Header with pattern info and controls
        header = QFrame()
        header.setStyleSheet("background-color: #252525; border-radius: 4px;")
        header_layout = QHBoxLayout(header)
        
        # Pattern name
        self.pattern_label = QLabel("No Pattern")
        self.pattern_label.setStyleSheet("color: #00aaff; font-weight: bold; font-size: 12px;")
        
        # Selected cell info
        self.selection_label = QLabel("Select a cell to edit")
        self.selection_label.setStyleSheet("color: #888; font-size: 10px; margin-left: 20px;")
        
        # Pattern length
        length_label = QLabel("Steps:")
        length_label.setStyleSheet("color: #888;")
        self.length_spin = QSpinBox()
        self.length_spin.setRange(4, 64)
        self.length_spin.setValue(16)
        self.length_spin.setStyleSheet("""
            QSpinBox { background: #333; color: white; border: 1px solid #444; padding: 2px; }
        """)
        self.length_spin.valueChanged.connect(self._on_length_changed)
        
        # Swing
        swing_label = QLabel("Swing:")
        swing_label.setStyleSheet("color: #888;")
        self.swing_slider = QSlider(Qt.Horizontal)
        self.swing_slider.setRange(-100, 100)
        self.swing_slider.setValue(0)
        self.swing_slider.setFixedWidth(80)
        
        # Clear pattern
        clear_btn = QPushButton("Clear")
        clear_btn.setStyleSheet("""
            QPushButton { background: #444; color: white; border: 1px solid #555; padding: 4px 12px; border-radius: 3px; }
            QPushButton:hover { background: #ff4444; }
        """)
        clear_btn.clicked.connect(self._clear_pattern)
        
        # Presets
        self.preset_combo = QComboBox()
        self.preset_combo.addItems([
            "Empty", "Four on Floor", "Breakbeat", "Trap", 
            "House", "Techno", "Hip-Hop", "Φ-Groove"
        ])
        self.preset_combo.setStyleSheet("""
            QComboBox { background: #333; color: white; border: 1px solid #444; padding: 4px; }
            QComboBox::drop-down { border: none; }
        """)
        self.preset_combo.currentTextChanged.connect(self._apply_preset)
        
        header_layout.addWidget(self.pattern_label)
        header_layout.addWidget(self.selection_label)
        header_layout.addStretch()
        
        # Velocity slider for selected step
        vel_label = QLabel("Velocity:")
        vel_label.setStyleSheet("color: #888; font-size: 10px;")
        self.velocity_slider = QSlider(Qt.Horizontal)
        self.velocity_slider.setRange(0, 100)
        self.velocity_slider.setValue(80)
        self.velocity_slider.setFixedWidth(80)
        self.velocity_slider.setEnabled(False)
        self.velocity_slider.valueChanged.connect(self._on_velocity_slider_changed)
        self.velocity_value = QLabel("80%")
        self.velocity_value.setStyleSheet("color: #00ffaa; font-size: 10px; min-width: 30px;")
        
        header_layout.addWidget(vel_label)
        header_layout.addWidget(self.velocity_slider)
        header_layout.addWidget(self.velocity_value)
        header_layout.addSpacing(10)
        
        header_layout.addWidget(length_label)
        header_layout.addWidget(self.length_spin)
        header_layout.addWidget(swing_label)
        header_layout.addWidget(self.swing_slider)
        header_layout.addWidget(QLabel("Preset:"))
        header_layout.addWidget(self.preset_combo)
        header_layout.addWidget(clear_btn)
        
        layout.addWidget(header)

        advanced_bar = QFrame()
        advanced_bar.setStyleSheet("background-color: #1a1a1a; border-radius: 4px;")
        adv_layout = QHBoxLayout(advanced_bar)
        adv_layout.setContentsMargins(8, 4, 8, 4)
        adv_layout.setSpacing(6)
        root_label = QLabel("Root:")
        root_label.setStyleSheet("color: #888;")
        adv_layout.addWidget(root_label)
        self.scale_root_combo = QComboBox()
        for idx, note in enumerate(self.NOTE_NAMES):
            self.scale_root_combo.addItem(note, idx)
        self.scale_root_combo.setStyleSheet("QComboBox { background: #222; color: #eee; border: 1px solid #00aaff; padding: 2px 6px; }")
        adv_layout.addWidget(self.scale_root_combo)
        scale_label = QLabel("Scale:")
        scale_label.setStyleSheet("color: #888;")
        adv_layout.addWidget(scale_label)
        self.scale_type_combo = QComboBox()
        self.scale_type_combo.addItems(list(self.SCALE_PRESETS.keys()))
        self.scale_type_combo.setStyleSheet("QComboBox { background: #222; color: #eee; border: 1px solid #00aaff; padding: 2px 6px; }")
        adv_layout.addWidget(self.scale_type_combo)
        self.scale_apply_btn = QPushButton("Quantize Scale")
        self.scale_apply_btn.setStyleSheet("QPushButton { background: #333; color: #00ffaa; border: 1px solid #00ffaa; padding: 4px 10px; border-radius: 3px; }")
        self.scale_apply_btn.clicked.connect(self._apply_scale_lock)
        adv_layout.addWidget(self.scale_apply_btn)
        adv_layout.addSpacing(12)
        prob_label = QLabel("Prob.:")
        prob_label.setStyleSheet("color: #888;")
        adv_layout.addWidget(prob_label)
        self.probability_slider = QSlider(Qt.Horizontal)
        self.probability_slider.setRange(0, 100)
        self.probability_slider.setValue(100)
        self.probability_slider.setFixedWidth(90)
        self.probability_slider.setEnabled(False)
        self.probability_slider.valueChanged.connect(self._on_probability_slider_changed)
        adv_layout.addWidget(self.probability_slider)
        self.probability_value = QLabel("100%")
        self.probability_value.setStyleSheet("color: #ffaa00; min-width: 36px;")
        adv_layout.addWidget(self.probability_value)
        offset_label = QLabel("Offset:")
        offset_label.setStyleSheet("color: #888;")
        adv_layout.addWidget(offset_label)
        self.offset_spin = QDoubleSpinBox()
        self.offset_spin.setRange(-0.5, 0.5)
        self.offset_spin.setSingleStep(0.01)
        self.offset_spin.setDecimals(2)
        self.offset_spin.setSuffix(" stp")
        self.offset_spin.setStyleSheet("QDoubleSpinBox { background: #222; color: #eee; border: 1px solid #444; padding: 2px 6px; }")
        self.offset_spin.setEnabled(False)
        self.offset_spin.valueChanged.connect(self._on_offset_changed)
        adv_layout.addWidget(self.offset_spin)
        decay_label = QLabel("Decay:")
        decay_label.setStyleSheet("color: #888;")
        adv_layout.addWidget(decay_label)
        self.decay_spin = QDoubleSpinBox()
        self.decay_spin.setRange(0.1, 4.0)
        self.decay_spin.setSingleStep(0.1)
        self.decay_spin.setDecimals(2)
        self.decay_spin.setSuffix(" x")
        self.decay_spin.setStyleSheet("QDoubleSpinBox { background: #222; color: #eee; border: 1px solid #444; padding: 2px 6px; }")
        self.decay_spin.setEnabled(False)
        self.decay_spin.valueChanged.connect(self._on_decay_changed)
        adv_layout.addWidget(self.decay_spin)
        self.slide_check = QCheckBox("Slide")
        self.slide_check.setStyleSheet("color: #ccc;")
        self.slide_check.setEnabled(False)
        self.slide_check.toggled.connect(self._on_slide_toggled)
        adv_layout.addWidget(self.slide_check)
        adv_layout.addSpacing(12)
        density_label = QLabel("Density:")
        density_label.setStyleSheet("color: #888;")
        adv_layout.addWidget(density_label)
        self.random_density_spin = QSpinBox()
        self.random_density_spin.setRange(5, 100)
        self.random_density_spin.setValue(60)
        self.random_density_spin.setSuffix(" %")
        self.random_density_spin.setStyleSheet("QSpinBox { background: #222; color: #eee; border: 1px solid #444; padding: 2px 6px; }")
        adv_layout.addWidget(self.random_density_spin)
        self.randomize_row_btn = QPushButton("Randomize Row")
        self.randomize_row_btn.setStyleSheet("QPushButton { background: #333; color: #ccc; border: 1px solid #444; padding: 4px 10px; border-radius: 3px; }")
        self.randomize_row_btn.clicked.connect(self._randomize_selected_row)
        adv_layout.addWidget(self.randomize_row_btn)
        self.randomize_pattern_btn = QPushButton("Randomize Pattern")
        self.randomize_pattern_btn.setStyleSheet("QPushButton { background: #333; color: #ccc; border: 1px solid #444; padding: 4px 10px; border-radius: 3px; }")
        self.randomize_pattern_btn.clicked.connect(self._randomize_all_rows)
        adv_layout.addWidget(self.randomize_pattern_btn)
        adv_layout.addStretch()

        layout.addWidget(advanced_bar)
        self._set_scale_controls_enabled(False)
        self._set_step_controls_enabled(False)
        
        # Scroll area for rows
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("""
            QScrollArea { border: none; background: transparent; }
            QScrollBar:vertical { background: #1a1a1a; width: 8px; }
            QScrollBar::handle:vertical { background: #444; border-radius: 4px; }
        """)
        
        self.rows_container = QWidget()
        self.rows_layout = QVBoxLayout(self.rows_container)
        self.rows_layout.setContentsMargins(0, 0, 0, 0)
        self.rows_layout.setSpacing(1)
        
        scroll.setWidget(self.rows_container)
        layout.addWidget(scroll)
    
    def _setup_shortcuts(self):
        """Setup keyboard shortcuts for pattern editor"""
        # Copy/Paste step
        copy = QShortcut(QKeySequence("Ctrl+C"), self)
        copy.activated.connect(self.copy_step)
        
        paste = QShortcut(QKeySequence("Ctrl+V"), self)
        paste.activated.connect(self.paste_step)
        
        # Copy/Paste row
        copy_row = QShortcut(QKeySequence("Ctrl+Shift+C"), self)
        copy_row.activated.connect(self.copy_row)
        
        paste_row = QShortcut(QKeySequence("Ctrl+Shift+V"), self)
        paste_row.activated.connect(self.paste_row)
        
        # Delete step
        delete = QShortcut(QKeySequence(Qt.Key_Delete), self)
        delete.activated.connect(self.clear_selected_step)
        
        # Toggle step
        space = QShortcut(QKeySequence(Qt.Key_Space), self)
        space.activated.connect(self.toggle_selected_step)
        
        # Arrow key navigation
        left = QShortcut(QKeySequence(Qt.Key_Left), self)
        left.activated.connect(lambda: self._navigate(-1, 0))
        
        right = QShortcut(QKeySequence(Qt.Key_Right), self)
        right.activated.connect(lambda: self._navigate(1, 0))
        
        up = QShortcut(QKeySequence(Qt.Key_Up), self)
        up.activated.connect(lambda: self._navigate(0, -1))
        
        down = QShortcut(QKeySequence(Qt.Key_Down), self)
        down.activated.connect(lambda: self._navigate(0, 1))
        
        # Velocity up/down
        vel_up = QShortcut(QKeySequence("Shift+Up"), self)
        vel_up.activated.connect(lambda: self._adjust_velocity(0.1))
        
        vel_down = QShortcut(QKeySequence("Shift+Down"), self)
        vel_down.activated.connect(lambda: self._adjust_velocity(-0.1))
        
        # Fill row with current step
        fill = QShortcut(QKeySequence("Ctrl+F"), self)
        fill.activated.connect(self.fill_row)
    
    def copy_step(self):
        """Copy selected step to clipboard"""
        if self.pattern and 0 <= self.selected_row < len(self.pattern.rows):
            row = self.pattern.rows[self.selected_row]
            if 0 <= self.selected_step < len(row.steps):
                orig = row.steps[self.selected_step]
                self._step_clipboard = PatternStep(
                    active=orig.active,
                    velocity=orig.velocity,
                    note=orig.note,
                    probability=orig.probability,
                    offset=orig.offset,
                    decay=orig.decay,
                    slide=orig.slide
                )
    
    def paste_step(self):
        """Paste clipboard step to selected position"""
        if self._step_clipboard is None:
            return
        if self.pattern and 0 <= self.selected_row < len(self.pattern.rows):
            row = self.pattern.rows[self.selected_row]
            if 0 <= self.selected_step < len(row.steps):
                step = row.steps[self.selected_step]
                step.active = self._step_clipboard.active
                step.velocity = self._step_clipboard.velocity
                step.note = self._step_clipboard.note
                step.probability = self._step_clipboard.probability
                step.offset = self._step_clipboard.offset
                step.decay = self._step_clipboard.decay
                step.slide = self._step_clipboard.slide
                
                # Refresh display
                if 0 <= self.selected_row < len(self.row_widgets):
                    self.row_widgets[self.selected_row].refresh_step(self.selected_step)
                self._populate_step_controls(step)
                self.pattern_changed.emit()
    
    def copy_row(self):
        """Copy entire selected row"""
        if self.pattern and 0 <= self.selected_row < len(self.pattern.rows):
            row = self.pattern.rows[self.selected_row]
            self._row_clipboard = [
                PatternStep(
                    active=s.active,
                    velocity=s.velocity,
                    note=s.note,
                    probability=s.probability,
                    offset=s.offset,
                    decay=s.decay,
                    slide=s.slide
                ) for s in row.steps
            ]
    
    def paste_row(self):
        """Paste clipboard row to selected row"""
        if not self._row_clipboard:
            return
        if self.pattern and 0 <= self.selected_row < len(self.pattern.rows):
            row = self.pattern.rows[self.selected_row]
            for i, clip in enumerate(self._row_clipboard):
                if i < len(row.steps):
                    step = row.steps[i]
                    step.active = clip.active
                    step.velocity = clip.velocity
                    step.note = clip.note
                    step.probability = clip.probability
                    step.offset = clip.offset
                    step.decay = clip.decay
                    step.slide = clip.slide
            
            # Refresh row display
            self._refresh_row(self.selected_row)
            self.pattern_changed.emit()
    
    def clear_selected_step(self):
        """Clear the selected step"""
        if self.pattern and 0 <= self.selected_row < len(self.pattern.rows):
            row = self.pattern.rows[self.selected_row]
            if 0 <= self.selected_step < len(row.steps):
                step = row.steps[self.selected_step]
                step.active = False
                step.velocity = 0.8
                step.probability = 1.0
                step.offset = 0.0
                step.decay = 1.0
                step.slide = False
                
                if 0 <= self.selected_row < len(self.row_widgets):
                    self.row_widgets[self.selected_row].refresh_step(self.selected_step)
                self._populate_step_controls(step)
                self.pattern_changed.emit()
    
    def toggle_selected_step(self):
        """Toggle the selected step active state"""
        if self.pattern and 0 <= self.selected_row < len(self.pattern.rows):
            row = self.pattern.rows[self.selected_row]
            if 0 <= self.selected_step < len(row.steps):
                step = row.steps[self.selected_step]
                step.active = not step.active
                
                if 0 <= self.selected_row < len(self.row_widgets):
                    self.row_widgets[self.selected_row].refresh_step(self.selected_step)
                
                # Trigger preview if activating
                if step.active:
                    self.step_preview.emit(self.selected_row, step.velocity)
                
                self.pattern_changed.emit()
    
    def fill_row(self):
        """Fill row by repeating clipboard step"""
        if not self._step_clipboard:
            return
        if self.pattern and 0 <= self.selected_row < len(self.pattern.rows):
            row = self.pattern.rows[self.selected_row]
            for step in row.steps:
                step.active = self._step_clipboard.active
                step.velocity = self._step_clipboard.velocity
                step.note = self._step_clipboard.note
                step.probability = self._step_clipboard.probability
                step.offset = self._step_clipboard.offset
                step.decay = self._step_clipboard.decay
                step.slide = self._step_clipboard.slide
            
            self._refresh_row(self.selected_row)
            self.pattern_changed.emit()
    
    def _navigate(self, dx: int, dy: int):
        """Navigate selection by delta"""
        if not self.pattern:
            return
        
        new_step = max(0, min(self.pattern.steps - 1, self.selected_step + dx))
        new_row = max(0, min(len(self.pattern.rows) - 1, self.selected_row + dy))
        
        if new_row != self.selected_row or new_step != self.selected_step:
            self._on_step_selected(new_row, new_step)
    
    def _adjust_velocity(self, delta: float):
        """Adjust velocity of selected step"""
        if self.pattern and 0 <= self.selected_row < len(self.pattern.rows):
            row = self.pattern.rows[self.selected_row]
            if 0 <= self.selected_step < len(row.steps):
                step = row.steps[self.selected_step]
                step.velocity = max(0.0, min(1.0, step.velocity + delta))
                
                if 0 <= self.selected_row < len(self.row_widgets):
                    self.row_widgets[self.selected_row].refresh_step(self.selected_step)
                self._populate_step_controls(step)
                self.pattern_changed.emit()
    
    def _refresh_row(self, row_index: int):
        """Refresh all steps in a row"""
        if 0 <= row_index < len(self.row_widgets):
            widget = self.row_widgets[row_index]
            for i in range(len(widget.step_buttons)):
                widget.refresh_step(i)
    
    def _set_scale_controls_enabled(self, enabled: bool):
        if hasattr(self, 'scale_root_combo'):
            for widget in (self.scale_root_combo, self.scale_type_combo, self.scale_apply_btn):
                widget.setEnabled(enabled)

    def _set_step_controls_enabled(self, enabled: bool):
        for widget in (
            self.velocity_slider,
            self.probability_slider,
            self.offset_spin,
            self.decay_spin,
            self.slide_check,
        ):
            widget.setEnabled(enabled)

    def _reset_step_control_values(self):
        self.velocity_slider.blockSignals(True)
        self.velocity_slider.setValue(0)
        self.velocity_slider.blockSignals(False)
        self.velocity_value.setText("0%")
        self.probability_slider.blockSignals(True)
        self.probability_slider.setValue(100)
        self.probability_slider.blockSignals(False)
        self.probability_value.setText("100%")
        self.offset_spin.blockSignals(True)
        self.offset_spin.setValue(0.0)
        self.offset_spin.blockSignals(False)
        self.decay_spin.blockSignals(True)
        self.decay_spin.setValue(1.0)
        self.decay_spin.blockSignals(False)
        self.slide_check.blockSignals(True)
        self.slide_check.setChecked(False)
        self.slide_check.blockSignals(False)

    def _populate_step_controls(self, step_data: PatternStep):
        self._set_step_controls_enabled(True)
        self.velocity_slider.blockSignals(True)
        self.velocity_slider.setValue(int(step_data.velocity * 100))
        self.velocity_slider.blockSignals(False)
        self.velocity_value.setText(f"{int(step_data.velocity * 100)}%")
        self.probability_slider.blockSignals(True)
        self.probability_slider.setValue(int(step_data.probability * 100))
        self.probability_slider.blockSignals(False)
        self.probability_value.setText(f"{int(step_data.probability * 100)}%")
        self.offset_spin.blockSignals(True)
        self.offset_spin.setValue(step_data.offset)
        self.offset_spin.blockSignals(False)
        self.decay_spin.blockSignals(True)
        self.decay_spin.setValue(step_data.decay)
        self.decay_spin.blockSignals(False)
        self.slide_check.blockSignals(True)
        self.slide_check.setChecked(step_data.slide)
        self.slide_check.blockSignals(False)

    def set_pattern(self, pattern: Pattern):
        """Load a pattern into the editor"""
        self.pattern = pattern
        self.pattern_label.setText(pattern.name)
        self.length_spin.setValue(pattern.steps)
        self.selected_row = -1
        self.selected_step = -1
        self.selection_label.setText("Select a cell to edit")
        self._set_step_controls_enabled(False)
        self._reset_step_control_values()
        
        # Clear existing rows
        for widget in self.row_widgets:
            widget.deleteLater()
        self.row_widgets.clear()
        
        # Create row widgets
        for i, row in enumerate(pattern.rows):
            row_widget = DrumRowWidget(i, row)
            row_widget.row_changed.connect(self._on_row_changed)
            row_widget.row_selected.connect(self._on_row_selected)
            row_widget.step_selected.connect(self._on_step_selected)
            row_widget.velocity_edit.connect(self._on_velocity_edit_request)
            row_widget.step_activated.connect(self._on_step_activated)
            self.row_widgets.append(row_widget)
            self.rows_layout.addWidget(row_widget)
        
        self.rows_layout.addStretch()
        self._set_scale_controls_enabled(not pattern.is_drum)
    
    def _get_selected_step(self):
        if not self.pattern:
            return None, None
        if self.selected_row < 0 or self.selected_row >= len(self.pattern.rows):
            return None, None
        row = self.pattern.rows[self.selected_row]
        if self.selected_step < 0 or self.selected_step >= len(row.steps):
            return row, None
        return row, row.steps[self.selected_step]

    def _on_velocity_slider_changed(self, value):
        """Velocity slider changed - update selected step"""
        self.velocity_value.setText(f"{value}%")
        
        if self.selected_row >= 0 and self.selected_step >= 0:
            if self.selected_row < len(self.row_widgets):
                row_widget = self.row_widgets[self.selected_row]
                if self.selected_step < len(row_widget.step_buttons):
                    row_widget.step_buttons[self.selected_step].set_velocity(value / 100.0)
                    self.pattern_changed.emit()

    def _on_probability_slider_changed(self, value: int):
        """Probability slider changed - update selected step trigger chance."""
        self.probability_value.setText(f"{value}%")
        if not self.pattern:
            return
        if self.selected_row < 0 or self.selected_step < 0:
            return
        if self.selected_row >= len(self.pattern.rows):
            return
        row = self.pattern.rows[self.selected_row]
        if self.selected_step >= len(row.steps):
            return
        row.steps[self.selected_step].probability = value / 100.0
        if 0 <= self.selected_row < len(self.row_widgets):
            row_widget = self.row_widgets[self.selected_row]
            if self.selected_step < len(row_widget.step_buttons):
                row_widget.step_buttons[self.selected_step].set_probability(value / 100.0)
        self.pattern_changed.emit()

    def _on_offset_changed(self, value: float):
        row, step = self._get_selected_step()
        if step is None:
            return
        step.offset = value
        if 0 <= self.selected_row < len(self.row_widgets):
            self.row_widgets[self.selected_row].refresh_step(self.selected_step)
        self.pattern_changed.emit()

    def _on_decay_changed(self, value: float):
        row, step = self._get_selected_step()
        if step is None:
            return
        step.decay = value
        if 0 <= self.selected_row < len(self.row_widgets):
            self.row_widgets[self.selected_row].refresh_step(self.selected_step)
        self.pattern_changed.emit()

    def _on_slide_toggled(self, checked: bool):
        row, step = self._get_selected_step()
        if step is None:
            return
        step.slide = checked
        if 0 <= self.selected_row < len(self.row_widgets):
            self.row_widgets[self.selected_row].refresh_step(self.selected_step)
        self.pattern_changed.emit()
    
    def _on_row_selected(self, row_index: int):
        """Row header was clicked"""
        # Deselect previous row
        if 0 <= self.selected_row < len(self.row_widgets):
            self.row_widgets[self.selected_row].set_selected(False)
        
        self.selected_row = row_index
        self.selected_step = -1
        self._set_step_controls_enabled(False)
        self._reset_step_control_values()
        if 0 <= row_index < len(self.row_widgets):
            self.row_widgets[row_index].set_selected(True)
            row_name = self.pattern.rows[row_index].name if self.pattern else "Row"
            self.selection_label.setText(f"🥁 {row_name}")
    
    def _on_step_selected(self, row: int, step: int):
        """A step was clicked - select it"""
        # Deselect previous
        if 0 <= self.selected_row < len(self.row_widgets):
            self.row_widgets[self.selected_row].select_step(-1)
        
        self.selected_row = row
        self.selected_step = step
        
        # Select new
        if 0 <= row < len(self.row_widgets):
            self.row_widgets[row].select_step(step)
            self.row_widgets[row].set_selected(True)
            
            # Update velocity slider
            if self.pattern and row < len(self.pattern.rows):
                step_data = self.pattern.rows[row].steps[step]
                row_name = self.pattern.rows[row].name
                self.selection_label.setText(f"🥁 {row_name} • Step {step + 1}")
                self._populate_step_controls(step_data)
            else:
                self._set_step_controls_enabled(False)
                self._reset_step_control_values()
        else:
            self._set_step_controls_enabled(False)
            self._reset_step_control_values()
        
        # Emit cell selection for routing to arrangement clip
        self.cell_selected.emit(row, step)
    
    def _on_velocity_edit_request(self, row: int, step: int):
        """Step was held - show velocity editing UI"""
        # Select the step first
        self._on_step_selected(row, step)
        # Focus on velocity slider
        self.velocity_slider.setFocus()
    
    def _on_row_changed(self, row_index: int):
        self.pattern_changed.emit()

    def _on_step_activated(self, row_index: int, velocity: float):
        """A step was turned on - emit preview signal so DAW can play the sound"""
        self.step_preview.emit(row_index, velocity)

    def _randomize_selected_row(self):
        if self.pattern is None or self.selected_row < 0:
            return
        self._randomize_rows([self.selected_row])

    def _randomize_all_rows(self):
        if self.pattern is None:
            return
        self._randomize_rows(list(range(len(self.pattern.rows))))

    def _randomize_rows(self, row_indices: List[int]):
        if not self.pattern:
            return
        density = self.random_density_spin.value() / 100.0
        for idx in row_indices:
            if idx >= len(self.pattern.rows):
                continue
            row = self.pattern.rows[idx]
            for step in row.steps:
                step.active = np.random.random() < density
                if step.active:
                    step.velocity = 0.45 + np.random.random() * 0.5
                    step.probability = 0.5 + np.random.random() * 0.5
                    step.offset = (np.random.random() - 0.5) * 0.1
                    step.decay = 0.7 + np.random.random() * 0.6
                    step.slide = np.random.random() < 0.1
                else:
                    step.velocity = 0.8
                    step.probability = 1.0
                    step.offset = 0.0
                    step.decay = 1.0
                    step.slide = False
            self._refresh_row(idx)
        self.pattern_changed.emit()
    
    def _on_length_changed(self, value: int):
        if self.pattern:
            self.pattern.resize(value)
            self.set_pattern(self.pattern)
            self.pattern_changed.emit()
    
    def _clear_pattern(self):
        if self.pattern:
            for row in self.pattern.rows:
                for step in row.steps:
                    step.active = False
                    step.velocity = 0.8
                    step.probability = 1.0
                    step.offset = 0.0
                    step.decay = 1.0
                    step.slide = False
            self.set_pattern(self.pattern)
            self.pattern_changed.emit()
    
    def _apply_preset(self, preset_name: str):
        if not self.pattern:
            return
        
        # Clear first
        for row in self.pattern.rows:
            for step in row.steps:
                step.active = False
                step.velocity = 0.8
                step.probability = 1.0
                step.offset = 0.0
                step.decay = 1.0
                step.slide = False
        
        presets = {
            "Four on Floor": {
                DrumType.KICK: [0, 4, 8, 12],
                DrumType.SNARE: [4, 12],
                DrumType.HIHAT_CLOSED: [0, 2, 4, 6, 8, 10, 12, 14],
            },
            "Breakbeat": {
                DrumType.KICK: [0, 6, 10],
                DrumType.SNARE: [4, 12],
                DrumType.HIHAT_CLOSED: [0, 2, 4, 6, 8, 10, 12, 14],
                DrumType.HIHAT_OPEN: [3, 11],
            },
            "Trap": {
                DrumType.KICK: [0, 7, 11],
                DrumType.SNARE: [4, 12],
                DrumType.HIHAT_CLOSED: list(range(16)),
                DrumType.CLAP: [4, 12],
            },
            "House": {
                DrumType.KICK: [0, 4, 8, 12],
                DrumType.CLAP: [4, 12],
                DrumType.HIHAT_CLOSED: [2, 6, 10, 14],
                DrumType.HIHAT_OPEN: [0, 4, 8, 12],
            },
            "Techno": {
                DrumType.KICK: [0, 4, 8, 12],
                DrumType.HIHAT_CLOSED: [2, 6, 10, 14],
                DrumType.CLAP: [4, 12],
                DrumType.RIMSHOT: [2, 10],
            },
            "Hip-Hop": {
                DrumType.KICK: [0, 5, 8, 13],
                DrumType.SNARE: [4, 12],
                DrumType.HIHAT_CLOSED: [0, 2, 4, 6, 8, 10, 12, 14],
            },
            "Φ-Groove": {
                DrumType.KICK: [0, 5, 10],  # Golden ratio positions
                DrumType.SNARE: [3, 8, 13],
                DrumType.HIHAT_CLOSED: [0, 2, 5, 7, 10, 12, 15],
                DrumType.PERC_1: [1, 6, 11],
                DrumType.PERC_2: [4, 9, 14],
            },
        }
        
        if preset_name in presets:
            for row in self.pattern.rows:
                if row.drum_type in presets[preset_name]:
                    for step_idx in presets[preset_name][row.drum_type]:
                        if step_idx < len(row.steps):
                            row.steps[step_idx].active = True
        
        self.set_pattern(self.pattern)
        self.pattern_changed.emit()
    
    def _apply_scale_lock(self):
        if not self.pattern or self.pattern.is_drum:
            return
        scale_name = self.scale_type_combo.currentText()
        intervals = self.SCALE_PRESETS.get(scale_name)
        if not intervals:
            return
        root_data = self.scale_root_combo.currentData()
        root = int(root_data) if root_data is not None else 0
        for idx, row in enumerate(self.pattern.rows):
            if row.drum_type is not None:
                continue
            row.note = self._quantize_note_to_scale(row.note, root, intervals)
            for step in row.steps:
                step.note = self._quantize_note_to_scale(step.note, root, intervals)
            self._refresh_row(idx)
        root_label = self.scale_root_combo.currentText()
        self.selection_label.setText(f"🎹 Scale locked to {root_label} {scale_name}")
        self.pattern_changed.emit()

    def _quantize_note_to_scale(self, note: int, root: int, intervals: List[int]) -> int:
        base_octave = (note - root) // 12
        candidates = []
        for octave_offset in (-1, 0, 1):
            for interval in intervals:
                candidates.append(root + interval + (base_octave + octave_offset) * 12)
        return min(candidates, key=lambda candidate: abs(candidate - note))

    def highlight_step(self, step: int):
        """Highlight current playhead step"""
        self.current_step = step
        for row_widget in self.row_widgets:
            row_widget.highlight_step(step)

    def _refresh_row(self, row_index: int):
        if 0 <= row_index < len(self.row_widgets):
            self.row_widgets[row_index].refresh()


# ═══════════════════════════════════════════════════════════════════════════════
# INSTRUMENT SELECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class InstrumentSelector(QWidget):
    """Widget to select instrument type for a track"""
    
    instrument_changed = pyqtSignal(str, dict)  # instrument_type, params
    
    INSTRUMENTS = {
        "Drums": {
            "808 Kit": {"kit": "808", "type": "drums"},
            "909 Kit": {"kit": "909", "type": "drums"},
            "Acoustic Kit": {"kit": "acoustic", "type": "drums"},
            "Φ-Kit": {"kit": "phi", "type": "drums"},
        },
        "Synth Lead": {
            "Φ-Lead": {"preset": "Φ-Lead", "type": "synth"},
            "Super Saw": {"preset": "Super Saw", "type": "synth"},
            "Sine Lead": {"preset": "Sine Lead", "type": "synth"},
            "Square Lead": {"preset": "Square Lead", "type": "synth"},
        },
        "Synth Bass": {
            "Deep Bass": {"preset": "Deep Bass", "type": "synth"},
            "Acid Bass": {"preset": "Acid Bass", "type": "synth"},
            "Sub Bass": {"preset": "Sub Bass", "type": "synth"},
            "Reese Bass": {"preset": "Reese Bass", "type": "synth"},
        },
        "Synth Pad": {
            "Warm Pad": {"preset": "Warm Pad", "type": "synth"},
            "Soft Pad": {"preset": "Soft Pad", "type": "synth"},
            "Bright Pad": {"preset": "Bright Pad", "type": "synth"},
            "Φ-Pad": {"preset": "Φ-Pad", "type": "synth"},
        },
        "Keys": {
            "Piano": {"preset": "Piano", "type": "synth"},
            "Electric Piano": {"preset": "Electric Piano", "type": "synth"},
            "Organ": {"preset": "Organ", "type": "synth"},
            "Pluck": {"preset": "Pluck", "type": "synth"},
        },
    }
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        
        # Category selector
        self.category_combo = QComboBox()
        self.category_combo.addItems(list(self.INSTRUMENTS.keys()))
        self.category_combo.setStyleSheet("""
            QComboBox { 
                background: #333; color: #00aaff; 
                border: 1px solid #00aaff; padding: 6px; 
                font-weight: bold;
            }
        """)
        self.category_combo.currentTextChanged.connect(self._update_instruments)
        
        # Instrument list
        self.instrument_list = QListWidget()
        self.instrument_list.setStyleSheet("""
            QListWidget {
                background: #1a1a1a;
                border: 1px solid #333;
                color: #ccc;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #2a2a2a;
            }
            QListWidget::item:hover {
                background: #2a2a2a;
            }
            QListWidget::item:selected {
                background: #00aaff;
                color: white;
            }
        """)
        self.instrument_list.itemClicked.connect(self._on_instrument_selected)
        
        layout.addWidget(QLabel("Category:"))
        layout.addWidget(self.category_combo)
        layout.addWidget(QLabel("Instrument:"))
        layout.addWidget(self.instrument_list)
        
        self._update_instruments(self.category_combo.currentText())
    
    def _update_instruments(self, category: str):
        self.instrument_list.clear()
        if category in self.INSTRUMENTS:
            for name in self.INSTRUMENTS[category].keys():
                self.instrument_list.addItem(name)
    
    def _on_instrument_selected(self, item: QListWidgetItem):
        category = self.category_combo.currentText()
        name = item.text()
        if category in self.INSTRUMENTS and name in self.INSTRUMENTS[category]:
            params = self.INSTRUMENTS[category][name]
            self.instrument_changed.emit(name, params)


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    'DrumType', 'DrumSound', 'DrumSynthesizer',
    'PatternStep', 'PatternRow', 'Pattern', 'PatternPlayer',
    'StepButton', 'DrumRowWidget', 'PatternEditorWidget',
    'InstrumentSelector',
]
