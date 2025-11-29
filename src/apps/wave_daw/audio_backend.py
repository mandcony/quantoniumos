"""
Wave DAW Audio Backend - Real-time audio I/O with sounddevice.

Integrates with Session/AudioEngine for proper DAW graph audio processing.
"""

import numpy as np
import threading
from typing import Optional, Callable, List
from dataclasses import dataclass

try:
    import sounddevice as sd
except ImportError:
    sd = None
    print("Warning: sounddevice not installed. Audio playback disabled.")

# Golden ratio for RFT processing
PHI = (1 + np.sqrt(5)) / 2


@dataclass
class AudioBackendConfig:
    """Audio backend configuration"""
    sample_rate: int = 44100
    block_size: int = 512
    channels: int = 2
    dtype: str = 'float32'


class AudioBackend:
    """
    Real-time audio backend using sounddevice.
    
    Integrates with Session/AudioEngine for:
    - Playing back DAW tracks through the audio graph
    - Synth input for real-time keyboard playing
    - Metronome and click track
    """
    
    def __init__(self, config: AudioBackendConfig = None, session=None):
        self.config = config or AudioBackendConfig()
        self.stream: Optional[sd.OutputStream] = None
        self.is_running = False
        
        # Session and engine references
        self.session = session
        self.engine = session.engine if session else None
        
        # Playback state (fallback when no session)
        self.playing = False
        self.position_samples = 0
        self.tempo_bpm = 120.0
        self.metronome_enabled = True
        
        # Audio generators
        self.generators: List[Callable] = []
        
        # Clips to play (start_sample, audio_data) - legacy fallback
        self.clips: List[tuple] = []
        
        # Preview sounds (one-shots triggered by UI for instant playback)
        self.preview_sounds: List[tuple] = []  # (offset, audio_data)
        
        # Synth for real-time keyboard playback
        self.synth = None
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Metronome state
        self._metro_phase = 0.0
        self._last_beat = -1
    
    def attach_session(self, session):
        """Attach a Session after construction"""
        with self.lock:
            self.session = session
            self.engine = session.engine if session else None
            if session:
                self.tempo_bpm = session.tempo
    
    def set_synth(self, synth):
        """Set the synth for real-time playback"""
        with self.lock:
            self.synth = synth
        
    def start(self):
        """Start the audio stream"""
        if sd is None:
            print("sounddevice not available")
            return False
            
        try:
            self.stream = sd.OutputStream(
                samplerate=self.config.sample_rate,
                blocksize=self.config.block_size,
                channels=self.config.channels,
                dtype=self.config.dtype,
                callback=self._audio_callback
            )
            self.stream.start()
            self.is_running = True
            print(f"Audio backend started: {self.config.sample_rate}Hz, {self.config.block_size} samples")
            return True
        except Exception as e:
            print(f"Failed to start audio: {e}")
            return False
    
    def stop(self):
        """Stop the audio stream"""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.is_running = False
        print("Audio backend stopped")
    
    def play(self):
        """Start playback"""
        with self.lock:
            self.playing = True
            if self.session:
                self.session.transport.playing = True
    
    def pause(self):
        """Pause playback"""
        with self.lock:
            self.playing = False
            if self.session:
                self.session.transport.playing = False
    
    def stop_playback(self):
        """Stop and reset to beginning"""
        with self.lock:
            self.playing = False
            self.position_samples = 0
            self._last_beat = -1
            if self.session:
                self.session.transport.playing = False
                self.session.transport.position = 0.0
    
    def set_tempo(self, bpm: float):
        """Set tempo in BPM"""
        bpm = max(20.0, min(300.0, bpm))
        with self.lock:
            self.tempo_bpm = bpm
            if self.session:
                self.session.tempo = bpm
    
    def set_position(self, beat: float):
        """Set position in beats"""
        samples_per_beat = (self.config.sample_rate * 60) / self.tempo_bpm
        with self.lock:
            self.position_samples = int(beat * samples_per_beat)
            self._last_beat = int(beat) - 1
            if self.session:
                self.session.transport.position = beat
    
    def get_position_beats(self) -> float:
        """Get current position in beats"""
        if self.session:
            return self.session.transport.position
        samples_per_beat = (self.config.sample_rate * 60) / self.tempo_bpm
        return self.position_samples / samples_per_beat
    
    def add_clip(self, start_beat: float, audio_data: np.ndarray):
        """Add an audio clip to play (legacy - prefer using Session tracks)"""
        samples_per_beat = (self.config.sample_rate * 60) / self.tempo_bpm
        start_sample = int(start_beat * samples_per_beat)
        with self.lock:
            self.clips.append((start_sample, audio_data))
    
    def clear_clips(self):
        """Clear all clips"""
        with self.lock:
            self.clips.clear()

    def play_preview(self, audio_data: np.ndarray):
        """Queue a one-shot preview sound for immediate playback (e.g. drum hit when step toggled)"""
        with self.lock:
            # Convert to float32 if needed
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            self.preview_sounds.append((0, audio_data))
    
    def _audio_callback(self, outdata: np.ndarray, frames: int, 
                        time_info, status):
        """
        Audio callback - called by sounddevice for each block.
        This runs in a separate thread!
        """
        if status:
            pass  # Suppress underflow messages
        
        # Start with silence
        outdata.fill(0)
        
        with self.lock:
            # Always render synth (for keyboard playing even when stopped)
            if self.synth is not None:
                try:
                    synth_audio = self.synth.render(frames)
                    # Convert mono to stereo
                    outdata[:, 0] += synth_audio
                    outdata[:, 1] += synth_audio
                except Exception:
                    pass  # Ignore synth errors

            # Always mix preview sounds (drum hits, etc.) even when stopped
            self._mix_preview_sounds(outdata, frames)
            
            # Use engine if available, otherwise fallback
            if self.engine and self.session:
                self._process_with_engine(outdata, frames)
            else:
                self._process_fallback(outdata, frames)
    
    def _process_with_engine(self, outdata: np.ndarray, frames: int):
        """Process audio through the Session's AudioEngine"""
        if not self.session.transport.playing:
            return
        
        try:
            # Get audio from the engine (processes all tracks and devices)
            engine_out = self.engine.process_block(frames)
            
            # Engine returns (channels, frames) or (frames,)
            if engine_out is not None:
                if engine_out.ndim == 1:
                    outdata[:, 0] += engine_out[:frames]
                    outdata[:, 1] += engine_out[:frames]
                else:
                    outdata[:, 0] += engine_out[0, :frames]
                    outdata[:, 1] += engine_out[1, :frames] if engine_out.shape[0] > 1 else engine_out[0, :frames]
            
            # Calculate timing for metronome
            samples_per_beat = (self.config.sample_rate * 60) / self.session.tempo
            current_beat = self.session.transport.position
            
            # Generate metronome clicks
            if self.metronome_enabled:
                self._generate_metronome(outdata, frames, current_beat, samples_per_beat)
            
            # Advance transport
            beat_increment = frames / samples_per_beat
            self.session.transport.position += beat_increment
            
        except Exception as e:
            print(f"Engine process error: {e}")
    
    def _process_fallback(self, outdata: np.ndarray, frames: int):
        """Fallback processing without Session (metronome + legacy clips)"""
        if not self.playing:
            return
        
        # Calculate timing
        samples_per_beat = (self.config.sample_rate * 60) / self.tempo_bpm
        current_beat = self.position_samples / samples_per_beat
        
        # Generate metronome clicks
        if self.metronome_enabled:
            self._generate_metronome(outdata, frames, current_beat, samples_per_beat)
        
        # Mix in any legacy clips
        for start_sample, audio_data in self.clips:
            self._mix_clip(outdata, frames, start_sample, audio_data)
        
        # Advance position
        self.position_samples += frames
    
    def _generate_metronome(self, outdata: np.ndarray, frames: int,
                            current_beat: float, samples_per_beat: float):
        """Generate metronome clicks"""
        current_beat_int = int(current_beat)
        
        # Check if we crossed a beat boundary
        if current_beat_int > self._last_beat:
            self._last_beat = current_beat_int
            
            # Determine click parameters
            is_downbeat = (current_beat_int % 4) == 0
            freq = 1000.0 if is_downbeat else 800.0
            amplitude = 0.5 if is_downbeat else 0.3
            click_samples = int(0.02 * self.config.sample_rate)  # 20ms click
            
            # Generate click
            t = np.arange(click_samples) / self.config.sample_rate
            
            # Sine with exponential decay
            envelope = np.exp(-t * 50)
            click = amplitude * np.sin(2 * np.pi * freq * t) * envelope
            
            # Add harmonics for richer sound
            click += amplitude * 0.3 * np.sin(2 * np.pi * freq * 2 * t) * envelope
            
            # Calculate where in the block the beat occurs
            beat_position_in_block = (current_beat - int(current_beat)) * samples_per_beat
            start_idx = max(0, int(frames - beat_position_in_block))
            
            # Mix click into output
            end_idx = min(frames, start_idx + len(click))
            click_len = end_idx - start_idx
            
            if click_len > 0 and start_idx < frames:
                outdata[start_idx:end_idx, 0] += click[:click_len]
                outdata[start_idx:end_idx, 1] += click[:click_len]
    
    def _mix_preview_sounds(self, outdata: np.ndarray, frames: int):
        """Mix one-shot preview sounds (drum hits, etc.) into output and advance their position"""
        new_previews = []
        for offset, audio in self.preview_sounds:
            if offset >= len(audio):
                continue  # Sound finished, discard
            end = min(offset + frames, len(audio))
            length = end - offset
            if length > 0:
                if audio.ndim == 1:
                    outdata[:length, 0] += audio[offset:end]
                    outdata[:length, 1] += audio[offset:end]
                else:
                    ch_count = min(audio.shape[0], 2)
                    for ch in range(ch_count):
                        outdata[:length, ch] += audio[ch, offset:end]
            # Keep if not finished
            if end < len(audio):
                new_previews.append((end, audio))
        self.preview_sounds = new_previews

    def _mix_clip(self, outdata: np.ndarray, frames: int,
                  start_sample: int, audio_data: np.ndarray):
        """Mix a clip into the output buffer (legacy)"""
        clip_start = start_sample - self.position_samples
        clip_end = clip_start + len(audio_data)
        
        # Check if clip overlaps with this block
        if clip_end < 0 or clip_start >= frames:
            return
        
        # Calculate overlap region
        out_start = max(0, clip_start)
        out_end = min(frames, clip_end)
        clip_offset = max(0, -clip_start)
        
        # Mix
        length = out_end - out_start
        if audio_data.ndim == 1:
            outdata[out_start:out_end, 0] += audio_data[clip_offset:clip_offset + length]
            outdata[out_start:out_end, 1] += audio_data[clip_offset:clip_offset + length]
        else:
            channels = min(audio_data.shape[0], 2)
            for ch in range(channels):
                outdata[out_start:out_end, ch] += audio_data[ch, clip_offset:clip_offset + length]


class TestToneGenerator:
    """Generate test tones for audio testing"""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
    
    def sine(self, freq: float = 440.0, duration: float = 1.0, 
             amplitude: float = 0.5) -> np.ndarray:
        """Generate a sine wave"""
        t = np.arange(int(duration * self.sample_rate)) / self.sample_rate
        return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    
    def phi_chord(self, root: float = 220.0, duration: float = 2.0,
                  amplitude: float = 0.3) -> np.ndarray:
        """Generate a chord based on golden ratio intervals"""
        t = np.arange(int(duration * self.sample_rate)) / self.sample_rate
        
        # Golden ratio frequency relationships
        freqs = [
            root,
            root * PHI,           # ~356 Hz
            root * PHI * PHI,     # ~576 Hz
            root * 2,             # Octave
        ]
        
        # Envelope
        attack = 0.1
        release = 0.5
        env = np.ones_like(t)
        attack_samples = int(attack * self.sample_rate)
        release_samples = int(release * self.sample_rate)
        env[:attack_samples] = np.linspace(0, 1, attack_samples)
        env[-release_samples:] = np.linspace(1, 0, release_samples)
        
        # Sum harmonics
        signal = np.zeros_like(t)
        for i, freq in enumerate(freqs):
            amp = amplitude / (i + 1)
            signal += amp * np.sin(2 * np.pi * freq * t)
        
        return (signal * env).astype(np.float32)
    
    def drum_hit(self, duration: float = 0.3) -> np.ndarray:
        """Generate a simple drum-like sound"""
        t = np.arange(int(duration * self.sample_rate)) / self.sample_rate
        
        # Pitch envelope (starts high, drops fast)
        pitch_env = 200 * np.exp(-t * 30) + 60
        
        # Amplitude envelope
        amp_env = np.exp(-t * 10)
        
        # Generate with frequency modulation
        phase = np.cumsum(2 * np.pi * pitch_env / self.sample_rate)
        signal = 0.6 * np.sin(phase) * amp_env
        
        # Add some noise for attack
        noise = np.random.randn(len(t)) * 0.2 * np.exp(-t * 50)
        signal += noise
        
        return signal.astype(np.float32)


# Singleton backend instance
_backend: Optional[AudioBackend] = None


def get_audio_backend() -> AudioBackend:
    """Get or create the global audio backend"""
    global _backend
    if _backend is None:
        _backend = AudioBackend()
    return _backend


def init_audio() -> bool:
    """Initialize the audio system"""
    backend = get_audio_backend()
    return backend.start()


def init_audio_with_session(session) -> AudioBackend:
    """Initialize audio with a Session for full DAW integration"""
    global _backend
    if _backend is None:
        _backend = AudioBackend(session=session)
    else:
        _backend.attach_session(session)
    _backend.start()
    return _backend


def shutdown_audio():
    """Shutdown the audio system"""
    global _backend
    if _backend:
        _backend.stop()
        _backend = None
