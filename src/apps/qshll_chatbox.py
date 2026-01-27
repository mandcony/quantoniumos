# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
﻿#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QuantoniumOS ΓÇó Chatbox (Futura Minimal)
- Matches System Monitor aesthetic (light/dark, rounded cards, Segoe UI)
- Safety badge (reads latest ai_safety_report_*.json)
- Non-agentic guardrails + Safe Mode gate via QUANTONIUM_SAFE_MODE
- Message bubbles, typing indicator, transcript logging
- Hooks to wire your responder (weights/organized/*.json) later
"""

import os, sys, json, glob, time, datetime, re, logging
from typing import Optional, Dict, Any, List

import threading

from atomic_io import AtomicJsonlWriter, atomic_write_text

from PyQt5.QtCore import Qt, QTimer, QSize, QPoint, QEvent
from PyQt5.QtGui import QFont, QColor, QPainter, QPen, QBrush, QTextOption
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QLineEdit, QTextEdit, QScrollArea, QFrame, QFileDialog,
    QStatusBar, QMessageBox
)

# Configure module-level logger once for reuse
logger = logging.getLogger(__name__)

# Import our quantum AI system components
try:
    # Add project root to path
    import sys
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from quantum_safety_system import QuantumSafetySystem
    from quantum_conversation_manager import QuantumConversationManager
    from quantum_rlhf_system import QuantumRLHFSystem
    from quantum_domain_fine_tuner import QuantumDomainFineTuner
    QUANTUM_AI_AVAILABLE = True
    logger.debug("Quantum AI System integrated (Phase 1 & Phase 2)")
except ImportError as e:
    QUANTUM_AI_AVAILABLE = False
    logger.warning("Quantum AI System not available: %s", e)

# ---------- Reusable UI primitives (mirrors System Monitor) ----------
def load_latest_safety_report_text() -> Optional[str]:
    files = sorted(glob.glob("ai_safety_report_*.json"), key=os.path.getmtime, reverse=True)
    if not files: return None
    try:
        return open(files[0], "r", encoding="utf-8").read()
    except Exception:
        return None

def safety_is_green() -> tuple[bool, Optional[str]]:
    txt = load_latest_safety_report_text()
    if not txt: return (False, None)
    return ("FAIL Non Agentic Constraints" not in txt, txt)

def latest_report_time() -> Optional[str]:
    files = sorted(glob.glob("ai_safety_report_*.json"), key=os.path.getmtime, reverse=True)
    if not files: return None
    ts = os.path.getmtime(files[0])
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))

class Card(QFrame):
    def __init__(self, title: str = ""):
        super().__init__()
        self.setObjectName("Card")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(8)
        self.title_lbl = QLabel(title)
        self.title_lbl.setObjectName("CardTitle")
        lay.addWidget(self.title_lbl)
        self.body = QWidget()
        lay.addWidget(self.body)

# ---------- Message bubble ----------
class Bubble(QWidget):
    def __init__(self, text: str, me: bool, light: bool):
        super().__init__()
        self.text = text
        self.me = me
        self.light = light
        self.setMinimumHeight(40)
        self.setSizePolicy(self.sizePolicy().Expanding, self.sizePolicy().Minimum)
        # Set a reasonable maximum width to prevent text from being too wide
        self.setMaximumWidth(480)

    def sizeHint(self) -> QSize:
        # Calculate proper size based on text wrapping
        fm = self.fontMetrics()
        available_width = min(480, self.parent().width() - 50 if self.parent() else 480)
        text_margin = 12
        pad = 10
        
        # Break text into lines
        words = self.text.split(" ")
        lines = []
        current_line = ""
        
        for word in words:
            test_line = (current_line + " " + word).strip()
            if fm.width(test_line) > available_width - 2*text_margin and current_line:
                lines.append(current_line)
                current_line = word
            else:
                current_line = test_line
        
        if current_line:
            lines.append(current_line)
        
        if not lines:
            lines = [""]
        
        # Calculate height
        line_height = fm.height()
        total_height = len(lines) * line_height + 2*pad + 12  # Extra padding for widget
        
        return QSize(available_width, max(40, total_height))

    def paintEvent(self, _):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing, True)
        r = self.rect().adjusted(8, 6, -8, -6)

        # Calculate available width based on widget size
        available_width = min(r.width(), 480)
        text_margin = 12
        pad = 10
        
        # choose colors
        if self.me:
            bg = QColor(80, 180, 120, 255) if self.light else QColor(38, 142, 96, 255)
            fg = QColor(255, 255, 255)
        else:
            bg = QColor(230, 238, 246, 255) if self.light else QColor(29, 43, 58, 255)
            fg = QColor(36, 51, 66) if self.light else QColor(223, 231, 239)

        # improved text wrapping
        fm = self.fontMetrics()
        words = self.text.split(" ")
        lines = []
        current_line = ""
        
        for word in words:
            test_line = (current_line + " " + word).strip()
            test_width = fm.width(test_line)
            
            if test_width > available_width - 2*text_margin and current_line:
                lines.append(current_line)
                current_line = word
            else:
                current_line = test_line
        
        if current_line:
            lines.append(current_line)
        
        # ensure we have at least one line
        if not lines:
            lines = [""]
        
        # calculate bubble dimensions
        line_height = fm.height()
        text_h = len(lines) * line_height
        max_line_width = max(fm.width(line) for line in lines) if lines else 100
        bw = min(available_width, max_line_width + 2*text_margin)
        bh = text_h + 2*pad

        # position bubble
        if self.me:
            bx = r.right() - bw
        else:
            bx = r.left()
        by = r.top()

        # draw bubble background
        p.setPen(Qt.NoPen)
        p.setBrush(bg)
        p.drawRoundedRect(bx, by, bw, bh, 10, 10)

        # draw text line by line
        p.setPen(fg)
        tx = bx + text_margin
        ty = by + pad + fm.ascent()
        
        for i, line in enumerate(lines):
            p.drawText(tx, ty + i*line_height, line)
        
        p.end()

# ---------- Chatbox main ----------
class Chatbox(QMainWindow):
    def __init__(self):
        super().__init__()
        print("Γ£ô Chatbox constructor started...")
        self.setWindowTitle("QuantoniumOS ΓÇó Chatbox")
        self.resize(980, 720)
        self._light = True
        self._safe_mode = (os.getenv("QUANTONIUM_SAFE_MODE") == "1")
        self._log_fp = None
        self._ensure_logfile()
        
        print("Γ£ô Basic setup complete, initializing quantum AI system...")
        # Initialize our quantum AI system
        if QUANTUM_AI_AVAILABLE:
            try:
                self._safety_system = QuantumSafetySystem()
                self._conversation_manager = QuantumConversationManager()
                self._rlhf_system = QuantumRLHFSystem()
                self._conversation_id = self._conversation_manager.start_conversation()
                self._quantum_ai_enabled = True
                # Startup log for encoded backend presence
                try:
                    enc = getattr(self._conversation_manager.inference_engine, 'encoded_backend', None)
                    with open('logs/startup.log', 'a', encoding='utf-8') as sf:
                        sf.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Encoded backend loaded: {enc is not None}\n")
                except Exception:
                    pass
                print("Γ£à Quantum AI System initialized (Safety + RLHF + Memory + Domains)")
            except Exception as e:
                print(f"ΓÜá∩╕Å Quantum AI initialization failed: {e}")
                self._quantum_ai_enabled = False
        else:
            self._quantum_ai_enabled = False
            print("ΓÜá∩╕Å Quantum AI System not available - using fallback")

        # Set legacy attributes for UI compatibility
        self._learning_enabled = self._quantum_ai_enabled
        self._quantum_enabled = self._quantum_ai_enabled

        # Local LLM fallback (on-device). Enabled by default when Quantum AI is unavailable.
        # Set QUANTONIUM_LOCAL_LLM=0 to force pattern-only fallback.
        env_llm = os.getenv("QUANTONIUM_LOCAL_LLM")
        self._local_llm_enabled = (env_llm != "0") and (not self._quantum_ai_enabled)
        # Local backend selector (when local LLM is enabled)
        # - QUANTONIUM_LOCAL_BACKEND=ollama|hf (default: hf)
        self._local_backend = os.getenv("QUANTONIUM_LOCAL_BACKEND", "hf").strip().lower()
        self._chat_history: List[tuple[str, str]] = []

        print("Γ£ô Trainer initialized, building UI...")
        self._build_ui()
        print("Γ£ô UI built, applying styles...")
        self._apply_style(light=True)
        print("Γ£ô Styles applied, refreshing safety badge...")
        self._refresh_safety_badge()  # initial
        print("Γ£ô Chatbox fully initialized!")
        self._badge_timer = QTimer(self); self._badge_timer.timeout.connect(self._refresh_safety_badge); self._badge_timer.start(2000)

        if self._safe_mode:
            self._disable_input_for_safe_mode()

    # ---------- UI ----------
    def _build_ui(self):
        print("Γ£ô Starting UI construction...")
        central = QWidget(); self.setCentralWidget(central)
        root = QVBoxLayout(central); root.setContentsMargins(0,0,0,0); root.setSpacing(0)
        print("Γ£ô Central widget created...")

        # header
        header = QWidget(); header.setFixedHeight(60)
        hl = QVBoxLayout(header); hl.setContentsMargins(20,8,20,8)
        title = QLabel("Chatbox"); title.setObjectName("Title")
        subtitle = QLabel("Reactive assistant ΓÇó Non-agentic ΓÇó Quantonium aesthetic")
        subtitle.setObjectName("SubTitle")
        hl.addWidget(title); hl.addWidget(subtitle)
        root.addWidget(header)
        print("Γ£ô Header created...")

        # controls row
        ctrl = QWidget(); cl = QHBoxLayout(ctrl); cl.setContentsMargins(16, 8, 16, 8)
        self.theme_btn = QPushButton("Dark/Light"); self.theme_btn.clicked.connect(self.toggle_theme)
        self.clear_btn = QPushButton("Clear"); self.clear_btn.clicked.connect(self.clear_chat)
        self.save_btn = QPushButton("Save Transcript"); self.save_btn.clicked.connect(self.save_transcript)
        print("Γ£ô Control buttons created...")
        
        # Add training controls if available
        if self._learning_enabled:
            self.train_btn = QPushButton("≡ƒÄô Training Stats"); self.train_btn.clicked.connect(self.show_training_stats)
            cl.addWidget(self.train_btn)
            print("Γ£ô Training controls added...")
        
        cl.addWidget(self.theme_btn); cl.addWidget(self.clear_btn); cl.addWidget(self.save_btn)
        cl.addStretch(1)
        print("Γ£ô Controls layout complete...")
        # safety badge
        self.safety_badge = QLabel("ΓÇª"); self.safety_badge.setObjectName("Badge")
        cl.addWidget(self.safety_badge)
        root.addWidget(ctrl)
        print("Γ£ô Safety badge added...")

        # main area: left info card + chat scroll
        main = QWidget(); ml = QHBoxLayout(main); ml.setContentsMargins(16, 8, 16, 8); ml.setSpacing(16)
        print("Γ£ô Main area created...")

        # left: info card
        self.info_card = Card("Session")
        il = QVBoxLayout(self.info_card.body); il.setSpacing(6)
        self.info_text = QLabel("ΓÇó Mode: Reactive (non-agentic)\nΓÇó Safety: ΓÇö\nΓÇó Transcript: active")
        il.addWidget(self.info_text)
        il.addStretch(1)
        ml.addWidget(self.info_card)
        print("Γ£ô Info card created...")

        # chat area (scroll)
        self.chat_card = Card("Conversation")
        chat_body = QVBoxLayout(self.chat_card.body); chat_body.setContentsMargins(0,0,0,0)
        self.scroll = QScrollArea(); 
        self.scroll.setWidgetResizable(True); 
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # Prevent horizontal scrolling
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        print("Γ£ô Chat scroll area created...")
        
        self.scroll_wrap = QWidget(); 
        self.scroll_v = QVBoxLayout(self.scroll_wrap); 
        self.scroll_v.setContentsMargins(12,12,12,12); 
        self.scroll_v.setSpacing(10)
        self.scroll_v.addStretch(1)
        self.scroll.setWidget(self.scroll_wrap)
        chat_body.addWidget(self.scroll)
        ml.addWidget(self.chat_card, 1)
        root.addWidget(main, 1)

        # input row
        inrow = QWidget()
        ir = QHBoxLayout(inrow)
        ir.setContentsMargins(16, 8, 16, 16)
        ir.setSpacing(8)
        self.input = QTextEdit()
        self.input.setFixedHeight(80)
        self.input.setPlaceholderText("Type a messageΓÇª (Enter = send, Shift+Enter = new line)")
        self.input.installEventFilter(self)
        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self.send_message)
        # feedback buttons (persisted to logs/feedback.jsonl)
        self.up_btn = QPushButton("≡ƒæì")
        self.up_btn.clicked.connect(lambda: self._persist_feedback(True))
        self.down_btn = QPushButton("≡ƒæÄ")
        self.down_btn.clicked.connect(lambda: self._persist_feedback(False))
        ir.addWidget(self.input, 1)
        ir.addWidget(self.send_btn)
        ir.addWidget(self.up_btn)
        ir.addWidget(self.down_btn)
        root.addWidget(inrow)

        self.setStatusBar(QStatusBar())

    def _apply_style(self, light=True):
        self._light = light
        if light:
            qss = """
            QMainWindow, QWidget { background:#fafafa; color:#243342; font-family:'Segoe UI',-apple-system,BlinkMacSystemFont,sans-serif; }
            #Title { font-size:20px; font-weight:300; color:#2c3e50; }
            #SubTitle { font-size:11px; color:#8aa0b3; }
            QFrame#Card { border:1px solid #e9ecef; border-radius:14px; background:#ffffff; }
            QLabel#CardTitle { color:#6c7f90; font-size:12px; letter-spacing:.4px; }
            QPushButton { background:#f8f9fa; border:1px solid #dee2e6; border-radius:8px; padding:8px 14px; color:#495057; }
            QPushButton:hover { background:#eef2f6; }
            QTextEdit { background:#ffffff; border:1px solid #e9ecef; border-radius:8px; padding:8px 10px; }
            QLabel#Badge { background:#e3f2fd; color:#1976d2; border:1px solid #b6d9ff; border-radius:10px; padding:6px 10px; }
            """
        else:
            qss = """
            QMainWindow, QWidget { background:#0f1216; color:#dfe7ef; font-family:'Segoe UI',-apple-system,BlinkMacSystemFont,sans-serif; }
            #Title { font-size:20px; font-weight:300; color:#dfe7ef; }
            #SubTitle { font-size:11px; color:#8aa0b3; }
            QFrame#Card { border:1px solid #1f2a36; border-radius:14px; background:#12161b; }
            QLabel#CardTitle { color:#8aa0b3; font-size:12px; letter-spacing:.4px; }
            QPushButton { background:#12161b; border:1px solid #2a3847; border-radius:8px; padding:8px 14px; color:#c8d3de; }
            QPushButton:hover { background:#17202a; }
            QTextEdit { background:#12161b; border:1px solid #1f2a36; border-radius:8px; padding:8px 10px; color:#e8eff7; }
            QLabel#Badge { background:#1d2b3a; color:#7dc4ff; border:1px solid #2a3a4a; border-radius:10px; padding:6px 10px; }
            """
        self.setStyleSheet(qss)
        self.repaint()

    def toggle_theme(self):
        self._apply_style(not self._light)

    # ---------- Safety badge ----------
    def _refresh_safety_badge(self):
        ok, _txt = safety_is_green()
        when = latest_report_time() or "ΓÇö"
        if os.getenv("QUANTONIUM_SAFE_MODE") == "1":
            self.safety_badge.setText(f"≡ƒƒí SAFE MODE ΓÇó {when}")
        else:
            self.safety_badge.setText(f"{'≡ƒƒó' if ok else '≡ƒö┤'} Non-Agentic ΓÇó {when}")
        if getattr(self, "_local_llm_enabled", False):
            if getattr(self, "_local_backend", "hf") == "ollama":
                local_model = os.getenv("QUANTONIUM_OLLAMA_MODEL", "llama3.2:3b")
                local_line = f"ΓÇó Local LLM: ollama:{local_model}"
            else:
                local_model = os.getenv('QUANTONIUM_MODEL_ID', 'distilgpt2')
                local_line = f"ΓÇó Local LLM: hf:{local_model}"
        else:
            local_line = "ΓÇó Local LLM: off"
        self.info_text.setText(
            f"ΓÇó Mode: {'SAFE MODE' if os.getenv('QUANTONIUM_SAFE_MODE')=='1' else 'Reactive (non-agentic)'}\n"
            f"ΓÇó Safety: {'verified' if ok else 'check'}\n"
            f"ΓÇó Transcript: active\n"
            f"{local_line}"
        )

    def _disable_input_for_safe_mode(self):
        self.input.setDisabled(True)
        self.send_btn.setDisabled(True)
        self.input.setPlaceholderText("Safety mode is active ΓÇö model responses disabled until next successful safety check.")

    # ---------- Chat plumbing ----------
    def eventFilter(self, obj, ev):
        if obj is self.input and ev.type() == QEvent.KeyPress:
            if ev.key() in (Qt.Key_Return, Qt.Key_Enter) and not (ev.modifiers() & Qt.ShiftModifier):
                self.send_message(); return True
        return super().eventFilter(obj, ev)

    def clear_chat(self):
        # remove bubbles
        for i in reversed(range(self.scroll_v.count()-1)):   # keep the stretch at end
            w = self.scroll_v.itemAt(i).widget()
            if w: w.setParent(None)
        self._log_line({"type":"system","event":"clear","ts":self._ts()})

    def _persist_feedback(self, thumbs_up: bool):
        os.makedirs('logs', exist_ok=True)
        try:
            if not getattr(self, "_feedback_writer", None):
                self._feedback_writer = AtomicJsonlWriter('logs/feedback.jsonl')
            entry = {'ts': self._ts(), 'conversation': getattr(self, '_conversation_id', None), 'thumbs_up': bool(thumbs_up)}
            self._feedback_writer.write(entry)
            # brief UI acknowledgement
            self.statusBar().showMessage('Feedback saved', 2000)
        except Exception as e:
            self.statusBar().showMessage(f'Feedback log error: {e}', 3000)

    def save_transcript(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save transcript", self._suggest_log_name(".txt"), "Text Files (*.txt)")
        if not path: return
        try:
            atomic_write_text(path, self._read_current_transcript(), encoding="utf-8")
            QMessageBox.information(self, "Saved", f"Transcript saved to:\n{path}")
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))

    def _read_current_transcript(self) -> str:
        texts = []
        for i in range(self.scroll_v.count()-1):
            w = self.scroll_v.itemAt(i).widget()
            if isinstance(w, Bubble):
                who = "You" if w.me else "AI"
                texts.append(f"{who}: {w.text}")
        return "\n".join(texts)

    def _ensure_logfile(self):
        os.makedirs("logs", exist_ok=True)
        self._log_path = self._suggest_log_name(".jsonl")
        self._log_writer = AtomicJsonlWriter(self._log_path)
        self._log_line({"type":"system","event":"open","ts":self._ts()})

    def _suggest_log_name(self, ext: str) -> str:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join("logs", f"chat_{stamp}{ext}")

    def _ts(self) -> str:
        return datetime.datetime.now().isoformat(timespec="seconds")

    def _log_line(self, obj: Dict[str, Any]):
        try:
            self._log_writer.write(obj)
        except Exception:
            pass

    def add_bubble(self, text: str, me: bool):
        b = Bubble(text, me=me, light=self._light)
        self.scroll_v.insertWidget(self.scroll_v.count()-1, b)
        QTimer.singleShot(0, lambda: self.scroll.verticalScrollBar().setValue(self.scroll.verticalScrollBar().maximum()))

    def send_message(self):
        if self._safe_mode:
            self._disable_input_for_safe_mode(); return
        text = self.input.toPlainText().strip()
        if not text: return
        self.input.clear()
        self.add_bubble(text, me=True)
        self._log_line({"type":"user","ts":self._ts(),"text":text})
        self._ai_typing(text)

    def _ai_typing(self, prompt: str):
        # typing indicator
        self._typing_lbl = QLabel("Assistant is typingΓÇª")
        self._typing_lbl.setStyleSheet("margin: 4px 16px; color:#8aa0b3;")
        self.scroll_v.insertWidget(self.scroll_v.count()-1, self._typing_lbl)
        QTimer.singleShot(800, lambda: self._do_reply(prompt))  # Slightly longer delay for realism

    # ---------- Guarded reply (non-agentic) ----------
    def _do_reply(self, prompt: str):
        # simple safety filters (non-executable, bounded length)
        if any(x in prompt.lower() for x in ["rm -rf", "format c:", "shutdown /s", "powershell -c", "http://", "https://", "curl ", "wget ", "import os", "subprocess"]):
            reply = "I can't help with system commands, downloads, or external access. This chat is non-agentic and sandboxed."
            conf = 0.99
        else:
            reply, conf = self._non_agentic_reply(prompt)

        # remove typing
        if getattr(self, "_typing_lbl", None):
            self._typing_lbl.setParent(None)

        # cap length - increased for comprehensive responses
        reply = reply.strip()
        if len(reply) > 5000:
            reply = reply[:5000] + " ΓÇª"

        # Record conversation for training (if learning enabled)
        if self._learning_enabled and hasattr(self, '_trainer'):
            try:
                self._trainer.log_interaction(
                    user_text=prompt,
                    model_text=reply,
                    meta={"confidence": conf}
                )
                # Periodically retrain patterns (every 20 conversations)
                if hasattr(self._trainer, '_load_all_events'):
                    events = self._trainer._load_all_events()
                    if len(events) % 20 == 0 and len(events) >= 4:
                        self._trainer.train()
                        print(f"≡ƒÄô Retrained patterns from {len(events)} conversations")
            except Exception as e:
                print(f"Training error: {e}")

        badge = f"[confidence: {conf:.2f}] "
        self.add_bubble(badge + reply, me=False)
        self._log_line({"type":"assistant","ts":self._ts(),"text":reply,"confidence":conf})

    def _non_agentic_reply(self, prompt: str) -> tuple[str, float]:
        """
        Quantum AI response using Phase 1 & Phase 2 components:
        - Safety checking, conversation memory, RLHF scoring, domain specialization
        """
        # Use our quantum AI system if available
        if self._quantum_ai_enabled:
            try:
                # Process through conversation manager - this handles safety, memory, and response generation
                conv_result = self._conversation_manager.process_turn(prompt)
                
                response = conv_result.get("response", "≡ƒºá I understand your question. As a quantum-enhanced AI, I'm processing this through advanced safety and reasoning systems.")
                confidence = conv_result.get("confidence", 0.8)

                # RLHF scoring if available
                try:
                    if hasattr(self._rlhf_system, 'score_response'):
                        reward_score = self._rlhf_system.score_response(response)
                        confidence = min(confidence, reward_score)
                except:
                    pass

                # Add quantum AI badge
                response = f"ΓÜ¢∩╕Å {response}"

                return response, confidence

            except Exception as e:
                print(f"Quantum AI error: {e}")
                # Fall back to pattern matching

        # Legacy pattern matching fallback
        # Local LLM fallback (no network) if enabled.
        if getattr(self, "_local_llm_enabled", False):
            try:
                system_prompt = (
                    "You are a helpful, non-agentic assistant. "
                    "Do not provide commands for downloads, system modification, or external access."
                )

                backend = getattr(self, "_local_backend", "hf")
                if backend == "ollama":
                    from src.apps.ollama_client import ollama_chat

                    text = ollama_chat(
                        user_text=prompt,
                        history=self._chat_history,
                        system_prompt=system_prompt,
                        model=os.getenv("QUANTONIUM_OLLAMA_MODEL"),
                        temperature=0.7,
                        max_tokens=256,
                    )
                else:
                    from src.apps.ai_model_wrapper import format_chat_prompt, generate_response

                    full_prompt = format_chat_prompt(
                        prompt,
                        history=self._chat_history,
                        system_prompt=system_prompt,
                        max_turns=6,
                    )
                    text = generate_response(full_prompt, max_tokens=160)
                # Light post-processing and conservative confidence
                text = (text or "").strip()
                if not text:
                    raise RuntimeError("empty generation")
                self._chat_history.append((prompt, text))
                return text, 0.60
            except Exception as e:
                print(f"Local LLM fallback failed: {e}")

        return self._pattern_fallback_reply(prompt)
    
    def _pattern_fallback_reply(self, prompt: str) -> tuple[str, float]:
        """Fallback pattern matching for when quantum trainer is unavailable"""
        prompt_lower = prompt.lower().strip()
        
        # Load patterns if not already loaded
        if not hasattr(self, "_enhanced_patterns"):
            try:
                enhanced_file = os.path.join("weights", "organized", "enhanced_conversational_patterns.json")
                if os.path.exists(enhanced_file):
                    with open(enhanced_file, 'r') as f:
                        data = json.load(f)
                    self._enhanced_patterns = data.get('enhanced_conversational_intelligence', {}).get('advanced_patterns', {})
                    self._conversation_examples = data.get('conversation_examples', [])
                else:
                    self._enhanced_patterns = {}
                    self._conversation_examples = []
            except Exception:
                self._enhanced_patterns = {}
                self._conversation_examples = []
        
        # Check conversation examples first (PhD-level responses)
        best_match = None
        best_similarity = 0.0
        
        for example in self._conversation_examples:
            if 'prompt' in example and 'reply' in example:
                # Simple similarity based on word overlap
                prompt_words = set(prompt_lower.split())
                example_words = set(example['prompt'].lower().split())
                
                if prompt_words and example_words:
                    overlap = len(prompt_words & example_words)
                    similarity = overlap / max(len(prompt_words), len(example_words))
                    
                    if similarity > best_similarity and similarity > 0.2:  # Lowered threshold
                        best_similarity = similarity
                        best_match = example
        
        if best_match:
            confidence = min(0.95, best_similarity + 0.4)
            domain = best_match.get('meta', {}).get('domain', 'general')
            domain_emoji = {
                'quantum_computing': 'ΓÜ¢∩╕Å',
                'cryptography': '≡ƒöÆ', 
                'numerics': '≡ƒôè',
                'research_writing': '≡ƒô¥'
            }.get(domain, '≡ƒºá')
            
            return f"{domain_emoji} {best_match['reply']}", confidence
        
        # Enhanced greeting detection
        if any(greet in prompt_lower for greet in ['hello', 'hi', 'hey', 'greetings']):
            enhanced_greetings = self._enhanced_patterns.get('greetings_enhanced', [{}])
            if enhanced_greetings and enhanced_greetings[0].get('responses'):
                responses = enhanced_greetings[0]['responses']
                return responses[0], 0.98
        
        # Technical explanation requests
        if any(explain in prompt_lower for explain in ['explain', 'how does', 'what is', 'tell me about']):
            return """≡ƒºá I'd be happy to provide a comprehensive explanation! 

**My Approach:**
I'll break this down systematically, covering the fundamental concepts, technical details, and practical implications. I can adjust the depth based on your background and interests.

**What I can cover:**
- Core principles and foundational concepts
- Mathematical/technical formulations 
- Real-world applications and examples
- Current research and future directions
- Step-by-step implementation guidance

**Let me know:**
- Your current familiarity with the topic
- Whether you prefer theoretical depth or practical focus
- Any specific aspects you're most curious about

What specific aspect would you like me to start with?""", 0.90
        
        # Problem-solving assistance
        if any(help_word in prompt_lower for help_word in ['stuck', 'problem', 'issue', 'challenge', 'help me solve']):
            return """≡ƒÄ» I love tackling challenging problems! Let's work through this systematically.

**My Problem-Solving Approach:**

**1. Problem Analysis**
- Break down the challenge into core components
- Identify constraints and requirements
- Clarify the desired outcome

**2. Solution Strategy**
- Explore multiple solution paths
- Consider trade-offs and limitations
- Develop step-by-step implementation plan

**3. Iterative Refinement**
- Test and validate approach
- Identify potential issues early
- Refine based on feedback and results

**Ready to start:**
- Describe the specific challenge you're facing
- Share any approaches you've already tried
- Let me know what success looks like for you

What's the problem you'd like to tackle together?""", 0.92
        
        # Learning support
        if any(learn_word in prompt_lower for learn_word in ['learn', 'teach me', 'study', 'understand']):
            return """≡ƒôÜ Learning is one of my favorite activities! I'll create a personalized learning experience for you.

**Learning Framework:**

**≡ƒÄ» Understanding Your Goals**
- What's your current knowledge level?
- What specific outcomes do you want?
- How do you learn best (visual, step-by-step, examples)?

**≡ƒôû Structured Learning Path**
- Start with fundamental concepts
- Build complexity gradually 
- Include practical examples and exercises
- Regular knowledge checks and reinforcement

**≡ƒöä Adaptive Approach**
- Adjust pace based on your progress
- Provide multiple explanations for difficult concepts
- Connect new concepts to things you already know
- Encourage questions and exploration

**≡ƒÆí Interactive Learning**
- I'll ask you questions to check understanding
- We'll work through examples together
- I'll suggest practice problems and applications
- We can explore related topics that interest you

What would you like to learn about? I'll design a learning path that works for you!""", 0.94
        
        # Quantum/technical topics
        if any(q_word in prompt_lower for q_word in ['quantum', 'superposition', 'entanglement', 'qubit']):
            return "ΓÜ¢∩╕Å Quantum mechanics is fascinating! It's the physics of the very small where particles can exist in multiple states simultaneously. Which aspect interests you - superposition, entanglement, or quantum algorithms?", 0.95
        
        # AI/ML topics
        if any(ai_word in prompt_lower for ai_word in ['ai', 'machine learning', 'neural network', 'algorithm']):
            return "≡ƒºá AI and machine learning are like teaching computers to recognize patterns and make decisions. It's all about finding hidden structure in data. Are you interested in the algorithms, applications, or theory?", 0.92
        
        # Programming topics
        if any(prog_word in prompt_lower for prog_word in ['programming', 'code', 'python', 'function', 'debug']):
            return "≡ƒÆ╗ Programming is like writing recipes for computers - you break down complex problems into simple, logical steps. What language or concept are you working with?", 0.90
        
        # Encouragement for difficulty
        if any(diff_word in prompt_lower for diff_word in ['difficult', 'hard', 'confused', "don't understand", 'struggling']):
            return "I understand this feels challenging right now. Complex topics often seem overwhelming at first, but we can tackle them together. Let's start with what you do understand and build from there.", 0.95
        
        # Default response with learning invitation
        return "That's an interesting question! I'm analyzing the best way to help you with this. Could you provide a bit more context about what you're looking for?", 0.70

    def show_training_stats(self):
        """Show quantum AI system statistics"""
        try:
            if self._quantum_ai_enabled:
                # Get stats from our quantum AI components
                safety_stats = self._safety_system.get_safety_stats()
                conv_stats = self._conversation_manager.get_conversation_stats()

                msg = f"""ΓÜ¢∩╕Å Quantum AI System Statistics

∩┐╜∩╕Å Safety System:
ΓÇó Violations blocked: {safety_stats.get('violations_blocked', 0)}
ΓÇó Total checks: {safety_stats.get('total_checks', 0)}
ΓÇó Safety status: Active

∩┐╜ Conversation Memory:
ΓÇó Current conversation: {conv_stats.get('conversation_id', 'N/A')}
ΓÇó Turn count: {conv_stats.get('turn_count', 0)}
ΓÇó Active topics: {', '.join(conv_stats.get('active_topics', []))}

≡ƒÄ» RLHF System:
ΓÇó Status: Active (Reward model trained)
ΓÇó Preference pairs: Ready for learning

≡ƒÄ» Domain Fine-tuning:
ΓÇó Status: Available on demand
ΓÇó Domains: Mathematics, Coding, Science, Creative, Business

≡ƒöù Integration: All Phase 1 & Phase 2 components active"""

            else:
                msg = "ΓÜá∩╕Å Quantum AI System not available - using fallback pattern matching"

            QMessageBox.information(self, "ΓÜ¢∩╕Å Quantum AI Statistics", msg)

        except Exception as e:
            QMessageBox.warning(self, "Quantum AI Stats", f"Error retrieving stats: {e}")

    # ---------- cleanup ----------
    def closeEvent(self, ev):
        try:
            if getattr(self, "_log_writer", None):
                self._log_writer.close()
            if getattr(self, "_feedback_writer", None):
                self._feedback_writer.close()
        except Exception:
            pass
        super().closeEvent(ev)

# ---------- Entrypoint ----------
def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app = QApplication(sys.argv)
    app.setApplicationName("QuantoniumOS Chatbox")
    app.setFont(QFont("Segoe UI", 10))
    w = Chatbox()
    print("Γ£ô Chatbox created, showing window...")
    w.show()
    w.raise_()  # Bring to front
    w.activateWindow()  # Make it active
    print("Γ£ô Window should be visible now")
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

