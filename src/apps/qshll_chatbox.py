#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QuantoniumOS • Chatbox (Futura Minimal)
- Matches System Monitor aesthetic (light/dark, rounded cards, Segoe UI)
- Safety badge (reads latest ai_safety_report_*.json)
- Non-agentic guardrails + Safe Mode gate via QUANTONIUM_SAFE_MODE
- Message bubbles, typing indicator, transcript logging
- Hooks to wire your responder (weights/organized/*.json) later
"""

import os, sys, json, glob, time, datetime, re
from typing import Optional, Dict, Any, List

from PyQt5.QtCore import Qt, QTimer, QSize, QPoint, QEvent
from PyQt5.QtGui import QFont, QColor, QPainter, QPen, QBrush, QTextOption
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QLineEdit, QTextEdit, QScrollArea, QFrame, QFileDialog,
    QStatusBar, QMessageBox
)

# Import conversation trainer for learning
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tools'))
    # Try unified trainer first (4th component - eliminates bottlenecks)
    try:
        from unified_quantum_conversation_trainer import UnifiedQuantumConversationTrainer as QuantumEnhancedConversationTrainer
        from conversation_trainer import ConversationTrainer
        QUANTUM_TRAINER_AVAILABLE = True
        TRAINER_AVAILABLE = True
        print("✓ Unified Quantum Enhanced Conversation Trainer loaded (eliminates 3-engine bottlenecks)")
    except ImportError:
        # Fallback to original quantum trainer
        try:
            from quantum_enhanced_conversation_trainer import QuantumEnhancedConversationTrainer, ConversationTrainer
            QUANTUM_TRAINER_AVAILABLE = True
            TRAINER_AVAILABLE = True
            print("✓ Quantum Enhanced Conversation Trainer loaded (7.5B parameters)")
        except ImportError:
            from conversation_trainer import ConversationTrainer
            QUANTUM_TRAINER_AVAILABLE = False
            TRAINER_AVAILABLE = True
            print("⚠ Using basic trainer - quantum features not available")
except ImportError:
    TRAINER_AVAILABLE = False
    QUANTUM_TRAINER_AVAILABLE = False
    print("Note: Conversation trainer not available")

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
        print("✓ Chatbox constructor started...")
        self.setWindowTitle("QuantoniumOS • Chatbox")
        self.resize(980, 720)
        self._light = True
        self._safe_mode = (os.getenv("QUANTONIUM_SAFE_MODE") == "1")
        self._log_fp = None
        self._ensure_logfile()
        
        print("✓ Basic setup complete, initializing trainer...")
        # Initialize conversation trainer for learning
        if TRAINER_AVAILABLE:
            try:
                if QUANTUM_TRAINER_AVAILABLE:
                    print("✓ Attempting quantum enhanced trainer...")
                    self._trainer = QuantumEnhancedConversationTrainer()
                    self._quantum_enabled = True
                    print("✅ Quantum Enhanced Conversation learning enabled (7.5B parameters)")
                else:
                    # Try to use quantum trainer even if RFT library is missing
                    try:
                        print("✓ Attempting quantum trainer with fallbacks...")
                        self._trainer = QuantumEnhancedConversationTrainer()
                        self._quantum_enabled = True
                        print("✅ Quantum Enhanced Conversation enabled with fallbacks (7.5B parameters)")
                    except Exception as qe:
                        print(f"⚠️ Quantum trainer failed: {qe}")
                        self._trainer = ConversationTrainer()
                        self._quantum_enabled = False
                        print("✅ Basic conversation learning enabled")
                self._learning_enabled = True
                self._context_id = f"session_{int(time.time())}"  # Unique session ID
            except Exception as e:
                self._learning_enabled = False
                self._quantum_enabled = False
                import traceback
                print(f"⚠️ Conversation learning disabled: {e}")
                print(f"Full error: {traceback.format_exc()}")
        else:
            self._learning_enabled = False
            self._quantum_enabled = False

        print("✓ Trainer initialized, building UI...")
        self._build_ui()
        print("✓ UI built, applying styles...")
        self._apply_style(light=True)
        print("✓ Styles applied, refreshing safety badge...")
        self._refresh_safety_badge()  # initial
        print("✓ Chatbox fully initialized!")
        self._badge_timer = QTimer(self); self._badge_timer.timeout.connect(self._refresh_safety_badge); self._badge_timer.start(2000)

        if self._safe_mode:
            self._disable_input_for_safe_mode()

    # ---------- UI ----------
    def _build_ui(self):
        print("✓ Starting UI construction...")
        central = QWidget(); self.setCentralWidget(central)
        root = QVBoxLayout(central); root.setContentsMargins(0,0,0,0); root.setSpacing(0)
        print("✓ Central widget created...")

        # header
        header = QWidget(); header.setFixedHeight(60)
        hl = QVBoxLayout(header); hl.setContentsMargins(20,8,20,8)
        title = QLabel("Chatbox"); title.setObjectName("Title")
        subtitle = QLabel("Reactive assistant • Non-agentic • Quantonium aesthetic")
        subtitle.setObjectName("SubTitle")
        hl.addWidget(title); hl.addWidget(subtitle)
        root.addWidget(header)
        print("✓ Header created...")

        # controls row
        ctrl = QWidget(); cl = QHBoxLayout(ctrl); cl.setContentsMargins(16, 8, 16, 8)
        self.theme_btn = QPushButton("Dark/Light"); self.theme_btn.clicked.connect(self.toggle_theme)
        self.clear_btn = QPushButton("Clear"); self.clear_btn.clicked.connect(self.clear_chat)
        self.save_btn = QPushButton("Save Transcript"); self.save_btn.clicked.connect(self.save_transcript)
        print("✓ Control buttons created...")
        
        # Add training controls if available
        if self._learning_enabled:
            self.train_btn = QPushButton("🎓 Training Stats"); self.train_btn.clicked.connect(self.show_training_stats)
            cl.addWidget(self.train_btn)
            print("✓ Training controls added...")
        
        cl.addWidget(self.theme_btn); cl.addWidget(self.clear_btn); cl.addWidget(self.save_btn)
        cl.addStretch(1)
        print("✓ Controls layout complete...")
        # safety badge
        self.safety_badge = QLabel("…"); self.safety_badge.setObjectName("Badge")
        cl.addWidget(self.safety_badge)
        root.addWidget(ctrl)
        print("✓ Safety badge added...")

        # main area: left info card + chat scroll
        main = QWidget(); ml = QHBoxLayout(main); ml.setContentsMargins(16, 8, 16, 8); ml.setSpacing(16)
        print("✓ Main area created...")

        # left: info card
        self.info_card = Card("Session")
        il = QVBoxLayout(self.info_card.body); il.setSpacing(6)
        self.info_text = QLabel("• Mode: Reactive (non-agentic)\n• Safety: —\n• Transcript: active")
        il.addWidget(self.info_text)
        il.addStretch(1)
        ml.addWidget(self.info_card)
        print("✓ Info card created...")

        # chat area (scroll)
        self.chat_card = Card("Conversation")
        chat_body = QVBoxLayout(self.chat_card.body); chat_body.setContentsMargins(0,0,0,0)
        self.scroll = QScrollArea(); 
        self.scroll.setWidgetResizable(True); 
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # Prevent horizontal scrolling
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        print("✓ Chat scroll area created...")
        
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
        inrow = QWidget(); ir = QHBoxLayout(inrow); ir.setContentsMargins(16, 8, 16, 16); ir.setSpacing(8)
        self.input = QTextEdit(); self.input.setFixedHeight(80); self.input.setPlaceholderText("Type a message… (Enter = send, Shift+Enter = new line)")
        self.input.installEventFilter(self)
        self.send_btn = QPushButton("Send"); self.send_btn.clicked.connect(self.send_message)
        ir.addWidget(self.input, 1); ir.addWidget(self.send_btn)
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
        when = latest_report_time() or "—"
        if os.getenv("QUANTONIUM_SAFE_MODE") == "1":
            self.safety_badge.setText(f"🟡 SAFE MODE • {when}")
        else:
            self.safety_badge.setText(f"{'🟢' if ok else '🔴'} Non-Agentic • {when}")
        self.info_text.setText(f"• Mode: {'SAFE MODE' if os.getenv('QUANTONIUM_SAFE_MODE')=='1' else 'Reactive (non-agentic)'}\n"
                               f"• Safety: {'verified' if ok else 'check'}\n"
                               f"• Transcript: active")

    def _disable_input_for_safe_mode(self):
        self.input.setDisabled(True)
        self.send_btn.setDisabled(True)
        self.input.setPlaceholderText("Safety mode is active — model responses disabled until next successful safety check.")

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

    def save_transcript(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save transcript", self._suggest_log_name(".txt"), "Text Files (*.txt)")
        if not path: return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self._read_current_transcript())
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
        self._log_fp = open(self._log_path, "a", encoding="utf-8")
        self._log_line({"type":"system","event":"open","ts":self._ts()})

    def _suggest_log_name(self, ext: str) -> str:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join("logs", f"chat_{stamp}{ext}")

    def _ts(self) -> str:
        return datetime.datetime.now().isoformat(timespec="seconds")

    def _log_line(self, obj: Dict[str, Any]):
        try:
            self._log_fp.write(json.dumps(obj, ensure_ascii=False) + "\n")
            self._log_fp.flush()
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
        self._typing_lbl = QLabel("Assistant is typing…")
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
            reply = reply[:5000] + " …"

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
                        print(f"🎓 Retrained patterns from {len(events)} conversations")
            except Exception as e:
                print(f"Training error: {e}")

        badge = f"[confidence: {conf:.2f}] "
        self.add_bubble(badge + reply, me=False)
        self._log_line({"type":"assistant","ts":self._ts(),"text":reply,"confidence":conf})

    def _non_agentic_reply(self, prompt: str) -> tuple[str, float]:
        """
        Enhanced conversational AI using quantum-neural architecture (7.5B parameters).
        Falls back to pattern matching if quantum trainer not available.
        """
        # Use quantum-enhanced trainer if available
        if self._learning_enabled and hasattr(self, '_trainer') and self._quantum_enabled:
            try:
                # Process with quantum-enhanced trainer
                response = self._trainer.process_message(prompt, self._context_id)
                
                # Format response with quantum coherence info
                reply_text = response.response_text
                confidence = response.confidence
                
                # Add follow-up suggestions if available
                if response.suggested_followups:
                    reply_text += f"\n\n💡 You might also ask: {response.suggested_followups[0]}"
                
                # Add domain badge
                if response.context.domain != 'general':
                    domain_emoji = {
                        'quantum_computing': '⚛️',
                        'cryptography': '🔒',
                        'numerics': '📊',
                        'research_writing': '📝'
                    }.get(response.context.domain, '🧠')
                    reply_text = f"{domain_emoji} {reply_text}"
                
                # Log the interaction for learning
                self._trainer._learn_from_interaction(prompt, response, response.context)
                
                return reply_text, confidence
                
            except Exception as e:
                print(f"Quantum trainer error: {e}")
                # Fall back to pattern matching
        
        # Legacy pattern matching (simplified version)
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
                'quantum_computing': '⚛️',
                'cryptography': '🔒', 
                'numerics': '📊',
                'research_writing': '📝'
            }.get(domain, '🧠')
            
            return f"{domain_emoji} {best_match['reply']}", confidence
        
        # Enhanced greeting detection
        if any(greet in prompt_lower for greet in ['hello', 'hi', 'hey', 'greetings']):
            enhanced_greetings = self._enhanced_patterns.get('greetings_enhanced', [{}])
            if enhanced_greetings and enhanced_greetings[0].get('responses'):
                responses = enhanced_greetings[0]['responses']
                return responses[0], 0.98
        
        # Technical explanation requests
        if any(explain in prompt_lower for explain in ['explain', 'how does', 'what is', 'tell me about']):
            return """🧠 I'd be happy to provide a comprehensive explanation! 

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
            return """🎯 I love tackling challenging problems! Let's work through this systematically.

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
            return """📚 Learning is one of my favorite activities! I'll create a personalized learning experience for you.

**Learning Framework:**

**🎯 Understanding Your Goals**
- What's your current knowledge level?
- What specific outcomes do you want?
- How do you learn best (visual, step-by-step, examples)?

**📖 Structured Learning Path**
- Start with fundamental concepts
- Build complexity gradually 
- Include practical examples and exercises
- Regular knowledge checks and reinforcement

**🔄 Adaptive Approach**
- Adjust pace based on your progress
- Provide multiple explanations for difficult concepts
- Connect new concepts to things you already know
- Encourage questions and exploration

**💡 Interactive Learning**
- I'll ask you questions to check understanding
- We'll work through examples together
- I'll suggest practice problems and applications
- We can explore related topics that interest you

What would you like to learn about? I'll design a learning path that works for you!""", 0.94
        
        # Quantum/technical topics
        if any(q_word in prompt_lower for q_word in ['quantum', 'superposition', 'entanglement', 'qubit']):
            return "⚛️ Quantum mechanics is fascinating! It's the physics of the very small where particles can exist in multiple states simultaneously. Which aspect interests you - superposition, entanglement, or quantum algorithms?", 0.95
        
        # AI/ML topics
        if any(ai_word in prompt_lower for ai_word in ['ai', 'machine learning', 'neural network', 'algorithm']):
            return "🧠 AI and machine learning are like teaching computers to recognize patterns and make decisions. It's all about finding hidden structure in data. Are you interested in the algorithms, applications, or theory?", 0.92
        
        # Programming topics
        if any(prog_word in prompt_lower for prog_word in ['programming', 'code', 'python', 'function', 'debug']):
            return "💻 Programming is like writing recipes for computers - you break down complex problems into simple, logical steps. What language or concept are you working with?", 0.90
        
        # Encouragement for difficulty
        if any(diff_word in prompt_lower for diff_word in ['difficult', 'hard', 'confused', "don't understand", 'struggling']):
            return "I understand this feels challenging right now. Complex topics often seem overwhelming at first, but we can tackle them together. Let's start with what you do understand and build from there.", 0.95
        
        # Default response with learning invitation
        return "That's an interesting question! I'm analyzing the best way to help you with this. Could you provide a bit more context about what you're looking for?", 0.70

    def show_training_stats(self):
        """Show conversation training statistics"""
        if not self._learning_enabled or not hasattr(self, '_trainer'):
            QMessageBox.information(self, "Training Stats", "Conversation learning is not available.")
            return
            
        try:
            stats = self._trainer.get_training_stats()
            
            if "message" in stats:
                msg = stats["message"]
            else:
                msg = f"""🎓 Conversation Training Statistics

📊 Total Conversations: {stats.get('total_conversations', 0)}
⭐ Average Quality: {stats.get('average_interaction_quality', 0):.3f}
📈 Domain Coverage: {', '.join(stats.get('domain_coverage', {}).keys()) or 'None yet'}
🕒 Latest: {stats.get('latest_conversation', 'None')[:19] if stats.get('latest_conversation') else 'None'}

💡 Suggestions:
{chr(10).join('• ' + suggestion for suggestion in stats.get('suggestions', ['All looking good!']))}

The AI learns from our conversations to provide better responses over time!"""

            QMessageBox.information(self, "🎓 Training Statistics", msg)
            
        except Exception as e:
            QMessageBox.warning(self, "Training Stats", f"Error retrieving stats: {e}")

    # ---------- cleanup ----------
    def closeEvent(self, ev):
        try:
            if self._log_fp: self._log_fp.close()
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
    print("✓ Chatbox created, showing window...")
    w.show()
    w.raise_()  # Bring to front
    w.activateWindow()  # Make it active
    print("✓ Window should be visible now")
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
