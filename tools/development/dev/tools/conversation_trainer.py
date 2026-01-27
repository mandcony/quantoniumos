#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
# -*- coding: utf-8 -*-
"""
ConversationTrainer (QuantoniumOS)
Non-agentic, offline pattern trainer for qshll_chatbox.py

- No network, subprocess, tools, or code execution.
- Reads/writes only under {logs/, weights/organized/}.
- Mines intents from your chat logs and produces reply patterns.
- Offers suggest() for candidate replies + confidence.
- Provides log_interaction() to append JSONL traces safely.

Expected import in your chatbox:
    from conversation_trainer import ConversationTrainer
"""

from __future__ import annotations
import os, re, json, time, math, hashlib
from dataclasses import dataclass, asdict
from collections import Counter, defaultdict
from typing import List, Dict, Optional, Tuple

SAFE_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOG_DIR   = os.path.join(SAFE_BASE, "logs", "conversations")
WEIGHTS_D = os.path.join(SAFE_BASE, "weights", "organized")
PATTERNS  = os.path.join(WEIGHTS_D, "enhanced_conversational_patterns.json")

STOP = {
    "the","and","you","your","are","for","with","that","this","what","when","how",
    "is","it","to","of","in","on","a","an","be","as","at","by","or","not","can",
    "we","us","our","me","i","my","from","about"
}
PII = re.compile(r"(\b(?:\d{3}-\d{2}-\d{4}|(?:\+?\d{1,3})?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})\b|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})")
PROFANITY = {"fuck","shit","bitch"}  # keep tiny & tame; extend if you want

def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())

def _sluggify(text: str, k: int = 4) -> str:
    toks = [t for t in re.findall(r"[A-Za-z0-9]+", text.lower()) if t not in STOP]
    return "-".join(toks[:k]) or "misc"

def _fingerprint(text: str) -> str:
    # stable, privacy-friendly hash (we also redact PII below)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

def _redact(s: str) -> str:
    s = PII.sub("[redacted]", s)
    return s

def _tokens(s: str) -> List[str]:
    return [t for t in re.findall(r"[A-Za-z0-9]{2,}", s.lower()) if t not in STOP]

def _cosine(a: Counter, b: Counter) -> float:
    if not a or not b: return 0.0
    dot = sum(a[k]*b.get(k,0) for k in a)
    na = math.sqrt(sum(v*v for v in a.values()))
    nb = math.sqrt(sum(v*v for v in b.values()))
    if na == 0 or nb == 0: return 0.0
    return dot/(na*nb)

@dataclass
class LogEvent:
    ts: str
    user: str
    prompt: str
    reply: str
    meta: Dict

@dataclass
class Pattern:
    intent: str
    match_terms: List[str]
    response: str
    confidence: float
    support: int
    sources: List[str]  # file:line fingerprints

class ConversationTrainer:
    """
    Minimal API used by the chatbox:
      - log_interaction(user_text, model_text, meta=None)
      - suggest(user_text, top_k=3) -> List[Dict]
      - train() -> writes enhanced_conversational_patterns.json
    """
    def __init__(self,
                 log_dir: str = LOG_DIR,
                 weights_dir: str = WEIGHTS_D,
                 patterns_path: str = PATTERNS):
        self.log_dir = log_dir
        self.weights_dir = weights_dir
        self.patterns_path = patterns_path
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.weights_dir, exist_ok=True)

        self._patterns: List[Pattern] = []
        if os.path.exists(self.patterns_path):
            try:
                data = json.load(open(self.patterns_path, "r", encoding="utf-8"))
                for p in data.get("patterns", []):
                    self._patterns.append(Pattern(**p))
            except Exception:
                # corrupt or first run: start fresh
                self._patterns = []

    # ---------- logging ----------
    def _logfile_today(self) -> str:
        fname = time.strftime("conv_%Y%m%d.jsonl", time.localtime())
        return os.path.join(self.log_dir, fname)

    def log_interaction(self, user_text: str, model_text: str, meta: Optional[Dict]=None) -> None:
        evt = LogEvent(
            ts=_now_iso(),
            user="local",
            prompt=_redact(user_text.strip()),
            reply=_redact(model_text.strip()),
            meta=meta or {}
        )
        line = json.dumps(asdict(evt), ensure_ascii=False)
        with open(self._logfile_today(), "a", encoding="utf-8") as f:
            f.write(line + "\n")

    # ---------- training (safe, offline) ----------
    def _load_all_events(self) -> List[LogEvent]:
        events: List[LogEvent] = []
        for name in sorted(os.listdir(self.log_dir)):
            if not name.endswith(".jsonl"): continue
            path = os.path.join(self.log_dir, name)
            with open(path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f, 1):
                    try:
                        obj = json.loads(line)
                        events.append(LogEvent(**obj))
                    except Exception:
                        # skip bad rows, keep going
                        continue
        return events

    def _cluster(self, prompts: List[str], min_support: int = 2) -> Dict[str, List[int]]:
        """
        Greedy cosine-based topic clustering; returns {cluster_id: [indices]}
        Deterministic; stdlib only; good enough for bootstrapping patterns.
        """
        vecs = [Counter(_tokens(p)) for p in prompts]
        clusters: Dict[str, List[int]] = {}
        centers: Dict[str, Counter] = {}
        for i, v in enumerate(vecs):
            # find best existing center
            best_id, best = None, 0.0
            for cid, cvec in centers.items():
                score = _cosine(v, cvec)
                if score > best:
                    best, best_id = score, cid
            if best is None or best < 0.42:  # threshold; tune if needed
                cid = f"{_sluggify(prompts[i])}-{_fingerprint(' '.join(v.keys()))[:6]}"
                clusters[cid] = [i]
                centers[cid]  = v.copy()
            else:
                clusters[best_id].append(i)
                # update center
                for k, x in v.items():
                    centers[best_id][k] += x

        # filter tiny clusters
        return {cid: idxs for cid, idxs in clusters.items() if len(idxs) >= min_support}

    def train(self) -> str:
        """
        Mines patterns from logs and writes the JSON pattern file.
        Returns path to the updated file.
        """
        evts = self._load_all_events()
        prompts = [e.prompt for e in evts]
        clusters = self._cluster(prompts)

        new_patterns: List[Pattern] = []
        for cid, idxs in clusters.items():
            # terms: top keywords
            all_toks = Counter()
            for i in idxs: all_toks.update(_tokens(prompts[i]))
            terms = [w for w,_ in all_toks.most_common(6)]

            # choose canonical response: most frequent reply in cluster
            replies = Counter(evts[i].reply for i in idxs)
            best_reply, support = replies.most_common(1)[0]

            # rough confidence: normalized support * term density
            dens = min(1.0, len(terms)/6.0)
            conf = round(min(0.99, 0.35 + 0.10*support + 0.25*dens), 3)

            srcs = [f"{_fingerprint(evts[i].prompt)}:{_fingerprint(evts[i].reply)}" for i in idxs]
            patt = Pattern(
                intent=cid,
                match_terms=terms[:4],
                response=best_reply,
                confidence=conf,
                support=support,
                sources=srcs[:8]
            )
            new_patterns.append(patt)

        # keep a small, high-signal set
        new_patterns.sort(key=lambda p: (p.support, p.confidence), reverse=True)
        final = new_patterns[:128]

        payload = {
            "updated": _now_iso(),
            "generator": "ConversationTrainer-stdlib",
            "patterns": [asdict(p) for p in final]
        }
        with open(self.patterns_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        self._patterns = final
        return self.patterns_path

    # ---------- inference (suggestions only) ----------
    def suggest(self, user_text: str, top_k: int = 3) -> List[Dict]:
        """
        Returns up to top_k pattern-based candidate replies with confidences.
        DOES NOT execute tools, browse, or call external code.
        """
        query_vec = Counter(_tokens(user_text))
        scored: List[Tuple[float, Pattern]] = []
        for p in self._patterns:
            pvec = Counter(p.match_terms)
            sim = _cosine(query_vec, pvec)
            if sim <= 0: continue
            # blend: similarity and pattern confidence
            score = 0.65*sim + 0.35*float(p.confidence)
            scored.append((score, p))
        scored.sort(reverse=True, key=lambda t: t[0])

        out = []
        for score, p in scored[:max(1, top_k)]:
            out.append({
                "intent": p.intent,
                "response": p.response,
                "confidence": round(min(0.99, score), 3),
                "terms": p.match_terms,
                "support": p.support
            })
        return out

    # ---------- Legacy compatibility methods ----------
    def record_conversation(self, user_input: str, ai_response: str, 
                          user_feedback: str = None, context: str = None):
        """Legacy method for compatibility with existing chatbox"""
        meta = {}
        if user_feedback:
            meta["feedback"] = user_feedback
        if context:
            meta["context"] = context
        self.log_interaction(user_input, ai_response, meta)

    def update_enhanced_patterns(self) -> int:
        """Legacy method - trains and returns number of patterns"""
        self.train()
        return len(self._patterns)

    def get_training_stats(self) -> Dict:
        """Legacy method for compatibility"""
        if not os.path.exists(self.log_dir):
            return {"message": "No training data available"}
        
        events = self._load_all_events()
        if not events:
            return {"message": "No conversations logged yet"}
            
        return {
            "total_conversations": len(events),
            "total_patterns": len(self._patterns),
            "latest_conversation": events[-1].ts if events else None,
            "log_directory": self.log_dir,
            "suggestions": ["Use the chatbox to have conversations and build training data!"]
        }

# ---------- CLI for manual training ----------
if __name__ == "__main__":
    trainer = ConversationTrainer()
    print("🎓 QuantoniumOS Conversation Trainer")
    print("=" * 50)
    
    if not os.path.exists(trainer.log_dir):
        print("📁 Creating conversation log directory...")
        os.makedirs(trainer.log_dir, exist_ok=True)
    
    print(f"📁 Log directory: {trainer.log_dir}")
    print(f"📁 Patterns file: {trainer.patterns_path}")
    
    # Check for existing data
    events = trainer._load_all_events()
    print(f"📊 Found {len(events)} conversation events")
    print(f"🎯 Current patterns: {len(trainer._patterns)}")
    
    if len(events) >= 4:  # Need minimum data to train
        print("\n🚀 Training patterns from conversation data...")
        path = trainer.train()
        print(f"✅ Updated patterns: {path}")
        print(f"🎯 New pattern count: {len(trainer._patterns)}")
    else:
        print("\n💡 Need more conversation data to train patterns.")
        print("   Use the chatbox to have conversations, then run this script again.")
    
    # Demo suggestion if we have patterns
    if trainer._patterns:
        print("\n🧪 Testing suggestion system...")
        for test_input in ["hello", "how do quantum computers work", "I need help"]:
            suggestions = trainer.suggest(test_input)
            print(f"\nInput: '{test_input}'")
            for i, sug in enumerate(suggestions, 1):
                print(f"  {i}. {sug['response'][:60]}... (conf: {sug['confidence']})")
