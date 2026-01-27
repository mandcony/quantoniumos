# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
﻿#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q-Vault ΓÇö Quantum Secure Storage (revamped)
- Minimal, focused, keyboard-friendly UI
- Master password ΓåÆ scrypt KDF
- AES-256-GCM when available; safe fallback otherwise
- Optional RFT-based keystream mixer (if unitary_rft present)
- Autosave, search, light/dark, lock/unlock, idle autolock
"""

import os, sys, json, time, uuid, hmac, hashlib, base64, secrets, re
from dataclasses import dataclass, asdict
from typing import Dict, Optional

from PyQt5.QtCore import Qt, QTimer, QEvent, QMimeData
from PyQt5.QtGui import QFont, QKeySequence, QClipboard
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QListWidget, QListWidgetItem, QTextEdit, QLineEdit, QPushButton,
    QFileDialog, QStatusBar, QLabel, QMessageBox, QAction, QInputDialog
)

# ---------- Optional dependencies ----------
try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM   # type: ignore
    _HAS_AESGCM = True
except Exception:
    _HAS_AESGCM = False

# RFT (optional): used only as a keystream mixer; NOT relied upon for security
try:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(BASE_DIR, "ASSEMBLY", "python_bindings"))
    import unitary_rft  # type: ignore
    _HAS_RFT = True
except Exception:
    _HAS_RFT = False

APP_NAME   = "Q-Vault"
VAULT_DIR  = os.path.join(os.path.expanduser("~"), "QuantoniumOS", "QVault")
INDEX_PATH = os.path.join(VAULT_DIR, "index.json")

# ---------- Model ----------

@dataclass
class VaultItem:
    id: str
    title: str
    updated: float
    filename: str
    # future: tags, meta

    @staticmethod
    def new(title: str) -> "VaultItem":
        nid = uuid.uuid4().hex[:10]
        safe = re.sub(r"[^a-zA-Z0-9\-_]+", "_", title).strip("_") or "item"
        fname = f"{safe}_{nid}.qv"
        return VaultItem(id=nid, title=title, updated=time.time(), filename=fname)

# ---------- Storage helpers ----------

def _ensure():
    os.makedirs(VAULT_DIR, exist_ok=True)

def load_index() -> Dict[str, VaultItem]:
    _ensure()
    if not os.path.exists(INDEX_PATH): return {}
    try:
        with open(INDEX_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {k: VaultItem(**v) for k, v in raw.items()}
    except Exception:
        return {}

def save_index(items: Dict[str, VaultItem]):
    _ensure()
    tmp = INDEX_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({k: asdict(v) for k, v in items.items()}, f, ensure_ascii=False, indent=2)
    os.replace(tmp, INDEX_PATH)

def item_path(item: VaultItem) -> str:
    return os.path.join(VAULT_DIR, item.filename)

# ---------- Crypto (KDF + AEAD or fallback) ----------

# KDF: scrypt (stdlib), strong + memory hard
def kdf_scrypt(password: str, salt: bytes) -> bytes:
    return hashlib.scrypt(password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1, dklen=32)

# RFT keystream mixer (deterministic, reversible mask; *not* relied upon for security)
def rft_mask(key: bytes, nonce: bytes, length: int) -> bytes:
    if not _HAS_RFT:
        return b"\x00" * length
    import numpy as np
    # seed vector from key+nonce
    seed = hashlib.sha256(key + nonce).digest()
    vec = np.frombuffer((seed * ((length // 32) + 1))[:length], dtype=np.uint8).astype(np.float64)
    # map bytes ΓåÆ [0,1)
    norm = vec / 255.0
    # size must match engine; pad to power-of-two >= length
    n = 1
    while n < max(32, len(norm)): n <<= 1
    st = np.zeros(n, dtype=np.complex128)
    st[:len(norm)] = norm + 0j
    eng = unitary_rft.UnitaryRFT(n)
    out = eng.forward(st)
    # pull bytes from real+imag (scaled & clipped)
    real = (np.clip(np.round((out.real % 1.0) * 255.0), 0, 255)).astype(np.uint8)
    imag = (np.clip(np.round((out.imag % 1.0) * 255.0), 0, 255)).astype(np.uint8)
    stream = (real ^ imag).tobytes()  # XOR combine
    return stream[:length]

def aead_encrypt(master_key: bytes, plaintext: bytes, aad: bytes=b"") -> bytes:
    nonce = secrets.token_bytes(12)
    if _HAS_AESGCM:
        aes = AESGCM(master_key)
        ct = aes.encrypt(nonce, plaintext, aad)
        return b"QV1\0" + nonce + ct  # header + nonce + ct
    # Fallback: XOR keystream + HMAC (not as strong; use AESGCM in prod)
    mask = hashlib.sha256(master_key + nonce).digest()
    ks   = (mask * ((len(plaintext)//32)+1))[:len(plaintext)]
    xored = bytes([a ^ b for a,b in zip(plaintext, ks)])
    tag = hmac.new(master_key, nonce + xored + aad, hashlib.sha256).digest()
    return b"QV0\0" + nonce + xored + tag

def aead_decrypt(master_key: bytes, blob: bytes, aad: bytes=b"") -> Optional[bytes]:
    if not blob or len(blob) < 4: return None
    magic = blob[:4]
    if magic == b"QV1\0":
        nonce = blob[4:16]
        ct = blob[16:]
        if not _HAS_AESGCM: return None
        try:
            return AESGCM(master_key).decrypt(nonce, ct, aad)
        except Exception:
            return None
    elif magic == b"QV0\0":
        if len(blob) < 4+12+32: return None
        nonce = blob[4:16]
        tag   = blob[-32:]
        xored = blob[16:-32]
        exp   = hmac.new(master_key, nonce + xored + aad, hashlib.sha256).digest()
        if not hmac.compare_digest(tag, exp): return None
        mask = hashlib.sha256(master_key + nonce).digest()
        ks   = (mask * ((len(xored)//32)+1))[:len(xored)]
        return bytes([a ^ b for a,b in zip(xored, ks)])
    return None

# High-level: master key + optional RFT mask
def encrypt_payload(master_key: bytes, plaintext: bytes, use_rft: bool) -> bytes:
    nonce = secrets.token_bytes(12)
    mixed = plaintext
    if use_rft:
        pad = rft_mask(master_key, nonce, len(plaintext))
        mixed = bytes([p ^ q for p,q in zip(plaintext, pad)])
    body = aead_encrypt(master_key, mixed, aad=b"QVAULT")
    return b"QHDR" + bytes([1 if use_rft else 0]) + nonce + body

def decrypt_payload(master_key: bytes, blob: bytes) -> Optional[bytes]:
    if not blob or len(blob) < 4+1+12: return None
    if blob[:4] != b"QHDR": return None
    use_rft = bool(blob[4])
    nonce   = blob[5:17]
    rest    = blob[17:]
    mixed = aead_decrypt(master_key, rest, aad=b"QVAULT")
    if mixed is None: return None
    if use_rft:
        pad = rft_mask(master_key, nonce, len(mixed))
        mixed = bytes([p ^ q for p,q in zip(mixed, pad)])
    return mixed

# ---------- UI ----------

class QVault(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1100, 720)

        self.items: Dict[str, VaultItem] = load_index()
        self.current: Optional[VaultItem] = None
        self.master_key: Optional[bytes] = None
        self._dirty = False
        self._light = True
        self._use_rft = _HAS_RFT  # default: on if available

        # idle lock timer (5 min)
        self.idle = QTimer(self); self.idle.setInterval(5*60*1000)
        self.idle.timeout.connect(self.lock_now)

        # autosave debounce
        self.autosave = QTimer(self); self.autosave.setInterval(700)
        self.autosave.setSingleShot(True); self.autosave.timeout.connect(self._save_silent)

        self._build_ui()
        self._apply_style(light=True)
        self._populate_list()
        self.statusBar().showMessage(self._status_text())

        # prompt for master password on load
        self.unlock_prompt()

    # --- UI Construction ---
    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        root = QVBoxLayout(central); root.setContentsMargins(0,0,0,0); root.setSpacing(0)

        # Header
        head = QWidget(); head.setFixedHeight(56)
        hlyt = QVBoxLayout(head); hlyt.setContentsMargins(20,8,20,8); hlyt.setSpacing(0)
        t = QLabel("Q-Vault"); t.setObjectName("Title")
        s = QLabel("Quantum-grade storage ΓÇó AES-GCM + optional RFT mixer"); s.setObjectName("SubTitle")
        hlyt.addWidget(t); hlyt.addWidget(s)
        root.addWidget(head)

        split = QSplitter(Qt.Horizontal); root.addWidget(split, 1)

        # Sidebar
        sidebar = QWidget(); slyt = QVBoxLayout(sidebar)
        slyt.setContentsMargins(16,16,16,16); slyt.setSpacing(10)

        self.search = QLineEdit(placeholderText="Search itemsΓÇª  (Ctrl+K)")
        self.search.textChanged.connect(self._filter_items)
        slyt.addWidget(self.search)

        self.list = QListWidget(); self.list.itemClicked.connect(self._select_from_list)
        slyt.addWidget(self.list, 1)

        row = QHBoxLayout()
        nb = QPushButton("New (Ctrl+N)"); nb.clicked.connect(self.new_item)
        db = QPushButton("Delete (Del)");  db.clicked.connect(self.delete_item)
        ib = QPushButton("Import");        ib.clicked.connect(self.import_file)
        row.addWidget(nb); row.addWidget(db); row.addWidget(ib); row.addStretch(1)
        slyt.addLayout(row)
        split.addWidget(sidebar)

        # Editor pane
        panel = QWidget(); plyt = QVBoxLayout(panel)
        plyt.setContentsMargins(20,20,20,20); plyt.setSpacing(10)

        self.titleEdit = QLineEdit(placeholderText="Title")
        self.titleEdit.textChanged.connect(self._rename_current)
        plyt.addWidget(self.titleEdit)

        self.editor = QTextEdit()
        self.editor.textChanged.connect(self._on_change)
        plyt.addWidget(self.editor, 1)

        bar = QHBoxLayout()
        saveb = QPushButton("Save (Ctrl+S)"); saveb.clicked.connect(self.save_dialog)
        expb  = QPushButton("ExportΓÇª");       expb.clicked.connect(self.export_plain)
        lockb = QPushButton("Lock (Ctrl+L)"); lockb.clicked.connect(self.lock_now)
        rftb  = QPushButton("RFT Mixer: ON" if self._use_rft else "RFT Mixer: OFF")
        rftb.clicked.connect(lambda: self._toggle_rft(rftb))
        theme = QPushButton("Dark/Light");    theme.clicked.connect(self.toggle_theme)
        copyb = QPushButton("Copy Γºë");        copyb.clicked.connect(self.copy_clipboard)
        bar.addWidget(saveb); bar.addWidget(expb); bar.addWidget(lockb)
        bar.addWidget(rftb); bar.addWidget(theme); bar.addWidget(copyb); bar.addStretch(1)

        self.wordLbl = QLabel("0 chars"); bar.addWidget(self.wordLbl)
        plyt.addLayout(bar)

        split.addWidget(panel); split.setSizes([320, 780])

        self.setStatusBar(QStatusBar())

        # Shortcuts
        QAction(self, shortcut=QKeySequence("Ctrl+N"), triggered=self.new_item).setParent(self)
        QAction(self, shortcut=QKeySequence("Ctrl+S"), triggered=self._save_silent).setParent(self)
        QAction(self, shortcut=QKeySequence("Ctrl+K"), triggered=lambda: self.search.setFocus()).setParent(self)
        QAction(self, shortcut=QKeySequence("Ctrl+L"), triggered=self.lock_now).setParent(self)
        QAction(self, shortcut=QKeySequence("Delete"), triggered=self.delete_item).setParent(self)

        # DnD import
        self.setAcceptDrops(True)

    def _apply_style(self, light=True):
        self._light = light
        if light:
            qss = """
            QMainWindow, QWidget { background:#fafafa; color:#243342; font-family:'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif; }
            #Title { font-size:20px; font-weight:300; color:#2c3e50; }
            #SubTitle { font-size:11px; color:#8aa0b3; }
            QLineEdit, QTextEdit { background:#ffffff; border:1px solid #e9ecef; border-radius:8px; padding:10px 12px; }
            QListWidget { background:#ffffff; border:1px solid #e9ecef; border-radius:8px; }
            QListWidget::item { padding:8px 10px; margin:1px; }
            QListWidget::item:selected { background:#e3f2fd; color:#1976d2; }
            QPushButton { background:#f8f9fa; border:1px solid #dee2e6; border-radius:6px; padding:8px 14px; color:#495057; }
            QPushButton:hover { background:#eef2f6; }
            """
        else:
            qss = """
            QMainWindow, QWidget { background:#0f1216; color:#dfe7ef; font-family:'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif; }
            #Title { font-size:20px; font-weight:300; color:#dfe7ef; }
            #SubTitle { font-size:11px; color:#8aa0b3; }
            QLineEdit, QTextEdit { background:#12161b; border:1px solid #1f2a36; border-radius:8px; padding:10px 12px; color:#e8eff7; }
            QListWidget { background:#12161b; border:1px solid #1f2a36; border-radius:8px; }
            QListWidget::item { padding:8px 10px; margin:1px; }
            QListWidget::item:selected { background:#1d2b3a; color:#7dc4ff; }
            QPushButton { background:#12161b; border:1px solid #2a3847; border-radius:6px; padding:8px 14px; color:#c8d3de; }
            QPushButton:hover { background:#17202a; }
            """
        self.setStyleSheet(qss)

    def toggle_theme(self):
        self._apply_style(not self._light)

    # --- Lock / Unlock ---
    def unlock_prompt(self):
        # Ask for master password; derive key with scrypt
        ok = False
        while not ok:
            pwd, ok = QInputDialog.getText(self, "Unlock Q-Vault", "Enter master password:", echo=QLineEdit.Password)
            if not ok:  # user cancelled
                break
            if not pwd:
                QMessageBox.warning(self, "Password required", "Please enter a password.")
                ok = False
        if ok and pwd:
            salt = self._get_global_salt()
            self.master_key = kdf_scrypt(pwd, salt)
            self.statusBar().showMessage(self._status_text())
            self._set_locked(False)
            self.idle.start()
        else:
            self._set_locked(True)

    def lock_now(self):
        # wipe key, disable editing
        self.master_key = None
        self._set_locked(True)
        self.statusBar().showMessage(self._status_text())

    def _set_locked(self, locked: bool):
        self.titleEdit.setEnabled(not locked)
        self.editor.setEnabled(not locked)
        if locked:
            self.titleEdit.clear(); self.editor.clear()
            self.current = None
        self._populate_list()

    def _get_global_salt(self) -> bytes:
        p = os.path.join(VAULT_DIR, ".salt")
        _ensure()
        if os.path.exists(p):
            return open(p, "rb").read()
        s = secrets.token_bytes(16)
        with open(p, "wb") as f: f.write(s)
        return s

    def _status_text(self) -> str:
        mode = "AES-GCM" if _HAS_AESGCM else "Fallback"
        rft  = "RFT ON" if self._use_rft and _HAS_RFT else "RFT OFF"
        lock = "LOCKED" if self.master_key is None else "UNLOCKED"
        return f"{lock} ΓÇó {mode} ΓÇó {rft}"

    # --- List / Search ---
    def _populate_list(self, keep: bool=False):
        current_id = self.current.id if self.current else None
        self.list.clear()
        items = sorted(self.items.values(), key=lambda x: x.updated, reverse=True)
        for it in items:
            lw = QListWidgetItem(it.title)
            lw.setData(Qt.UserRole, it.id)
            self.list.addItem(lw)
            if keep and it.id == current_id:
                self.list.setCurrentItem(lw)

    def _filter_items(self):
        q = self.search.text().lower().strip()
        for i in range(self.list.count()):
            it = self.list.item(i)
            vid = it.data(Qt.UserRole)
            v = self.items.get(vid)
            hay = (v.title if v else "").lower()
            it.setHidden(q not in hay)

    def _select_from_list(self, item: QListWidgetItem):
        if self.master_key is None: return
        vid = item.data(Qt.UserRole)
        v = self.items.get(vid); ifnot = (v is None)
        if v is None: return
        self._load_item(v)

    # --- Item CRUD ---
    def new_item(self):
        if self.master_key is None: 
            self.unlock_prompt(); 
            if self.master_key is None: return
        base = "Secret"
        title = base; i = 1
        existing = {x.title for x in self.items.values()}
        while title in existing: i += 1; title = f"{base} {i}"
        v = VaultItem.new(title)
        self.items[v.id] = v; save_index(self.items)
        self._populate_list()
        self._load_item(v)
        self.titleEdit.selectAll()

    def delete_item(self):
        if self.master_key is None or not self.current: return
        if QMessageBox.question(self, "Delete item",
                                f"Delete '{self.current.title}' permanently?",
                                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        try: os.remove(item_path(self.current))
        except FileNotFoundError: pass
        self.items.pop(self.current.id, None); save_index(self.items)
        self.current = None; self.titleEdit.clear(); self.editor.clear()
        self._populate_list(); self.statusBar().showMessage("Deleted")

    def _rename_current(self):
        if self.master_key is None or not self.current: return
        new_title = self.titleEdit.text().strip()
        if not new_title or new_title == self.current.title: return
        # filename stays with same id; allow cosmetic rename only
        self.current.title = new_title; self.current.updated = time.time()
        save_index(self.items); self._populate_list(keep=True)

    def _load_item(self, v: VaultItem):
        # read + decrypt payload
        path = item_path(v)
        plain = ""
        if os.path.exists(path):
            blob = open(path, "rb").read()
            if self.master_key:
                out = decrypt_payload(self.master_key, blob)
                plain = out.decode("utf-8", errors="replace") if out is not None else "[Decryption failed]"
        self.current = v
        self.titleEdit.blockSignals(True); self.editor.blockSignals(True)
        self.titleEdit.setText(v.title); self.editor.setPlainText(plain)
        self.titleEdit.blockSignals(False); self.editor.blockSignals(False)
        self._dirty = False; self._update_count()
        self.statusBar().showMessage(self._status_text() + f" ΓÇó Opened: {v.title}")
        self.idle.start()

    def _on_change(self):
        if self.master_key is None: return
        self._dirty = True; self._update_count()
        self.autosave.start(); self.idle.start()

    def _update_count(self):
        self.wordLbl.setText(f"{len(self.editor.toPlainText())} chars")

    def _save_silent(self):
        if self.master_key is None or not self.current or not self._dirty: return
        text = self.editor.toPlainText().encode("utf-8")
        blob = encrypt_payload(self.master_key, text, use_rft=self._use_rft and _HAS_RFT)
        p = item_path(self.current); tmp = p + ".tmp"
        with open(tmp, "wb") as f: f.write(blob)
        os.replace(tmp, p)
        self.current.updated = time.time(); save_index(self.items)
        self._dirty = False
        self.statusBar().showMessage(self._status_text() + " ΓÇó Saved")

    def save_dialog(self):
        """Export current item as encrypted .qvault file."""
        if self.master_key is None or not self.current: return
        path, _ = QFileDialog.getSaveFileName(self, "Save Encrypted", f"{self.current.title}.qvault",
                                              "Q-Vault (*.qvault);;All Files (*.*)")
        if not path: return
        # ensure latest save
        self._save_silent()
        with open(item_path(self.current), "rb") as f:
            blob = f.read()
        with open(path, "wb") as f:
            f.write(blob)
        self.statusBar().showMessage("Exported encrypted file")

    def export_plain(self):
        """Export decrypted text (be careful)."""
        if self.master_key is None or not self.current: return
        path, _ = QFileDialog.getSaveFileName(self, "Export Plaintext", f"{self.current.title}.txt",
                                              "Text (*.txt);;All Files (*.*)")
        if not path: return
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.editor.toPlainText())
        self.statusBar().showMessage("Exported plaintext")

    def import_file(self):
        if self.master_key is None: self.unlock_prompt()
        if self.master_key is None: return
        path, _ = QFileDialog.getOpenFileName(self, "Import Text", "", "Text (*.txt *.md);;All Files (*.*)")
        if not path: return
        try:
            text = open(path, "r", encoding="utf-8").read()
        except Exception as e:
            QMessageBox.warning(self, "Import failed", str(e)); return
        v = VaultItem.new(os.path.splitext(os.path.basename(path))[0])
        self.items[v.id] = v; save_index(self.items)
        self._populate_list(); self._load_item(v)
        self.editor.setPlainText(text); self._save_silent()

    # --- Clipboard ---
    def copy_clipboard(self):
        txt = self.editor.toPlainText()
        if not txt: return
        cb: QClipboard = QApplication.clipboard()
        cb.setText(txt, mode=QClipboard.Clipboard)
        self.statusBar().showMessage("Copied ΓÇó will clear in 12s")
        QTimer.singleShot(12000, lambda: cb.clear(mode=QClipboard.Clipboard))

    # --- RFT toggle ---
    def _toggle_rft(self, btn: QPushButton):
        self._use_rft = not self._use_rft
        btn.setText("RFT Mixer: ON" if self._use_rft else "RFT Mixer: OFF")
        self.statusBar().showMessage(self._status_text())

    # --- DnD import ---
    def dragEnterEvent(self, e: QEvent):
        if e.mimeData().hasUrls(): e.acceptProposedAction()
    def dropEvent(self, e: QEvent):
        for url in e.mimeData().urls():
            p = url.toLocalFile()
            if not p: continue
            try:
                text = open(p, "r", encoding="utf-8").read()
                v = VaultItem.new(os.path.splitext(os.path.basename(p))[0])
                self.items[v.id] = v; save_index(self.items)
                with open(item_path(v), "wb") as _f:
                    blob = encrypt_payload(self.master_key or secrets.token_bytes(32), text.encode("utf-8"), use_rft=False)
                    _f.write(blob)
            except Exception: pass
        self._populate_list()

    # --- Events ---
    def event(self, e: QEvent):
        # reset idle timer on most interactions
        if e.type() in (QEvent.MouseButtonPress, QEvent.KeyPress):
            self.idle.start()
        return super().event(e)

# ---------- Entrypoint ----------

def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setFont(QFont("Segoe UI", 10))

    w = QVault()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

