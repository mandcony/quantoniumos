#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q-Notes — Quantum Text Editor (revamped)
- Minimal, fast, persistent
- Debounced autosave to disk
- Search + keyboard shortcuts
- Optional Markdown preview (if 'markdown' is installed)
- Light/Dark theme toggle
"""

import os, sys, json, time, re, uuid
from dataclasses import dataclass, asdict
from typing import Dict, Optional

from PyQt5.QtCore import Qt, QTimer, QSize, QEvent
from PyQt5.QtGui import QFont, QIcon, QCloseEvent, QKeySequence
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QListWidget, QListWidgetItem, QTextEdit, QTextBrowser, QLineEdit,
    QPushButton, QFileDialog, QStatusBar, QLabel, QMessageBox, QAction
)

# Optional Markdown preview
try:
    import markdown as _md
except Exception:
    _md = None

APP_NAME = "Q-Notes"
NOTE_ROOT = os.path.join(os.path.expanduser("~"), "QuantoniumOS", "QNotes")
INDEX_PATH = os.path.join(NOTE_ROOT, "index.json")

# ---------- Models ----------

@dataclass
class Note:
    id: str
    title: str
    text: str = ""
    created: float = 0.0
    updated: float = 0.0
    filename: str = ""  # .md on disk

    @staticmethod
    def new(title: str) -> "Note":
        nid = uuid.uuid4().hex[:10]
        safe = re.sub(r"[^a-zA-Z0-9\-_]+", "_", title).strip("_") or "note"
        fname = f"{safe}_{nid}.md"
        now = time.time()
        return Note(id=nid, title=title, text="", created=now, updated=now, filename=fname)

# ---------- Storage ----------

def ensure_dirs():
    os.makedirs(NOTE_ROOT, exist_ok=True)

def load_index() -> Dict[str, Note]:
    ensure_dirs()
    if not os.path.exists(INDEX_PATH):
        return {}
    try:
        with open(INDEX_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        notes = {}
        for nid, meta in raw.items():
            notes[nid] = Note(**meta)
        return notes
    except Exception:
        return {}

def save_index(notes: Dict[str, Note]) -> None:
    ensure_dirs()
    serial = {nid: asdict(n) for nid, n in notes.items()}
    tmp = INDEX_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(serial, f, ensure_ascii=False, indent=2)
    os.replace(tmp, INDEX_PATH)

def note_path(note: Note) -> str:
    return os.path.join(NOTE_ROOT, note.filename)

def write_note_text(note: Note) -> None:
    ensure_dirs()
    tmp = note_path(note) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(note.text)
    os.replace(tmp, note_path(note))

def read_note_text(note: Note) -> str:
    p = note_path(note)
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return f.read()
    return ""

def delete_note_files(note: Note) -> None:
    try:
        os.remove(note_path(note))
    except FileNotFoundError:
        pass

# ---------- UI ----------

class QNotes(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setObjectName("AppWindow")
        self.setWindowTitle(APP_NAME)
        self.resize(1100, 720)

        # State
        self.notes: Dict[str, Note] = load_index()
        self.current: Optional[Note] = None
        self._dirty = False

        # Autosave timer (debounced)
        self.autosave = QTimer(self)
        self.autosave.setInterval(600)   # ms
        self.autosave.setSingleShot(True)
        self.autosave.timeout.connect(self._save_current_silent)

        self._build_ui()
        self._apply_style(light=True)
        self._populate_list()
        self.statusBar().showMessage("Ready")

    # ----- UI construction -----
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0,0,0,0)
        root.setSpacing(0)

        # Header
        header = self._header()
        root.addWidget(header)

        # Content split
        split = QSplitter(Qt.Horizontal)
        root.addWidget(split, 1)

        # Sidebar
        sidebar = QWidget()
        slyt = QVBoxLayout(sidebar); slyt.setContentsMargins(16,16,16,16); slyt.setSpacing(10)

        self.search = QLineEdit()
        self.search.setPlaceholderText("Search notes…  (Ctrl+K)")
        self.search.textChanged.connect(self._filter_notes)
        slyt.addWidget(self.search)

        self.list = QListWidget()
        self.list.itemClicked.connect(self._select_from_list)
        self.list.setAlternatingRowColors(True)
        slyt.addWidget(self.list, 1)

        btnrow = QHBoxLayout()
        newb = QPushButton("New (Ctrl+N)"); newb.clicked.connect(self.new_note)
        delb = QPushButton("Delete (Del)");  delb.clicked.connect(self.delete_note)
        impb = QPushButton("Import");        impb.clicked.connect(self.import_file)
        btnrow.addWidget(newb); btnrow.addWidget(delb); btnrow.addWidget(impb); btnrow.addStretch(1)
        slyt.addLayout(btnrow)

        split.addWidget(sidebar)

        # Editor + (optional) Preview
        editorPane = QWidget()
        elyt = QVBoxLayout(editorPane); elyt.setContentsMargins(20,20,20,20); elyt.setSpacing(10)

        self.titleEdit = QLineEdit(placeholderText="Title")
        self.titleEdit.textChanged.connect(self._rename_current)
        elyt.addWidget(self.titleEdit)

        self.editor = QTextEdit()
        self.editor.textChanged.connect(self._on_text_change)
        elyt.addWidget(self.editor, 1)

        # Markdown preview (optional)
        self.preview = QTextBrowser()
        self.preview.setOpenExternalLinks(True)
        self.preview.setVisible(False)  # hidden by default
        elyt.addWidget(self.preview, 1)

        row = QHBoxLayout()
        saveb   = QPushButton("Save (Ctrl+S)"); saveb.clicked.connect(self.save_note_dialog)
        exportb = QPushButton("Export…");       exportb.clicked.connect(self.export_current)
        toggleb = QPushButton("Preview ⌘");     toggleb.clicked.connect(self.toggle_preview)
        themeb  = QPushButton("Dark/Light");    themeb.clicked.connect(self.toggle_theme)
        row.addWidget(saveb); row.addWidget(exportb); row.addWidget(toggleb); row.addWidget(themeb)
        row.addStretch(1)

        # Word count / status
        self.wordLbl = QLabel("0 words")
        row.addWidget(self.wordLbl)

        elyt.addLayout(row)

        split.addWidget(editorPane)
        split.setSizes([320, 780])

        # Status bar
        self.setStatusBar(QStatusBar())

        # Shortcuts
        self._bind_shortcuts()

        # Drag & drop import
        self.setAcceptDrops(True)

    def _header(self) -> QWidget:
        w = QWidget(); w.setFixedHeight(56)
        l = QVBoxLayout(w); l.setContentsMargins(20,8,20,8); l.setSpacing(0)
        t = QLabel(APP_NAME); t.setObjectName("Title")
        s = QLabel("Secure, minimal notes • autosave • local storage"); s.setObjectName("SubTitle")
        l.addWidget(t); l.addWidget(s)
        return w

    def _apply_style(self, light=True):
        # Minimal Quantonium aesthetic
        if light:
            base = """
            QMainWindow, QWidget { background:#fafafa; color:#243342; font-family:'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif; }
            #Title { font-size:20px; font-weight:300; color:#2c3e50; }
            #SubTitle { font-size:11px; color:#8aa0b3; }
            QLineEdit, QTextEdit, QTextBrowser { background:#ffffff; border:1px solid #e9ecef; border-radius:8px; padding:10px 12px; }
            QListWidget { background:#ffffff; border:1px solid #e9ecef; border-radius:8px; }
            QListWidget::item { padding:8px 10px; margin:1px; }
            QListWidget::item:selected { background:#e3f2fd; color:#1976d2; }
            QPushButton { background:#f8f9fa; border:1px solid #dee2e6; border-radius:6px; padding:8px 14px; color:#495057; }
            QPushButton:hover { background:#eef2f6; }
            """
        else:
            base = """
            QMainWindow, QWidget { background:#0f1216; color:#dfe7ef; font-family:'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif; }
            #Title { font-size:20px; font-weight:300; color:#dfe7ef; }
            #SubTitle { font-size:11px; color:#8aa0b3; }
            QLineEdit, QTextEdit, QTextBrowser { background:#12161b; border:1px solid #1f2a36; border-radius:8px; padding:10px 12px; color:#e8eff7; }
            QListWidget { background:#12161b; border:1px solid #1f2a36; border-radius:8px; }
            QListWidget::item { padding:8px 10px; margin:1px; }
            QListWidget::item:selected { background:#1d2b3a; color:#7dc4ff; }
            QPushButton { background:#12161b; border:1px solid #2a3847; border-radius:6px; padding:8px 14px; color:#c8d3de; }
            QPushButton:hover { background:#17202a; }
            """
        self.setStyleSheet(base)
        self._light = light

    def toggle_theme(self):
        self._apply_style(not getattr(self, "_light", True))

    def _bind_shortcuts(self):
        QAction(self, shortcut=QKeySequence("Ctrl+N"), triggered=self.new_note).setParent(self)
        QAction(self, shortcut=QKeySequence("Ctrl+S"), triggered=self._save_current_silent).setParent(self)
        QAction(self, shortcut=QKeySequence("Delete"), triggered=self.delete_note).setParent(self)
        QAction(self, shortcut=QKeySequence("Ctrl+K"), triggered=lambda: self.search.setFocus()).setParent(self)
        QAction(self, shortcut=QKeySequence("Ctrl+P"), triggered=self.toggle_preview).setParent(self)

    # ----- Notes list management -----
    def _populate_list(self, keep_selection: bool=False):
        current_id = self.current.id if (self.current) else None
        self.list.clear()
        # sort by updated desc
        for note in sorted(self.notes.values(), key=lambda n: n.updated, reverse=True):
            item = QListWidgetItem(f"{note.title}")
            item.setData(Qt.UserRole, note.id)
            self.list.addItem(item)
            if keep_selection and note.id == current_id:
                self.list.setCurrentItem(item)

        if self.list.count() and not keep_selection and not self.current:
            self.list.setCurrentRow(0)
            self._select_from_list(self.list.item(0))

    def _select_from_list(self, item: QListWidgetItem):
        nid = item.data(Qt.UserRole)
        note = self.notes.get(nid)
        if not note:
            return
        self._load_into_editor(note)

    def _load_into_editor(self, note: Note):
        # Save previous if dirty
        if self._dirty:
            self._save_current_silent()

        self.current = note
        note.text = read_note_text(note)  # ensure latest from disk

        self.titleEdit.blockSignals(True)
        self.editor.blockSignals(True)
        self.titleEdit.setText(note.title)
        self.editor.setPlainText(note.text)
        self.titleEdit.blockSignals(False)
        self.editor.blockSignals(False)

        self._dirty = False
        self._update_counts()
        self.statusBar().showMessage(f"Opened: {note.title}")

    # ----- CRUD -----
    def new_note(self):
        base = "New Note"
        title = base
        i = 1
        existing = {n.title for n in self.notes.values()}
        while title in existing:
            i += 1
            title = f"{base} {i}"
        note = Note.new(title)
        self.notes[note.id] = note
        save_index(self.notes)
        write_note_text(note)
        self._populate_list()
        # select the new
        items = self.list.findItems(note.title, Qt.MatchExactly)
        if items:
            self.list.setCurrentItem(items[0])
            self._load_into_editor(note)
            self.titleEdit.selectAll()

    def delete_note(self):
        if not self.current: return
        reply = QMessageBox.question(self, "Delete note?",
                                     f"Delete '{self.current.title}' permanently?",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply != QMessageBox.Yes: return
        delete_note_files(self.current)
        self.notes.pop(self.current.id, None)
        save_index(self.notes)
        self.current = None
        self._populate_list()
        self.titleEdit.clear(); self.editor.clear()
        self.statusBar().showMessage("Note deleted")

    def _rename_current(self):
        if not self.current: return
        new_title = self.titleEdit.text().strip()
        if not new_title: return
        if new_title == self.current.title: return
        # update filename only if stem changed (keep id)
        old_path = note_path(self.current)
        stem = re.sub(r"[^a-zA-Z0-9\-_]+", "_", new_title).strip("_") or "note"
        new_fname = f"{stem}_{self.current.id}.md"
        self.current.title = new_title
        if new_fname != self.current.filename:
            self.current.filename = new_fname
            # rename file if exists
            if os.path.exists(old_path):
                os.replace(old_path, note_path(self.current))
        self.current.updated = time.time()
        save_index(self.notes)
        self._populate_list(keep_selection=True)

    def _on_text_change(self):
        self._dirty = True
        self._update_counts()
        self.autosave.start()  # debounce

    def _update_counts(self):
        text = self.editor.toPlainText()
        words = len([w for w in re.findall(r"\b\w+\b", text)])
        self.wordLbl.setText(f"{words} words")
        if self.preview.isVisible() and _md:
            html = _md.markdown(text, extensions=["extra", "sane_lists"])
            self.preview.setHtml(html)

    def _save_current_silent(self):
        if not self.current: return
        if not self._dirty: return
        self.current.text = self.editor.toPlainText()
        self.current.updated = time.time()
        write_note_text(self.current)
        save_index(self.notes)
        self._dirty = False
        self.statusBar().showMessage(f"Saved at {time.strftime('%H:%M:%S')}")

    def save_note_dialog(self):
        """Export current note to a chosen file (leaves the local store intact)."""
        if not self.current: return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Note", f"{self.current.title}.md",
            "Markdown (*.md);;Text (*.txt);;All Files (*.*)"
        )
        if not path: return
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.editor.toPlainText())
        self.statusBar().showMessage(f"Exported: {path}")

    def export_current(self):
        self.save_note_dialog()

    def import_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import Text", "", "Text/Markdown (*.txt *.md);;All Files (*.*)")
        if not path: return
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            QMessageBox.warning(self, "Import failed", str(e))
            return
        title = os.path.splitext(os.path.basename(path))[0]
        note = Note.new(title)
        note.text = text
        self.notes[note.id] = note
        write_note_text(note)
        save_index(self.notes)
        self._populate_list()
        # select imported
        items = self.list.findItems(note.title, Qt.MatchExactly)
        if items:
            self.list.setCurrentItem(items[0])
            self._load_into_editor(note)

    # ----- Search / Filter -----
    def _filter_notes(self):
        q = self.search.text().lower().strip()
        for i in range(self.list.count()):
            item = self.list.item(i)
            nid = item.data(Qt.UserRole)
            n = self.notes.get(nid)
            hay = (n.title + " " + (n.text[:500] if n else "")).lower()
            item.setHidden(q not in hay)

    # ----- Preview -----
    def toggle_preview(self):
        if not _md:
            QMessageBox.information(self, "Markdown", "Install the 'markdown' package to enable live preview.\n\npip install markdown")
            return
        self.preview.setVisible(not self.preview.isVisible())
        self._update_counts()

    # ----- DnD import -----
    def dragEnterEvent(self, e: QEvent):
        if e.mimeData().hasUrls(): e.acceptProposedAction()
    def dropEvent(self, e: QEvent):
        for url in e.mimeData().urls():
            p = url.toLocalFile()
            if not p: continue
            try:
                with open(p, "r", encoding="utf-8") as f:
                    text = f.read()
                title = os.path.splitext(os.path.basename(p))[0]
                note = Note.new(title); note.text = text
                self.notes[note.id] = note
                write_note_text(note)
            except Exception:
                continue
        save_index(self.notes)
        self._populate_list()

    # ----- Lifecycle -----
    def closeEvent(self, event: QCloseEvent):
        self._save_current_silent()
        event.accept()

# ---------- Entrypoint ----------

def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)

    # Futuristic but readable default font
    app.setFont(QFont("Segoe UI", 10))

    win = QNotes()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
