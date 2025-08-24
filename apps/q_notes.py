"""
QuantoniumOS - Quantum Notes
Advanced note-taking with quantum encryption
"""

import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

try:
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal
    from PyQt5.QtGui import QFont, QIcon, QPixmap
    from PyQt5.QtWidgets import (QComboBox, QGridLayout, QGroupBox,
                                 QHBoxLayout, QInputDialog, QLabel, QLineEdit,
                                 QListWidget, QListWidgetItem, QMessageBox,
                                 QPushButton, QSplitter, QTextEdit,
                                 QVBoxLayout, QWidget)

    PYQT5_AVAILABLE = True
except ImportError:
    PYQT5_AVAILABLE = False


class QuantumNotes(QWidget if PYQT5_AVAILABLE else object):
    """Quantum-encrypted note-taking application"""

    def __init__(self):
        if not PYQT5_AVAILABLE:
            print("⚠️ PyQt5 required for Quantum Notes GUI")
            return

        super().__init__()
        self.notes_data = {}
        self.current_note_id = None
        self.notes_file = "quantum_notes_data.json"
        self.quantum_key = None

        self.init_ui()
        self.load_notes()
        self.generate_quantum_key()

        # Auto-save timer
        self.autosave_timer = QTimer()
        self.autosave_timer.timeout.connect(self.autosave_current_note)
        self.autosave_timer.start(30000)  # Auto-save every 30 seconds

    def init_ui(self):
        """Initialize the quantum notes interface"""
        self.setWindowTitle("📝 QuantoniumOS - Quantum Notes")
        self.setGeometry(300, 300, 1200, 800)

        # Apply cream design styling
        self.setStyleSheet(
            """
            QWidget {
                background-color: #f0ead6;
                color: #2d2d2d;
                font-family: "Segoe UI", Arial, sans-serif;
                font-size: 9pt;
            }
            QGroupBox {
                border: 1px solid #c0c0c0;
                border-radius: 4px;
                padding: 10px;
                margin-top: 10px;
                font-weight: normal;
                background-color: #f8f6f0;
            }
            QGroupBox::title {
                color: #2d2d2d;
                padding: 5px;
                font-weight: bold;
            }
            QPushButton {
                background-color: #0078d4;
                border: 1px solid #005a9e;
                border-radius: 3px;
                padding: 6px 12px;
                color: white;
                font-weight: normal;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #106ebe;
                border: 1px solid #005a9e;
            }
            QLineEdit {
                background-color: white;
                border: 1px solid #c0c0c0;
                border-radius: 3px;
                padding: 5px;
                color: #2d2d2d;
                font-size: 9pt;
            }
            QTextEdit {
                background-color: white;
                border: 1px solid #c0c0c0;
                border-radius: 3px;
                color: #2d2d2d;
                font-size: 9pt;
                padding: 8px;
            }
            QListWidget {
                background: rgba(255, 255, 255, 0.8);
                border: 1px solid rgba(255, 255, 255, 0.3);
                border-radius: 8px;
                color: #1a1a1a;
                selection-background-color: #7c3aed;
                outline: none;
            }
            QListWidget::item {
                padding: 10px;
                border-bottom: 1px solid rgba(255, 255, 255, 0.2);
            }
            QListWidget::item:hover {
                background: rgba(124, 58, 237, 0.1);
            }
            QListWidget::item:selected {
                background: #7c3aed;
                color: white;
            }
            QComboBox {
                background: rgba(255, 255, 255, 0.8);
                border: 1px solid rgba(255, 255, 255, 0.3);
                border-radius: 6px;
                padding: 6px;
                color: #1a1a1a;
            }
            QLabel {
                color: #1a1a1a;
            }
        """
        )

        layout = QVBoxLayout(self)

        # Title
        title = QLabel("📝 Quantum Notes - Encrypted Note System")
        title.setFont(QFont("Consolas", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #1a1a1a; margin: 12px; font-weight: 600;")
        layout.addWidget(title)

        # Main toolbar
        self.create_toolbar(layout)

        # Main content with splitter
        splitter = QSplitter(Qt.Horizontal)

        # Left side - Notes list
        self.create_notes_list(splitter)

        # Right side - Note editor
        self.create_note_editor(splitter)

        splitter.setSizes([300, 900])
        layout.addWidget(splitter)

        # Status bar
        self.create_status_bar(layout)

    def create_toolbar(self, parent_layout):
        """Create main toolbar"""
        toolbar_group = QGroupBox("🛠️ Note Tools")
        toolbar_layout = QHBoxLayout(toolbar_group)

        # File operations
        self.new_note_btn = QPushButton("📝 New Note")
        self.new_note_btn.clicked.connect(self.create_new_note)
        toolbar_layout.addWidget(self.new_note_btn)

        self.save_note_btn = QPushButton("💾 Save")
        self.save_note_btn.clicked.connect(self.save_current_note)
        toolbar_layout.addWidget(self.save_note_btn)

        self.delete_note_btn = QPushButton("🗑️ Delete")
        self.delete_note_btn.clicked.connect(self.delete_current_note)
        toolbar_layout.addWidget(self.delete_note_btn)

        toolbar_layout.addWidget(QLabel("|"))

        # Encryption operations
        self.encrypt_btn = QPushButton("🔒 Encrypt")
        self.encrypt_btn.clicked.connect(self.encrypt_current_note)
        toolbar_layout.addWidget(self.encrypt_btn)

        self.decrypt_btn = QPushButton("🔓 Decrypt")
        self.decrypt_btn.clicked.connect(self.decrypt_current_note)
        toolbar_layout.addWidget(self.decrypt_btn)

        toolbar_layout.addWidget(QLabel("|"))

        # Search and filter
        toolbar_layout.addWidget(QLabel("🔍 Search:"))
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search notes...")
        self.search_input.textChanged.connect(self.filter_notes)
        toolbar_layout.addWidget(self.search_input)

        # Category filter
        toolbar_layout.addWidget(QLabel("📂 Category:"))
        self.category_filter = QComboBox()
        self.category_filter.addItems(
            ["All", "Personal", "Work", "Ideas", "Quantum", "Encrypted"]
        )
        self.category_filter.currentTextChanged.connect(self.filter_notes)
        toolbar_layout.addWidget(self.category_filter)

        toolbar_layout.addStretch()

        parent_layout.addWidget(toolbar_group)

    def create_notes_list(self, parent_splitter):
        """Create notes list panel"""
        notes_group = QGroupBox("📚 Notes Library")
        notes_layout = QVBoxLayout(notes_group)

        # Notes list
        self.notes_list = QListWidget()
        self.notes_list.itemClicked.connect(self.load_selected_note)
        notes_layout.addWidget(self.notes_list)

        # Notes count
        self.notes_count_label = QLabel("📊 Notes: 0")
        self.notes_count_label.setStyleSheet(
            "color: #7c3aed; font-weight: 600; margin: 6px;"
        )
        notes_layout.addWidget(self.notes_count_label)

        parent_splitter.addWidget(notes_group)

    def create_note_editor(self, parent_splitter):
        """Create note editor panel"""
        editor_group = QGroupBox("✏️ Note Editor")
        editor_layout = QVBoxLayout(editor_group)

        # Note metadata
        meta_layout = QHBoxLayout()

        meta_layout.addWidget(QLabel("📋 Title:"))
        self.title_input = QLineEdit()
        self.title_input.setPlaceholderText("Enter note title...")
        meta_layout.addWidget(self.title_input)

        meta_layout.addWidget(QLabel("📂 Category:"))
        self.category_input = QComboBox()
        self.category_input.setEditable(True)
        self.category_input.addItems(
            ["Personal", "Work", "Ideas", "Quantum", "Encrypted"]
        )
        meta_layout.addWidget(self.category_input)

        editor_layout.addLayout(meta_layout)

        # Note content
        editor_layout.addWidget(QLabel("📝 Content:"))
        self.content_editor = QTextEdit()
        self.content_editor.setPlaceholderText(
            "Start typing your note here...\n\nQuantum encryption available for sensitive content."
        )
        editor_layout.addWidget(self.content_editor)

        # Note statistics
        stats_layout = QHBoxLayout()

        self.char_count_label = QLabel("📊 Characters: 0")
        self.char_count_label.setStyleSheet("color: #7c3aed;")
        stats_layout.addWidget(self.char_count_label)

        self.word_count_label = QLabel("📝 Words: 0")
        self.word_count_label.setStyleSheet("color: #7c3aed;")
        stats_layout.addWidget(self.word_count_label)

        self.encryption_status = QLabel("🔓 Unencrypted")
        self.encryption_status.setStyleSheet("color: #f59e0b;")
        stats_layout.addWidget(self.encryption_status)

        stats_layout.addStretch()

        editor_layout.addLayout(stats_layout)

        # Connect text change event
        self.content_editor.textChanged.connect(self.update_statistics)

        parent_splitter.addWidget(editor_group)

    def create_status_bar(self, parent_layout):
        """Create status bar"""
        status_layout = QHBoxLayout()

        self.status_label = QLabel("✅ Quantum Notes Ready")
        self.status_label.setStyleSheet(
            "color: #9d4edd; font-weight: bold; margin: 5px;"
        )
        status_layout.addWidget(self.status_label)

        status_layout.addStretch()

        self.quantum_key_status = QLabel("🔑 Quantum Key: Active")
        self.quantum_key_status.setStyleSheet(
            "color: #00ff88; font-weight: bold; margin: 5px;"
        )
        status_layout.addWidget(self.quantum_key_status)

        self.autosave_status = QLabel("💾 Auto-save: Enabled")
        self.autosave_status.setStyleSheet(
            "color: #00ffcc; font-weight: bold; margin: 5px;"
        )
        status_layout.addWidget(self.autosave_status)

        parent_layout.addLayout(status_layout)

    def generate_quantum_key(self):
        """Generate quantum encryption key"""
        # Simulate quantum key generation
        import random

        random.seed(42)  # For demonstration
        self.quantum_key = "".join([hex(random.randint(0, 15))[2:] for _ in range(64)])
        self.quantum_key_status.setText("🔑 Quantum Key: Generated")
        self.status_label.setText("🔑 Quantum encryption key generated")

    def create_new_note(self):
        """Create a new note"""
        note_id = f"note_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        title, ok = QInputDialog.getText(self, "📝 New Note", "Enter note title:")
        if not ok or not title.strip():
            title = f"New Note {datetime.now().strftime('%H:%M')}"

        new_note = {
            "id": note_id,
            "title": title,
            "content": "",
            "category": "Personal",
            "created": datetime.now().isoformat(),
            "modified": datetime.now().isoformat(),
            "encrypted": False,
            "tags": [],
        }

        self.notes_data[note_id] = new_note
        self.current_note_id = note_id

        self.refresh_notes_list()
        self.load_note_to_editor(new_note)
        self.status_label.setText(f"📝 Created new note: {title}")

    def save_current_note(self):
        """Save the current note"""
        if not self.current_note_id:
            self.status_label.setText("❌ No note selected to save")
            return

        note = self.notes_data[self.current_note_id]
        note["title"] = self.title_input.text() or "Untitled"
        note["content"] = self.content_editor.toPlainText()
        note["category"] = self.category_input.currentText()
        note["modified"] = datetime.now().isoformat()

        self.save_notes()
        self.refresh_notes_list()
        self.status_label.setText(f"💾 Saved note: {note['title']}")

    def delete_current_note(self):
        """Delete the current note"""
        if not self.current_note_id:
            self.status_label.setText("❌ No note selected to delete")
            return

        note = self.notes_data[self.current_note_id]
        reply = QMessageBox.question(
            self,
            "🗑️ Delete Note",
            f"Are you sure you want to delete '{note['title']}'?",
            QMessageBox.Yes | QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            del self.notes_data[self.current_note_id]
            self.current_note_id = None
            self.clear_editor()
            self.save_notes()
            self.refresh_notes_list()
            self.status_label.setText(f"🗑️ Deleted note: {note['title']}")

    def encrypt_current_note(self):
        """Encrypt the current note"""
        if not self.current_note_id:
            self.status_label.setText("❌ No note selected to encrypt")
            return

        note = self.notes_data[self.current_note_id]
        if note["encrypted"]:
            self.status_label.setText("🔒 Note is already encrypted")
            return

        # Simulate quantum encryption
        content = self.content_editor.toPlainText()
        encrypted_content = self.quantum_encrypt(content)

        note["content"] = encrypted_content
        note["encrypted"] = True
        note["modified"] = datetime.now().isoformat()

        self.content_editor.setPlainText(encrypted_content)
        self.encryption_status.setText("🔒 Encrypted")
        self.encryption_status.setStyleSheet("color: #00ff88;")

        self.save_notes()
        self.status_label.setText("🔒 Note encrypted with quantum key")

    def decrypt_current_note(self):
        """Decrypt the current note"""
        if not self.current_note_id:
            self.status_label.setText("❌ No note selected to decrypt")
            return

        note = self.notes_data[self.current_note_id]
        if not note["encrypted"]:
            self.status_label.setText("🔓 Note is not encrypted")
            return

        # Simulate quantum decryption
        encrypted_content = note["content"]
        decrypted_content = self.quantum_decrypt(encrypted_content)

        note["content"] = decrypted_content
        note["encrypted"] = False
        note["modified"] = datetime.now().isoformat()

        self.content_editor.setPlainText(decrypted_content)
        self.encryption_status.setText("🔓 Unencrypted")
        self.encryption_status.setStyleSheet("color: #ffaa00;")

        self.save_notes()
        self.status_label.setText("🔓 Note decrypted successfully")

    def quantum_encrypt(self, text):
        """Simulate quantum encryption"""
        if not text:
            return text

        # Simple XOR encryption with quantum key (for demonstration)
        encrypted = ""
        for i, char in enumerate(text):
            key_char = self.quantum_key[i % len(self.quantum_key)]
            encrypted_char = chr(ord(char) ^ int(key_char, 16))
            encrypted += encrypted_char

        # Encode to hex for safe storage
        return encrypted.encode("utf-8", errors="replace").hex()

    def quantum_decrypt(self, encrypted_hex):
        """Simulate quantum decryption"""
        if not encrypted_hex:
            return encrypted_hex

        try:
            # Decode from hex
            encrypted = bytes.fromhex(encrypted_hex).decode("utf-8", errors="replace")

            # XOR decryption
            decrypted = ""
            for i, char in enumerate(encrypted):
                key_char = self.quantum_key[i % len(self.quantum_key)]
                decrypted_char = chr(ord(char) ^ int(key_char, 16))
                decrypted += decrypted_char

            return decrypted
        except:
            return "[DECRYPTION ERROR - Quantum key mismatch]"

    def load_selected_note(self, item):
        """Load selected note from list"""
        note_id = item.data(Qt.UserRole)
        if note_id in self.notes_data:
            self.current_note_id = note_id
            note = self.notes_data[note_id]
            self.load_note_to_editor(note)
            self.status_label.setText(f"📖 Loaded note: {note['title']}")

    def load_note_to_editor(self, note):
        """Load note data to editor"""
        self.title_input.setText(note["title"])
        self.content_editor.setPlainText(note["content"])
        self.category_input.setCurrentText(note["category"])

        if note["encrypted"]:
            self.encryption_status.setText("🔒 Encrypted")
            self.encryption_status.setStyleSheet("color: #00ff88;")
        else:
            self.encryption_status.setText("🔓 Unencrypted")
            self.encryption_status.setStyleSheet("color: #ffaa00;")

        self.update_statistics()

    def clear_editor(self):
        """Clear the editor"""
        self.title_input.clear()
        self.content_editor.clear()
        self.category_input.setCurrentText("Personal")
        self.encryption_status.setText("🔓 Unencrypted")
        self.encryption_status.setStyleSheet("color: #ffaa00;")

    def filter_notes(self):
        """Filter notes based on search and category"""
        search_text = self.search_input.text().lower()
        category_filter = self.category_filter.currentText()

        self.notes_list.clear()

        filtered_notes = []
        for note in self.notes_data.values():
            # Category filter
            if category_filter != "All" and note["category"] != category_filter:
                continue

            # Search filter
            if (
                search_text
                and search_text not in note["title"].lower()
                and search_text not in note["content"].lower()
            ):
                continue

            filtered_notes.append(note)

        # Sort by modification date (newest first)
        filtered_notes.sort(key=lambda x: x["modified"], reverse=True)

        for note in filtered_notes:
            item = QListWidgetItem()

            # Create display text
            title = note["title"]
            category = note["category"]
            modified = datetime.fromisoformat(note["modified"]).strftime("%m/%d %H:%M")
            encrypted_icon = "🔒" if note["encrypted"] else "📝"

            item.setText(f"{encrypted_icon} {title}\n📂 {category} • {modified}")
            item.setData(Qt.UserRole, note["id"])

            self.notes_list.addItem(item)

        self.notes_count_label.setText(f"📊 Notes: {len(filtered_notes)}")

    def refresh_notes_list(self):
        """Refresh the notes list"""
        self.filter_notes()

    def update_statistics(self):
        """Update character and word count"""
        content = self.content_editor.toPlainText()
        char_count = len(content)
        word_count = len(content.split()) if content.strip() else 0

        self.char_count_label.setText(f"📊 Characters: {char_count}")
        self.word_count_label.setText(f"📝 Words: {word_count}")

    def autosave_current_note(self):
        """Auto-save current note"""
        if self.current_note_id and self.title_input.text().strip():
            self.save_current_note()
            self.autosave_status.setText("💾 Auto-saved")
            QTimer.singleShot(
                2000, lambda: self.autosave_status.setText("💾 Auto-save: Enabled")
            )

    def load_notes(self):
        """Load notes from file"""
        try:
            if os.path.exists(self.notes_file):
                with open(self.notes_file, "r", encoding="utf-8") as f:
                    self.notes_data = json.load(f)
            else:
                self.notes_data = {}

            self.refresh_notes_list()
            self.status_label.setText(f"📚 Loaded {len(self.notes_data)} notes")
        except Exception as e:
            self.status_label.setText(f"❌ Error loading notes: {str(e)}")
            self.notes_data = {}

    def save_notes(self):
        """Save notes to file"""
        try:
            with open(self.notes_file, "w", encoding="utf-8") as f:
                json.dump(self.notes_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.status_label.setText(f"❌ Error saving notes: {str(e)}")


def main():
    """Main entry point"""
    if not PYQT5_AVAILABLE:
        print("❌ PyQt5 required for Quantum Notes")
        return

    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = QuantumNotes()
    window.show()

    return app.exec_()


# Alias for app controller compatibility
QNotesApp = QuantumNotes

if __name__ == "__main__":
    main()
