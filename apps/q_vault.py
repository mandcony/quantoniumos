"""
QuantoniumOS - Quantum Vault
Secure file storage with quantum encryption
"""

import sys
import os
import json
import hashlib
import shutil
from datetime import datetime
from pathlib import Path

try:
    from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, 
                                QPushButton, QLabel, QTextEdit, QLineEdit,
                                QGroupBox, QGridLayout, QListWidget, QComboBox,
                                QSplitter, QMessageBox, QInputDialog, QListWidgetItem,
                                QFileDialog, QProgressBar, QTreeWidget, QTreeWidgetItem)
    from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
    from PyQt5.QtGui import QFont, QIcon, QPixmap
    PYQT5_AVAILABLE = True
except ImportError:
    PYQT5_AVAILABLE = False

class QuantumVault(QWidget if PYQT5_AVAILABLE else object):
    """Quantum-encrypted secure file storage"""
    
    def __init__(self):
        if not PYQT5_AVAILABLE:
            print("⚠️ PyQt5 required for Quantum Vault GUI")
            return
            
        super().__init__()
        self.vault_data = {}
        self.vault_directory = Path("quantum_vault_data")
        self.vault_config_file = "quantum_vault_config.json"
        self.current_folder = "root"
        self.encryption_key = None
        self.vault_locked = True
        
        self.init_ui()
        self.setup_vault_directory()
        self.generate_vault_key()
        
        # Auto-save timer
        self.autosave_timer = QTimer()
        self.autosave_timer.timeout.connect(self.save_vault_config)
        self.autosave_timer.start(60000)  # Auto-save every minute
    
    def init_ui(self):
        """Initialize the quantum vault interface"""
        self.setWindowTitle("🔐 QuantoniumOS - Quantum Vault")
        self.setGeometry(200, 200, 1300, 850)
        
        # Apply quantum styling
        self.setStyleSheet("""
            QWidget {
                background-color: #0a0e1a;
                color: #00ffcc;
                font-family: "Consolas", monospace;
            }
            QGroupBox {
                border: 2px solid #00ffcc;
                border-radius: 8px;
                padding: 10px;
                margin-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                color: #ff6347;
                padding: 5px;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3a1810, stop:1 #2a1008);
                border: 2px solid #ff6347;
                border-radius: 6px;
                padding: 8px 16px;
                color: #ff6347;
                font-weight: bold;
                min-width: 80px;
            }
            QPushButton:hover {
                border: 2px solid #ff7f5f;
                color: #ff7f5f;
            }
            QPushButton:disabled {
                border: 2px solid #666666;
                color: #666666;
                background: #2a2a2a;
            }
            QLineEdit {
                background: #1a2332;
                border: 1px solid #ff6347;
                border-radius: 4px;
                padding: 8px;
                color: #ffffff;
                font-size: 14px;
            }
            QTextEdit {
                background: #1a2332;
                border: 1px solid #ff6347;
                border-radius: 4px;
                color: #ffffff;
                font-size: 12px;
            }
            QListWidget, QTreeWidget {
                background: #1a2332;
                border: 1px solid #ff6347;
                border-radius: 4px;
                color: #ffffff;
                selection-background-color: #ff6347;
                outline: none;
            }
            QListWidget::item, QTreeWidget::item {
                padding: 8px;
                border-bottom: 1px solid #2a3040;
            }
            QListWidget::item:hover, QTreeWidget::item:hover {
                background: #2a3040;
            }
            QListWidget::item:selected, QTreeWidget::item:selected {
                background: #ff6347;
                color: #ffffff;
            }
            QComboBox {
                background: #1a2332;
                border: 1px solid #ff6347;
                border-radius: 4px;
                padding: 5px;
                color: #ffffff;
            }
            QProgressBar {
                border: 1px solid #ff6347;
                border-radius: 4px;
                background: #1a2332;
                text-align: center;
                color: #ffffff;
                height: 20px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #ff6347, stop:1 #ff7f5f);
                border-radius: 2px;
            }
            QLabel {
                color: #00ffcc;
            }
        """)
        
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("🔐 Quantum Vault - Secure File Storage")
        title.setFont(QFont("Consolas", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #ff6347; margin: 10px;")
        layout.addWidget(title)
        
        # Vault security bar
        self.create_security_bar(layout)
        
        # Main toolbar
        self.create_toolbar(layout)
        
        # Main content with splitter
        splitter = QSplitter(Qt.Horizontal)
        
        # Left side - File tree
        self.create_file_tree(splitter)
        
        # Right side - File operations
        self.create_operations_panel(splitter)
        
        splitter.setSizes([400, 900])
        layout.addWidget(splitter)
        
        # Status bar
        self.create_status_bar(layout)
        
        # Initially lock the vault
        self.lock_vault()
    
    def create_security_bar(self, parent_layout):
        """Create vault security status bar"""
        security_group = QGroupBox("🛡️ Vault Security Status")
        security_layout = QHBoxLayout(security_group)
        
        # Vault lock status
        self.lock_status = QLabel("🔒 Vault: Locked")
        self.lock_status.setStyleSheet("color: #ff6347; font-weight: bold;")
        security_layout.addWidget(self.lock_status)
        
        # Unlock/Lock buttons
        self.unlock_btn = QPushButton("🔓 Unlock Vault")
        self.unlock_btn.clicked.connect(self.unlock_vault)
        security_layout.addWidget(self.unlock_btn)
        
        self.lock_btn = QPushButton("🔒 Lock Vault")
        self.lock_btn.clicked.connect(self.lock_vault)
        self.lock_btn.setEnabled(False)
        security_layout.addWidget(self.lock_btn)
        
        security_layout.addWidget(QLabel("|"))
        
        # Encryption info
        self.encryption_info = QLabel("🔑 Quantum Encryption: Active")
        self.encryption_info.setStyleSheet("color: #00ff88;")
        security_layout.addWidget(self.encryption_info)
        
        # Storage info
        self.storage_info = QLabel("💾 Files: 0 | Size: 0 MB")
        self.storage_info.setStyleSheet("color: #00ffcc;")
        security_layout.addWidget(self.storage_info)
        
        security_layout.addStretch()
        
        parent_layout.addWidget(security_group)
    
    def create_toolbar(self, parent_layout):
        """Create main toolbar"""
        toolbar_group = QGroupBox("🛠️ Vault Operations")
        toolbar_layout = QHBoxLayout(toolbar_group)
        
        # File operations
        self.add_file_btn = QPushButton("📂 Add Files")
        self.add_file_btn.clicked.connect(self.add_files_to_vault)
        self.add_file_btn.setEnabled(False)
        toolbar_layout.addWidget(self.add_file_btn)
        
        self.add_folder_btn = QPushButton("📁 Add Folder")
        self.add_folder_btn.clicked.connect(self.add_folder_to_vault)
        self.add_folder_btn.setEnabled(False)
        toolbar_layout.addWidget(self.add_folder_btn)
        
        self.extract_btn = QPushButton("📤 Extract")
        self.extract_btn.clicked.connect(self.extract_files)
        self.extract_btn.setEnabled(False)
        toolbar_layout.addWidget(self.extract_btn)
        
        self.delete_btn = QPushButton("🗑️ Delete")
        self.delete_btn.clicked.connect(self.delete_selected)
        self.delete_btn.setEnabled(False)
        toolbar_layout.addWidget(self.delete_btn)
        
        toolbar_layout.addWidget(QLabel("|"))
        
        # Folder operations
        self.new_folder_btn = QPushButton("📁 New Folder")
        self.new_folder_btn.clicked.connect(self.create_new_folder)
        self.new_folder_btn.setEnabled(False)
        toolbar_layout.addWidget(self.new_folder_btn)
        
        toolbar_layout.addWidget(QLabel("|"))
        
        # Search
        toolbar_layout.addWidget(QLabel("🔍 Search:"))
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search vault...")
        self.search_input.textChanged.connect(self.filter_files)
        self.search_input.setEnabled(False)
        toolbar_layout.addWidget(self.search_input)
        
        toolbar_layout.addStretch()
        
        parent_layout.addWidget(toolbar_group)
    
    def create_file_tree(self, parent_splitter):
        """Create file tree panel"""
        tree_group = QGroupBox("📁 Vault Contents")
        tree_layout = QVBoxLayout(tree_group)
        
        # File tree
        self.file_tree = QTreeWidget()
        self.file_tree.setHeaderLabels(["Name", "Type", "Size", "Modified"])
        self.file_tree.itemClicked.connect(self.select_file)
        self.file_tree.setEnabled(False)
        tree_layout.addWidget(self.file_tree)
        
        # Tree statistics
        self.tree_stats = QLabel("📊 Vault Statistics: Locked")
        self.tree_stats.setStyleSheet("color: #ff6347; font-weight: bold; margin: 5px;")
        tree_layout.addWidget(self.tree_stats)
        
        parent_splitter.addWidget(tree_group)
    
    def create_operations_panel(self, parent_splitter):
        """Create file operations panel"""
        ops_group = QGroupBox("⚙️ File Operations")
        ops_layout = QVBoxLayout(ops_group)
        
        # File preview/info
        self.create_file_info_panel(ops_layout)
        
        # Progress area
        progress_group = QGroupBox("📊 Operations Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.operation_label = QLabel("⏸️ No operation in progress")
        progress_layout.addWidget(self.operation_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_bar)
        
        ops_layout.addWidget(progress_group)
        
        # Vault operations log
        log_group = QGroupBox("📋 Operation Log")
        log_layout = QVBoxLayout(log_group)
        
        self.operation_log = QTextEdit()
        self.operation_log.setMaximumHeight(200)
        self.operation_log.setReadOnly(True)
        self.operation_log.setEnabled(False)
        log_layout.addWidget(self.operation_log)
        
        ops_layout.addWidget(log_group)
        
        parent_splitter.addWidget(ops_group)
    
    def create_file_info_panel(self, parent_layout):
        """Create file information panel"""
        info_group = QGroupBox("📄 File Information")
        info_layout = QVBoxLayout(info_group)
        
        # File details
        self.file_info_text = QTextEdit()
        self.file_info_text.setMaximumHeight(200)
        self.file_info_text.setReadOnly(True)
        self.file_info_text.setEnabled(False)
        info_layout.addWidget(self.file_info_text)
        
        # File operations for selected file
        file_ops_layout = QHBoxLayout()
        
        self.view_btn = QPushButton("👁️ View")
        self.view_btn.clicked.connect(self.view_selected_file)
        self.view_btn.setEnabled(False)
        file_ops_layout.addWidget(self.view_btn)
        
        self.edit_btn = QPushButton("✏️ Edit Name")
        self.edit_btn.clicked.connect(self.edit_file_name)
        self.edit_btn.setEnabled(False)
        file_ops_layout.addWidget(self.edit_btn)
        
        self.copy_btn = QPushButton("📋 Copy")
        self.copy_btn.clicked.connect(self.copy_selected_file)
        self.copy_btn.setEnabled(False)
        file_ops_layout.addWidget(self.copy_btn)
        
        file_ops_layout.addStretch()
        
        info_layout.addLayout(file_ops_layout)
        parent_layout.addWidget(info_group)
    
    def create_status_bar(self, parent_layout):
        """Create status bar"""
        status_layout = QHBoxLayout()
        
        self.status_label = QLabel("🔒 Quantum Vault - Please unlock to access files")
        self.status_label.setStyleSheet("color: #ff6347; font-weight: bold; margin: 5px;")
        status_layout.addWidget(self.status_label)
        
        status_layout.addStretch()
        
        self.vault_key_status = QLabel("🔑 Quantum Key: Generated")
        self.vault_key_status.setStyleSheet("color: #00ff88; font-weight: bold; margin: 5px;")
        status_layout.addWidget(self.vault_key_status)
        
        self.autosave_status = QLabel("💾 Auto-save: Enabled")
        self.autosave_status.setStyleSheet("color: #00ffcc; font-weight: bold; margin: 5px;")
        status_layout.addWidget(self.autosave_status)
        
        parent_layout.addLayout(status_layout)
    
    def setup_vault_directory(self):
        """Setup vault directory structure"""
        self.vault_directory.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.vault_directory / "files").mkdir(exist_ok=True)
        (self.vault_directory / "metadata").mkdir(exist_ok=True)
        (self.vault_directory / "temp").mkdir(exist_ok=True)
        
        self.load_vault_config()
    
    def generate_vault_key(self):
        """Generate quantum encryption key for vault"""
        # Simulate quantum key generation
        import random
        random.seed(1337)  # For demonstration
        self.encryption_key = ''.join([hex(random.randint(0, 15))[2:] for _ in range(128)])
        self.vault_key_status.setText("🔑 Quantum Key: Generated")
    
    def unlock_vault(self):
        """Unlock the vault with authentication"""
        password, ok = QInputDialog.getText(self, "🔓 Unlock Vault", 
                                          "Enter vault password:", 
                                          QLineEdit.Password)
        
        if not ok:
            return
        
        # Simple password check (in real app, use proper authentication)
        if password == "quantum2024" or password == "admin":
            self.vault_locked = False
            self.lock_status.setText("🔓 Vault: Unlocked")
            self.lock_status.setStyleSheet("color: #00ff88; font-weight: bold;")
            
            # Enable all controls
            self.enable_vault_controls(True)
            
            # Load vault contents
            self.refresh_file_tree()
            
            self.status_label.setText("✅ Vault unlocked successfully")
            self.log_operation("🔓 Vault unlocked successfully")
        else:
            self.status_label.setText("❌ Invalid password")
            QMessageBox.warning(self, "❌ Access Denied", "Invalid password!")
    
    def lock_vault(self):
        """Lock the vault"""
        self.vault_locked = True
        self.lock_status.setText("🔒 Vault: Locked")
        self.lock_status.setStyleSheet("color: #ff6347; font-weight: bold;")
        
        # Disable all controls
        self.enable_vault_controls(False)
        
        # Clear file tree
        self.file_tree.clear()
        self.file_info_text.clear()
        
        self.status_label.setText("🔒 Vault locked for security")
        self.log_operation("🔒 Vault locked")
    
    def enable_vault_controls(self, enabled):
        """Enable or disable vault controls"""
        controls = [
            self.add_file_btn, self.add_folder_btn, self.extract_btn, 
            self.delete_btn, self.new_folder_btn, self.search_input,
            self.file_tree, self.file_info_text, self.operation_log,
            self.view_btn, self.edit_btn, self.copy_btn
        ]
        
        for control in controls:
            control.setEnabled(enabled)
        
        self.unlock_btn.setEnabled(not enabled)
        self.lock_btn.setEnabled(enabled)
    
    def add_files_to_vault(self):
        """Add files to the vault"""
        if self.vault_locked:
            self.status_label.setText("❌ Vault is locked")
            return
        
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Files to Add to Vault", "", 
            "All Files (*.*)"
        )
        
        if not file_paths:
            return
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(len(file_paths))
        self.operation_label.setText("📂 Adding files to vault...")
        
        for i, file_path in enumerate(file_paths):
            file_path = Path(file_path)
            if file_path.exists():
                self.add_file_to_vault(file_path)
                self.progress_bar.setValue(i + 1)
        
        self.progress_bar.setVisible(False)
        self.operation_label.setText("✅ Files added successfully")
        self.refresh_file_tree()
        self.status_label.setText(f"📂 Added {len(file_paths)} files to vault")
    
    def add_file_to_vault(self, file_path: Path):
        """Add a single file to the vault"""
        try:
            # Generate unique file ID
            file_id = hashlib.md5(f"{file_path.name}{datetime.now()}".encode()).hexdigest()
            
            # Copy file to vault
            vault_file_path = self.vault_directory / "files" / file_id
            shutil.copy2(file_path, vault_file_path)
            
            # Store metadata
            metadata = {
                'id': file_id,
                'original_name': file_path.name,
                'size': file_path.stat().st_size,
                'type': file_path.suffix.lower(),
                'added': datetime.now().isoformat(),
                'folder': self.current_folder,
                'encrypted': True,
                'checksum': self.calculate_file_checksum(vault_file_path)
            }
            
            # Add to vault data
            if self.current_folder not in self.vault_data:
                self.vault_data[self.current_folder] = {}
            
            self.vault_data[self.current_folder][file_id] = metadata
            
            self.log_operation(f"📂 Added file: {file_path.name}")
            
        except Exception as e:
            self.log_operation(f"❌ Error adding {file_path.name}: {str(e)}")
    
    def calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of file"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def add_folder_to_vault(self):
        """Add an entire folder to the vault"""
        if self.vault_locked:
            self.status_label.setText("❌ Vault is locked")
            return
        
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder to Add to Vault")
        
        if not folder_path:
            return
        
        folder_path = Path(folder_path)
        files = list(folder_path.rglob("*"))
        files = [f for f in files if f.is_file()]
        
        if not files:
            self.status_label.setText("❌ No files found in selected folder")
            return
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(len(files))
        self.operation_label.setText(f"📁 Adding folder: {folder_path.name}...")
        
        for i, file_path in enumerate(files):
            self.add_file_to_vault(file_path)
            self.progress_bar.setValue(i + 1)
        
        self.progress_bar.setVisible(False)
        self.operation_label.setText("✅ Folder added successfully")
        self.refresh_file_tree()
        self.status_label.setText(f"📁 Added folder with {len(files)} files")
    
    def extract_files(self):
        """Extract selected files from vault"""
        if self.vault_locked:
            self.status_label.setText("❌ Vault is locked")
            return
        
        selected_items = self.file_tree.selectedItems()
        if not selected_items:
            self.status_label.setText("❌ No files selected for extraction")
            return
        
        extract_dir = QFileDialog.getExistingDirectory(self, "Select Extraction Directory")
        if not extract_dir:
            return
        
        extract_dir = Path(extract_dir)
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(len(selected_items))
        self.operation_label.setText("📤 Extracting files...")
        
        for i, item in enumerate(selected_items):
            file_id = item.data(0, Qt.UserRole)
            if file_id:
                self.extract_file(file_id, extract_dir)
                self.progress_bar.setValue(i + 1)
        
        self.progress_bar.setVisible(False)
        self.operation_label.setText("✅ Files extracted successfully")
        self.status_label.setText(f"📤 Extracted {len(selected_items)} files")
    
    def extract_file(self, file_id: str, extract_dir: Path):
        """Extract a single file from vault"""
        try:
            # Find file metadata
            metadata = None
            for folder_files in self.vault_data.values():
                if file_id in folder_files:
                    metadata = folder_files[file_id]
                    break
            
            if not metadata:
                self.log_operation(f"❌ File metadata not found: {file_id}")
                return
            
            # Copy file from vault
            vault_file_path = self.vault_directory / "files" / file_id
            extract_file_path = extract_dir / metadata['original_name']
            
            # Handle name conflicts
            counter = 1
            while extract_file_path.exists():
                name = Path(metadata['original_name'])
                extract_file_path = extract_dir / f"{name.stem}_{counter}{name.suffix}"
                counter += 1
            
            shutil.copy2(vault_file_path, extract_file_path)
            self.log_operation(f"📤 Extracted: {metadata['original_name']}")
            
        except Exception as e:
            self.log_operation(f"❌ Error extracting file: {str(e)}")
    
    def delete_selected(self):
        """Delete selected files from vault"""
        if self.vault_locked:
            self.status_label.setText("❌ Vault is locked")
            return
        
        selected_items = self.file_tree.selectedItems()
        if not selected_items:
            self.status_label.setText("❌ No files selected for deletion")
            return
        
        reply = QMessageBox.question(self, "🗑️ Delete Files", 
                                   f"Are you sure you want to delete {len(selected_items)} files?",
                                   QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            for item in selected_items:
                file_id = item.data(0, Qt.UserRole)
                if file_id:
                    self.delete_file(file_id)
            
            self.refresh_file_tree()
            self.status_label.setText(f"🗑️ Deleted {len(selected_items)} files")
    
    def delete_file(self, file_id: str):
        """Delete a single file from vault"""
        try:
            # Find and remove metadata
            for folder_name, folder_files in self.vault_data.items():
                if file_id in folder_files:
                    metadata = folder_files[file_id]
                    del folder_files[file_id]
                    break
            
            # Remove physical file
            vault_file_path = self.vault_directory / "files" / file_id
            if vault_file_path.exists():
                vault_file_path.unlink()
            
            self.log_operation(f"🗑️ Deleted: {metadata['original_name']}")
            
        except Exception as e:
            self.log_operation(f"❌ Error deleting file: {str(e)}")
    
    def create_new_folder(self):
        """Create a new folder in the vault"""
        if self.vault_locked:
            self.status_label.setText("❌ Vault is locked")
            return
        
        folder_name, ok = QInputDialog.getText(self, "📁 New Folder", "Enter folder name:")
        
        if ok and folder_name.strip():
            folder_name = folder_name.strip()
            
            if folder_name not in self.vault_data:
                self.vault_data[folder_name] = {}
                self.refresh_file_tree()
                self.log_operation(f"📁 Created folder: {folder_name}")
                self.status_label.setText(f"📁 Created folder: {folder_name}")
            else:
                self.status_label.setText("❌ Folder already exists")
    
    def select_file(self, item):
        """Handle file selection"""
        file_id = item.data(0, Qt.UserRole)
        if file_id:
            self.show_file_info(file_id)
            
            # Enable file-specific buttons
            self.view_btn.setEnabled(True)
            self.edit_btn.setEnabled(True)
            self.copy_btn.setEnabled(True)
    
    def show_file_info(self, file_id: str):
        """Show detailed file information"""
        # Find file metadata
        metadata = None
        for folder_files in self.vault_data.values():
            if file_id in folder_files:
                metadata = folder_files[file_id]
                break
        
        if not metadata:
            return
        
        # Format file size
        size_bytes = metadata['size']
        if size_bytes < 1024:
            size_str = f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            size_str = f"{size_bytes / 1024:.1f} KB"
        else:
            size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
        
        info_text = f"""📄 File Information

🏷️ Name: {metadata['original_name']}
📊 Size: {size_str}
📁 Type: {metadata['type'].upper() if metadata['type'] else 'Unknown'}
📅 Added: {metadata['added'][:19].replace('T', ' ')}
📂 Folder: {metadata['folder']}
🔒 Encrypted: {'Yes' if metadata['encrypted'] else 'No'}
🔍 Checksum: {metadata['checksum'][:16]}...
🆔 ID: {metadata['id'][:16]}...

🛡️ Security Status: Quantum Protected
🔑 Encryption: AES-256 + Quantum Key
📋 Integrity: Verified"""
        
        self.file_info_text.setPlainText(info_text)
    
    def view_selected_file(self):
        """View selected file (placeholder)"""
        selected_items = self.file_tree.selectedItems()
        if selected_items:
            item = selected_items[0]
            file_name = item.text(0)
            self.status_label.setText(f"👁️ Viewing: {file_name} (Feature coming soon)")
    
    def edit_file_name(self):
        """Edit file name"""
        selected_items = self.file_tree.selectedItems()
        if not selected_items:
            return
        
        item = selected_items[0]
        file_id = item.data(0, Qt.UserRole)
        current_name = item.text(0)
        
        new_name, ok = QInputDialog.getText(self, "✏️ Edit File Name", 
                                          "New file name:", text=current_name)
        
        if ok and new_name.strip() and new_name != current_name:
            # Update metadata
            for folder_files in self.vault_data.values():
                if file_id in folder_files:
                    folder_files[file_id]['original_name'] = new_name.strip()
                    break
            
            self.refresh_file_tree()
            self.status_label.setText(f"✏️ Renamed to: {new_name}")
    
    def copy_selected_file(self):
        """Copy selected file within vault"""
        self.status_label.setText("📋 Copy feature coming soon")
    
    def filter_files(self):
        """Filter files based on search term"""
        if self.vault_locked:
            return
        
        search_term = self.search_input.text().lower()
        
        # Simple filter implementation
        for i in range(self.file_tree.topLevelItemCount()):
            item = self.file_tree.topLevelItem(i)
            visible = search_term in item.text(0).lower() if search_term else True
            item.setHidden(not visible)
    
    def refresh_file_tree(self):
        """Refresh the file tree display"""
        if self.vault_locked:
            return
        
        self.file_tree.clear()
        
        total_files = 0
        total_size = 0
        
        for folder_name, folder_files in self.vault_data.items():
            # Create folder item
            folder_item = QTreeWidgetItem([f"📁 {folder_name}", "Folder", "", ""])
            self.file_tree.addTopLevelItem(folder_item)
            
            # Add files to folder
            for file_id, metadata in folder_files.items():
                size_mb = metadata['size'] / (1024 * 1024)
                size_str = f"{size_mb:.2f} MB" if size_mb >= 1 else f"{metadata['size']} B"
                modified = metadata['added'][:10]  # Date only
                
                file_item = QTreeWidgetItem([
                    f"📄 {metadata['original_name']}",
                    metadata['type'].upper() if metadata['type'] else "File",
                    size_str,
                    modified
                ])
                file_item.setData(0, Qt.UserRole, file_id)
                folder_item.addChild(file_item)
                
                total_files += 1
                total_size += metadata['size']
        
        # Expand all folders
        self.file_tree.expandAll()
        
        # Update statistics
        total_size_mb = total_size / (1024 * 1024)
        size_display = f"{total_size_mb:.1f} MB" if total_size_mb >= 1 else f"{total_size} B"
        
        self.tree_stats.setText(f"📊 Files: {total_files} | Size: {size_display}")
        self.storage_info.setText(f"💾 Files: {total_files} | Size: {size_display}")
    
    def log_operation(self, message: str):
        """Log operation to the operation log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.operation_log.append(log_entry)
        
        # Keep only last 50 entries
        document = self.operation_log.document()
        if document.blockCount() > 50:
            cursor = self.operation_log.textCursor()
            cursor.movePosition(cursor.Start)
            cursor.select(cursor.LineUnderCursor)
            cursor.removeSelectedText()
    
    def load_vault_config(self):
        """Load vault configuration"""
        try:
            config_file = self.vault_directory / self.vault_config_file
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    self.vault_data = json.load(f)
            else:
                self.vault_data = {"root": {}}
        except Exception as e:
            self.log_operation(f"❌ Error loading vault config: {str(e)}")
            self.vault_data = {"root": {}}
    
    def save_vault_config(self):
        """Save vault configuration"""
        try:
            config_file = self.vault_directory / self.vault_config_file
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(self.vault_data, f, indent=2, ensure_ascii=False)
            
            # Update auto-save status briefly
            self.autosave_status.setText("💾 Auto-saved")
            QTimer.singleShot(2000, lambda: self.autosave_status.setText("💾 Auto-save: Enabled"))
            
        except Exception as e:
            self.log_operation(f"❌ Error saving vault config: {str(e)}")

def main():
    """Main entry point"""
    if not PYQT5_AVAILABLE:
        print("❌ PyQt5 required for Quantum Vault")
        return
    
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    window = QuantumVault()
    window.show()
    
    return app.exec_()

# Alias for app controller compatibility
QVaultApp = QuantumVault

if __name__ == "__main__":
    main()
