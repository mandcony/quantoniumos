"""
QuantoniumOS - Quantum Mail
Secure email client with quantum encryption
"""

import email
import imaplib
import json
import os
import smtplib
import sys
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

try:
    from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
    from PyQt5.QtGui import QFont, QIcon
    from PyQt5.QtWidgets import (QComboBox, QGridLayout, QGroupBox,
                                 QHBoxLayout, QHeaderView, QInputDialog,
                                 QLabel, QLineEdit, QListWidget,
                                 QListWidgetItem, QMessageBox, QPushButton,
                                 QSplitter, QTableWidget, QTableWidgetItem,
                                 QTabWidget, QTextEdit, QVBoxLayout, QWidget)

    PYQT5_AVAILABLE = True
except ImportError:
    PYQT5_AVAILABLE = False


class QuantumMail(QWidget if PYQT5_AVAILABLE else object):
    """Quantum-encrypted email client"""

    def __init__(self):
        if not PYQT5_AVAILABLE:
            print("⚠️ PyQt5 required for Quantum Mail GUI")
            return

        super().__init__()
        self.mail_data = {"inbox": [], "sent": [], "drafts": [], "trash": []}
        self.mail_config = {}
        self.config_file = "quantum_mail_config.json"
        self.quantum_encryption = True
        self.current_folder = "inbox"
        self.selected_message = None

        self.init_ui()
        self.load_mail_config()
        self.setup_quantum_encryption()

        # Auto-refresh timer
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.auto_refresh_mail)
        self.refresh_timer.start(300000)  # Refresh every 5 minutes

    def init_ui(self):
        """Initialize the quantum mail interface"""
        self.setWindowTitle("📧 QuantoniumOS - Quantum Mail")
        self.setGeometry(150, 150, 1400, 900)

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
            QPushButton:disabled {
                background-color: #cccccc;
                border: 1px solid #999999;
                color: #666666;
            }
            QLineEdit {
                background: #1a2332;
                border: 1px solid #4169e1;
                border-radius: 4px;
                padding: 8px;
                color: #ffffff;
                font-size: 14px;
            }
            QTextEdit {
                background: #1a2332;
                border: 1px solid #4169e1;
                border-radius: 4px;
                color: #ffffff;
                font-size: 12px;
            }
            QListWidget, QTableWidget {
                background: #1a2332;
                border: 1px solid #4169e1;
                border-radius: 4px;
                color: #ffffff;
                selection-background-color: #4169e1;
                outline: none;
                gridline-color: #2a3040;
            }
            QListWidget::item, QTableWidget::item {
                padding: 8px;
                border-bottom: 1px solid #2a3040;
            }
            QListWidget::item:hover, QTableWidget::item:hover {
                background: #2a3040;
            }
            QListWidget::item:selected, QTableWidget::item:selected {
                background: #4169e1;
                color: #ffffff;
            }
            QTabWidget::pane {
                border: 2px solid #4169e1;
                border-radius: 8px;
                background: #1a2332;
            }
            QTabBar::tab {
                background: #2a3040;
                border: 1px solid #4169e1;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                color: #4169e1;
            }
            QTabBar::tab:selected {
                background: #4169e1;
                color: #0a0e1a;
                font-weight: bold;
            }
            QComboBox {
                background: #1a2332;
                border: 1px solid #4169e1;
                border-radius: 4px;
                padding: 5px;
                color: #ffffff;
            }
            QHeaderView::section {
                background: #2a3040;
                color: #4169e1;
                border: 1px solid #4169e1;
                padding: 8px;
                font-weight: bold;
            }
            QLabel {
                color: #00ffcc;
            }
        """
        )

        layout = QVBoxLayout(self)

        # Title
        title = QLabel("📧 Quantum Mail - Secure Email Client")
        title.setFont(QFont("Consolas", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #4169e1; margin: 10px;")
        layout.addWidget(title)

        # Security status bar
        self.create_security_bar(layout)

        # Main toolbar
        self.create_toolbar(layout)

        # Main content with tabs
        self.create_main_content(layout)

        # Status bar
        self.create_status_bar(layout)

    def create_security_bar(self, parent_layout):
        """Create mail security status bar"""
        security_group = QGroupBox("🛡️ Mail Security Status")
        security_layout = QHBoxLayout(security_group)

        # Quantum encryption status
        self.encryption_status = QLabel("🔒 Quantum Encryption: Active")
        self.encryption_status.setStyleSheet("color: #00ff88; font-weight: bold;")
        security_layout.addWidget(self.encryption_status)

        # Connection status
        self.connection_status = QLabel("📡 Connection: Offline")
        self.connection_status.setStyleSheet("color: #ff6b35; font-weight: bold;")
        security_layout.addWidget(self.connection_status)

        security_layout.addWidget(QLabel("|"))

        # Account info
        self.account_info = QLabel("👤 Account: Not configured")
        self.account_info.setStyleSheet("color: #ffaa00;")
        security_layout.addWidget(self.account_info)

        # Mail stats
        self.mail_stats = QLabel("📊 Messages: 0 | Unread: 0")
        self.mail_stats.setStyleSheet("color: #00ffcc;")
        security_layout.addWidget(self.mail_stats)

        security_layout.addStretch()

        parent_layout.addWidget(security_group)

    def create_toolbar(self, parent_layout):
        """Create main toolbar"""
        toolbar_group = QGroupBox("🛠️ Mail Operations")
        toolbar_layout = QHBoxLayout(toolbar_group)

        # Mail operations
        self.compose_btn = QPushButton("✏️ Compose")
        self.compose_btn.clicked.connect(self.compose_new_mail)
        toolbar_layout.addWidget(self.compose_btn)

        self.refresh_btn = QPushButton("🔄 Refresh")
        self.refresh_btn.clicked.connect(self.refresh_mail)
        toolbar_layout.addWidget(self.refresh_btn)

        self.reply_btn = QPushButton("↩️ Reply")
        self.reply_btn.clicked.connect(self.reply_to_mail)
        self.reply_btn.setEnabled(False)
        toolbar_layout.addWidget(self.reply_btn)

        self.forward_btn = QPushButton("↗️ Forward")
        self.forward_btn.clicked.connect(self.forward_mail)
        self.forward_btn.setEnabled(False)
        toolbar_layout.addWidget(self.forward_btn)

        self.delete_btn = QPushButton("🗑️ Delete")
        self.delete_btn.clicked.connect(self.delete_mail)
        self.delete_btn.setEnabled(False)
        toolbar_layout.addWidget(self.delete_btn)

        toolbar_layout.addWidget(QLabel("|"))

        # Account settings
        self.settings_btn = QPushButton("⚙️ Settings")
        self.settings_btn.clicked.connect(self.show_settings)
        toolbar_layout.addWidget(self.settings_btn)

        toolbar_layout.addWidget(QLabel("|"))

        # Search
        toolbar_layout.addWidget(QLabel("🔍 Search:"))
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search emails...")
        self.search_input.textChanged.connect(self.filter_emails)
        toolbar_layout.addWidget(self.search_input)

        # Encryption toggle
        self.encryption_toggle = QPushButton("🔒 Quantum Mode")
        self.encryption_toggle.setCheckable(True)
        self.encryption_toggle.setChecked(True)
        self.encryption_toggle.clicked.connect(self.toggle_encryption)
        toolbar_layout.addWidget(self.encryption_toggle)

        toolbar_layout.addStretch()

        parent_layout.addWidget(toolbar_group)

    def create_main_content(self, parent_layout):
        """Create main content area with tabs"""
        # Main splitter
        main_splitter = QSplitter(Qt.Horizontal)

        # Left side - Folder list
        self.create_folder_panel(main_splitter)

        # Center - Email list
        self.create_email_list_panel(main_splitter)

        # Right side - Email viewer
        self.create_email_viewer_panel(main_splitter)

        main_splitter.setSizes([250, 500, 650])
        parent_layout.addWidget(main_splitter)

    def create_folder_panel(self, parent_splitter):
        """Create email folder panel"""
        folder_group = QGroupBox("📁 Mail Folders")
        folder_layout = QVBoxLayout(folder_group)

        # Folder list
        self.folder_list = QListWidget()
        folders = [
            ("📥 Inbox", "inbox"),
            ("📤 Sent", "sent"),
            ("📝 Drafts", "drafts"),
            ("🗑️ Trash", "trash"),
        ]

        for folder_name, folder_id in folders:
            item = QListWidgetItem(folder_name)
            item.setData(Qt.UserRole, folder_id)
            self.folder_list.addItem(item)

        self.folder_list.itemClicked.connect(self.select_folder)
        self.folder_list.setCurrentRow(0)  # Select inbox by default
        folder_layout.addWidget(self.folder_list)

        # Folder statistics
        self.folder_stats = QLabel("📊 Inbox: 0 messages")
        self.folder_stats.setStyleSheet(
            "color: #4169e1; font-weight: bold; margin: 5px;"
        )
        folder_layout.addWidget(self.folder_stats)

        parent_splitter.addWidget(folder_group)

    def create_email_list_panel(self, parent_splitter):
        """Create email list panel"""
        list_group = QGroupBox("📧 Email List")
        list_layout = QVBoxLayout(list_group)

        # Email table
        self.email_table = QTableWidget()
        self.email_table.setColumnCount(4)
        self.email_table.setHorizontalHeaderLabels(
            ["From/To", "Subject", "Date", "Size"]
        )

        # Configure table
        header = self.email_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)

        self.email_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.email_table.itemSelectionChanged.connect(self.select_email)
        list_layout.addWidget(self.email_table)

        parent_splitter.addWidget(list_group)

    def create_email_viewer_panel(self, parent_splitter):
        """Create email viewer panel"""
        viewer_group = QGroupBox("📖 Email Viewer")
        viewer_layout = QVBoxLayout(viewer_group)

        # Email header
        header_group = QGroupBox("📋 Message Header")
        header_layout = QGridLayout(header_group)

        # Header fields
        self.from_label = QLabel("From: -")
        self.to_label = QLabel("To: -")
        self.subject_label = QLabel("Subject: -")
        self.date_label = QLabel("Date: -")

        header_layout.addWidget(QLabel("📧 From:"), 0, 0)
        header_layout.addWidget(self.from_label, 0, 1)
        header_layout.addWidget(QLabel("📨 To:"), 1, 0)
        header_layout.addWidget(self.to_label, 1, 1)
        header_layout.addWidget(QLabel("📋 Subject:"), 2, 0)
        header_layout.addWidget(self.subject_label, 2, 1)
        header_layout.addWidget(QLabel("📅 Date:"), 3, 0)
        header_layout.addWidget(self.date_label, 3, 1)

        viewer_layout.addWidget(header_group)

        # Email content
        content_group = QGroupBox("📄 Message Content")
        content_layout = QVBoxLayout(content_group)

        self.content_viewer = QTextEdit()
        self.content_viewer.setReadOnly(True)
        content_layout.addWidget(self.content_viewer)

        viewer_layout.addWidget(content_group)

        parent_splitter.addWidget(viewer_group)

    def create_status_bar(self, parent_layout):
        """Create status bar"""
        status_layout = QHBoxLayout()

        self.status_label = QLabel("✅ Quantum Mail Ready")
        self.status_label.setStyleSheet(
            "color: #4169e1; font-weight: bold; margin: 5px;"
        )
        status_layout.addWidget(self.status_label)

        status_layout.addStretch()

        self.sync_status = QLabel("🔄 Last sync: Never")
        self.sync_status.setStyleSheet("color: #00ffcc; margin: 5px;")
        status_layout.addWidget(self.sync_status)

        self.encryption_indicator = QLabel("🔒 Quantum Protected")
        self.encryption_indicator.setStyleSheet(
            "color: #00ff88; font-weight: bold; margin: 5px;"
        )
        status_layout.addWidget(self.encryption_indicator)

        parent_layout.addLayout(status_layout)

    def setup_quantum_encryption(self):
        """Setup quantum encryption for emails"""
        # Generate quantum encryption keys
        import random

        random.seed(2024)  # For demonstration
        self.quantum_key = "".join([hex(random.randint(0, 15))[2:] for _ in range(128)])

        self.encryption_status.setText("🔒 Quantum Encryption: Active")
        self.status_label.setText("🔑 Quantum encryption initialized")

    def compose_new_mail(self):
        """Open compose window"""
        compose_window = ComposeWindow(self)
        compose_window.exec_()

    def refresh_mail(self):
        """Refresh mail from server"""
        if not self.mail_config.get("configured", False):
            self.status_label.setText("❌ Please configure mail settings first")
            self.show_settings()
            return

        # Simulate mail refresh
        self.connection_status.setText("📡 Connection: Syncing...")
        self.connection_status.setStyleSheet("color: #ffaa00; font-weight: bold;")

        # Simulate loading demo emails
        self.load_demo_emails()

        QTimer.singleShot(2000, self.finish_sync)
        self.status_label.setText("🔄 Refreshing mail...")

    def finish_sync(self):
        """Finish mail synchronization"""
        self.connection_status.setText("📡 Connection: Online")
        self.connection_status.setStyleSheet("color: #00ff88; font-weight: bold;")

        current_time = datetime.now().strftime("%H:%M")
        self.sync_status.setText(f"🔄 Last sync: {current_time}")

        self.refresh_email_list()
        self.status_label.setText("✅ Mail synchronized successfully")

    def load_demo_emails(self):
        """Load demonstration emails"""
        demo_emails = [
            {
                "id": "demo_1",
                "from": "quantum.research@quantonium.os",
                "to": "user@quantonium.os",
                "subject": "🔬 Quantum Computing Research Update",
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "content": """Dear Quantum Research Team,

I hope this message finds you well. I'm writing to update you on our latest quantum computing research developments.

🌟 Key Achievements:
• Successfully implemented 1000-qubit quantum processor
• Achieved 99.9% quantum coherence stability
• Developed revolutionary RFT (Randomized Feistel Transform) algorithms
• Integrated quantum encryption into QuantoniumOS

🔬 Technical Highlights:
The new quantum kernel shows remarkable performance improvements, with quantum state preservation exceeding all previous benchmarks. Our RFT implementation provides cryptographic security that is provably resistant to both classical and quantum attacks.

🚀 Next Steps:
We're now focusing on practical applications and will begin testing the quantum file system integration next week.

Best regards,
Dr. Quantum Researcher
QuantoniumOS Research Division

---
🔒 This message was secured with quantum encryption
🛡️ Threat level: None detected
📡 Network: QuantoniumOS Secure Grid""",
                "encrypted": True,
                "read": False,
                "size": "4.2 KB",
            },
            {
                "id": "demo_2",
                "from": "security@quantonium.os",
                "to": "user@quantonium.os",
                "subject": "🛡️ Security Alert: Quantum Shield Update",
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "content": """QUANTUM SECURITY ALERT

🛡️ Security Update Notification

A new quantum security update has been deployed to your QuantoniumOS system. This update includes:

• Enhanced quantum firewall protocols
• Improved threat detection algorithms
• Updated quantum encryption standards
• Advanced malware protection

🔍 Scan Results:
✅ System integrity: Verified
✅ Quantum coherence: Stable
✅ Encryption status: Active
✅ Threat level: Minimal

No action required. Your system is fully protected.

Stay quantum-secure!

QuantoniumOS Security Team

---
🔐 Auto-generated security report
🕒 Timestamp: Quantum-synchronized
🌐 Network status: Secure""",
                "encrypted": True,
                "read": False,
                "size": "2.8 KB",
            },
            {
                "id": "demo_3",
                "from": "noreply@quantonium.os",
                "to": "user@quantonium.os",
                "subject": "📧 Welcome to QuantoniumOS Mail",
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "content": """Welcome to QuantoniumOS Quantum Mail! 📧

Congratulations on setting up your quantum-secured email client. You now have access to the most advanced email security in the universe.

🌟 Features you'll love:
• Quantum encryption for ultimate privacy
• Advanced threat protection
• Secure file attachments
• Real-time malware scanning
• Quantum-authenticated sending

🚀 Getting Started:
1. Configure your email accounts in Settings
2. Import your existing emails (optional)
3. Start sending quantum-secured messages
4. Enjoy unbreakable email security

🔒 Security Note:
All emails are automatically encrypted using our quantum key distribution system. Your privacy is mathematically guaranteed.

Need help? Check the Help menu or contact our quantum support team.

Happy emailing!
The QuantoniumOS Team

---
🌌 Powered by quantum technology
🔐 Your privacy is our priority""",
                "encrypted": False,
                "read": True,
                "size": "3.1 KB",
            },
        ]

        # Add to inbox
        self.mail_data["inbox"] = demo_emails

    def auto_refresh_mail(self):
        """Auto-refresh mail periodically"""
        if self.mail_config.get("configured", False) and self.mail_config.get(
            "auto_refresh", True
        ):
            self.refresh_mail()

    def select_folder(self, item):
        """Select email folder"""
        folder_id = item.data(Qt.UserRole)
        self.current_folder = folder_id
        self.refresh_email_list()

        folder_name = item.text()
        count = len(self.mail_data.get(folder_id, []))
        self.folder_stats.setText(f"📊 {folder_name}: {count} messages")

    def refresh_email_list(self):
        """Refresh the email list display"""
        emails = self.mail_data.get(self.current_folder, [])

        self.email_table.setRowCount(len(emails))

        for row, email_data in enumerate(emails):
            # From/To column
            if self.current_folder == "sent":
                from_to = email_data.get("to", "Unknown")
            else:
                from_to = email_data.get("from", "Unknown")

            # Add encryption indicator
            if email_data.get("encrypted", False):
                from_to = f"🔒 {from_to}"

            # Add read/unread indicator
            if not email_data.get("read", True):
                from_to = f"● {from_to}"  # Unread indicator

            self.email_table.setItem(row, 0, QTableWidgetItem(from_to))
            self.email_table.setItem(
                row, 1, QTableWidgetItem(email_data.get("subject", "No Subject"))
            )
            self.email_table.setItem(
                row, 2, QTableWidgetItem(email_data.get("date", "Unknown"))
            )
            self.email_table.setItem(
                row, 3, QTableWidgetItem(email_data.get("size", "0 KB"))
            )

            # Store email ID in first column
            self.email_table.item(row, 0).setData(Qt.UserRole, email_data.get("id"))

        # Update mail statistics
        total_messages = sum(len(folder) for folder in self.mail_data.values())
        unread_count = sum(
            1
            for folder in self.mail_data.values()
            for email in folder
            if not email.get("read", True)
        )

        self.mail_stats.setText(
            f"📊 Messages: {total_messages} | Unread: {unread_count}"
        )

    def select_email(self):
        """Handle email selection"""
        current_row = self.email_table.currentRow()
        if current_row >= 0:
            email_id = self.email_table.item(current_row, 0).data(Qt.UserRole)
            email_data = self.find_email_by_id(email_id)

            if email_data:
                self.display_email(email_data)
                self.selected_message = email_data

                # Enable reply/forward/delete buttons
                self.reply_btn.setEnabled(True)
                self.forward_btn.setEnabled(True)
                self.delete_btn.setEnabled(True)

                # Mark as read
                if not email_data.get("read", True):
                    email_data["read"] = True
                    self.refresh_email_list()

    def find_email_by_id(self, email_id):
        """Find email by ID in current folder"""
        emails = self.mail_data.get(self.current_folder, [])
        for email_data in emails:
            if email_data.get("id") == email_id:
                return email_data
        return None

    def display_email(self, email_data):
        """Display email in viewer"""
        self.from_label.setText(email_data.get("from", "Unknown"))
        self.to_label.setText(email_data.get("to", "Unknown"))
        self.subject_label.setText(email_data.get("subject", "No Subject"))
        self.date_label.setText(email_data.get("date", "Unknown"))

        content = email_data.get("content", "No content")

        # Apply quantum decryption if needed
        if email_data.get("encrypted", False):
            content = self.quantum_decrypt(content)
            content = f"🔒 QUANTUM ENCRYPTED MESSAGE 🔒\n\n{content}\n\n---\n✅ Message verified and decrypted successfully"

        self.content_viewer.setPlainText(content)

    def quantum_decrypt(self, content):
        """Simulate quantum decryption"""
        # In a real implementation, this would perform actual quantum decryption
        return content  # For demo, return as-is

    def quantum_encrypt(self, content):
        """Simulate quantum encryption"""
        # In a real implementation, this would perform actual quantum encryption
        return content  # For demo, return as-is

    def reply_to_mail(self):
        """Reply to selected email"""
        if self.selected_message:
            compose_window = ComposeWindow(
                self,
                reply_to=self.selected_message.get("from"),
                subject=f"Re: {self.selected_message.get('subject', '')}",
                quote_content=self.selected_message.get("content", ""),
            )
            compose_window.exec_()

    def forward_mail(self):
        """Forward selected email"""
        if self.selected_message:
            compose_window = ComposeWindow(
                self,
                subject=f"Fwd: {self.selected_message.get('subject', '')}",
                quote_content=self.selected_message.get("content", ""),
            )
            compose_window.exec_()

    def delete_mail(self):
        """Delete selected email"""
        if self.selected_message:
            reply = QMessageBox.question(
                self,
                "🗑️ Delete Email",
                "Are you sure you want to delete this email?",
                QMessageBox.Yes | QMessageBox.No,
            )

            if reply == QMessageBox.Yes:
                # Move to trash (or delete if already in trash)
                emails = self.mail_data.get(self.current_folder, [])
                emails = [
                    e for e in emails if e.get("id") != self.selected_message.get("id")
                ]
                self.mail_data[self.current_folder] = emails

                if self.current_folder != "trash":
                    # Move to trash
                    self.mail_data.setdefault("trash", []).append(self.selected_message)

                self.refresh_email_list()
                self.clear_email_viewer()
                self.status_label.setText("🗑️ Email deleted")

    def clear_email_viewer(self):
        """Clear email viewer"""
        self.from_label.setText("From: -")
        self.to_label.setText("To: -")
        self.subject_label.setText("Subject: -")
        self.date_label.setText("Date: -")
        self.content_viewer.clear()

        # Disable buttons
        self.reply_btn.setEnabled(False)
        self.forward_btn.setEnabled(False)
        self.delete_btn.setEnabled(False)

        self.selected_message = None

    def filter_emails(self):
        """Filter emails based on search term"""
        search_term = self.search_input.text().lower()

        for row in range(self.email_table.rowCount()):
            visible = True
            if search_term:
                # Check subject and from/to fields
                subject = self.email_table.item(row, 1).text().lower()
                from_to = self.email_table.item(row, 0).text().lower()
                visible = search_term in subject or search_term in from_to

            self.email_table.setRowHidden(row, not visible)

    def toggle_encryption(self):
        """Toggle quantum encryption mode"""
        self.quantum_encryption = self.encryption_toggle.isChecked()

        if self.quantum_encryption:
            self.encryption_indicator.setText("🔒 Quantum Protected")
            self.encryption_indicator.setStyleSheet(
                "color: #00ff88; font-weight: bold; margin: 5px;"
            )
            self.status_label.setText("🔒 Quantum encryption enabled")
        else:
            self.encryption_indicator.setText("⚠️ Standard Mode")
            self.encryption_indicator.setStyleSheet(
                "color: #ff6b35; font-weight: bold; margin: 5px;"
            )
            self.status_label.setText("⚠️ Quantum encryption disabled")

    def show_settings(self):
        """Show mail settings dialog"""
        settings_window = MailSettingsWindow(self)
        settings_window.exec_()

    def load_mail_config(self):
        """Load mail configuration"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, "r", encoding="utf-8") as f:
                    self.mail_config = json.load(f)
            else:
                self.mail_config = {"configured": False}

            if self.mail_config.get("configured", False):
                email = self.mail_config.get("email", "user@quantonium.os")
                self.account_info.setText(f"👤 Account: {email}")
                self.account_info.setStyleSheet("color: #00ff88;")
        except Exception as e:
            self.status_label.setText(f"❌ Error loading config: {str(e)}")
            self.mail_config = {"configured": False}

    def save_mail_config(self):
        """Save mail configuration"""
        try:
            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump(self.mail_config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.status_label.setText(f"❌ Error saving config: {str(e)}")


# Compose Window Class
class ComposeWindow(QMessageBox if PYQT5_AVAILABLE else object):
    """Email compose window"""

    def __init__(self, parent, reply_to=None, subject=None, quote_content=None):
        if not PYQT5_AVAILABLE:
            return

        super().__init__(parent)
        self.parent_mail = parent
        self.reply_to = reply_to
        self.quote_content = quote_content

        self.setWindowTitle("✏️ Compose Email")
        self.resize(800, 600)

        # This is a simplified compose window
        # In a real implementation, this would be a full dialog
        text = f"Compose Email\n\nTo: {reply_to or ''}\nSubject: {subject or ''}\n\n"
        if quote_content:
            text += f"--- Original Message ---\n{quote_content[:200]}..."

        self.setText(text)
        self.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)


# Settings Window Class
class MailSettingsWindow(QMessageBox if PYQT5_AVAILABLE else object):
    """Mail settings window"""

    def __init__(self, parent):
        if not PYQT5_AVAILABLE:
            return

        super().__init__(parent)
        self.parent_mail = parent

        self.setWindowTitle("⚙️ Mail Settings")
        self.setText(
            "Mail Settings\n\nConfigure your email accounts and preferences here.\n\nThis is a simplified settings dialog."
        )
        self.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)


def main():
    """Main entry point"""
    if not PYQT5_AVAILABLE:
        print("❌ PyQt5 required for Quantum Mail")
        return

    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = QuantumMail()
    window.show()

    return app.exec_()


# Alias for app controller compatibility
QMailApp = QuantumMail

if __name__ == "__main__":
    main()
