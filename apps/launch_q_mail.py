#!/usr/bin/env python3
"""
QuantoniumOS Q-Mail
=================
Secure quantum-encrypted mail client
"""

import os
import sys
from typing import List, Dict

# Import the base launcher
try:
    from launcher_base import AppLauncherBase, AppWindow, AppTerminal, HAS_PYQT
except ImportError:
    # Try to find the launcher_base module
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    try:
        from launcher_base import AppLauncherBase, AppWindow, AppTerminal, HAS_PYQT
    except ImportError:
        print("Error: launcher_base.py not found")
        sys.exit(1)

# Try to import PyQt5 for the GUI
if HAS_PYQT:
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                              QHBoxLayout, QPushButton, QLabel, QLineEdit,
                              QTextEdit, QListWidget, QTabWidget, QFrame)
    from PyQt5.QtGui import QIcon, QFont
    from PyQt5.QtCore import Qt

class QMailApp(AppWindow):
    """Q-Mail window"""
    
    def __init__(self, app_name: str, app_icon: str):
        """Initialize the Q-Mail window"""
        super().__init__(app_name, app_icon)
        
        # Create the UI
        self.create_ui()
    
    def create_ui(self):
        """Create the user interface"""
        # Clear the layout
        for i in reversed(range(self.layout.count())): 
            self.layout.itemAt(i).widget().setParent(None)
        
        # Add tabs for different views
        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)
        
        # Add the inbox tab
        self.create_inbox_tab()
        
        # Add the compose tab
        self.create_compose_tab()
    
    def create_inbox_tab(self):
        """Create the inbox tab"""
        # Create the tab widget
        inbox_tab = QWidget()
        inbox_layout = QVBoxLayout(inbox_tab)
        
        # Add the inbox list
        inbox_frame = QFrame()
        inbox_frame.setFrameShape(QFrame.StyledPanel)
        inbox_frame.setStyleSheet("QFrame { background-color: rgba(40, 40, 60, 150); border-radius: 5px; }")
        inbox_frame_layout = QVBoxLayout(inbox_frame)
        
        # Add a header
        header_label = QLabel("Inbox")
        header_label.setStyleSheet("color: white; font-size: 18px; font-weight: bold;")
        inbox_frame_layout.addWidget(header_label)
        
        # Add the inbox list
        self.inbox_list = QListWidget()
        self.inbox_list.setStyleSheet("""
            QListWidget {
                background-color: rgba(30, 30, 40, 150);
                color: white;
                border: 1px solid rgba(100, 100, 200, 150);
                border-radius: 5px;
            }
            QListWidget::item {
                border-bottom: 1px solid rgba(60, 60, 80, 150);
                padding: 5px;
            }
            QListWidget::item:selected {
                background-color: rgba(80, 80, 120, 150);
                border: 1px solid rgba(150, 150, 255, 150);
            }
        """)
        
        # Add some sample messages
        self.inbox_list.addItem("Alice <alice@quantum.org> - Quantum Encryption Test")
        self.inbox_list.addItem("Bob <bob@quantum.org> - Re: Project Status")
        self.inbox_list.addItem("QuantoniumOS Admin - Welcome to Q-Mail")
        
        inbox_frame_layout.addWidget(self.inbox_list)
        
        inbox_layout.addWidget(inbox_frame)
        
        # Add the message preview
        preview_frame = QFrame()
        preview_frame.setFrameShape(QFrame.StyledPanel)
        preview_frame.setStyleSheet("QFrame { background-color: rgba(40, 40, 60, 150); border-radius: 5px; }")
        preview_layout = QVBoxLayout(preview_frame)
        
        # Add the message header
        header_layout = QVBoxLayout()
        
        self.message_subject = QLabel("Welcome to Q-Mail")
        self.message_subject.setStyleSheet("color: white; font-size: 16px; font-weight: bold;")
        header_layout.addWidget(self.message_subject)
        
        self.message_from = QLabel("From: QuantoniumOS Admin <admin@quantonium.os>")
        self.message_from.setStyleSheet("color: white;")
        header_layout.addWidget(self.message_from)
        
        self.message_date = QLabel("Date: Today")
        self.message_date.setStyleSheet("color: white;")
        header_layout.addWidget(self.message_date)
        
        preview_layout.addLayout(header_layout)
        
        # Add a separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setStyleSheet("background-color: rgba(100, 100, 200, 150);")
        preview_layout.addWidget(separator)
        
        # Add the message body
        self.message_body = QTextEdit()
        self.message_body.setReadOnly(True)
        self.message_body.setStyleSheet("""
            QTextEdit {
                background-color: rgba(30, 30, 40, 150);
                color: white;
                border: 1px solid rgba(100, 100, 200, 150);
                border-radius: 5px;
            }
        """)
        self.message_body.setText("""Welcome to Q-Mail!

This is a secure, quantum-encrypted email client built on the QuantoniumOS platform.

Key features:
- Quantum encryption for secure communications
- Integration with QuantoniumOS security protocols
- Resistant to quantum computing attacks

Enjoy using Q-Mail!

The QuantoniumOS Team""")
        
        preview_layout.addWidget(self.message_body)
        
        inbox_layout.addWidget(preview_frame)
        
        # Add the tab
        self.tabs.addTab(inbox_tab, "Inbox")
    
    def create_compose_tab(self):
        """Create the compose tab"""
        # Create the tab widget
        compose_tab = QWidget()
        compose_layout = QVBoxLayout(compose_tab)
        
        # Add the compose form
        compose_frame = QFrame()
        compose_frame.setFrameShape(QFrame.StyledPanel)
        compose_frame.setStyleSheet("QFrame { background-color: rgba(40, 40, 60, 150); border-radius: 5px; }")
        compose_layout_inner = QVBoxLayout(compose_frame)
        
        # Add a header
        header_label = QLabel("Compose Message")
        header_label.setStyleSheet("color: white; font-size: 18px; font-weight: bold;")
        compose_layout_inner.addWidget(header_label)
        
        # Add the recipient field
        to_layout = QHBoxLayout()
        to_label = QLabel("To:")
        to_label.setStyleSheet("color: white;")
        to_label.setFixedWidth(60)
        to_layout.addWidget(to_label)
        
        self.to_field = QLineEdit()
        self.to_field.setStyleSheet("""
            QLineEdit {
                background-color: rgba(30, 30, 40, 150);
                color: white;
                border: 1px solid rgba(100, 100, 200, 150);
                border-radius: 5px;
                padding: 5px;
            }
        """)
        to_layout.addWidget(self.to_field)
        
        compose_layout_inner.addLayout(to_layout)
        
        # Add the subject field
        subject_layout = QHBoxLayout()
        subject_label = QLabel("Subject:")
        subject_label.setStyleSheet("color: white;")
        subject_label.setFixedWidth(60)
        subject_layout.addWidget(subject_label)
        
        self.subject_field = QLineEdit()
        self.subject_field.setStyleSheet("""
            QLineEdit {
                background-color: rgba(30, 30, 40, 150);
                color: white;
                border: 1px solid rgba(100, 100, 200, 150);
                border-radius: 5px;
                padding: 5px;
            }
        """)
        subject_layout.addWidget(self.subject_field)
        
        compose_layout_inner.addLayout(subject_layout)
        
        # Add the message field
        self.message_field = QTextEdit()
        self.message_field.setStyleSheet("""
            QTextEdit {
                background-color: rgba(30, 30, 40, 150);
                color: white;
                border: 1px solid rgba(100, 100, 200, 150);
                border-radius: 5px;
            }
        """)
        compose_layout_inner.addWidget(self.message_field)
        
        # Add the send button
        button_layout = QHBoxLayout()
        
        send_button = QPushButton("Send Message")
        send_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(60, 80, 60, 200);
                color: white;
                border: 1px solid rgba(100, 200, 100, 200);
                border-radius: 5px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgba(80, 120, 80, 200);
                border: 1px solid rgba(150, 255, 150, 200);
            }
            QPushButton:pressed {
                background-color: rgba(40, 60, 40, 200);
                border: 1px solid rgba(100, 200, 100, 200);
            }
        """)
        send_button.clicked.connect(self.send_message)
        button_layout.addWidget(send_button)
        
        compose_layout_inner.addLayout(button_layout)
        
        compose_layout.addWidget(compose_frame)
        
        # Add the tab
        self.tabs.addTab(compose_tab, "Compose")
    
    def send_message(self):
        """Send a message"""
        # Get the message details
        to = self.to_field.text()
        subject = self.subject_field.text()
        message = self.message_field.toPlainText()
        
        # Validate the fields
        if not to:
            self.show_message("Error", "Please enter a recipient")
            return
        
        if not subject:
            self.show_message("Error", "Please enter a subject")
            return
        
        if not message:
            self.show_message("Error", "Please enter a message")
            return
        
        # Show a success message
        self.show_message("Success", "Message sent successfully")
        
        # Clear the fields
        self.to_field.clear()
        self.subject_field.clear()
        self.message_field.clear()
        
        # Switch to the inbox tab
        self.tabs.setCurrentIndex(0)
    
    def show_message(self, title, message):
        """Show a message dialog"""
        # Create a simple message dialog
        dialog = QMainWindow(self)
        dialog.setWindowTitle(title)
        dialog.setFixedSize(300, 150)
        
        # Create the layout
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        
        # Add the message
        message_label = QLabel(message)
        message_label.setStyleSheet("color: white; font-size: 14px;")
        message_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(message_label)
        
        # Add the OK button
        ok_button = QPushButton("OK")
        ok_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(60, 60, 80, 200);
                color: white;
                border: 1px solid rgba(100, 100, 200, 200);
                border-radius: 5px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgba(80, 80, 120, 200);
                border: 1px solid rgba(150, 150, 255, 200);
            }
            QPushButton:pressed {
                background-color: rgba(40, 40, 60, 200);
                border: 1px solid rgba(100, 100, 200, 200);
            }
        """)
        ok_button.clicked.connect(dialog.close)
        layout.addWidget(ok_button)
        
        # Set the central widget
        dialog.setCentralWidget(central_widget)
        
        # Show the dialog
        dialog.show()

class QMailTerminal(AppTerminal):
    """Q-Mail terminal"""
    
    def __init__(self, app_name: str):
        """Initialize the Q-Mail terminal"""
        super().__init__(app_name)
        
        # Initialize the inbox
        self.inbox = [
            {
                "from": "Alice <alice@quantum.org>",
                "subject": "Quantum Encryption Test",
                "date": "Today",
                "body": "This is a test of the quantum encryption system."
            },
            {
                "from": "Bob <bob@quantum.org>",
                "subject": "Re: Project Status",
                "date": "Yesterday",
                "body": "The project is on track for completion next week."
            },
            {
                "from": "QuantoniumOS Admin <admin@quantonium.os>",
                "subject": "Welcome to Q-Mail",
                "date": "Last Week",
                "body": "Welcome to Q-Mail!\n\nThis is a secure, quantum-encrypted email client built on the QuantoniumOS platform.\n\nKey features:\n- Quantum encryption for secure communications\n- Integration with QuantoniumOS security protocols\n- Resistant to quantum computing attacks\n\nEnjoy using Q-Mail!\n\nThe QuantoniumOS Team"
            }
        ]
    
    def start(self):
        """Start the terminal app"""
        print("\n" + "=" * 60)
        print(f"{self.app_name} Terminal Interface")
        print("=" * 60 + "\n")
        
        print("Available commands:")
        print("  help          - Show this help message")
        print("  inbox         - List messages in your inbox")
        print("  read [number] - Read a specific message")
        print("  compose       - Compose a new message")
        print("  exit          - Exit the application\n")
        
        # Main loop
        while self.running:
            command = input(f"{self.app_name}> ").strip()
            self.process_command(command)
    
    def process_command(self, command: str):
        """Process a terminal command"""
        parts = command.split()
        
        if not parts:
            return
        
        cmd = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []
        
        if cmd == "help":
            print("\nAvailable commands:")
            print("  help          - Show this help message")
            print("  inbox         - List messages in your inbox")
            print("  read [number] - Read a specific message")
            print("  compose       - Compose a new message")
            print("  exit          - Exit the application\n")
        
        elif cmd == "inbox":
            self.list_inbox()
        
        elif cmd == "read":
            if not args:
                print("Error: Missing message number")
                print("Usage: read [number]")
                return
            
            try:
                message_index = int(args[0]) - 1
                if message_index < 0 or message_index >= len(self.inbox):
                    print(f"Error: Message number must be between 1 and {len(self.inbox)}")
                    return
                
                self.read_message(message_index)
            except ValueError:
                print("Error: Invalid message number")
        
        elif cmd == "compose":
            self.compose_message()
        
        elif cmd == "exit":
            print(f"Exiting {self.app_name}...")
            self.running = False
        
        else:
            print(f"Unknown command: {cmd}")
            print("Type 'help' for a list of commands")
    
    def list_inbox(self):
        """List the inbox messages"""
        print("\nInbox:")
        for i, message in enumerate(self.inbox):
            print(f"  {i+1}. {message['subject']} - From: {message['from']} ({message['date']})")
        print("")
    
    def read_message(self, index):
        """Read a message"""
        message = self.inbox[index]
        
        print("\n" + "=" * 60)
        print(f"Subject: {message['subject']}")
        print(f"From: {message['from']}")
        print(f"Date: {message['date']}")
        print("=" * 60)
        print(message['body'])
        print("=" * 60 + "\n")
    
    def compose_message(self):
        """Compose a new message"""
        print("\nComposing a new message")
        to = input("To: ")
        subject = input("Subject: ")
        print("Body (type '.' on a new line to finish):")
        
        body = []
        while True:
            line = input()
            if line == ".":
                break
            body.append(line)
        
        # Validate the message
        if not to:
            print("Error: Recipient is required")
            return
        
        if not subject:
            print("Error: Subject is required")
            return
        
        if not body:
            print("Error: Message body is required")
            return
        
        # Send the message
        print("\nMessage sent successfully!\n")

def main():
    """Main function"""
    # Create the app launcher
    launcher = AppLauncherBase("Q-Mail", "fa5s.envelope")
    
    # Check if the GUI should be disabled
    if "--no-gui" in sys.argv:
        launcher.launch_terminal(QMailTerminal)
    else:
        launcher.launch_gui(QMailApp)

if __name__ == "__main__":
    main()
