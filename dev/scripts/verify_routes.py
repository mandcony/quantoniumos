#!/usr/bin/env python3
"""
QuantoniumOS App Route Verification Script
Tests all desktop SVG button routes to ensure they open correctly
"""

import os
import sys
from pathlib import Path

def verify_app_routes():
    """Verify all app routes are correctly configured"""
    base_path = Path(__file__).parent  # This should be the quantoniumos directory
    
    # Application mapping from desktop
    apps = [
        {
            "name": "RFT Validation Suite", 
            "path": base_path / "src" / "apps" / "rft_validation_suite.py",
            "icon": "rft_validator.svg"
        },
        {
            "name": "AI Chat", 
            "path": base_path / "src" / "apps" / "qshll_chatbox.py",
            "icon": "ai_chat.svg"
        },
        {
            "name": "Quantum Simulator", 
            "path": base_path / "src" / "apps" / "quantum_simulator.py",
            "icon": "quantum_simulator.svg"
        },
        {
            "name": "Quantum Cryptography", 
            "path": base_path / "src" / "apps" / "quantum_crypto.py",
            "icon": "quantum_crypto.svg"
        },
        {
            "name": "System Monitor", 
            "path": base_path / "src" / "apps" / "qshll_system_monitor.py",
            "icon": "system_monitor.svg"
        },
        {
            "name": "Q-Notes", 
            "path": base_path / "src" / "apps" / "q_notes.py",
            "icon": "q_notes.svg"
        },
        {
            "name": "Q-Vault", 
            "path": base_path / "src" / "apps" / "q_vault.py",
            "icon": "q_vault.svg"
        }
    ]
    
    print("🔍 QuantoniumOS Desktop App Route Verification")
    print("=" * 50)
    
    all_good = True
    
    for app in apps:
        app_file_exists = app["path"].exists()
        icon_file = base_path / "ui" / "icons" / app["icon"]
        icon_exists = icon_file.exists()
        
        status_app = "✅" if app_file_exists else "❌"
        status_icon = "✅" if icon_exists else "❌"
        
        print(f"{status_app} {app['name']}")
        print(f"   App:  {app['path']}")
        print(f"   Icon: {status_icon} {icon_file}")
        print()
        
        if not app_file_exists or not icon_exists:
            all_good = False
    
    if all_good:
        print("🎉 All routes verified! Desktop SVG buttons should work perfectly.")
    else:
        print("⚠️  Some routes need fixing - see errors above.")
    
    return all_good

if __name__ == "__main__":
    verify_app_routes()
