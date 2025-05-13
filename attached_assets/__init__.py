"""
QuantoniumOS Desktop Applications Package

This package contains all the desktop applications for QuantoniumOS.
Desktop apps are launched from the OS interface and run in separate windows.
"""

import os
import sys
import platform

# Add parent directory to path to ensure imports work correctly
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

def setup_headless_environment():
    """
    Configure the environment to run Qt applications in headless mode.
    This is needed when running in container environments like Replit.
    
    Call this function before creating a QApplication instance.
    """
    # Set QT_QPA_PLATFORM to 'offscreen' for headless environments
    if 'DISPLAY' not in os.environ or not os.environ['DISPLAY']:
        os.environ['QT_QPA_PLATFORM'] = 'offscreen'
        print("QuantoniumOS: Running in headless mode with offscreen rendering")
    else:
        print(f"QuantoniumOS: Display available at {os.environ['DISPLAY']}")
    
    # Configure log path
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    return {
        'platform': platform.system(),
        'headless': 'QT_QPA_PLATFORM' in os.environ and os.environ['QT_QPA_PLATFORM'] == 'offscreen',
        'log_dir': log_dir
    }