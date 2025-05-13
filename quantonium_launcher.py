"""
QuantoniumOS Launcher

This script launches the QuantoniumOS desktop environment by importing
and running the actual QuantoniumOS code from attached_assets.
"""

import os
import sys
import importlib.util

def launch_quantonium_os():
    """
    Import the quantonium_os_main module from attached_assets
    and launch the QuantoniumOS desktop environment.
    """
    # Get the path to the quantonium_os_main.py file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os_main_path = os.path.join(current_dir, "attached_assets", "quantonium_os_main.py")
    
    # Check if the file exists
    if not os.path.exists(os_main_path):
        print(f"ERROR: quantonium_os_main.py not found at {os_main_path}")
        return False
    
    try:
        # Add the attached_assets directory to sys.path
        assets_dir = os.path.join(current_dir, "attached_assets")
        sys.path.insert(0, assets_dir)
        
        # Create a directory for app files if it doesn't exist
        app_dir = os.path.join(assets_dir, "apps")
        os.makedirs(app_dir, exist_ok=True)
        
        # Import and run the code from quantonium_os_main.py
        spec = importlib.util.spec_from_file_location("quantonium_os_main", os_main_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # The module is executed, which should launch the QuantoniumOS
        return True
    
    except ImportError as e:
        print(f"ERROR: Failed to import QuantoniumOS - {str(e)}")
        if "PyQt5" in str(e):
            print("PyQt5 is required. Install it with: pip install pyqt5 qtawesome")
        return False
    
    except Exception as e:
        print(f"ERROR: Failed to launch QuantoniumOS - {str(e)}")
        return False

if __name__ == "__main__":
    launch_quantonium_os()