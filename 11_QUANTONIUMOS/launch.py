#!/usr/bin/env python3
"""
QuantoniumOS Unified Launcher
===========================

This is the MAIN entry point for QuantoniumOS.
Use this script to launch the system in different modes.

USAGE:
    python launch.py                    # Interactive mode selection
    python launch.py web                # Web interface (Flask)  
    python launch.py gui                # Desktop GUI (Qt)
    python launch.py test               # Run validation tests
    python launch.py apps               # Launch application menu

ENTRY POINTS HIERARCHY:
    1. This file (launch.py) - MAIN LAUNCHER
    2. 03_RUNNING_SYSTEMS/app.py - Web interface
    3. 11_QUANTONIUMOS/quantonium_os_unified.py - Desktop GUI
    4. 02_CORE_VALIDATORS/run_all_validators.py - Validation tests
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def get_project_root():
    """Get the project root directory"""
    return Path(__file__).parent


def launch_web():
    """Launch the web interface"""
    print("🌐 Launching QuantoniumOS Web Interface...")
    web_app = get_project_root() / "03_RUNNING_SYSTEMS" / "app.py"
    
    try:
        # Try to launch the web app
        os.chdir(get_project_root() / "03_RUNNING_SYSTEMS")
        subprocess.run([sys.executable, str(web_app)])
    except Exception as e:
        print(f"❌ Web interface failed to start: {e}")
        print("💡 Try running directly: cd 03_RUNNING_SYSTEMS && python app.py")


def launch_gui():
    """Launch the desktop GUI"""
    print("🖥️ Launching QuantoniumOS Desktop GUI...")
    gui_app = get_project_root() / "11_QUANTONIUMOS" / "quantonium_os_unified.py"
    
    try:
        # Try to launch the GUI
        os.chdir(get_project_root() / "11_QUANTONIUMOS")
        subprocess.run([sys.executable, str(gui_app)])
    except Exception as e:
        print(f"❌ Desktop GUI failed to start: {e}")
        print("💡 Try running directly: cd 11_QUANTONIUMOS && python quantonium_os_unified.py")


def run_tests():
    """Run validation tests"""
    print("🧪 Running QuantoniumOS Validation Tests...")
    test_runner = get_project_root() / "02_CORE_VALIDATORS" / "run_all_validators.py"
    
    try:
        subprocess.run([sys.executable, str(test_runner)])
    except Exception as e:
        print(f"❌ Tests failed to run: {e}")
        print("💡 Try running directly: cd 02_CORE_VALIDATORS && python run_all_validators.py")


def launch_apps():
    """Show available applications"""
    print("📱 QuantoniumOS Applications:")
    apps_dir = get_project_root() / "apps"
    
    if apps_dir.exists():
        print("\nAvailable applications:")
        for app_file in apps_dir.glob("launch_*.py"):
            app_name = app_file.stem.replace("launch_", "")
            print(f"  • {app_name}")
        print(f"\nTo launch an app: cd apps && python launch_<app_name>.py")
    else:
        print("❌ Apps directory not found")


def interactive_mode():
    """Interactive mode for selecting launch options"""
    print("🌟 QuantoniumOS Launcher")
    print("=" * 40)
    print("1. Web Interface (Flask)")
    print("2. Desktop GUI (Qt)")  
    print("3. Run Tests")
    print("4. Show Apps")
    print("5. Exit")
    
    while True:
        try:
            choice = input("\nSelect option (1-5): ").strip()
            
            if choice == "1":
                launch_web()
                break
            elif choice == "2":
                launch_gui()
                break
            elif choice == "3":
                run_tests()
                break
            elif choice == "4":
                launch_apps()
                break
            elif choice == "5":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please select 1-5.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="QuantoniumOS Unified Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python launch.py           # Interactive mode
  python launch.py web       # Web interface
  python launch.py gui       # Desktop GUI
  python launch.py test      # Run tests
  python launch.py apps      # Show applications
        """
    )
    
    parser.add_argument(
        "mode",
        nargs="?",
        choices=["web", "gui", "test", "apps"],
        help="Launch mode (if not provided, interactive mode is used)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "web":
        launch_web()
    elif args.mode == "gui":
        launch_gui()
    elif args.mode == "test":
        run_tests()
    elif args.mode == "apps":
        launch_apps()
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
