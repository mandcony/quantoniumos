#!/usr/bin/env python3
"""
🚀 QUANTONIUMOS MAIN LAUNCHER  
Updated launcher with centered display and modern UI
"""

import sys
import os
from pathlib import Path

# Add frontend path (already in frontend directory)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = BASE_DIR  # Already in frontend directory
sys.path.insert(0, FRONTEND_DIR)

def launch_quantonium_os():
    """Launch QuantoniumOS with unified frontend"""
    
    print("🚀 LAUNCHING QUANTONIUMOS")
    print("=" * 50)
    print("Unified Quantum Operating System")
    print("Version 1.0.0 - Streamlined Architecture")
    print("=" * 50)
    
    try:
        # Import the unified desktop UI
        from quantonium_desktop import QuantoniumDesktop
        from PyQt5.QtWidgets import QApplication
        
        # Create application
        app = QApplication(sys.argv)
        app.setApplicationName("QuantoniumOS")
        app.setApplicationVersion("1.0.0")
        
        print("✅ UI System initialized")
        print("✅ Unified quantum desktop loading...")
        
        # Create and show main window
        main_window = QuantoniumDesktop()
        main_window.show()
        
        print("✅ QuantoniumOS launched successfully!")
        print("   • Centered quantum logo with expandable apps")
        print("   • QuantoniumOS branding at bottom")
        print("   • System time display on right")
        print("   • Scientific golden ratio design")
        print("   • Single unified frontend")
        
        # Run application
        sys.exit(app.exec_())
        
    except ImportError as e:
        print(f"❌ Error importing UI components: {e}")
        print("   Checking frontend availability...")
        console_mode()
    
    except Exception as e:
        print(f"❌ Error launching QuantoniumOS: {e}")
        console_mode()

def console_mode():
    """Fallback console mode"""
    
    print("\n🖥️  QUANTONIUMOS CONSOLE MODE")
    print("=" * 40)
    
    print("\n📱 Available Apps:")
    apps_dir = Path(BASE_DIR).parent / "apps"
    if apps_dir.exists():
        for app_file in apps_dir.glob("*.py"):
            if not app_file.name.startswith("launcher_"):
                app_name = app_file.stem.replace("_", " ").title()
                print(f"   • {app_name}")
    
    print(f"\n🔬 Validation Suite:")
    validation_dir = Path(BASE_DIR).parent / "validation"
    if validation_dir.exists():
        print(f"   • Benchmarks: validation/benchmarks/")
        print(f"   • Analysis: validation/analysis/")
        print(f"   • Results: validation/results/")
    
    print(f"\n🚀 To launch GUI mode:")
    print(f"   python engine/launch_quantonium_os.py")
    
    print(f"\n✅ QuantoniumOS Console Ready")

def main():
    """Main entry point"""
    
    # Check if GUI launch is requested
    if len(sys.argv) > 1 and sys.argv[1] == "--console":
        console_mode()
    else:
        launch_quantonium_os()

if __name__ == "__main__":
    main()
