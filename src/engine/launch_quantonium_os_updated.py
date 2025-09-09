#!/usr/bin/env python3
"""
🚀 QUANTONIUMOS MAIN LAUNCHER  
Updated launcher with centered display and modern UI
"""

import sys
import os
from pathlib import Path

# Add frontend path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(os.path.dirname(BASE_DIR), "frontend")
sys.path.insert(0, FRONTEND_DIR)

def launch_quantonium_os():
    """Launch QuantoniumOS with intro animation followed by desktop"""
    
    print("🚀 LAUNCHING QUANTONIUMOS")
    print("=" * 50)
    print("Symbolic Quantum-Inspired Computing Engine")
    print("Version 1.0.0 - Production Ready")
    print("=" * 50)
    
    try:
        # Import the main desktop UI and intro
        from quantonium_desktop import QuantoniumDesktop
        from quantonium_intro import QuantoniumIntro
        from PyQt5.QtWidgets import QApplication
        
        # Create application
        app = QApplication(sys.argv)
        app.setApplicationName("QuantoniumOS")
        app.setApplicationVersion("1.0.0")
        
        print("✅ UI System initialized")
        print("✅ Starting intro animation...")
        
        # Create intro window
        intro = QuantoniumIntro()
        
        # Create desktop window (hidden initially)
        main_window = QuantoniumDesktop(show_immediately=False)
        
        def show_desktop():
            """Complete the transition"""
            # Just close intro - desktop is already visible and faded in
            intro.close()
        
        def start_crossfade():
            """Begin seamless cross-fade transition"""
            print("🌊 Starting seamless cross-fade transition...")
            print("✅ QuantoniumOS launched successfully!")
            print("   • Animated intro complete")
            print("   • Centered quantum logo display")
            print("   • Expandable app dock")
            print("   • Scientific minimal design")
            print("   • Golden ratio proportions")
            
            # Start desktop fade-in
            main_window.start_fadein()
            
            # Start intro fade-out
            intro.start_fadeout()
        
        # Connect signals for seamless transition
        intro.start_crossfade.connect(start_crossfade)
        intro.animation_finished.connect(show_desktop)
        
        # Show intro and start animation
        intro.show()
        intro.start_animation()
        
        # Run application
        sys.exit(app.exec_())
        
    except ImportError as e:
        print(f"❌ Error importing UI components: {e}")
        print("   Falling back to basic launcher...")
        
        # Basic fallback
        try:
            from quantonium_desktop import QuantoniumDesktop
            from PyQt5.QtWidgets import QApplication
            
            app = QApplication(sys.argv)
            main_window = QuantoniumDesktop()
            main_window.show()
            sys.exit(app.exec_())
            
        except ImportError:
            print("❌ UI components not available")
            print("   Running in console mode...")
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
