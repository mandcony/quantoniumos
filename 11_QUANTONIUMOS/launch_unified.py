#!/usr/bin/env python3
"""
QuantoniumOS Unified Launcher
Quick launcher for the unified quantum operating system
"""

import os
import sys
from pathlib import Path

# Add the QuantoniumOS directory to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(current_dir))  # Add the current directory first
sys.path.insert(1, str(project_root))  # Then add the parent directory


def main():
    print("🌌 QuantoniumOS - Unified Quantum Operating System Launcher")
    print("=" * 60)
    print()
    print("Available launch modes:")
    print("  1. Desktop GUI    - Full desktop interface")
    print("  2. Web Interface  - Browser-based quantum interface")
    print("  3. CLI Mode       - Command-line interface")
    print("  4. Full Mode      - All interfaces simultaneously")
    print("  5. Demo Mode      - Quick demonstration")
    print()

    try:
        choice = input("Select mode (1-5) or press Enter for Desktop GUI: ").strip()

        if choice == "":
            choice = "1"

        mode_map = {"1": "desktop", "2": "web", "3": "cli", "4": "full", "5": "demo"}

        if choice in mode_map:
            mode = mode_map[choice]
            print(f"\n🚀 Launching QuantoniumOS in {mode} mode...")

            # Import and launch
            from core.quantonium_os_unified import core.main as main
            if mode == "desktop":
                print("Launching desktop interface...")
                main()
                os_system.launch_desktop_gui()
            elif mode == "web":
                port = input("Enter port (default 5000): ").strip()
                port = int(port) if port else 5000
                url = os_system.launch_web_interface(port)
                if url:
                    print(f"🌐 Web interface available at: {url}")
                    print("Press Ctrl+C to stop")
                    import time

                    try:
                        while True:
                            time.sleep(1)
                    except KeyboardInterrupt:
                        print("\n🛑 Shutting down...")
            elif mode == "cli":
                os_system.launch_cli()
            elif mode == "full":
                url = os_system.launch_web_interface(5000)
                if url:
                    print(f"🌐 Web interface: {url}")
                os_system.launch_desktop_gui()
            elif mode == "demo":
                import json
                import time

                # Quick demo
                for gate in ["H", "X", "Y", "Z"]:
                    os_system._apply_quantum_gate(gate)
                    time.sleep(0.5)

                rft_result = os_system._run_patent_demo("RFT")
                quantum_result = os_system._run_patent_demo("Quantum")

                print(f"\n✅ Demo completed!")
                print(
                    f"📊 Quantum state: {json.dumps(os_system.quantum_state, indent=2)}"
                )

                launch_gui = input("\nLaunch GUI? (y/n): ")
                if launch_gui.lower() == "y":
                    os_system.launch_desktop_gui()
        else:
            print("Invalid choice. Launching desktop GUI...")
            from core.quantonium_os_unified import QuantoniumOSUnified

            os_system = QuantoniumOSUnified()
            os_system.launch_desktop_gui()

    except KeyboardInterrupt:
        print("\n🛑 Launcher cancelled")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Falling back to desktop mode...")
        try:
            from core.quantonium_os_unified import QuantoniumOSUnified

            os_system = QuantoniumOSUnified()
            os_system.launch_desktop_gui()
        except Exception as e2:
            print(f"❌ Could not launch: {e2}")


if __name__ == "__main__":
    main()
