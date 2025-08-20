#!/usr/bin/env python3
"""
QuantoniumOS Simple Launcher

Quick launcher for QuantoniumOS with menu selection
"""

import os
import sys
import subprocess
from pathlib import Path

def print_logo():
    """Print QuantoniumOS logo"""
    print("""
╔═══════════════════════════════════════════════════════════╗
║                     🌌 QuantoniumOS                     ║
║           1000-Qubit Quantum Operating System            ║
║                                                           ║
║        World's First Vertex-Based Quantum OS            ║
║              🔮 Phase 1 - Now Available 🔮              ║
╚═══════════════════════════════════════════════════════════╝
    """)

def show_menu():
    """Show main menu"""
    print("\n🚀 QUANTONIUMOS LAUNCHER")
    print("═" * 50)
    print("1. 🎬 Demo Mode           - Show system capabilities")
    print("2. 🖥️  Desktop GUI        - Launch desktop interface")
    print("3. 💻 CLI Mode            - Command line interface")
    print("4. 📁 Filesystem Test     - Test quantum filesystem")
    print("5. 🔬 Kernel Only         - Just quantum kernel")
    print("6. 📊 System Info         - Show component status")
    print("7. ❓ Help               - Show documentation")
    print("8. 🚪 Exit               - Quit launcher")
    print("═" * 50)

def check_components():
    """Check which components are available"""
    base_path = Path(__file__).parent
    components = {
        "kernel": (base_path / "kernel" / "quantum_vertex_kernel.py").exists(),
        "desktop": (base_path / "gui" / "desktop.py").exists(),
        "filesystem": (base_path / "filesystem" / "quantum_fs.py").exists(),
        "integration": (base_path / "kernel" / "patent_integration.py").exists(),
        "complete": (base_path / "quantonium_os_complete.py").exists()
    }
    return components

def run_component(component, args=""):
    """Run a QuantoniumOS component"""
    base_path = Path(__file__).parent
    
    component_paths = {
        "demo": "quantonium_os_complete.py demo",
        "desktop": "gui\\desktop.py",
        "cli": "quantonium_os_complete.py cli",
        "filesystem": "filesystem\\quantum_fs.py",
        "kernel": "kernel\\quantum_vertex_kernel.py",
        "complete": f"quantonium_os_complete.py {args}"
    }
    
    if component in component_paths:
        cmd = f"python {component_paths[component]}"
        print(f"\n🚀 Launching: {cmd}")
        print("─" * 50)
        
        try:
            subprocess.run(cmd, shell=True, cwd=base_path)
        except KeyboardInterrupt:
            print("\n🔄 Component stopped by user")
        except Exception as e:
            print(f"❌ Error running component: {e}")
    else:
        print(f"❌ Unknown component: {component}")

def show_system_info():
    """Show system information"""
    components = check_components()
    
    print("\n📊 QUANTONIUMOS SYSTEM INFORMATION")
    print("═" * 60)
    print("🎯 Version: QuantoniumOS v1.0 - Phase 1")
    print("🔮 Architecture: 1000-qubit quantum vertex network")
    print("📍 Location: c:\\quantoniumos-1\\11_QUANTONIUMOS")
    print("\n🏗️ AVAILABLE COMPONENTS:")
    
    status_icons = {True: "✅", False: "❌"}
    
    for component, available in components.items():
        status = status_icons[available]
        print(f"   {status} {component.title()}: {'Available' if available else 'Missing'}")
    
    print("\n📋 FEATURES:")
    print("   • 1000-qubit quantum vertex network")
    print("   • Real-time quantum process management")
    print("   • RFT (Resonant Frequency Transform) integration")
    print("   • Quantum-safe cryptographic engine")
    print("   • Patent-protected quantum algorithms")
    print("   • Quantum-aware file system")
    print("   • Desktop GUI with real-time monitoring")
    print("   • Command-line interface")
    
    print("\n🎯 BREAKTHROUGH ACHIEVEMENT:")
    print("   First operating system to run quantum processes")
    print("   directly on quantum vertices with RFT enhancement.")
    print("═" * 60)

def show_help():
    """Show help documentation"""
    print("\n📖 QUANTONIUMOS HELP")
    print("═" * 60)
    print("🎯 GETTING STARTED:")
    print("   1. Run 'Demo Mode' to see system capabilities")
    print("   2. Try 'Desktop GUI' for graphical interface")
    print("   3. Use 'CLI Mode' for command-line control")
    print("   4. Test 'Filesystem' for quantum file operations")
    
    print("\n🔮 QUANTUM OPERATIONS:")
    print("   • Spawn Process: Create quantum processes on vertices")
    print("   • Apply Gates: Execute quantum gate operations (H, X, Z)")
    print("   • Evolve System: Advance quantum state evolution")
    print("   • RFT Transform: Apply resonant frequency transformations")
    
    print("\n📁 FILE SYSTEM:")
    print("   • .qstate files: Quantum state data")
    print("   • .rft files: RFT transformation data")
    print("   • .qkey files: Quantum cryptographic keys")
    print("   • .qcircuit files: Quantum circuit definitions")
    
    print("\n⚙️ SYSTEM REQUIREMENTS:")
    print("   • Python 3.12+ (installed)")
    print("   • tkinter for desktop GUI (included)")
    print("   • Optional: Flask for web interface")
    
    print("\n🆘 TROUBLESHOOTING:")
    print("   • If components fail to load, check Python path")
    print("   • For web interface: pip install flask flask-socketio")
    print("   • All quantum operations work without external deps")
    print("═" * 60)

def main():
    """Main launcher"""
    print_logo()
    
    # Check if we're in the right directory
    if not Path("kernel").exists():
        print("❌ Error: Please run from QuantoniumOS directory")
        print("   Expected location: c:\\quantoniumos-1\\11_QUANTONIUMOS")
        return
    
    while True:
        show_menu()
        
        try:
            choice = input("\nSelect option (1-8): ").strip()
            
            if choice == "1":
                run_component("demo")
            elif choice == "2":
                run_component("desktop")
            elif choice == "3":
                run_component("cli")
            elif choice == "4":
                run_component("filesystem")
            elif choice == "5":
                run_component("kernel")
            elif choice == "6":
                show_system_info()
            elif choice == "7":
                show_help()
            elif choice == "8":
                break
            else:
                print("❌ Invalid choice. Please select 1-8.")
            
            if choice in ["1", "2", "3", "4", "5"]:
                input("\nPress Enter to return to menu...")
                
        except KeyboardInterrupt:
            print("\n\n👋 Thanks for using QuantoniumOS!")
            break
        except EOFError:
            break
    
    print("🔄 QuantoniumOS Launcher exiting...")

if __name__ == "__main__":
    main()
