#!/usr/bin/env python3
"""
QuantoniumOS Complete Operating System Launcher

Full operating system with:
- Desktop GUI (tkinter)
- Web interface (Flask)
- Quantum kernel (1000-qubit)
- File system
- Applications
- System services

Launch modes:
- Desktop: GUI desktop environment
- Web: Browser-based interface
- CLI: Command-line interface
- Full: All interfaces simultaneously
"""

import argparse
import sys
import threading
import time
from pathlib import Path

# Add all component paths
base_path = Path(__file__).parent
sys.path.insert(0, str(base_path / "kernel"))
sys.path.insert(0, str(base_path / "gui"))
sys.path.insert(0, str(base_path / "web"))
sys.path.insert(0, str(base_path / "filesystem"))

print("🚀 QUANTONIUMOS COMPLETE OPERATING SYSTEM")
print("🌌 1000-Qubit Quantum Vertex Operating System")
print("=" * 60)

# Import components
try:
    from patent_integration import QuantoniumOSIntegration
    from quantum_vertex_kernel import QuantoniumKernel

    kernel_available = True
    print("✅ Quantum kernel modules loaded")
except ImportError as e:
    print(f"⚠️ Kernel modules unavailable: {e}")
    kernel_available = False

try:
    from desktop import QuantoniumDesktop

    desktop_available = True
    print("✅ Desktop GUI modules loaded")
except ImportError as e:
    print(f"⚠️ Desktop GUI unavailable: {e}")
    desktop_available = False

try:
    from quantum_fs import QuantumFileSystem

    filesystem_available = True
    print("✅ Quantum filesystem modules loaded")
except ImportError as e:
    print(f"⚠️ Filesystem unavailable: {e}")
    filesystem_available = False

# Web components (optional dependencies)
try:
    import flask
    import flask_socketio

    web_deps_available = True
    print("✅ Web dependencies available")
except ImportError:
    web_deps_available = False
    print("⚠️ Web dependencies unavailable (pip install flask flask-socketio)")

try:
    if web_deps_available:
        from core.app import app, initialize_quantum_system, socketio

        web_available = True
        print("✅ Web interface modules loaded")
    else:
        web_available = False
except ImportError as e:
    print(f"⚠️ Web interface unavailable: {e}")
    web_available = False


class QuantoniumOSManager:
    """
    Main QuantoniumOS system manager
    Orchestrates all components of the operating system
    """

    def __init__(self):
        self.kernel = None
        self.integration = None
        self.filesystem = None
        self.desktop = None
        self.services = {}
        self.running = False

        print("\n🔧 Initializing QuantoniumOS System Manager...")

    def initialize_core_system(self):
        """Initialize core quantum system"""
        if not kernel_available:
            print("❌ Cannot start QuantoniumOS without quantum kernel")
            return False

        try:
            print("🔹 Initializing quantum kernel...")
            self.kernel = QuantoniumKernel()

            print("🔹 Initializing patent integration...")
            self.integration = QuantoniumOSIntegration()

            if filesystem_available:
                print("🔹 Initializing quantum filesystem...")
                self.filesystem = QuantumFileSystem()

            self.running = True
            print("✅ QuantoniumOS core system initialized")
            return True

        except Exception as e:
            print(f"❌ Core system initialization failed: {e}")
            return False

    def start_desktop_interface(self):
        """Start desktop GUI interface"""
        if not desktop_available:
            print("❌ Desktop interface unavailable")
            return False

        try:
            print("🖥️ Starting desktop interface...")
            self.desktop = QuantoniumDesktop()

            # Run desktop in main thread
            self.desktop.run()
            return True

        except Exception as e:
            print(f"❌ Desktop interface failed: {e}")
            return False

    def start_web_interface(self, host="localhost", port=5000):
        """Start web interface"""
        if not web_available:
            print("❌ Web interface unavailable")
            return False

        try:
            print(f"🌐 Starting web interface on http://{host}:{port}")

            # Initialize web backend with our kernel
            if self.kernel:
                # Connect web interface to our kernel
                import core.app as appas web_app

                web_app.quantum_kernel = self.kernel
                web_app.quantum_integration = self.integration
                web_app.system_running = True

            # Start web server in thread
            def run_web():
                socketio.run(app, host=host, port=port, debug=False)

            web_thread = threading.Thread(target=run_web, daemon=True)
            web_thread.start()

            print(f"✅ Web interface started at http://{host}:{port}")
            return True

        except Exception as e:
            print(f"❌ Web interface failed: {e}")
            return False

    def start_cli_interface(self):
        """Start command-line interface"""
        print("💻 QuantoniumOS Command-Line Interface")
        print("Type 'help' for commands, 'exit' to quit")
        print("-" * 40)

        while self.running:
            try:
                command = input("QuantoniumOS> ").strip().lower()

                if command == "exit" or command == "quit":
                    break
                elif command == "help":
                    self.show_cli_help()
                elif command == "status":
                    self.show_system_status()
                elif command == "spawn":
                    self.cli_spawn_process()
                elif command == "gate":
                    self.cli_apply_gate()
                elif command == "evolve":
                    self.cli_evolve_system()
                elif command == "files":
                    self.cli_list_files()
                elif command == "info":
                    self.show_system_info()
                elif command == "":
                    continue
                else:
                    print(f"Unknown command: {command}")
                    print("Type 'help' for available commands")

            except KeyboardInterrupt:
                print("\nUse 'exit' to quit QuantoniumOS")
            except EOFError:
                break

        print("👋 QuantoniumOS CLI session ended")

    def show_cli_help(self):
        """Show CLI help"""
        print(
            """
QuantoniumOS CLI Commands:
══════════════════════════════════════════════════════════
help        - Show this help message
status      - Show quantum system status
spawn       - Spawn quantum process (interactive)
gate        - Apply quantum gate (interactive)
evolve      - Evolve quantum system
files       - List filesystem contents
info        - Show detailed system information
exit/quit   - Exit QuantoniumOS
══════════════════════════════════════════════════════════
"""
        )

    def show_system_status(self):
        """Show system status"""
        if self.kernel:
            status = self.kernel.get_system_status()
            print("\n📊 QuantoniumOS System Status:")
            print("═" * 40)
            for key, value in status.items():
                print(f"  {key}: {value}")
        else:
            print("❌ Quantum kernel offline")

    def cli_spawn_process(self):
        """CLI spawn process"""
        if not self.kernel:
            print("❌ Quantum kernel offline")
            return

        try:
            vertex_id = int(input("Enter vertex ID (0-999): "))
            if 0 <= vertex_id < 1000:
                pid = self.kernel.spawn_quantum_process(vertex_id)
                print(f"✅ Process {pid} spawned on vertex {vertex_id}")
            else:
                print("❌ Invalid vertex ID")
        except ValueError:
            print("❌ Invalid input")
        except Exception as e:
            print(f"❌ Error: {e}")

    def cli_apply_gate(self):
        """CLI apply gate"""
        if not self.kernel:
            print("❌ Quantum kernel offline")
            return

        try:
            vertex_id = int(input("Enter vertex ID (0-999): "))
            gate = input("Enter gate type (H/X/Z): ").upper()

            if 0 <= vertex_id < 1000 and gate in ["H", "X", "Z"]:
                success = self.kernel.apply_quantum_gate(vertex_id, gate)
                if success:
                    print(f"✅ {gate} gate applied to vertex {vertex_id}")
                else:
                    print("❌ Gate application failed")
            else:
                print("❌ Invalid input")
        except ValueError:
            print("❌ Invalid input")
        except Exception as e:
            print(f"❌ Error: {e}")

    def cli_evolve_system(self):
        """CLI evolve system"""
        if not self.kernel:
            print("❌ Quantum kernel offline")
            return

        try:
            steps = int(input("Enter evolution steps (default 5): ") or "5")
            print(f"🌊 Evolving system ({steps} steps)...")
            self.kernel.evolve_quantum_system(time_steps=steps)
            print("✅ Evolution complete")
        except ValueError:
            print("❌ Invalid input")
        except Exception as e:
            print(f"❌ Error: {e}")

    def cli_list_files(self):
        """CLI list files"""
        if not self.filesystem:
            print("❌ Filesystem unavailable")
            return

        try:
            directory = input("Enter directory (default /): ") or ""
            files = self.filesystem.list_directory(directory)

            print(f"\n📁 Directory: /{directory}")
            print("═" * 40)
            for qfile in files:
                print(f"  {qfile.name} ({qfile.file_type}) - {qfile.size} bytes")
            print(f"\nTotal: {len(files)} files")

        except Exception as e:
            print(f"❌ Error: {e}")

    def show_system_info(self):
        """Show detailed system information"""
        print("\n🌌 QuantoniumOS System Information")
        print("═" * 60)
        print("🎯 Version: QuantoniumOS v1.0 - Phase 1")
        print("🔮 Architecture: 1000-qubit quantum vertex network")
        print("🏗️ Components:")
        print(f"   • Quantum Kernel: {'✅ Online' if self.kernel else '❌ Offline'}")
        print(
            f"   • Patent Integration: {'✅ Active' if self.integration else '❌ Inactive'}"
        )
        print(f"   • Filesystem: {'✅ Mounted' if self.filesystem else '❌ Unmounted'}")
        print(
            f"   • Desktop GUI: {'✅ Available' if desktop_available else '❌ Unavailable'}"
        )
        print(
            f"   • Web Interface: {'✅ Available' if web_available else '❌ Unavailable'}"
        )

        if self.filesystem:
            stats = self.filesystem.get_filesystem_stats()
            print(
                f"📁 Filesystem: {stats['total_files']} files, {stats['total_size_bytes']} bytes"
            )

        print("═" * 60)

    def shutdown(self):
        """Shutdown QuantoniumOS"""
        print("🔄 Shutting down QuantoniumOS...")

        if self.kernel:
            self.kernel.shutdown()

        self.running = False
        print("✅ QuantoniumOS shutdown complete")


def main():
    """Main QuantoniumOS launcher"""
    parser = argparse.ArgumentParser(
        description="QuantoniumOS - 1000-Qubit Quantum Operating System"
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="desktop",
        choices=["desktop", "web", "cli", "full", "demo"],
        help="Launch mode (default: desktop)",
    )
    parser.add_argument("--host", default="localhost", help="Web interface host")
    parser.add_argument("--port", type=int, default=5000, help="Web interface port")
    parser.add_argument(
        "--no-kernel", action="store_true", help="Skip kernel initialization"
    )

    args = parser.parse_args()

    print(f"🎯 Launch mode: {args.mode.upper()}")
    print()

    # Initialize system manager
    os_manager = QuantoniumOSManager()

    # Initialize core system unless skipped
    if not args.no_kernel:
        if not os_manager.initialize_core_system():
            print("❌ Failed to initialize core system")
            if args.mode not in ["demo", "cli"]:
                return 1

    try:
        if args.mode == "desktop":
            # Desktop GUI mode
            if not os_manager.start_desktop_interface():
                print("❌ Desktop interface failed, trying CLI...")
                os_manager.start_cli_interface()

        elif args.mode == "web":
            # Web interface mode
            if os_manager.start_web_interface(args.host, args.port):
                print("🌐 Web interface running. Press Ctrl+C to stop.")
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    pass
            else:
                print("❌ Web interface failed")

        elif args.mode == "cli":
            # Command-line interface mode
            os_manager.start_cli_interface()

        elif args.mode == "full":
            # All interfaces
            print("🚀 Starting all QuantoniumOS interfaces...")

            # Start web interface in background
            if web_available:
                os_manager.start_web_interface(args.host, args.port)
                time.sleep(2)  # Give web server time to start

            # Start desktop interface (blocking)
            if desktop_available:
                print("🖥️ Starting desktop interface...")
                os_manager.start_desktop_interface()
            else:
                print("🖥️ Desktop unavailable, using CLI...")
                os_manager.start_cli_interface()

        elif args.mode == "demo":
            # Demo mode - show capabilities
            print("🎬 QuantoniumOS Demo Mode")
            print("═" * 40)

            # Show system info
            os_manager.show_system_info()

            if os_manager.kernel:
                # Show system status
                print("\n📊 Live System Status:")
                os_manager.show_system_status()

                # Demo operations
                print("\n🔬 Demo Operations:")
                print("   🚀 Spawning process on vertex 0...")
                pid = os_manager.kernel.spawn_quantum_process(0)
                print(f"   ✅ Process {pid} spawned")

                print("   🚪 Applying Hadamard gate to vertex 0...")
                os_manager.kernel.apply_quantum_gate(0, "H")
                print("   ✅ Gate applied")

                print("   🌊 Evolving system (3 steps)...")
                os_manager.kernel.evolve_quantum_system(time_steps=3)
                print("   ✅ Evolution complete")

            if os_manager.filesystem:
                print("\n📁 Filesystem Demo:")
                stats = os_manager.filesystem.get_filesystem_stats()
                print(f"   📊 {stats['total_files']} files in quantum filesystem")
                files = os_manager.filesystem.list_directory("")
                print(f"   📋 Root directory has {len(files)} items")

            print("\n🎯 Demo complete! Use other modes for interactive operation.")

    except KeyboardInterrupt:
        print("\n🔄 Shutdown requested...")
    except Exception as e:
        print(f"❌ QuantoniumOS error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        os_manager.shutdown()

    return 0


if __name__ == "__main__":
    exit(main())
