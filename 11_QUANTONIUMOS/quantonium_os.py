#!/usr/bin/env python3
"""
QuantoniumOS Launcher - Phase 1

Complete 1000-qubit quantum operating system launcher that combines:
- Quantum vertex kernel (1000 qubits in 32x32 grid)
- Patent integration layer (RFT, crypto, quantum engines)
- System monitoring and process management
- Interactive OS shell interface

This is the world's first vertex-based quantum operating system.
"""

import sys
import time
from pathlib import Path

# Import QuantoniumOS components
try:
    from kernel.patent_integration import (QuantoniumOSIntegration,
                                           demonstrate_integration)
    from kernel.quantum_vertex_kernel import (QuantoniumKernel,
                                              demonstrate_quantonium_kernel)
except ImportError as e:
    print(f"❌ Failed to import QuantoniumOS components: {e}")
    # Try alternative import path
    try:
        sys.path.insert(0, str(Path(__file__).parent / "kernel"))
        from patent_integration import (QuantoniumOSIntegration,
                                        demonstrate_integration)
        from quantum_vertex_kernel import (QuantoniumKernel,
                                           demonstrate_quantonium_kernel)

        print("✅ Using alternative import path")
    except ImportError as e2:
        print(f"❌ Alternative import also failed: {e2}")
        sys.exit(1)


class QuantoniumOSShell:
    """
    Interactive shell interface for QuantoniumOS
    """

    def __init__(self, kernel: QuantoniumKernel, integration: QuantoniumOSIntegration):
        self.kernel = kernel
        self.integration = integration
        self.running = True

        self.commands = {
            "help": self.cmd_help,
            "status": self.cmd_status,
            "spawn": self.cmd_spawn_process,
            "gate": self.cmd_apply_gate,
            "evolve": self.cmd_evolve_system,
            "enhance": self.cmd_enhance_vertex,
            "patents": self.cmd_patents_status,
            "vertices": self.cmd_list_vertices,
            "processes": self.cmd_list_processes,
            "shutdown": self.cmd_shutdown,
            "exit": self.cmd_shutdown,
        }

    def cmd_help(self, args):
        """Show available commands"""
        print("\n🌟 QUANTONIUMOS SHELL COMMANDS:")
        print("   help        - Show this help message")
        print("   status      - Show system status")
        print("   spawn <vid> - Spawn process on vertex ID")
        print("   gate <vid> <gate> - Apply quantum gate (H, X, Z)")
        print("   evolve <steps> - Evolve quantum system")
        print("   enhance <vid> - Enhance vertex with patents")
        print("   patents     - Show patent integration status")
        print("   vertices    - List quantum vertices")
        print("   processes   - List active processes")
        print("   shutdown    - Shutdown QuantoniumOS")
        print("   exit        - Exit QuantoniumOS")

    def cmd_status(self, args):
        """Show system status"""
        status = self.kernel.get_system_status()
        print("\n📊 QUANTONIUMOS SYSTEM STATUS:")
        for key, value in status.items():
            print(f"   • {key}: {value}")

    def cmd_spawn_process(self, args):
        """Spawn quantum process"""
        if not args:
            print("❌ Usage: spawn <vertex_id>")
            return

        try:
            vertex_id = int(args[0])
            pid = self.kernel.spawn_quantum_process(vertex_id)
            if pid is not None:
                print(f"✅ Process {pid} spawned on vertex {vertex_id}")
            else:
                print(f"❌ Failed to spawn process on vertex {vertex_id}")
        except ValueError:
            print("❌ Invalid vertex ID")

    def cmd_apply_gate(self, args):
        """Apply quantum gate"""
        if len(args) < 2:
            print("❌ Usage: gate <vertex_id> <gate_type>")
            print("   Gate types: H (Hadamard), X (Pauli-X), Z (Pauli-Z)")
            return

        try:
            vertex_id = int(args[0])
            gate = args[1].upper()

            if self.kernel.apply_quantum_gate(vertex_id, gate):
                print(f"✅ {gate} gate applied to vertex {vertex_id}")
            else:
                print(f"❌ Failed to apply {gate} gate to vertex {vertex_id}")
        except ValueError:
            print("❌ Invalid vertex ID")

    def cmd_evolve_system(self, args):
        """Evolve quantum system"""
        steps = 5
        if args:
            try:
                steps = int(args[0])
            except ValueError:
                print("❌ Invalid step count, using default: 5")

        print(f"🌊 Evolving quantum system ({steps} steps)...")
        self.kernel.evolve_quantum_system(time_steps=steps)

    def cmd_enhance_vertex(self, args):
        """Enhance vertex with patent technologies"""
        if not args:
            print("❌ Usage: enhance <vertex_id>")
            return

        try:
            vertex_id = int(args[0])
            if vertex_id not in self.kernel.vertices:
                print(f"❌ Vertex {vertex_id} does not exist")
                return

            vertex = self.kernel.vertices[vertex_id]
            vertex_state = vertex.alpha + 1j * vertex.beta

            enhanced = self.integration.enhance_vertex_with_patents(
                vertex_state, vertex_id
            )

            print(f"\n🚀 VERTEX {vertex_id} ENHANCEMENT:")
            print(f"   📊 Original: {enhanced['original_state']}")
            print(f"   📊 RFT Enhanced: {enhanced['rft_enhanced']}")
            print(f"   📊 Crypto Enhanced: {enhanced['crypto_enhanced']}")
            print(f"   ✅ Enhancement Applied: {enhanced['enhancement_applied']}")

        except ValueError:
            print("❌ Invalid vertex ID")

    def cmd_patents_status(self, args):
        """Show patent integration status"""
        report = self.integration.get_integration_report()
        print("\n🔗 PATENT INTEGRATION STATUS:")
        for key, value in report.items():
            if key != "patent_modules":
                print(f"   • {key}: {value}")

    def cmd_list_vertices(self, args):
        """List quantum vertices"""
        count = 10  # Default show first 10
        if args:
            try:
                count = int(args[0])
            except ValueError:
                print("❌ Invalid count, showing first 10")

        print(f"\n🔮 QUANTUM VERTICES (showing first {count}):")
        for i, (vid, vertex) in enumerate(self.kernel.vertices.items()):
            if i >= count:
                break
            state_str = f"α={vertex.alpha:.3f}, β={vertex.beta:.3f}"
            proc_count = len([p for p in vertex.processes if p.state == "running"])
            print(f"   • Vertex {vid}: {state_str} | Processes: {proc_count}")

        total = len(self.kernel.vertices)
        if count < total:
            print(f"   ... and {total - count} more vertices")

    def cmd_list_processes(self, args):
        """List active processes"""
        print("\n⚙️  ACTIVE QUANTUM PROCESSES:")
        total_procs = 0

        for vid, vertex in self.kernel.vertices.items():
            active_procs = [p for p in vertex.processes if p.state == "running"]
            for proc in active_procs:
                print(f"   • PID {proc.pid} on Vertex {vid}: Priority {proc.priority}")
                total_procs += 1

        if total_procs == 0:
            print("   No active processes")
        else:
            print(f"   Total: {total_procs} active processes")

    def cmd_shutdown(self, args):
        """Shutdown QuantoniumOS"""
        print("\n🔄 Shutting down QuantoniumOS...")
        self.kernel.shutdown()
        self.running = False
        print("✅ QuantoniumOS shutdown complete")

    def run(self):
        """Run the interactive shell"""
        print("\n" + "=" * 60)
        print("🌟 QUANTONIUMOS INTERACTIVE SHELL")
        print("🎯 1000-Qubit Quantum Vertex Operating System")
        print("Type 'help' for available commands")
        print("=" * 60)

        while self.running:
            try:
                user_input = input("\nQuantoniumOS> ").strip()
                if not user_input:
                    continue

                parts = user_input.split()
                command = parts[0].lower()
                args = parts[1:] if len(parts) > 1 else []

                if command in self.commands:
                    self.commands[command](args)
                else:
                    print(f"❌ Unknown command: {command}")
                    print("Type 'help' for available commands")

            except KeyboardInterrupt:
                print("\n\n🔄 Received interrupt signal...")
                self.cmd_shutdown([])
                break
            except EOFError:
                print("\n\n🔄 EOF received...")
                self.cmd_shutdown([])
                break


class QuantoniumOS:
    """
    Main QuantoniumOS class - 1000-qubit quantum operating system
    """

    def __init__(self):
        self.kernel = None
        self.integration = None
        self.shell = None

        print("🚀 LAUNCHING QUANTONIUMOS - PHASE 1")
        print("🎯 1000-Qubit Quantum Vertex Operating System")
        print("=" * 60)

        self._initialize_os()

    def _initialize_os(self):
        """Initialize all QuantoniumOS components"""
        print("\n🔧 INITIALIZING QUANTONIUMOS COMPONENTS...")

        # Initialize quantum kernel
        print("\n1️⃣ QUANTUM KERNEL INITIALIZATION:")
        self.kernel = QuantoniumKernel()

        # Initialize patent integration
        print("\n2️⃣ PATENT INTEGRATION INITIALIZATION:")
        self.integration = QuantoniumOSIntegration()

        # Initialize shell interface
        print("\n3️⃣ SHELL INTERFACE INITIALIZATION:")
        self.shell = QuantoniumOSShell(self.kernel, self.integration)

        print("\n✅ QUANTONIUMOS INITIALIZATION COMPLETE!")
        self._print_startup_summary()

    def _print_startup_summary(self):
        """Print OS startup summary"""
        status = self.kernel.get_system_status()
        integration_report = self.integration.get_integration_report()

        print("\n" + "=" * 60)
        print("🎯 QUANTONIUMOS STARTUP SUMMARY")
        print("=" * 60)
        print(f"🔮 Quantum Vertices: {status['quantum_vertices']}")
        print(f"🌐 Grid Topology: {status['grid_size']}")
        print(f"🔗 Quantum Connections: {status['quantum_connections']}")
        print(f"📊 Memory Usage: {status['memory_mb']:.2f} MB")
        print(f"⚡ Boot Time: {status['uptime_seconds']:.3f} seconds")
        print(f"🔗 Patent Integrations: {integration_report['total_integrations']}")
        print(f"✅ System Status: {integration_report['status'].upper()}")
        print("=" * 60)

    def run_interactive(self):
        """Run QuantoniumOS in interactive mode"""
        if self.shell:
            self.shell.run()

    def run_demo(self):
        """Run QuantoniumOS demonstration mode"""
        print("\n🎬 RUNNING QUANTONIUMOS DEMONSTRATION...")

        # Run kernel demo
        demonstrate_quantonium_kernel()

        # Run integration demo
        demonstrate_integration()

        print("\n🎯 QUANTONIUMOS DEMONSTRATION COMPLETE!")


def main():
    """Main QuantoniumOS launcher"""
    import sys

    # Check for demo mode
    demo_mode = "--demo" in sys.argv or "-d" in sys.argv

    try:
        # Launch QuantoniumOS
        os = QuantoniumOS()

        if demo_mode:
            # Run demonstration
            os.run_demo()
        else:
            # Run interactive shell
            os.run_interactive()

    except Exception as e:
        print(f"\n❌ QUANTONIUMOS STARTUP FAILED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
