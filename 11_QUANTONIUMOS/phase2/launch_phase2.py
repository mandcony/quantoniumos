#!/usr/bin/env python3
"""
QuantoniumOS Phase 2 - Comprehensive Launcher

Unified launcher for all Phase 2 components:
- Advanced Web GUI Framework
- Real-time 3D Vertex Visualization  
- Interactive Patent Demonstration Suite
- Multi-server coordination
- Performance monitoring
"""

import os
import sys
import time
import threading
import subprocess
from pathlib import Path
import signal
from datetime import datetime

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "11_QUANTONIUMOS" / "kernel"))

try:
    from quantum_vertex_kernel import QuantoniumKernel
    from patent_integration import QuantoniumOSIntegration
    kernel_available = True
except ImportError:
    kernel_available = False

class Phase2Coordinator:
    """Coordinates all Phase 2 components"""
    
    def __init__(self):
        self.components = {}
        self.processes = {}
        self.running = False
        self.kernel = None
        self.integration = None
        
        # Initialize quantum backend
        if kernel_available:
            try:
                print("🔧 Initializing quantum backend...")
                self.kernel = QuantoniumKernel()
                self.integration = QuantoniumOSIntegration()
                print("✅ Quantum backend ready")
            except Exception as e:
                print(f"⚠️ Running without quantum backend: {e}")
        
        self.setup_components()
    
    def setup_components(self):
        """Setup all Phase 2 components"""
        
        # Web GUI Framework
        self.components['web_gui'] = {
            'name': 'Advanced Web GUI Framework',
            'script': str(Path(__file__).parent / "web_gui" / "quantum_web_interface.py"),
            'port': 8080,
            'status': 'ready',
            'description': 'React-style web interface with real-time quantum monitoring'
        }
        
        # 3D Visualization Engine  
        self.components['visualization'] = {
            'name': 'Real-time 3D Vertex Visualization',
            'script': str(Path(__file__).parent / "visualization" / "quantum_3d_engine.py"),
            'port': 8081,
            'status': 'ready',
            'description': 'WebGL-accelerated 3D quantum vertex visualization'
        }
        
        # Patent Demo Suite
        self.components['patent_demos'] = {
            'name': 'Interactive Patent Demonstration Suite',
            'script': str(Path(__file__).parent / "patent_demos" / "patent_demo_suite.py"),
            'port': None,  # CLI-based
            'status': 'ready',
            'description': 'Comprehensive patent implementation demonstrations'
        }
        
        print(f"🎯 Phase 2 components configured: {len(self.components)}")
    
    def start_all_components(self):
        """Start all Phase 2 components"""
        print("\n🚀 STARTING QUANTONIUMOS PHASE 2")
        print("=" * 60)
        print("🌌 Advanced Quantum Operating System Interface")
        print("🔮 Web GUI + 3D Visualization + Patent Demos")
        print("=" * 60)
        
        self.running = True
        
        # Start each component
        for comp_id, component in self.components.items():
            self.start_component(comp_id, component)
            time.sleep(2)  # Stagger startup
        
        # Monitor components
        self.monitor_components()
    
    def start_component(self, comp_id, component):
        """Start individual component"""
        try:
            print(f"\n🔄 Starting {component['name']}...")
            
            if comp_id == 'patent_demos':
                # Patent demos run in CLI mode
                print(f"📋 {component['name']}: Available for CLI interaction")
                print(f"   Run: python {component['script']}")
                component['status'] = 'available'
                return
            
            # Start web-based components
            env = os.environ.copy()
            env['PYTHONPATH'] = str(project_root)
            
            # Use Python directly for better compatibility
            process = subprocess.Popen([
                sys.executable, 
                component['script']
            ], 
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            self.processes[comp_id] = process
            component['status'] = 'starting'
            
            # Give it a moment to start
            time.sleep(3)
            
            if process.poll() is None:
                component['status'] = 'running'
                port_info = f" on port {component['port']}" if component['port'] else ""
                print(f"✅ {component['name']} started{port_info}")
                
                if component['port']:
                    print(f"   🌐 Access at: http://localhost:{component['port']}")
            else:
                component['status'] = 'failed'
                stdout, stderr = process.communicate()
                print(f"❌ {component['name']} failed to start")
                if stderr:
                    print(f"   Error: {stderr[:200]}...")
                    
        except Exception as e:
            component['status'] = 'error'
            print(f"❌ Error starting {component['name']}: {e}")
    
    def monitor_components(self):
        """Monitor component health"""
        print(f"\n📊 PHASE 2 STATUS DASHBOARD")
        print("=" * 40)
        
        while self.running:
            try:
                self.display_status()
                time.sleep(10)  # Update every 10 seconds
            except KeyboardInterrupt:
                break
    
    def display_status(self):
        """Display current status of all components"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        print(f"\n[{timestamp}] 🔍 Component Status:")
        
        for comp_id, component in self.components.items():
            status_icon = self.get_status_icon(component['status'])
            name = component['name']
            status = component['status']
            
            print(f"  {status_icon} {name}: {status}")
            
            # Check process health for running components
            if comp_id in self.processes:
                process = self.processes[comp_id]
                if process.poll() is not None and component['status'] == 'running':
                    component['status'] = 'crashed'
                    print(f"    ⚠️ Process exited with code {process.returncode}")
        
        # Display access information
        print(f"\n🌐 Access URLs:")
        for comp_id, component in self.components.items():
            if component['port'] and component['status'] == 'running':
                print(f"  • {component['name']}: http://localhost:{component['port']}")
        
        # Display quantum backend status
        if self.kernel:
            vertex_count = len(self.kernel.vertices)
            process_count = sum(len([p for p in v.processes if p.state == 'running']) 
                              for v in self.kernel.vertices.values())
            print(f"\n🔮 Quantum Backend: {vertex_count} vertices, {process_count} processes")
    
    def get_status_icon(self, status):
        """Get status icon for component"""
        icons = {
            'ready': '⏸️',
            'starting': '🔄',
            'running': '🟢',
            'available': '📋',
            'failed': '❌',
            'crashed': '💥',
            'error': '🔴'
        }
        return icons.get(status, '❓')
    
    def stop_all_components(self):
        """Stop all components gracefully"""
        print(f"\n🔄 Shutting down Phase 2 components...")
        self.running = False
        
        for comp_id, process in self.processes.items():
            try:
                if process.poll() is None:
                    print(f"   Stopping {self.components[comp_id]['name']}...")
                    process.terminate()
                    
                    # Give it 5 seconds to shut down gracefully
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        print(f"   Force killing {self.components[comp_id]['name']}...")
                        process.kill()
                        
            except Exception as e:
                print(f"   Error stopping {comp_id}: {e}")
        
        print("✅ Phase 2 shutdown complete")
    
    def interactive_menu(self):
        """Interactive menu for Phase 2 control"""
        while self.running:
            print(f"\n🎮 QUANTONIUMOS PHASE 2 CONTROL MENU")
            print("=" * 40)
            print("1. 🌐 Open Web GUI (port 8080)")
            print("2. 🎬 Open 3D Visualization (port 8081)")
            print("3. 🔬 Run Patent Demos (CLI)")
            print("4. 📊 Show Detailed Status")
            print("5. 🔄 Restart Component")
            print("6. 🛑 Stop All Components")
            print("7. ❓ Help")
            print("8. 🚪 Exit")
            
            choice = input("\n🎯 Select option: ").strip()
            
            if choice == '1':
                self.open_web_interface('web_gui')
            elif choice == '2':
                self.open_web_interface('visualization')
            elif choice == '3':
                self.run_patent_demos()
            elif choice == '4':
                self.show_detailed_status()
            elif choice == '5':
                self.restart_component_menu()
            elif choice == '6':
                self.stop_all_components()
                break
            elif choice == '7':
                self.show_help()
            elif choice == '8':
                self.stop_all_components()
                break
            else:
                print("❌ Invalid option")
    
    def open_web_interface(self, comp_id):
        """Open web interface in browser"""
        component = self.components[comp_id]
        
        if component['status'] != 'running':
            print(f"❌ {component['name']} is not running (status: {component['status']})")
            return
        
        port = component['port']
        url = f"http://localhost:{port}"
        
        try:
            import webbrowser
            webbrowser.open(url)
            print(f"🌐 Opening {component['name']} at {url}")
        except:
            print(f"🌐 Please open your browser to: {url}")
    
    def run_patent_demos(self):
        """Run patent demonstrations"""
        script_path = self.components['patent_demos']['script']
        
        print(f"\n🔬 Starting Patent Demonstration Suite...")
        print(f"   Running: {script_path}")
        
        try:
            env = os.environ.copy()
            env['PYTHONPATH'] = str(project_root)
            
            # Run patent demos interactively
            subprocess.run([sys.executable, script_path], env=env)
        except Exception as e:
            print(f"❌ Error running patent demos: {e}")
    
    def show_detailed_status(self):
        """Show detailed status information"""
        print(f"\n📊 DETAILED PHASE 2 STATUS REPORT")
        print("=" * 50)
        
        for comp_id, component in self.components.items():
            print(f"\n🔧 {component['name']}:")
            print(f"   Status: {component['status']}")
            print(f"   Description: {component['description']}")
            
            if component['port']:
                print(f"   Port: {component['port']}")
                print(f"   URL: http://localhost:{component['port']}")
            
            if comp_id in self.processes:
                process = self.processes[comp_id]
                print(f"   Process ID: {process.pid}")
                print(f"   Process Status: {'Running' if process.poll() is None else 'Stopped'}")
        
        # Show quantum backend details
        if self.kernel:
            print(f"\n🔮 Quantum Backend Details:")
            status = self.kernel.get_system_status()
            for key, value in status.items():
                print(f"   {key}: {value}")
    
    def restart_component_menu(self):
        """Menu for restarting components"""
        print(f"\n🔄 RESTART COMPONENT")
        
        components = [(comp_id, comp) for comp_id, comp in self.components.items() 
                     if comp_id in self.processes]
        
        if not components:
            print("❌ No restartable components found")
            return
        
        for i, (comp_id, component) in enumerate(components, 1):
            print(f"  {i}. {component['name']} ({component['status']})")
        
        choice = input("\n🎯 Select component to restart: ").strip()
        
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(components):
                comp_id, component = components[idx]
                self.restart_component(comp_id, component)
            else:
                print("❌ Invalid selection")
        else:
            print("❌ Invalid input")
    
    def restart_component(self, comp_id, component):
        """Restart a specific component"""
        print(f"🔄 Restarting {component['name']}...")
        
        # Stop existing process
        if comp_id in self.processes:
            process = self.processes[comp_id]
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
            del self.processes[comp_id]
        
        # Restart component
        time.sleep(2)
        self.start_component(comp_id, component)
    
    def show_help(self):
        """Show help information"""
        print(f"\n❓ QUANTONIUMOS PHASE 2 HELP")
        print("=" * 40)
        print("🌌 QuantoniumOS Phase 2 provides an advanced quantum computing interface")
        print("   combining web technologies with quantum simulation.")
        print("")
        print("🔧 Components:")
        print("   • Web GUI Framework: Modern React-style interface for quantum operations")
        print("   • 3D Visualization: Real-time WebGL visualization of quantum vertex networks")
        print("   • Patent Demos: Interactive demonstrations of all patent implementations")
        print("")
        print("🌐 Web Interfaces:")
        print("   • Open http://localhost:8080 for the main quantum interface")
        print("   • Open http://localhost:8081 for 3D visualization")
        print("")
        print("🔮 Features:")
        print("   • Real-time quantum state monitoring")
        print("   • Interactive quantum gate operations")
        print("   • 3D quantum vertex network visualization")
        print("   • Patent demonstration suite")
        print("   • Performance analytics")
        print("")
        print("🎯 Usage Tips:")
        print("   • Use the web interfaces for visual interaction")
        print("   • Run patent demos for algorithm demonstrations")
        print("   • Check detailed status for troubleshooting")
        print("   • Components can be restarted individually if needed")


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    print(f"\n🔄 Received shutdown signal {signum}")
    if hasattr(signal_handler, 'coordinator'):
        signal_handler.coordinator.stop_all_components()
    sys.exit(0)


def main():
    """Main Phase 2 launcher"""
    print("🌌 QUANTONIUMOS PHASE 2 LAUNCHER")
    print("=" * 50)
    print("🚀 Advanced Quantum Operating System Interface")
    print("🎯 Loading Phase 2 components...")
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create coordinator
    coordinator = Phase2Coordinator()
    signal_handler.coordinator = coordinator  # Store for signal handler
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("⚠️ Python 3.8+ recommended for optimal performance")
    
    try:
        # Start all components
        coordinator.start_all_components()
        
        print(f"\n🎉 QUANTONIUMOS PHASE 2 READY!")
        print("=" * 40)
        print("🌐 Web GUI: http://localhost:8080")
        print("🎬 3D Visualization: http://localhost:8081")
        print("🔬 Patent Demos: Available via menu")
        print("")
        print("💡 TIP: Open both URLs in your browser for the full experience!")
        print("📱 Use Ctrl+C anytime to access the control menu")
        
        # Run interactive menu
        coordinator.interactive_menu()
        
    except KeyboardInterrupt:
        print(f"\n🔄 User requested shutdown...")
        coordinator.stop_all_components()
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        coordinator.stop_all_components()
        sys.exit(1)
    
    print("\n👋 QuantoniumOS Phase 2 session ended")


if __name__ == "__main__":
    main()
