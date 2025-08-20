"""
Phase 4: Quantum Cryptography Playground
Interactive quantum cryptography protocols and education
"""

import tkinter as tk
from tkinter import ttk
import random

class QuantumCryptographyPlayground:
    """Quantum Cryptography Interactive Playground"""
    
    def __init__(self, parent=None):
        self.parent = parent
        self.setup_window()
        
    def setup_window(self):
        """Setup the playground window"""
        self.root = tk.Toplevel(self.parent) if self.parent else tk.Tk()
        self.root.title("Quantum Cryptography Playground")
        self.root.geometry("900x700")
        self.root.configure(bg='#2d1b69')
        
        # Header
        header = tk.Label(self.root, text="🔐 Quantum Cryptography Playground", 
                         font=('Arial', 16, 'bold'), 
                         fg='#9c27b0', bg='#2d1b69')
        header.pack(pady=20)
        
        # Protocol selector
        self.setup_protocol_selector()
        
        # Main area
        self.setup_main_area()
        
        # Controls
        self.setup_controls()
        
    def setup_protocol_selector(self):
        """Setup protocol selection"""
        selector_frame = tk.Frame(self.root, bg='#2d1b69')
        selector_frame.pack(pady=10)
        
        tk.Label(selector_frame, text="Select Protocol:", 
                fg='#ffffff', bg='#2d1b69').pack(side=tk.LEFT)
        
        self.protocol_var = tk.StringVar(value="BB84")
        protocols = ["BB84", "E91", "SARG04", "Quantum Teleportation"]
        
        protocol_menu = ttk.Combobox(selector_frame, textvariable=self.protocol_var,
                                   values=protocols, state="readonly")
        protocol_menu.pack(side=tk.LEFT, padx=10)
        
    def setup_main_area(self):
        """Setup main simulation area"""
        main_frame = tk.Frame(self.root, bg='#2d1b69')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Simulation display
        self.sim_text = tk.Text(main_frame, height=20, bg='#1a0d3d', fg='#e1bee7',
                               font=('Consolas', 10))
        self.sim_text.pack(fill=tk.BOTH, expand=True)
        
        # Initial text
        self.sim_text.insert(tk.END, "Welcome to Quantum Cryptography Playground!\n\n")
        self.sim_text.insert(tk.END, "Available protocols:\n")
        self.sim_text.insert(tk.END, "• BB84: Quantum Key Distribution\n")
        self.sim_text.insert(tk.END, "• E91: Entanglement-based QKD\n")
        self.sim_text.insert(tk.END, "• SARG04: Modified BB84 protocol\n")
        self.sim_text.insert(tk.END, "• Quantum Teleportation: State transfer\n\n")
        self.sim_text.insert(tk.END, "Select a protocol and click 'Run Simulation' to begin!\n")
        
    def setup_controls(self):
        """Setup control buttons"""
        controls = tk.Frame(self.root, bg='#2d1b69')
        controls.pack(pady=10)
        
        tk.Button(controls, text="Run Simulation", 
                 command=self.run_simulation,
                 bg='#9c27b0', fg='white', font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=5)
        
        tk.Button(controls, text="Clear Log", 
                 command=self.clear_log,
                 bg='#f44336', fg='white').pack(side=tk.LEFT, padx=5)
        
        tk.Button(controls, text="Educational Mode", 
                 command=self.educational_mode,
                 bg='#4caf50', fg='white').pack(side=tk.LEFT, padx=5)
        
    def run_simulation(self):
        """Run the selected protocol simulation"""
        protocol = self.protocol_var.get()
        self.sim_text.insert(tk.END, f"\n{'='*50}\n")
        self.sim_text.insert(tk.END, f"Running {protocol} Protocol Simulation\n")
        self.sim_text.insert(tk.END, f"{'='*50}\n\n")
        
        if protocol == "BB84":
            self.simulate_bb84()
        elif protocol == "E91":
            self.simulate_e91()
        elif protocol == "SARG04":
            self.simulate_sarg04()
        elif protocol == "Quantum Teleportation":
            self.simulate_teleportation()
            
        self.sim_text.see(tk.END)
        
    def simulate_bb84(self):
        """Simulate BB84 protocol"""
        self.sim_text.insert(tk.END, "1. Alice generates random bits and bases\n")
        
        # Generate random data
        bits = [random.randint(0, 1) for _ in range(8)]
        bases = [random.choice(['rectilinear', 'diagonal']) for _ in range(8)]
        
        self.sim_text.insert(tk.END, f"   Alice's bits: {bits}\n")
        self.sim_text.insert(tk.END, f"   Alice's bases: {bases}\n\n")
        
        self.sim_text.insert(tk.END, "2. Alice sends quantum states to Bob\n")
        self.sim_text.insert(tk.END, "3. Bob randomly chooses measurement bases\n")
        
        bob_bases = [random.choice(['rectilinear', 'diagonal']) for _ in range(8)]
        self.sim_text.insert(tk.END, f"   Bob's bases: {bob_bases}\n\n")
        
        self.sim_text.insert(tk.END, "4. Bob measures and gets results\n")
        self.sim_text.insert(tk.END, "5. Alice and Bob compare bases publicly\n")
        
        # Calculate matching bases
        matching = [i for i in range(8) if bases[i] == bob_bases[i]]
        self.sim_text.insert(tk.END, f"   Matching bases at positions: {matching}\n")
        
        shared_key = [bits[i] for i in matching]
        self.sim_text.insert(tk.END, f"6. Shared key: {shared_key}\n\n")
        
    def simulate_e91(self):
        """Simulate E91 protocol"""
        self.sim_text.insert(tk.END, "1. Entangled photon pairs generated\n")
        self.sim_text.insert(tk.END, "2. Alice and Bob measure with random bases\n")
        self.sim_text.insert(tk.END, "3. Bell inequality test for eavesdropping\n")
        self.sim_text.insert(tk.END, "4. Key extraction from correlated results\n\n")
        
    def simulate_sarg04(self):
        """Simulate SARG04 protocol"""
        self.sim_text.insert(tk.END, "1. Enhanced BB84 with improved security\n")
        self.sim_text.insert(tk.END, "2. Alice announces two non-orthogonal states\n")
        self.sim_text.insert(tk.END, "3. Bob deduces bit value from announcement\n")
        self.sim_text.insert(tk.END, "4. Improved resistance to PNS attacks\n\n")
        
    def simulate_teleportation(self):
        """Simulate quantum teleportation"""
        self.sim_text.insert(tk.END, "1. Alice has unknown quantum state |ψ⟩\n")
        self.sim_text.insert(tk.END, "2. Alice and Bob share entangled pair\n")
        self.sim_text.insert(tk.END, "3. Alice performs Bell measurement\n")
        self.sim_text.insert(tk.END, "4. Alice sends classical bits to Bob\n")
        self.sim_text.insert(tk.END, "5. Bob applies correction and recovers |ψ⟩\n\n")
        
    def clear_log(self):
        """Clear the simulation log"""
        self.sim_text.delete(1.0, tk.END)
        
    def educational_mode(self):
        """Enter educational mode with explanations"""
        self.sim_text.insert(tk.END, "\n🎓 EDUCATIONAL MODE\n")
        self.sim_text.insert(tk.END, "=" * 30 + "\n\n")
        self.sim_text.insert(tk.END, "Quantum Cryptography Principles:\n\n")
        self.sim_text.insert(tk.END, "• Quantum states cannot be cloned\n")
        self.sim_text.insert(tk.END, "• Measurement disturbs quantum systems\n")
        self.sim_text.insert(tk.END, "• Entanglement provides correlations\n")
        self.sim_text.insert(tk.END, "• No-cloning theorem ensures security\n\n")

if __name__ == "__main__":
    app = QuantumCryptographyPlayground()
    app.root.mainloop()
