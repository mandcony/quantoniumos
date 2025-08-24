"""
QuantoniumOS Phase 4: Quantum Cryptography Playground
Interactive environment for quantum cryptography experiments and education
"""

import hashlib
import json
import logging
import random
import threading
import time
import tkinter as tk
from datetime import datetime
from tkinter import messagebox, scrolledtext, ttk
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class QuantumCryptographyPlayground:
    """
    Interactive quantum cryptography playground for education and experimentation
    """

    def __init__(self, parent=None):
        self.logger = logging.getLogger(__name__)

        # Create main window
        if parent:
            self.root = tk.Toplevel(parent)
        else:
            self.root = tk.Tk()

        self.root.title("QuantoniumOS - Quantum Cryptography Playground")
        self.root.geometry("1600x1000")
        self.root.configure(bg="#1e1e1e")

        # Quantum states and protocols
        self.quantum_states = {}
        self.protocol_results = {}
        self.simulation_history = []

        # Cryptographic parameters
        self.key_length = 128
        self.error_rate = 0.1
        self.eavesdropper_present = False

        # Protocol implementations
        self.protocols = {
            "BB84": self.simulate_bb84,
            "B92": self.simulate_b92,
            "E91": self.simulate_e91,
            "SARG04": self.simulate_sarg04,
            "Quantum_Teleportation": self.simulate_quantum_teleportation,
            "Quantum_Key_Distribution": self.simulate_qkd,
            "Post_Quantum_Crypto": self.simulate_post_quantum,
        }

        self.setup_ui()
        self.setup_plots()

        self.logger.info("Quantum Cryptography Playground initialized")

    def setup_ui(self):
        """Setup the user interface"""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Create tabs
        self.setup_protocol_tab()
        self.setup_simulation_tab()
        self.setup_analysis_tab()
        self.setup_education_tab()
        self.setup_tools_tab()

    def setup_protocol_tab(self):
        """Setup protocol selection and configuration tab"""
        protocol_frame = ttk.Frame(self.notebook)
        self.notebook.add(protocol_frame, text="Protocols")

        # Protocol selection
        selection_frame = ttk.LabelFrame(
            protocol_frame, text="Protocol Selection", padding=10
        )
        selection_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(selection_frame, text="Select Protocol:").pack(side=tk.LEFT)
        self.protocol_var = tk.StringVar(value="BB84")
        protocol_combo = ttk.Combobox(
            selection_frame,
            textvariable=self.protocol_var,
            values=list(self.protocols.keys()),
            state="readonly",
            width=20,
        )
        protocol_combo.pack(side=tk.LEFT, padx=(10, 0))
        protocol_combo.bind("<<ComboboxSelected>>", self.on_protocol_selected)

        # Protocol parameters
        params_frame = ttk.LabelFrame(protocol_frame, text="Parameters", padding=10)
        params_frame.pack(fill=tk.X, pady=(0, 10))

        # Key length
        key_frame = ttk.Frame(params_frame)
        key_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(key_frame, text="Key Length:").pack(side=tk.LEFT)
        self.key_length_var = tk.StringVar(value="128")
        ttk.Entry(key_frame, textvariable=self.key_length_var, width=10).pack(
            side=tk.LEFT, padx=(10, 0)
        )

        # Error rate
        error_frame = ttk.Frame(params_frame)
        error_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(error_frame, text="Channel Error Rate:").pack(side=tk.LEFT)
        self.error_rate_var = tk.StringVar(value="0.1")
        ttk.Entry(error_frame, textvariable=self.error_rate_var, width=10).pack(
            side=tk.LEFT, padx=(10, 0)
        )

        # Eavesdropper
        ttk.Checkbutton(
            params_frame, text="Eavesdropper Present", variable=tk.BooleanVar()
        ).pack(anchor=tk.W, pady=(0, 5))

        # Advanced parameters
        advanced_frame = ttk.LabelFrame(params_frame, text="Advanced", padding=5)
        advanced_frame.pack(fill=tk.X, pady=(5, 0))

        # Noise model
        noise_frame = ttk.Frame(advanced_frame)
        noise_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(noise_frame, text="Noise Model:").pack(side=tk.LEFT)
        self.noise_model_var = tk.StringVar(value="Depolarizing")
        noise_combo = ttk.Combobox(
            noise_frame,
            textvariable=self.noise_model_var,
            values=["Depolarizing", "Amplitude Damping", "Phase Damping", "Bit Flip"],
            state="readonly",
            width=15,
        )
        noise_combo.pack(side=tk.LEFT, padx=(10, 0))

        # Protocol description
        desc_frame = ttk.LabelFrame(
            protocol_frame, text="Protocol Description", padding=10
        )
        desc_frame.pack(fill=tk.BOTH, expand=True)

        self.protocol_description = scrolledtext.ScrolledText(
            desc_frame, height=10, bg="#2e2e2e", fg="white", insertbackground="white"
        )
        self.protocol_description.pack(fill=tk.BOTH, expand=True)

        # Update description
        self.update_protocol_description()

    def setup_simulation_tab(self):
        """Setup simulation control tab"""
        sim_frame = ttk.Frame(self.notebook)
        self.notebook.add(sim_frame, text="Simulation")

        # Control panel
        control_frame = ttk.LabelFrame(sim_frame, text="Simulation Control", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        # Simulation buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(
            button_frame, text="Run Single", command=self.run_single_simulation
        ).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(
            button_frame, text="Run Batch", command=self.run_batch_simulation
        ).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(
            button_frame, text="Real-time", command=self.toggle_realtime_simulation
        ).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Stop", command=self.stop_simulation).pack(
            side=tk.LEFT, padx=(0, 10)
        )
        ttk.Button(button_frame, text="Clear Results", command=self.clear_results).pack(
            side=tk.LEFT, padx=(0, 5)
        )

        # Progress and status
        progress_frame = ttk.Frame(control_frame)
        progress_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(progress_frame, text="Progress:").pack(side=tk.LEFT)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            progress_frame, variable=self.progress_var, maximum=100, length=200
        )
        self.progress_bar.pack(side=tk.LEFT, padx=(10, 0))

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(progress_frame, textvariable=self.status_var).pack(
            side=tk.LEFT, padx=(20, 0)
        )

        # Results display
        results_frame = ttk.LabelFrame(sim_frame, text="Simulation Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True)

        # Results tree
        self.results_tree = ttk.Treeview(
            results_frame,
            columns=("Protocol", "Status", "Key Rate", "Error Rate", "Time"),
            show="tree headings",
            height=15,
        )

        self.results_tree.heading("#0", text="Run ID")
        self.results_tree.heading("Protocol", text="Protocol")
        self.results_tree.heading("Status", text="Status")
        self.results_tree.heading("Key Rate", text="Key Rate")
        self.results_tree.heading("Error Rate", text="Error Rate")
        self.results_tree.heading("Time", text="Execution Time")

        self.results_tree.column("#0", width=100)
        self.results_tree.column("Protocol", width=120)
        self.results_tree.column("Status", width=100)
        self.results_tree.column("Key Rate", width=100)
        self.results_tree.column("Error Rate", width=100)
        self.results_tree.column("Time", width=120)

        self.results_tree.pack(fill=tk.BOTH, expand=True)

        # Scrollbar for results
        scrollbar = ttk.Scrollbar(
            results_frame, orient=tk.VERTICAL, command=self.results_tree.yview
        )
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_tree.configure(yscrollcommand=scrollbar.set)

    def setup_analysis_tab(self):
        """Setup analysis and visualization tab"""
        analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(analysis_frame, text="Analysis")

        # Analysis controls
        control_frame = ttk.LabelFrame(
            analysis_frame, text="Analysis Controls", padding=10
        )
        control_frame.pack(fill=tk.X, pady=(0, 10))

        analysis_buttons = ttk.Frame(control_frame)
        analysis_buttons.pack(fill=tk.X)

        ttk.Button(
            analysis_buttons,
            text="Security Analysis",
            command=self.run_security_analysis,
        ).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(
            analysis_buttons,
            text="Performance Analysis",
            command=self.run_performance_analysis,
        ).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(
            analysis_buttons, text="Error Analysis", command=self.run_error_analysis
        ).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(
            analysis_buttons, text="Export Results", command=self.export_results
        ).pack(side=tk.LEFT, padx=(0, 10))

        # Plot area
        self.plot_frame = ttk.Frame(analysis_frame)
        self.plot_frame.pack(fill=tk.BOTH, expand=True)

    def setup_education_tab(self):
        """Setup educational content tab"""
        edu_frame = ttk.Frame(self.notebook)
        self.notebook.add(edu_frame, text="Education")

        # Educational content
        content_frame = ttk.LabelFrame(edu_frame, text="Learning Modules", padding=10)
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Create educational notebook
        edu_notebook = ttk.Notebook(content_frame)
        edu_notebook.pack(fill=tk.BOTH, expand=True)

        # Concepts tab
        concepts_frame = ttk.Frame(edu_notebook)
        edu_notebook.add(concepts_frame, text="Concepts")

        concepts_text = scrolledtext.ScrolledText(
            concepts_frame, bg="#2e2e2e", fg="white", insertbackground="white"
        )
        concepts_text.pack(fill=tk.BOTH, expand=True)
        concepts_text.insert(tk.END, self.get_concepts_content())

        # Tutorials tab
        tutorials_frame = ttk.Frame(edu_notebook)
        edu_notebook.add(tutorials_frame, text="Tutorials")

        tutorials_text = scrolledtext.ScrolledText(
            tutorials_frame, bg="#2e2e2e", fg="white", insertbackground="white"
        )
        tutorials_text.pack(fill=tk.BOTH, expand=True)
        tutorials_text.insert(tk.END, self.get_tutorials_content())

        # Examples tab
        examples_frame = ttk.Frame(edu_notebook)
        edu_notebook.add(examples_frame, text="Examples")

        examples_text = scrolledtext.ScrolledText(
            examples_frame, bg="#2e2e2e", fg="white", insertbackground="white"
        )
        examples_text.pack(fill=tk.BOTH, expand=True)
        examples_text.insert(tk.END, self.get_examples_content())

    def setup_tools_tab(self):
        """Setup cryptographic tools tab"""
        tools_frame = ttk.Frame(self.notebook)
        self.notebook.add(tools_frame, text="Tools")

        # Tools selection
        tools_selection = ttk.LabelFrame(
            tools_frame, text="Cryptographic Tools", padding=10
        )
        tools_selection.pack(fill=tk.X, pady=(0, 10))

        tools_buttons = ttk.Frame(tools_selection)
        tools_buttons.pack(fill=tk.X)

        ttk.Button(
            tools_buttons, text="Random Number Generator", command=self.open_rng_tool
        ).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(
            tools_buttons, text="Hash Calculator", command=self.open_hash_tool
        ).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(
            tools_buttons, text="Key Generator", command=self.open_key_gen_tool
        ).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(
            tools_buttons, text="Entropy Analyzer", command=self.open_entropy_tool
        ).pack(side=tk.LEFT, padx=(0, 5))

        # Tool workspace
        self.tools_workspace = ttk.Frame(tools_frame)
        self.tools_workspace.pack(fill=tk.BOTH, expand=True)

    def setup_plots(self):
        """Setup matplotlib plots for analysis"""
        # This will be called when analysis tab is selected
        pass

    def on_protocol_selected(self, event=None):
        """Handle protocol selection change"""
        self.update_protocol_description()

    def update_protocol_description(self):
        """Update protocol description text"""
        protocol = self.protocol_var.get()
        descriptions = {
            "BB84": """
BB84 Protocol (Bennett-Brassard 1984)

The BB84 protocol is the first quantum key distribution protocol and remains one of the most important.

Key Features:
- Uses four quantum states: |0⟩, |1⟩, |+⟩, |-⟩
- Two conjugate bases: rectilinear (Z) and diagonal (X)
- Security based on no-cloning theorem
- Can detect eavesdropping through increased error rate

Protocol Steps:
1. Alice randomly chooses bits and bases
2. Alice prepares quantum states according to her choices
3. Bob randomly chooses measurement bases
4. Alice and Bob publicly compare their basis choices
5. They keep bits where bases matched
6. Error checking reveals eavesdropping
7. Privacy amplification produces final key

Security: Information-theoretic security proven under ideal conditions.
            """,
            "B92": """
B92 Protocol (Bennett 1992)

The B92 protocol uses only two non-orthogonal states, making it simpler than BB84.

Key Features:
- Uses only two states: |0⟩ and |+⟩
- Single measurement basis
- More efficient than BB84 in terms of state preparation
- Higher error rate tolerance

Protocol Steps:
1. Alice randomly chooses between |0⟩ and |+⟩ states
2. Bob measures all states in a specific basis
3. Bob announces which measurements gave no result
4. Alice and Bob correlate their data
5. Error correction and privacy amplification

Security: Secure against individual attacks, requires careful implementation.
            """,
            "E91": """
E91 Protocol (Ekert 1991)

The E91 protocol uses quantum entanglement for key distribution.

Key Features:
- Based on EPR pairs (entangled photons)
- Uses Bell's inequality violations for security
- No need to trust the source
- Intrinsic randomness from quantum measurements

Protocol Steps:
1. Source generates entangled photon pairs
2. Alice and Bob receive one photon each
3. They measure in randomly chosen bases
4. They test Bell's inequality on subset of data
5. Violation confirms security
6. Remaining data forms the key

Security: Security based on fundamental quantum mechanics principles.
            """,
            "SARG04": """
SARG04 Protocol (Scarani-Acin-Ribordy-Gisin 2004)

SARG04 is a modification of BB84 with improved security properties.

Key Features:
- Same four states as BB84
- Different information reconciliation
- Better performance against photon-number-splitting attacks
- Optimized for practical implementations

Protocol Steps:
1. Similar to BB84 for state preparation and measurement
2. Alice announces two possible states for each bit
3. Bob determines which state was sent
4. Error correction and privacy amplification
5. Improved security analysis

Security: Enhanced security against realistic attack scenarios.
            """,
        }

        description = descriptions.get(protocol, "Protocol description not available.")

        self.protocol_description.delete(1.0, tk.END)
        self.protocol_description.insert(tk.END, description)

    def run_single_simulation(self):
        """Run a single protocol simulation"""
        try:
            protocol = self.protocol_var.get()
            self.status_var.set(f"Running {protocol} simulation...")

            # Update parameters
            self.key_length = int(self.key_length_var.get())
            self.error_rate = float(self.error_rate_var.get())

            # Run simulation in thread
            threading.Thread(
                target=self._run_simulation_thread, args=(protocol,), daemon=True
            ).start()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to start simulation: {e}")
            self.logger.error(f"Simulation error: {e}")

    def run_batch_simulation(self):
        """Run batch simulations for comparison"""
        try:
            protocols = list(self.protocols.keys())
            self.status_var.set("Running batch simulations...")

            threading.Thread(
                target=self._run_batch_thread, args=(protocols,), daemon=True
            ).start()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to start batch simulation: {e}")

    def toggle_realtime_simulation(self):
        """Toggle real-time simulation mode"""
        # Implementation for real-time simulation
        messagebox.showinfo("Info", "Real-time simulation not yet implemented")

    def stop_simulation(self):
        """Stop running simulation"""
        self.status_var.set("Simulation stopped")

    def clear_results(self):
        """Clear all simulation results"""
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        self.simulation_history.clear()
        self.status_var.set("Results cleared")

    def _run_simulation_thread(self, protocol):
        """Run simulation in background thread"""
        try:
            start_time = time.time()

            # Execute protocol simulation
            if protocol in self.protocols:
                result = self.protocols[protocol]()

                execution_time = time.time() - start_time

                # Add result to tree
                run_id = f"Run_{len(self.simulation_history) + 1}"
                self.root.after(
                    0,
                    self._add_result_to_tree,
                    run_id,
                    protocol,
                    result,
                    execution_time,
                )

                # Store in history
                self.simulation_history.append(
                    {
                        "run_id": run_id,
                        "protocol": protocol,
                        "result": result,
                        "execution_time": execution_time,
                        "timestamp": datetime.now().isoformat(),
                    }
                )

                self.root.after(
                    0, lambda: self.status_var.set(f"{protocol} simulation completed")
                )
            else:
                self.root.after(
                    0, lambda: self.status_var.set(f"Unknown protocol: {protocol}")
                )

        except Exception as e:
            self.root.after(0, lambda: self.status_var.set(f"Simulation failed: {e}"))
            self.logger.error(f"Simulation thread error: {e}")

    def _run_batch_thread(self, protocols):
        """Run batch simulations in background thread"""
        try:
            total = len(protocols)
            for i, protocol in enumerate(protocols):
                if protocol in self.protocols:
                    result = self.protocols[protocol]()

                    run_id = f"Batch_{i+1}"
                    self.root.after(
                        0, self._add_result_to_tree, run_id, protocol, result, 0
                    )

                # Update progress
                progress = (i + 1) / total * 100
                self.root.after(0, lambda p=progress: self.progress_var.set(p))

            self.root.after(
                0, lambda: self.status_var.set("Batch simulation completed")
            )

        except Exception as e:
            self.root.after(
                0, lambda: self.status_var.set(f"Batch simulation failed: {e}")
            )

    def _add_result_to_tree(self, run_id, protocol, result, execution_time):
        """Add simulation result to tree view"""
        self.results_tree.insert(
            "",
            tk.END,
            text=run_id,
            values=(
                protocol,
                result.get("status", "Unknown"),
                f"{result.get('key_rate', 0):.3f}",
                f"{result.get('error_rate', 0):.3f}",
                f"{execution_time:.3f}s",
            ),
        )

    # Protocol simulation methods
    def simulate_bb84(self):
        """Simulate BB84 protocol"""
        n_bits = self.key_length * 2  # Extra bits for sifting

        # Alice's random choices
        alice_bits = np.random.randint(0, 2, n_bits)
        alice_bases = np.random.randint(0, 2, n_bits)  # 0: Z basis, 1: X basis

        # Bob's random basis choices
        bob_bases = np.random.randint(0, 2, n_bits)

        # Quantum channel with errors
        received_bits = alice_bits.copy()
        for i in range(n_bits):
            if np.random.random() < self.error_rate:
                received_bits[i] = 1 - received_bits[i]  # Bit flip

        # Basis comparison and sifting
        matching_bases = alice_bases == bob_bases
        sifted_key_alice = alice_bits[matching_bases]
        sifted_key_bob = received_bits[matching_bases]

        # Error rate calculation
        if len(sifted_key_alice) > 0:
            bit_errors = np.sum(sifted_key_alice != sifted_key_bob)
            qber = bit_errors / len(sifted_key_alice)
        else:
            qber = 1.0

        # Key rate (simplified)
        if qber < 0.11:  # Below security threshold
            key_rate = len(sifted_key_alice) / n_bits * (1 - 2 * qber)
            status = "Success"
        else:
            key_rate = 0
            status = "Failed - High Error Rate"

        return {
            "status": status,
            "key_rate": max(0, key_rate),
            "error_rate": qber,
            "raw_key_length": len(sifted_key_alice),
            "sifting_efficiency": len(sifted_key_alice) / n_bits,
        }

    def simulate_b92(self):
        """Simulate B92 protocol"""
        n_bits = self.key_length * 3  # More bits needed for B92

        # Alice's random bit choices (0 or 1)
        alice_bits = np.random.randint(0, 2, n_bits)

        # Bob's measurements (some bits lost due to inconclusive results)
        bob_detections = []
        bob_bits = []

        for bit in alice_bits:
            if np.random.random() < 0.5:  # 50% detection rate
                detected_bit = bit
                if np.random.random() < self.error_rate:
                    detected_bit = 1 - detected_bit
                bob_detections.append(True)
                bob_bits.append(detected_bit)
            else:
                bob_detections.append(False)
                bob_bits.append(0)  # Placeholder

        # Keep only detected bits
        alice_final = [
            alice_bits[i] for i in range(len(alice_bits)) if bob_detections[i]
        ]
        bob_final = [bob_bits[i] for i in range(len(bob_bits)) if bob_detections[i]]

        # Calculate error rate
        if len(alice_final) > 0:
            errors = sum(a != b for a, b in zip(alice_final, bob_final))
            qber = errors / len(alice_final)
        else:
            qber = 1.0

        # Key rate
        if qber < 0.146:  # B92 threshold
            key_rate = len(alice_final) / n_bits * (1 - 2 * qber)
            status = "Success"
        else:
            key_rate = 0
            status = "Failed - High Error Rate"

        return {
            "status": status,
            "key_rate": max(0, key_rate),
            "error_rate": qber,
            "raw_key_length": len(alice_final),
            "detection_efficiency": len(alice_final) / n_bits,
        }

    def simulate_e91(self):
        """Simulate E91 protocol using entanglement"""
        n_pairs = self.key_length * 2

        # Generate entangled pairs (simplified simulation)
        alice_results = []
        bob_results = []
        alice_bases = []
        bob_bases = []

        for _ in range(n_pairs):
            # Random basis choices
            alice_basis = np.random.randint(
                0, 3
            )  # 0, 1, 2 for three measurement angles
            bob_basis = np.random.randint(0, 3)

            alice_bases.append(alice_basis)
            bob_bases.append(bob_basis)

            # Simulate correlated measurements
            if alice_basis == bob_basis:
                # Perfect correlation for same basis
                result = np.random.randint(0, 2)
                alice_results.append(result)
                bob_results.append(result)
            else:
                # Anti-correlation with quantum mechanics prediction
                alice_result = np.random.randint(0, 2)
                # Add quantum correlation based on basis difference
                correlation_prob = np.cos(np.pi * abs(alice_basis - bob_basis) / 6) ** 2
                if np.random.random() < correlation_prob:
                    bob_result = alice_result
                else:
                    bob_result = 1 - alice_result

                alice_results.append(alice_result)
                bob_results.append(bob_result)

        # Keep only matching bases for key generation
        key_bits_alice = [
            alice_results[i]
            for i in range(len(alice_results))
            if alice_bases[i] == bob_bases[i]
        ]
        key_bits_bob = [
            bob_results[i]
            for i in range(len(bob_results))
            if alice_bases[i] == bob_bases[i]
        ]

        # Calculate error rate
        if len(key_bits_alice) > 0:
            errors = sum(a != b for a, b in zip(key_bits_alice, key_bits_bob))
            qber = errors / len(key_bits_alice)
        else:
            qber = 1.0

        # Key rate
        if qber < 0.11:
            key_rate = len(key_bits_alice) / n_pairs * (1 - 2 * qber)
            status = "Success"
        else:
            key_rate = 0
            status = "Failed - High Error Rate"

        return {
            "status": status,
            "key_rate": max(0, key_rate),
            "error_rate": qber,
            "raw_key_length": len(key_bits_alice),
            "basis_matching_rate": len(key_bits_alice) / n_pairs,
        }

    def simulate_sarg04(self):
        """Simulate SARG04 protocol"""
        # Similar to BB84 but with different information reconciliation
        result = self.simulate_bb84()

        # SARG04 has slightly different security threshold
        if result["error_rate"] < 0.13:  # SARG04 threshold
            result["status"] = "Success"
            result["key_rate"] = max(0, result["key_rate"] * 0.9)  # Slightly lower rate
        else:
            result["status"] = "Failed - High Error Rate"
            result["key_rate"] = 0

        return result

    def simulate_quantum_teleportation(self):
        """Simulate quantum teleportation protocol"""
        # Simplified simulation
        success_rate = 1 - self.error_rate

        if success_rate > 0.8:
            status = "Success"
        else:
            status = "Failed - High Error Rate"

        return {
            "status": status,
            "key_rate": 0,  # Not applicable for teleportation
            "error_rate": self.error_rate,
            "fidelity": success_rate,
            "success_probability": success_rate,
        }

    def simulate_qkd(self):
        """Simulate general QKD protocol"""
        # Combination of multiple protocols
        bb84_result = self.simulate_bb84()
        b92_result = self.simulate_b92()

        # Take average performance
        avg_key_rate = (bb84_result["key_rate"] + b92_result["key_rate"]) / 2
        avg_error_rate = (bb84_result["error_rate"] + b92_result["error_rate"]) / 2

        if avg_error_rate < 0.11:
            status = "Success"
        else:
            status = "Failed - High Error Rate"

        return {
            "status": status,
            "key_rate": avg_key_rate,
            "error_rate": avg_error_rate,
            "protocol_type": "Hybrid QKD",
        }

    def simulate_post_quantum(self):
        """Simulate post-quantum cryptography"""
        # Classical post-quantum simulation
        lattice_security = np.random.uniform(0.8, 1.0)
        hash_security = np.random.uniform(0.9, 1.0)

        overall_security = min(lattice_security, hash_security)

        if overall_security > 0.85:
            status = "Success"
        else:
            status = "Failed - Low Security"

        return {
            "status": status,
            "key_rate": 1.0,  # Classical protocols don't have quantum key rate limitations
            "error_rate": 1 - overall_security,
            "security_level": overall_security,
            "algorithm_type": "Lattice-based",
        }

    # Analysis methods
    def run_security_analysis(self):
        """Run security analysis on simulation results"""
        if not self.simulation_history:
            messagebox.showwarning("Warning", "No simulation results available")
            return

        # Create analysis window
        analysis_window = tk.Toplevel(self.root)
        analysis_window.title("Security Analysis")
        analysis_window.geometry("800x600")

        # Perform analysis
        security_data = []
        for result in self.simulation_history:
            protocol = result["protocol"]
            error_rate = result["result"].get("error_rate", 0)
            key_rate = result["result"].get("key_rate", 0)

            # Calculate security metrics
            if protocol == "BB84":
                security_threshold = 0.11
            elif protocol == "B92":
                security_threshold = 0.146
            elif protocol == "SARG04":
                security_threshold = 0.13
            else:
                security_threshold = 0.11

            security_margin = security_threshold - error_rate
            security_data.append(
                {
                    "protocol": protocol,
                    "error_rate": error_rate,
                    "key_rate": key_rate,
                    "security_margin": security_margin,
                    "secure": error_rate < security_threshold,
                }
            )

        # Display analysis results
        analysis_text = scrolledtext.ScrolledText(
            analysis_window, bg="#2e2e2e", fg="white", insertbackground="white"
        )
        analysis_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        analysis_report = "SECURITY ANALYSIS REPORT\n"
        analysis_report += "=" * 50 + "\n\n"

        secure_count = sum(1 for data in security_data if data["secure"])
        total_count = len(security_data)

        analysis_report += f"Total Simulations: {total_count}\n"
        analysis_report += f"Secure Protocols: {secure_count}\n"
        analysis_report += f"Security Rate: {secure_count/total_count*100:.1f}%\n\n"

        analysis_report += "PROTOCOL DETAILS:\n"
        analysis_report += "-" * 30 + "\n"

        for data in security_data:
            analysis_report += f"Protocol: {data['protocol']}\n"
            analysis_report += f"  Error Rate: {data['error_rate']:.4f}\n"
            analysis_report += f"  Key Rate: {data['key_rate']:.4f}\n"
            analysis_report += f"  Security Margin: {data['security_margin']:.4f}\n"
            analysis_report += (
                f"  Status: {'SECURE' if data['secure'] else 'INSECURE'}\n\n"
            )

        analysis_text.insert(tk.END, analysis_report)

    def run_performance_analysis(self):
        """Run performance analysis"""
        if not self.simulation_history:
            messagebox.showwarning("Warning", "No simulation results available")
            return

        messagebox.showinfo(
            "Info", "Performance analysis visualization not yet implemented"
        )

    def run_error_analysis(self):
        """Run error analysis"""
        if not self.simulation_history:
            messagebox.showwarning("Warning", "No simulation results available")
            return

        messagebox.showinfo("Info", "Error analysis visualization not yet implemented")

    def export_results(self):
        """Export simulation results"""
        if not self.simulation_history:
            messagebox.showwarning("Warning", "No simulation results to export")
            return

        try:
            from tkinter import filedialog

            filename = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            )

            if filename:
                with open(filename, "w") as f:
                    json.dump(self.simulation_history, f, indent=2)
                messagebox.showinfo("Success", f"Results exported to {filename}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to export results: {e}")

    # Tool methods
    def open_rng_tool(self):
        """Open random number generator tool"""
        messagebox.showinfo("Info", "RNG tool not yet implemented")

    def open_hash_tool(self):
        """Open hash calculator tool"""
        messagebox.showinfo("Info", "Hash tool not yet implemented")

    def open_key_gen_tool(self):
        """Open key generator tool"""
        messagebox.showinfo("Info", "Key generator tool not yet implemented")

    def open_entropy_tool(self):
        """Open entropy analyzer tool"""
        messagebox.showinfo("Info", "Entropy analyzer tool not yet implemented")

    # Educational content
    def get_concepts_content(self):
        """Get quantum cryptography concepts content"""
        return """
QUANTUM CRYPTOGRAPHY CONCEPTS
============================

1. QUANTUM KEY DISTRIBUTION (QKD)
   - Uses quantum mechanics for secure key exchange
   - Security based on fundamental physics
   - No-cloning theorem prevents perfect copying
   - Measurement disturbs quantum states

2. QUANTUM STATES
   - Qubits can be in superposition
   - Orthogonal states: |0⟩ and |1⟩ 
   - Non-orthogonal states: |+⟩ and |-⟩
   - Entangled states show quantum correlations

3. SECURITY PRINCIPLES
   - Information-theoretic security
   - Eavesdropping detection through increased errors
   - Privacy amplification removes partial information
   - Error correction handles channel noise

4. PRACTICAL CONSIDERATIONS
   - Channel loss and noise
   - Detector efficiency
   - Source imperfections
   - Implementation vulnerabilities

5. POST-QUANTUM CRYPTOGRAPHY
   - Classical algorithms secure against quantum attacks
   - Lattice-based cryptography
   - Hash-based signatures
   - Multivariate cryptography
        """

    def get_tutorials_content(self):
        """Get tutorials content"""
        return """
QUANTUM CRYPTOGRAPHY TUTORIALS
==============================

TUTORIAL 1: BB84 PROTOCOL
1. Understanding the Protocol
   - Alice prepares random qubits in random bases
   - Bob measures in random bases
   - Public basis comparison reveals matching cases
   - Shared key formed from matching measurements

2. Hands-on Exercise
   - Use the simulation to run BB84
   - Vary error rates and observe security
   - Compare with and without eavesdropping

TUTORIAL 2: ENTANGLEMENT-BASED QKD
1. E91 Protocol
   - Uses entangled photon pairs
   - Bell inequality tests for security
   - No need to trust the source

2. Exercise
   - Simulate E91 protocol
   - Observe Bell inequality violations
   - Compare with prepare-and-measure protocols

TUTORIAL 3: SECURITY ANALYSIS
1. Error Rate Thresholds
   - Different protocols have different limits
   - QBER vs. key rate tradeoffs
   - Security proofs and assumptions

2. Exercise
   - Run security analysis tool
   - Test different attack scenarios
   - Understand security margins
        """

    def get_examples_content(self):
        """Get examples content"""
        return """
PRACTICAL EXAMPLES
==================

EXAMPLE 1: BANK-TO-BANK COMMUNICATION
Scenario: Two banks need to exchange sensitive financial data
Solution: Implement QKD for symmetric key generation
- Use BB84 for high-security key exchange
- Error correction for noisy channels
- Privacy amplification for perfect secrecy

EXAMPLE 2: GOVERNMENT COMMUNICATIONS
Scenario: Secure communications between government facilities
Solution: Long-distance QKD with trusted repeaters
- E91 protocol for source-independent security
- Quantum repeaters for long distances
- Integration with existing infrastructure

EXAMPLE 3: QUANTUM INTERNET
Scenario: Future quantum network infrastructure
Solution: Hybrid quantum-classical protocols
- Quantum key distribution for initial security
- Classical post-quantum crypto for bulk data
- Quantum digital signatures for authentication

EXAMPLE 4: IOT SECURITY
Scenario: Securing Internet of Things devices
Solution: Lightweight quantum-inspired protocols
- Quantum random number generation
- Post-quantum cryptographic algorithms
- Quantum-enhanced authentication
        """

    def run(self):
        """Run the quantum cryptography playground"""
        self.root.mainloop()


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    playground = QuantumCryptographyPlayground()
    playground.run()
