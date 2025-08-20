#!/usr/bin/env python3
"""
Clean Crypto Playground with Dual Input Fields
Enhanced QuantoniumOS Cryptography Interface
"""

import tkinter as tk
from tkinter import ttk
import time
import sys
import os

# Add the crypto module to path
sys.path.append(os.path.join(os.path.dirname(__file__), '06_CRYPTOGRAPHY'))

def create_crypto_playground():
    """Create enhanced crypto playground with dual inputs"""
    
    root = tk.Tk()
    root.title("QuantoniumOS - Enhanced Crypto Playground")
    root.geometry("1200x800")
    root.configure(bg="#0a0a0a")
    
    # Header
    header = tk.Label(root, text="Enhanced Resonance Cryptography Playground", 
                     font=("Arial", 20, "bold"), 
                     bg="#0a0a0a", fg="#00ff00")
    header.pack(pady=10)
    
    # Main container
    main_frame = tk.Frame(root, bg="#0a0a0a")
    main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
    
    # Create notebook for tabbed interface
    notebook = ttk.Notebook(main_frame)
    notebook.pack(fill=tk.BOTH, expand=True)
    
    # Encrypt/Decrypt Tab
    encrypt_frame = ttk.Frame(notebook)
    notebook.add(encrypt_frame, text="🔐 Encrypt & Decrypt")
    
    # Input area - Dual inputs for better UX
    input_frame = ttk.LabelFrame(encrypt_frame, text="Input Data", padding=10)
    input_frame.pack(fill=tk.X, pady=5)
    
    # Plaintext input
    tk.Label(input_frame, text="Plaintext Message:").grid(row=0, column=0, sticky="w")
    message_entry = tk.Entry(input_frame, width=80, font=("Courier", 10))
    message_entry.grid(row=0, column=1, padx=10, sticky="ew")
    message_entry.insert(0, "Secret quantum message using resonance encryption")
    
    # Key input
    tk.Label(input_frame, text="Encryption Key:").grid(row=1, column=0, sticky="w", pady=5)
    key_entry = tk.Entry(input_frame, width=80, font=("Courier", 10))
    key_entry.grid(row=1, column=1, padx=10, pady=5, sticky="ew")
    key_entry.insert(0, "resonance_quantum_key_2024")
    
    # Encrypted data input (for decryption testing)
    tk.Label(input_frame, text="Encrypted Data:").grid(row=2, column=0, sticky="w")
    encrypted_text = tk.Text(input_frame, height=4, width=80, font=("Courier", 8))
    encrypted_text.grid(row=2, column=1, padx=10, pady=5, sticky="ew")
    
    input_frame.grid_columnconfigure(1, weight=1)
    
    # Control buttons
    button_frame = ttk.Frame(encrypt_frame)
    button_frame.pack(fill=tk.X, pady=10)
    
    # Results area
    result_text = tk.Text(encrypt_frame, height=15, bg="#1a1a1a", fg="#00ff00", 
                         font=("Courier", 10), wrap=tk.WORD)
    result_text.pack(fill=tk.BOTH, expand=True, pady=10)
    
    def encrypt_message():
        """Encrypt message and populate encrypted data field"""
        try:
            message = message_entry.get()
            key = key_entry.get()
            
            if not message or not key:
                result_text.delete(1.0, tk.END)
                result_text.insert(tk.END, "❌ Please provide both message and key")
                return
            
            result_text.delete(1.0, tk.END)
            result_text.insert(tk.END, "ENCRYPTION OPERATION\n")
            result_text.insert(tk.END, "=" * 25 + "\n\n")
            
            result_text.insert(tk.END, f"📝 Original: {message}\n")
            result_text.insert(tk.END, f"🔑 Key: {key}\n\n")
            
            # Use the TRUE RFT RESONANCE ENCRYPTION
            try:
                import sys
                sys.path.append('06_CRYPTOGRAPHY')
                from resonance_encryption import resonance_encrypt
                
                start_time = time.time()
                encrypted = resonance_encrypt(message, key)
                encrypt_time = time.time() - start_time
                
                # Populate encrypted data field
                encrypted_text.delete(1.0, tk.END)
                encrypted_text.insert(1.0, encrypted)
                
                result_text.insert(tk.END, f"🔒 TRUE RFT ENCRYPTED: {encrypted[:50]}{'...' if len(encrypted) > 50 else ''}\n")
                result_text.insert(tk.END, f"⏱️ Time: {encrypt_time:.6f} seconds\n")
                result_text.insert(tk.END, f"📏 Size: {len(encrypted)} bytes\n")
                result_text.insert(tk.END, f"🌊 Engine: Novel RFT Resonance Cipher\n")
                result_text.insert(tk.END, f"🔐 Wave-HMAC: Integrated\n\n")
                result_text.insert(tk.END, "✅ TRUE RFT encryption completed - data ready for decryption\n")
                
            except ImportError as e:
                result_text.insert(tk.END, f"❌ Failed to import TRUE RFT cipher: {e}\n")
                # Fallback simple encryption for demo
                import base64
                encoded = base64.b64encode((message + key).encode()).decode()
                encrypted_text.delete(1.0, tk.END)
                encrypted_text.insert(1.0, encoded)
                
                result_text.insert(tk.END, f"🔒 Demo Encrypted: {encoded[:50]}{'...' if len(encoded) > 50 else ''}\n")
                result_text.insert(tk.END, "⚠️ Using demo encryption - TRUE RFT cipher not available\n")
                
        except Exception as e:
            result_text.delete(1.0, tk.END)
            result_text.insert(tk.END, f"❌ Encryption error: {str(e)}")
    
    def decrypt_message():
        """Decrypt data from encrypted data field"""
        try:
            encrypted_data = encrypted_text.get(1.0, tk.END).strip()
            key = key_entry.get()
            
            if not encrypted_data or not key:
                result_text.delete(1.0, tk.END)
                result_text.insert(tk.END, "❌ Please provide both encrypted data and key")
                return
            
            result_text.delete(1.0, tk.END)
            result_text.insert(tk.END, "DECRYPTION OPERATION\n")
            result_text.insert(tk.END, "=" * 25 + "\n\n")
            
            result_text.insert(tk.END, f"🔐 Encrypted: {encrypted_data[:50]}{'...' if len(encrypted_data) > 50 else ''}\n")
            result_text.insert(tk.END, f"🔑 Key: {key}\n\n")
            
            # Use the TRUE RFT RESONANCE DECRYPTION
            try:
                import sys
                sys.path.append('06_CRYPTOGRAPHY')
                from resonance_encryption import resonance_decrypt
                
                start_time = time.time()
                decrypted = resonance_decrypt(encrypted_data, key)
                decrypt_time = time.time() - start_time
                
                result_text.insert(tk.END, f"🔓 TRUE RFT DECRYPTED: {decrypted}\n")
                result_text.insert(tk.END, f"⏱️ Time: {decrypt_time:.6f} seconds\n")
                result_text.insert(tk.END, f"🌊 Engine: Novel RFT Resonance Cipher\n")
                result_text.insert(tk.END, f"🔐 Wave-HMAC: Verified\n\n")
                result_text.insert(tk.END, "✅ TRUE RFT decryption completed successfully\n")
                
            except ImportError:
                result_text.insert(tk.END, "❌ TRUE RFT cipher not available for decryption\n")
                # Fallback demo decryption
                import base64
                try:
                    decoded = base64.b64decode(encrypted_data.encode()).decode()
                    result_text.insert(tk.END, f"🔓 Demo Decrypted: {decoded}\n")
                    result_text.insert(tk.END, "⚠️ Using demo decryption\n")
                except:
                    result_text.insert(tk.END, "❌ Demo decryption failed - invalid data\n")
                
        except Exception as e:
            result_text.delete(1.0, tk.END)
            result_text.insert(tk.END, f"❌ Decryption error: {str(e)}\n")
            result_text.insert(tk.END, "This could indicate:\n")
            result_text.insert(tk.END, "• Wrong decryption key\n")
            result_text.insert(tk.END, "• Corrupted encrypted data\n")
            result_text.insert(tk.END, "• HMAC verification failure\n")
    
    def full_cycle_test():
        """Complete encrypt→decrypt cycle test"""
        try:
            message = message_entry.get()
            key = key_entry.get()
            
            result_text.delete(1.0, tk.END)
            result_text.insert(tk.END, "FULL ENCRYPTION CYCLE TEST\n")
            result_text.insert(tk.END, "=" * 35 + "\n\n")
            
            result_text.insert(tk.END, f"📝 Original: {message}\n")
            result_text.insert(tk.END, f"🔑 Key: {key}\n\n")
            
            # Use TRUE RFT RESONANCE CIPHER for full cycle
            try:
                import sys
                sys.path.append('06_CRYPTOGRAPHY')
                from resonance_encryption import resonance_encrypt, resonance_decrypt
                
                # Encrypt with TRUE RFT
                start_time = time.time()
                encrypted = resonance_encrypt(message, key)
                encrypt_time = time.time() - start_time
                
                result_text.insert(tk.END, f"🔒 TRUE RFT ENCRYPTED: {encrypted[:60]}{'...' if len(encrypted) > 60 else ''}\n")
                result_text.insert(tk.END, f"⏱️ Encrypt Time: {encrypt_time:.6f}s\n\n")
                
                # Decrypt with TRUE RFT
                start_time = time.time()
                decrypted = resonance_decrypt(encrypted, key)
                decrypt_time = time.time() - start_time
                
                result_text.insert(tk.END, f"🔓 TRUE RFT DECRYPTED: {decrypted}\n")
                result_text.insert(tk.END, f"⏱️ Decrypt Time: {decrypt_time:.6f}s\n\n")
                
                # Verify
                success = (message == decrypted)
                result_text.insert(tk.END, f"🧪 Verification: {'✅ SUCCESS' if success else '❌ FAILED'}\n")
                
                if success:
                    result_text.insert(tk.END, "🎯 Round-trip integrity: PERFECT\n")
                    result_text.insert(tk.END, f"⚡ Total Time: {encrypt_time + decrypt_time:.6f}s\n")
                    result_text.insert(tk.END, "\n🌊 Engine: TRUE NOVEL RFT RESONANCE CIPHER\n")
                    result_text.insert(tk.END, "🔐 Wave-HMAC: Authenticated\n")
                    result_text.insert(tk.END, "🧬 Quantum-Enhanced Waveform Generation\n")
                
            except ImportError as e:
                result_text.insert(tk.END, f"❌ TRUE RFT cipher not available: {e}\n")
                result_text.insert(tk.END, "Using demo mode for testing\n")
                
        except Exception as e:
            result_text.delete(1.0, tk.END)
            result_text.insert(tk.END, f"❌ Cycle test error: {str(e)}")
    
    # Create buttons
    tk.Button(button_frame, text="🔒 Encrypt Message", 
             command=encrypt_message,
             bg="#2a5a2a", fg="#ffffff", font=("Arial", 11)).pack(side=tk.LEFT, padx=5)
    tk.Button(button_frame, text="🔓 Decrypt Data", 
             command=decrypt_message,
             bg="#5a2a2a", fg="#ffffff", font=("Arial", 11)).pack(side=tk.LEFT, padx=5)
    tk.Button(button_frame, text="🔄 Full Cycle Test", 
             command=full_cycle_test,
             bg="#2a2a5a", fg="#ffffff", font=("Arial", 11)).pack(side=tk.LEFT, padx=5)
    tk.Button(button_frame, text="🧹 Clear", 
             command=lambda: result_text.delete(1.0, tk.END),
             bg="#5a5a2a", fg="#ffffff", font=("Arial", 11)).pack(side=tk.LEFT, padx=5)
    
    # Initialize display
    result_text.insert(tk.END, "ENHANCED RESONANCE CRYPTOGRAPHY PLAYGROUND\n")
    result_text.insert(tk.END, "=" * 50 + "\n\n")
    result_text.insert(tk.END, "🔐 Features Available:\n")
    result_text.insert(tk.END, "• Dual-Input Encryption/Decryption\n")
    result_text.insert(tk.END, "• Separate Encrypt & Decrypt Operations\n")
    result_text.insert(tk.END, "• Full Cycle Testing\n")
    result_text.insert(tk.END, "• Real-time Performance Metrics\n\n")
    result_text.insert(tk.END, "🎯 Best Practice Usage:\n")
    result_text.insert(tk.END, "1. Enter plaintext message and key\n")
    result_text.insert(tk.END, "2. Click 'Encrypt' to generate encrypted data\n")
    result_text.insert(tk.END, "3. Encrypted data appears in the field above\n")
    result_text.insert(tk.END, "4. Modify encrypted data if testing corruption\n")
    result_text.insert(tk.END, "5. Click 'Decrypt' to test decryption\n")
    result_text.insert(tk.END, "6. Use 'Full Cycle' for complete testing\n\n")
    result_text.insert(tk.END, "🚀 DUAL INPUT DESIGN = BEST PRACTICE!\n")
    result_text.insert(tk.END, "✅ Verification, ✅ Testing, ✅ Education\n\n")
    result_text.insert(tk.END, "Ready for cryptographic operations! 🎯\n")
    
    root.mainloop()

if __name__ == "__main__":
    create_crypto_playground()
