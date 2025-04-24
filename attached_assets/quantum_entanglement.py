# quantum_entanglement.py

import random

def entangle_processes(proc1, proc2):
    """
    Simulate quantum-inspired entanglement between two processes.
    Sets both processes' entanglement flags and links them as entangled pairs.
    """
    proc1.is_entangled = True
    proc2.is_entangled = True
    proc1.entangled_process = proc2
    proc2.entangled_process = proc1
    return

def process_is_entangled(proc):
    """
    Checks if the given process is entangled.
    Returns True if the process has been entangled with another process.
    """
    return getattr(proc, 'is_entangled', False)

def get_entangled_pair(proc):
    """
    Returns the process that is entangled with the given process, if any.
    """
    return getattr(proc, 'entangled_process', None)
