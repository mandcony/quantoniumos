import math

def monitor_quantum_nova(containers, dt):
    print("Monitoring quantum nova...")
    for c in containers:
        c.resonance += dt
    return containers