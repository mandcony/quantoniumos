import math

def parse_symbolic_amplitude(expr: str) -> complex:
    try:
        real, imag = map(float, expr.split('+'))
        return complex(real, imag)
    except ValueError:
        print(f"Error parsing amplitude: {expr}")
        return complex(1.0, 0.0)

def validate_amplitudes(amps, dt):
    print("Validating symbolic amplitudes...")
    return [parse_symbolic_amplitude(a) for a in amps]