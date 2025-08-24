"""
QuantoniumOS Python Utilities
"""


def get_timestamp():
    """Get current timestamp"""
    import time

    return time.time()


def format_scientific_notation(value, precision=6):
    """Format a value in scientific notation"""
    return f"{value:.{precision}e}"
