import math
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuantumLink:
    def __init__(self):
        self.components = []

    def add_component(self, comp):
        self.components.append(comp)

    def synchronize_states(self):
        total_norm = sum(abs(c.amplitude) ** 2 for c in self.components) ** 0.5
        if total_norm != 0:
            for c in self.components:
                c.amplitude /= total_norm
        logger.info("Synchronized states in QuantumLink")
        return total_norm

    def validate_link(self):
        logger.info("Validated QuantumLink")
        return len(self.components) > 0