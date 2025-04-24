import math
from geometric_container import GeometricContainer
import config

class ShardContainer(GeometricContainer):
    def __init__(self, id: str, amplitude: complex, resonance: float):
        super().__init__(amplitude, resonance)
        self.id = id

def validate_shard(shards, dt):
    cfg = config.Config()
    spread = cfg.data.get("resonance_spread", 0.1)
    print(f"Validating shards with spread {spread}...")
    for s in shards:
        s.resonance *= spread
    return shards