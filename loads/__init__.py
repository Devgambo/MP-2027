"""Part 4: load generator (earthquake records + irregular wave loads on a monopile)."""
from .combined import LoadCase, generate_loads, load_h5, save_h5

__all__ = ["LoadCase", "generate_loads", "load_h5", "save_h5"]
