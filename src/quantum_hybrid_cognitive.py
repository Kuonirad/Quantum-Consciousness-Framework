"""Convenience wrapper for the core quantum_hybrid_cognitive module."""

import os
import sys

_src_path = os.path.join(os.path.dirname(__file__), "src")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

from quantum_consciousness.core.quantum_hybrid_cognitive import *

__all__ = [name for name in globals() if not name.startswith("_")]
