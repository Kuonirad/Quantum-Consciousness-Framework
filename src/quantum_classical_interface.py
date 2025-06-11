"""Convenience wrapper for the core quantum_classical_interface module."""

import os
import sys

# Ensure the package in ``src`` is importable when the project is not installed.
_src_path = os.path.join(os.path.dirname(__file__), "src")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

from quantum_consciousness.core.quantum_classical_interface import *

__all__ = [name for name in globals() if not name.startswith("_")]
