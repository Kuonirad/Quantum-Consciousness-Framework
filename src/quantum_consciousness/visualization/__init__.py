"""
Quantum Consciousness Framework - Visualization Module
==================================================

This module provides visualization tools for quantum states, dynamics, and consciousness measures.
"""

from typing import Dict, Any

# Version info
__version__ = '0.1.0'

# Default visualization settings
DEFAULT_SETTINGS: Dict[str, Any] = {
    'colormap': 'viridis',
    'plot_style': 'dark_background',
    'dpi': 300,
    'interactive': True,
    'animation_fps': 30,
    'quantum_state_alpha': 0.8,
    'surface_plot_resolution': 100,
    'phase_colormap': 'hsv',
    'probability_colormap': 'plasma',
    'entanglement_colormap': 'coolwarm'
}

# Export settings
__all__ = ['DEFAULT_SETTINGS']
