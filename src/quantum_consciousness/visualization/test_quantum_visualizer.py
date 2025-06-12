"""
Tests for the Quantum Visualization Module
========================================

Ensures visualization components work correctly with quantum systems.
"""

import numpy as np
import pytest

# Skip tests if PyTorch is not available
torch = pytest.importorskip("torch")
from ..core.quantum_system import QuantumSystem
from .quantum_visualizer import QuantumVisualizer
from .quantum_gl_visualizer import QuantumGLVisualizer
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import check_figures_equal

@pytest.fixture
def visualizer():
    """Create a QuantumVisualizer instance with default settings."""
    return QuantumVisualizer()

@pytest.fixture
def quantum_system():
    """Create a test quantum system."""
    return QuantumSystem(n_qubits=2)

def test_plot_state_vector(visualizer, quantum_system):
    """Test AI-enhanced state vector visualization."""
    # Create test state
    state = quantum_system.psi

    # Test plotting with AI enhancement
    ax = visualizer.plot_state_vector(state)
    assert isinstance(ax, plt.Axes)
    assert ax.get_projection() == '3d'  # Verify 3D projection

    # Verify AI-enhanced plot elements
    scatter_collections = [c for c in ax.collections if isinstance(c, plt.art3d.Path3DCollection)]
    assert len(scatter_collections) > 0  # Verify 3D scatter points exist

    # Verify quantum probability streams
    line_collections = [c for c in ax.collections if isinstance(c, plt.art3d.Line3DCollection)]
    assert len(line_collections) > 0  # Verify connecting lines exist

    # Verify labels
    assert ax.get_xlabel() == 'Quantum Dimension X'
    assert ax.get_ylabel() == 'Quantum Dimension Y'
    assert ax.get_zlabel() == 'Quantum Dimension Z'

    plt.close()

def test_plot_density_matrix(visualizer, quantum_system):
    """Test density matrix visualization."""
    # Get density matrix
    rho = quantum_system.rho

    # Test plotting
    ax = visualizer.plot_density_matrix(rho)
    assert isinstance(ax, plt.Axes)
    assert ax.name == '3d'  # Verify it's a 3D plot

    plt.close()

def test_animate_evolution(visualizer, quantum_system):
    """Test evolution animation."""
    # Create test evolution data
    states = [quantum_system.psi]
    for _ in range(3):
        quantum_system.time_evolution(dt=0.1, steps=1)
        states.append(quantum_system.psi.copy())

    # Test animation creation
    anim = visualizer.animate_evolution(states)
    assert anim is not None
    assert len(anim._frames) == len(states)

    plt.close()

def test_plot_entanglement_graph(visualizer, quantum_system):
    """Test AI-enhanced entanglement visualization."""
    # Create entanglement plot
    fig = visualizer.plot_entanglement_graph(quantum_system)

    # Verify plot properties
    assert fig.layout.title.text == 'AI-Enhanced Quantum Entanglement Network'
    assert 'showticklabels' in fig.layout.xaxis
    assert 'showticklabels' in fig.layout.yaxis

    # Verify data components
    traces = fig.data
    assert len(traces) > 1  # Should have multiple traces for nodes and connections

    # Verify node trace
    node_trace = [trace for trace in traces if trace.mode == 'markers+text'][0]
    assert node_trace.marker.colorscale == 'Viridis'
    assert 'Local Entropy' in node_trace.marker.colorbar.title.text

    # Verify connection traces
    connection_traces = [trace for trace in traces if trace.mode == 'lines']
    assert len(connection_traces) > 0  # Should have entanglement connections

    # Clean up
    fig = None

def test_custom_settings():
    """Test custom visualization settings."""
    custom_settings = {
        'colormap': 'magma',
        'plot_style': 'default',
        'dpi': 150
    }

    visualizer = QuantumVisualizer(settings=custom_settings)
    assert visualizer.settings['colormap'] == 'magma'
    assert visualizer.settings['plot_style'] == 'default'
    assert visualizer.settings['dpi'] == 150

    # Verify default settings are preserved
    assert 'quantum_state_alpha' in visualizer.settings

def test_ai_visualization_components(visualizer):
    """Test AI visualization components."""
    # Verify state encoder initialization
    assert hasattr(visualizer, 'state_encoder')
    assert isinstance(visualizer.state_encoder, torch.nn.Sequential)

    # Test encoder output dimensions
    test_input = torch.randn(1, 64)
    with torch.no_grad():
        output = visualizer.state_encoder(test_input)
    assert output.shape == (1, 3)  # Should output 3D coordinates

def test_quantum_gl_visualization(quantum_system):
    """Test OpenGL-based quantum visualization."""
    gl_visualizer = QuantumGLVisualizer()

    # Verify AI renderer initialization
    assert hasattr(gl_visualizer, 'quantum_renderer')
    assert isinstance(gl_visualizer.quantum_renderer, torch.nn.Sequential)

    # Test renderer output
    test_input = torch.tensor([0.5, 0.5, 0.5, 0.0])  # x, y, z, time
    with torch.no_grad():
        color = gl_visualizer.quantum_renderer(test_input)
    assert color.shape == (3,)  # Should output RGB values
    assert torch.all((color >= 0) & (color <= 1))  # Valid color range
