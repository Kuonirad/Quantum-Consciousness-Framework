"""
Tests for quantum GNS (Gelfand-Naimark-Segal) construction implementation.
"""

import numpy as np
import pytest
from quantum_gns_construction import QuantumGNSConstruction, GNSState

def test_initialization():
    """Test GNS construction initialization."""
    dim = 4
    gns = QuantumGNSConstruction(dim)
    assert gns.dim == dim
    assert gns.hilbert_space.shape == (dim, dim)

def test_construct_state():
    """Test GNS state construction."""
    gns = QuantumGNSConstruction(2)

    # Define test functional
    def test_functional(matrix):
        return np.trace(matrix)

    state = gns.construct_state(test_functional)

    # Verify state properties
    assert isinstance(state, GNSState)
    assert np.abs(np.linalg.norm(state.vector) - 1.0) < 1e-10
    assert state.algebra_element.shape == (2, 2)
    assert state.representation.shape == (2, 2)

def test_inner_product():
    """Test GNS inner product calculation."""
    gns = QuantumGNSConstruction(2)

    # Create test states
    def func1(m): return np.trace(m)
    def func2(m): return np.trace(m @ m)

    state1 = gns.construct_state(func1)
    state2 = gns.construct_state(func2)

    # Calculate inner product
    prod = gns.inner_product(state1, state2)

    # Verify properties
    assert isinstance(prod, complex)
    assert np.isfinite(prod)

def test_geometric_phase():
    """Test geometric phase calculation."""
    gns = QuantumGNSConstruction(2)

    # Create test state
    def test_func(m): return np.trace(m)
    state = gns.construct_state(test_func)

    # Calculate geometric phase
    phase = gns.geometric_phase(state, 1.0)

    # Verify phase properties
    assert isinstance(phase, float)
    assert -np.pi <= phase <= np.pi

def test_modular_flow():
    """Test modular flow construction."""
    gns = QuantumGNSConstruction(2)

    # Create test state
    def test_func(m): return np.trace(m)
    state = gns.construct_state(test_func)

    # Calculate modular flow
    flow = gns.construct_modular_flow(state, 1.0)

    # Verify flow properties
    assert flow.shape == (2, 2)
    assert np.allclose(flow @ flow.conj().T, np.eye(2), atol=1e-10)

def test_tomita_takesaki():
    """Test Tomita-Takesaki modular theory implementation."""
    gns = QuantumGNSConstruction(2)

    # Create test state
    def test_func(m): return np.trace(m)
    state = gns.construct_state(test_func)

    # Calculate modular operator and conjugation
    delta, J = gns.tomita_takesaki_modular(state)

    # Verify properties
    assert delta.shape == (2, 2)
    assert J.shape == (2, 2)
    assert np.allclose(J @ J.conj().T, np.eye(2), atol=1e-10)

def test_relative_entropy():
    """Test relative modular entropy calculation."""
    gns = QuantumGNSConstruction(2)

    # Create test states
    def func1(m): return np.trace(m)
    def func2(m): return np.trace(m @ m)

    state1 = gns.construct_state(func1)
    state2 = gns.construct_state(func2)

    # Calculate relative entropy
    entropy = gns.relative_entropy(state1, state2)

    # Verify entropy properties
    assert isinstance(entropy, float)
    assert entropy >= 0

def test_crossed_product():
    """Test crossed product algebra construction."""
    gns = QuantumGNSConstruction(2)

    # Create test state and action
    def test_func(m): return np.trace(m)
    state = gns.construct_state(test_func)
    action = [np.eye(2), np.array([[0, 1], [1, 0]])]

    # Calculate crossed product
    crossed = gns.construct_crossed_product(state, action)

    # Verify properties
    assert crossed.shape == (4, 4)
    assert np.allclose(crossed @ crossed.conj().T, crossed.conj().T @ crossed)

def test_jones_index():
    """Test Jones index computation."""
    gns = QuantumGNSConstruction(2)

    # Create test state
    def test_func(m): return np.trace(m)
    state = gns.construct_state(test_func)

    # Calculate index
    index = gns.compute_index(state)

    # Verify index properties
    assert isinstance(index, float)
    assert index >= 1.0

def test_jones_tower():
    """Test Jones tower construction."""
    gns = QuantumGNSConstruction(2)

    # Create test state
    def test_func(m): return np.trace(m)
    state = gns.construct_state(test_func)

    # Construct tower
    depth = 3
    tower = gns.construct_tower(state, depth)

    # Verify tower properties
    assert len(tower) == depth + 1
    for i in range(depth + 1):
        dim = 2 ** (i + 1)
        assert tower[i].shape == (dim, dim)

def test_categorical_trace():
    """Test categorical trace computation."""
    gns = QuantumGNSConstruction(2)

    # Create test morphism
    morphism = np.array([[1, 0], [0, -1]])

    # Calculate categorical trace
    trace = gns.compute_categorical_trace(morphism)

    # Verify trace properties
    assert isinstance(trace, complex)
    assert np.isfinite(trace)

def test_error_handling():
    """Test error handling for invalid inputs."""
    gns = QuantumGNSConstruction(2)

    # Test invalid dimension
    with pytest.raises(ValueError):
        QuantumGNSConstruction(0)

    # Test invalid functional
    with pytest.raises(ValueError):
        def invalid_func(m): return "invalid"
        gns.construct_state(invalid_func)

def test_edge_cases():
    """Test edge cases and boundary conditions."""
    gns = QuantumGNSConstruction(2)

    # Test with identity functional
    def id_func(m): return 1.0
    state = gns.construct_state(id_func)

    # Verify state is well-defined
    assert np.all(np.isfinite(state.algebra_element))
    assert np.all(np.isfinite(state.representation))

    # Test with zero time parameter
    phase = gns.geometric_phase(state, 0.0)
    assert np.abs(phase) < 1e-10
