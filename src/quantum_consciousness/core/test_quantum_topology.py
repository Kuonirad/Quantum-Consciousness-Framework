"""
Tests for quantum topology implementation.
"""

import numpy as np
import pytest
from quantum_topology import QuantumTopology

def test_initialization():
    """Test quantum topology initialization."""
    # Test valid initialization
    qt = QuantumTopology(dim=4)
    assert qt.dim == 4

    # Test invalid initialization
    with pytest.raises(ValueError):
        QuantumTopology(dim=0)
    with pytest.raises(ValueError):
        QuantumTopology(dim=3)  # Must be even

def test_braiding_operators():
    """Test initialization of braiding operators."""
    qt = QuantumTopology(dim=4)

    # Test R-matrix properties
    assert qt.R.shape == (4, 4)
    assert np.allclose(qt.R @ qt.R.conj().T, np.eye(4))  # Unitarity

    # Test F-matrix properties
    assert qt.F.shape == (4, 4)
    assert np.allclose(qt.F @ qt.F.conj().T, np.eye(4))  # Unitarity

def test_braid_anyons():
    """Test braiding operations on anyonic states."""
    qt = QuantumTopology(dim=4)

    # Create test state
    state = np.array([1, 0, 0, 0], dtype=complex)

    # Test single braiding
    braided = qt.braid_anyons(state, [(0, 1)])
    assert len(braided) == 4
    assert np.abs(np.linalg.norm(braided) - 1.0) < 1e-10  # Normalization

    # Test invalid positions
    with pytest.raises(ValueError):
        qt.braid_anyons(state, [(0, 4)])

def test_jones_polynomial():
    """Test Jones polynomial computation."""
    qt = QuantumTopology(dim=4)

    # Simple braid word
    braid_word = [(0, 1), (1, 2)]

    # Compute polynomial
    jones = qt.compute_jones_polynomial(braid_word)

    # Test properties
    assert isinstance(jones, np.ndarray)
    assert np.isfinite(jones).all()

def test_tqft_invariant():
    """Test TQFT invariant computation."""
    qt = QuantumTopology(dim=4)

    # Create simple manifold data
    manifold_data = {
        'vertices': np.array([[0, 1], [1, 0]]),
        'edges': np.array([[0, 1], [1, 2]]),
        'faces': np.array([[0, 1, 2]])
    }

    # Compute invariant
    inv = qt.compute_tqft_invariant(manifold_data)

    # Test properties
    assert isinstance(inv, complex)
    assert np.isfinite(inv)

def test_linking_number():
    """Test linking number computation."""
    qt = QuantumTopology(dim=4)

    # Test simple braid word
    braid_word = [(0, 1), (1, 0)]
    link_num = qt.compute_linking_number(braid_word)

    # Test properties
    assert isinstance(link_num, int)
    assert link_num == 0  # Canceling crossings

def test_anyonic_state():
    """Test anyonic state creation."""
    qt = QuantumTopology(dim=4)

    # Create simple anyonic state
    particle_types = [0, 1]
    state = qt.create_anyonic_state(particle_types)

    # Test properties
    assert len(state) == 4
    assert np.abs(np.linalg.norm(state) - 1.0) < 1e-10  # Normalization

    # Test invalid input
    with pytest.raises(ValueError):
        qt.create_anyonic_state([0, 1, 2, 3, 4])

def test_topological_entropy():
    """Test topological entanglement entropy computation."""
    qt = QuantumTopology(dim=4)

    # Create test state
    state = np.array([1, 1, 0, 0], dtype=complex) / np.sqrt(2)

    # Compute entropy
    S = qt.compute_topological_entropy(state, [0, 1])

    # Test properties
    assert isinstance(S, float)
    assert S >= 0  # Non-negative
    assert np.isfinite(S)

    # Test invalid partition
    with pytest.raises(ValueError):
        qt.compute_topological_entropy(state, [4])

def test_partial_trace():
    """Test partial trace computation."""
    qt = QuantumTopology(dim=4)

    # Create test state
    state = np.array([1, 0, 0, 0], dtype=complex)
    rho = np.outer(state, state.conj())

    # Compute partial trace
    rho_A = qt._partial_trace(rho, [0, 1])

    # Test properties
    assert rho_A.shape == (2, 2)
    assert np.allclose(rho_A, rho_A.conj().T)  # Hermiticity
    assert np.abs(np.trace(rho_A) - 1.0) < 1e-10  # Trace preservation

def test_numerical_stability():
    """Test numerical stability of computations."""
    qt = QuantumTopology(dim=4)

    # Test with near-zero states
    small_state = np.array([1e-8, 1e-8, 1-2e-8, 0], dtype=complex)
    small_state = small_state / np.linalg.norm(small_state)

    # Ensure computations remain stable
    S = qt.compute_topological_entropy(small_state, [0, 1])
    assert np.isfinite(S)

def test_edge_cases():
    """Test edge cases and boundary conditions."""
    qt = QuantumTopology(dim=4)

    # Test with maximally entangled state
    bell_state = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)

    # Compute entropy
    S = qt.compute_topological_entropy(bell_state, [0, 1])
    assert np.isfinite(S)
    assert S >= 0

def test_consistency():
    """Test consistency of topological operations."""
    qt = QuantumTopology(dim=4)

    # Test Yang-Baxter equation for R-matrix
    R = qt.R
    I = np.eye(4)

    # Create R13 using tensor product permutation
    R13 = np.zeros((16, 16), dtype=complex)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                for l in range(4):
                    R13[4*i+k, 4*j+l] = R[i,j] if k == l else 0

    # R₁₂R₁₃R₂₃ = R₂₃R₁₃R₁₂
    R12 = np.kron(R, I)
    R23 = np.kron(I, R)

    lhs = R12 @ R13 @ R23
    rhs = R23 @ R13 @ R12

    assert np.allclose(lhs, rhs)  # Yang-Baxter equation
