"""
Tests for quantum cup product implementation.
"""

import numpy as np
import pytest
from quantum_cup_product import QuantumCupProduct

def test_initialization():
    """Test quantum cup product initialization."""
    # Test valid initialization
    qcp = QuantumCupProduct(dim=4)
    assert qcp.dim == 4

    # Test invalid initialization
    with pytest.raises(ValueError):
        QuantumCupProduct(dim=0)
    with pytest.raises(ValueError):
        QuantumCupProduct(dim=-1)

def test_operator_properties():
    """Test quantum operators' properties."""
    qcp = QuantumCupProduct(dim=4)

    # Test Perception operator
    assert qcp.P.shape == (4, 4)
    assert np.allclose(qcp.P, qcp.P.conj().T)  # Hermiticity
    assert np.abs(np.trace(qcp.P) - 1.0) < 1e-10  # Normalization

    # Test Attention operator
    assert qcp.A.shape == (4, 4)
    assert np.allclose(qcp.A, qcp.A.conj().T)  # Hermiticity

    # Test Memory operator
    assert qcp.M.shape == (4, 4)
    assert not np.allclose(qcp.M, np.zeros_like(qcp.M))  # Non-trivial

def test_cup_product():
    """Test quantum cup product computation."""
    qcp = QuantumCupProduct(dim=4)

    # Create test states
    state_a = np.array([1, 0, 0, 0], dtype=complex)
    state_b = np.array([0, 1, 0, 0], dtype=complex)

    # Compute cup product
    result = qcp.compute_cup_product(state_a, state_b)

    # Test properties
    assert len(result) == 4
    assert np.abs(np.linalg.norm(result) - 1.0) < 1e-10  # Normalization

    # Test invalid inputs
    with pytest.raises(ValueError):
        qcp.compute_cup_product(state_a, np.ones(5))

def test_cognitive_fidelity():
    """Test cognitive fidelity computation."""
    qcp = QuantumCupProduct(dim=4)

    # Create test states
    state_a = np.array([1, 0, 0, 0], dtype=complex)
    state_b = np.array([1, 0, 0, 0], dtype=complex)

    # Test self-fidelity
    F = qcp.compute_cognitive_fidelity(state_a, state_b)
    assert 0 <= F <= 1  # Bounded
    assert np.abs(F - 1.0) < 1e-10  # Perfect fidelity for identical states

    # Test orthogonal states
    state_c = np.array([0, 1, 0, 0], dtype=complex)
    F_orth = qcp.compute_cognitive_fidelity(state_a, state_c)
    assert F_orth < F  # Lower fidelity for orthogonal states

def test_cognitive_evolution():
    """Test cognitive state evolution."""
    qcp = QuantumCupProduct(dim=4)

    # Initial state
    psi_0 = np.array([1, 0, 0, 0], dtype=complex)

    # Evolve state
    steps = 10
    evolution = qcp.evolve_cognitive_state(psi_0, steps)

    # Test evolution properties
    assert len(evolution) == steps + 1  # Including initial state
    for state in evolution:
        assert np.abs(np.linalg.norm(state) - 1.0) < 1e-10  # Normalization
        assert len(state) == 4  # Dimension preservation

    # Test invalid input
    with pytest.raises(ValueError):
        qcp.evolve_cognitive_state(np.ones(5), steps)

def test_cognitive_entropy():
    """Test cognitive entropy computation."""
    qcp = QuantumCupProduct(dim=4)

    # Test pure state
    pure_state = np.array([1, 0, 0, 0], dtype=complex)
    S_pure = qcp.compute_cognitive_entropy(pure_state)
    assert S_pure >= 0  # Non-negative

    # Test superposition state
    super_state = np.array([1, 1, 0, 0], dtype=complex) / np.sqrt(2)
    S_super = qcp.compute_cognitive_entropy(super_state)
    assert S_super >= 0
    assert np.isfinite(S_super)

def test_numerical_stability():
    """Test numerical stability of computations."""
    qcp = QuantumCupProduct(dim=4)

    # Test with near-zero states
    small_state = np.array([1e-8, 1e-8, 1e-8, 1-3e-8], dtype=complex)
    small_state = small_state / np.linalg.norm(small_state)

    # Ensure computations remain stable
    result = qcp.compute_cup_product(small_state, small_state)
    assert np.isfinite(result).all()
    assert np.abs(np.linalg.norm(result) - 1.0) < 1e-10

    F = qcp.compute_cognitive_fidelity(small_state, small_state)
    assert np.isfinite(F)
    assert 0 <= F <= 1

def test_edge_cases():
    """Test edge cases and boundary conditions."""
    qcp = QuantumCupProduct(dim=4)

    # Test with maximally mixed state
    mixed_state = np.ones(4, dtype=complex) / 2

    # Ensure operations remain valid
    result = qcp.compute_cup_product(mixed_state, mixed_state)
    assert np.isfinite(result).all()

    S = qcp.compute_cognitive_entropy(mixed_state)
    assert np.isfinite(S)
    assert S >= 0

def test_operator_composition():
    """Test composition of cognitive operators."""
    qcp = QuantumCupProduct(dim=4)

    # Test P·A composition
    PA = qcp.P @ qcp.A
    assert PA.shape == (4, 4)
    assert np.allclose(PA, (PA @ PA.conj().T) @ PA)  # Consistency

    # Test A·M composition
    AM = qcp.A @ qcp.M
    assert AM.shape == (4, 4)
    assert not np.allclose(AM, np.zeros_like(AM))  # Non-trivial
