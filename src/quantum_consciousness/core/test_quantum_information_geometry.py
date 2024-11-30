"""
Tests for quantum information geometry implementation.
"""

import numpy as np
import pytest
from quantum_information_geometry import QuantumInfoGeometry

def test_initialization():
    """Test quantum information geometry initialization."""
    # Test valid initialization
    qig = QuantumInfoGeometry(dim=4)
    assert qig.dim == 4

    # Test invalid initialization
    with pytest.raises(ValueError):
        QuantumInfoGeometry(dim=0)
    with pytest.raises(ValueError):
        QuantumInfoGeometry(dim=-1)

def test_fubini_study_metric():
    """Test Fubini-Study metric computation."""
    qig = QuantumInfoGeometry(dim=2)

    # Create test state and derivatives
    psi = np.array([1, 0], dtype=complex)
    dpsi_i = np.array([0, 1], dtype=complex)
    dpsi_j = np.array([0, 1], dtype=complex)

    # Compute metric
    g = qig.fubini_study_metric(psi, dpsi_i, dpsi_j)

    # Test properties
    assert isinstance(g, complex)
    assert np.isfinite(g)
    assert g.real >= 0  # Positive semi-definite

def test_quantum_fisher_metric():
    """Test quantum Fisher information metric."""
    qig = QuantumInfoGeometry(dim=2)

    # Create test density matrix and observable
    rho = np.array([[0.8, 0], [0, 0.2]], dtype=complex)
    A = np.array([[0, 1], [1, 0]], dtype=complex)

    # Compute Fisher information
    F = qig.quantum_fisher_metric(rho, A)

    # Test properties
    assert isinstance(F, float)
    assert F >= 0  # Non-negative
    assert np.isfinite(F)

def test_bures_metric():
    """Test Bures metric computation."""
    qig = QuantumInfoGeometry(dim=2)

    # Create test density matrices
    rho = np.array([[1, 0], [0, 0]], dtype=complex)
    sigma = np.array([[0, 0], [0, 1]], dtype=complex)

    # Compute Bures distance
    D = qig.bures_metric(rho, sigma)

    # Test properties
    assert isinstance(D, float)
    assert 0 <= D <= np.sqrt(2)  # Bounded
    assert np.isfinite(D)

def test_parallel_transport():
    """Test parallel transport implementation."""
    qig = QuantumInfoGeometry(dim=2)

    # Create test state and connection
    psi = np.array([1, 0], dtype=complex)
    connection = np.array([[0, 1], [-1, 0]], dtype=complex)

    # Compute transport
    transported = qig.parallel_transport(psi, connection)

    # Test properties
    assert len(transported) == 2
    assert np.abs(np.linalg.norm(transported) - 1.0) < 1e-10  # Normalization

def test_geometric_phase():
    """Test geometric phase calculation."""
    qig = QuantumInfoGeometry(dim=2)

    # Create cyclic path of states
    states = [
        np.array([1, 0], dtype=complex),
        np.array([1, 1], dtype=complex) / np.sqrt(2),
        np.array([0, 1], dtype=complex),
        np.array([1, 0], dtype=complex)
    ]

    # Compute phase
    phase = qig.geometric_phase(states)

    # Test properties
    assert isinstance(phase, float)
    assert -np.pi <= phase <= np.pi
    assert np.isfinite(phase)

def test_quantum_principal_bundle():
    """Test quantum principal bundle construction."""
    qig = QuantumInfoGeometry(dim=2)

    # Create test state and gauge field
    state = np.array([1, 0], dtype=complex)
    gauge = np.array([[0, 1], [-1, 0]], dtype=complex)

    # Compute bundle components
    H, V = qig.quantum_principal_bundle(state, gauge)

    # Test properties
    assert H.shape == (2, 2)
    assert V.shape == (2, 2)
    assert np.allclose(H + V, gauge)  # Decomposition

def test_information_metric_tensor():
    """Test information metric tensor computation."""
    qig = QuantumInfoGeometry(dim=2)

    # Create test density matrix and basis operators
    rho = np.array([[0.8, 0], [0, 0.2]], dtype=complex)
    basis = [
        np.array([[1, 0], [0, -1]], dtype=complex),
        np.array([[0, 1], [1, 0]], dtype=complex)
    ]

    # Compute metric tensor
    g = qig.information_metric_tensor(rho, basis)

    # Test properties
    assert g.shape == (2, 2)
    assert np.allclose(g, g.conj().T)  # Hermiticity
    eigenvals = np.linalg.eigvalsh(g)
    assert np.all(eigenvals >= -1e-10)  # Positive semi-definite

def test_symplectic_structure():
    """Test symplectic structure computation."""
    qig = QuantumInfoGeometry(dim=2)

    # Create test density matrix and basis operators
    rho = np.array([[0.8, 0], [0, 0.2]], dtype=complex)
    basis = [
        np.array([[1, 0], [0, -1]], dtype=complex),
        np.array([[0, 1], [1, 0]], dtype=complex)
    ]

    # Compute symplectic form
    omega = qig.symplectic_structure(rho, basis)

    # Test properties
    assert omega.shape == (2, 2)
    assert np.allclose(omega, -omega.conj().T)  # Anti-symmetry

def test_numerical_stability():
    """Test numerical stability of computations."""
    qig = QuantumInfoGeometry(dim=2)

    # Test with near-zero states
    small_state = np.array([1e-8, 1-1e-8], dtype=complex)
    small_state = small_state / np.linalg.norm(small_state)

    # Ensure computations remain stable
    dpsi = np.array([1e-8, -1e-8], dtype=complex)
    g = qig.fubini_study_metric(small_state, dpsi, dpsi)
    assert np.isfinite(g)

def test_edge_cases():
    """Test edge cases and boundary conditions."""
    qig = QuantumInfoGeometry(dim=2)

    # Test with maximally mixed state
    mixed_state = np.array([[0.5, 0], [0, 0.5]], dtype=complex)
    pure_state = np.array([[1, 0], [0, 0]], dtype=complex)

    # Compute Bures distance
    D = qig.bures_metric(mixed_state, pure_state)
    assert np.isfinite(D)
    assert 0 <= D <= np.sqrt(2)

def test_geometric_consistency():
    """Test geometric consistency relations."""
    qig = QuantumInfoGeometry(dim=2)

    # Create test states
    psi = np.array([1, 0], dtype=complex)
    dpsi = np.array([0, 1], dtype=complex)

    # Test metric positivity
    g = qig.fubini_study_metric(psi, dpsi, dpsi)
    assert g.real >= -1e-10  # Account for numerical precision
