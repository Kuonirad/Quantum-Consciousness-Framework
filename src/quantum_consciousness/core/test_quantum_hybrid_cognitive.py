"""
Tests for quantum hybrid cognitive implementation.
"""

import numpy as np
import pytest
from quantum_hybrid_cognitive import QuantumHybridCognitive

def test_initialization():
    """Test quantum hybrid cognitive initialization."""
    # Test valid initialization
    qhc = QuantumHybridCognitive(n_qubits=2, n_classical=4)
    assert qhc.n_qubits == 2
    assert qhc.n_classical == 4
    assert qhc.dim_quantum == 4

    # Test invalid initialization
    with pytest.raises(ValueError):
        QuantumHybridCognitive(n_qubits=0, n_classical=4)
    with pytest.raises(ValueError):
        QuantumHybridCognitive(n_qubits=2, n_classical=0)

def test_operator_properties():
    """Test quantum cognitive operators."""
    qhc = QuantumHybridCognitive(n_qubits=2, n_classical=4)

    # Test Perception operator
    assert qhc.P.shape == (4, 4)
    assert np.allclose(qhc.P, qhc.P.conj().T)  # Hermiticity

    # Test Attention operator
    assert qhc.A.shape == (4, 4)
    assert np.allclose(qhc.A, qhc.A.conj().T)  # Hermiticity

    # Test Memory operator
    assert qhc.M.shape == (4, 4)
    assert not np.allclose(qhc.M, np.zeros_like(qhc.M))  # Non-trivial

def test_cognitive_processing():
    """Test hybrid cognitive processing."""
    qhc = QuantumHybridCognitive(n_qubits=2, n_classical=4)

    # Create test states
    quantum_state = np.array([1, 0, 0, 0], dtype=complex)
    classical_data = np.array([1, 0, 0, 0])

    # Process states
    q_out, c_out = qhc.process_cognitive_input(quantum_state, classical_data)

    # Test properties
    assert len(q_out) == 4
    assert len(c_out) == 4
    assert np.abs(np.linalg.norm(q_out) - 1.0) < 1e-10  # Normalization
    assert np.abs(np.linalg.norm(c_out) - 1.0) < 1e-10  # Normalization

def test_cognitive_coherence():
    """Test cognitive coherence computation."""
    qhc = QuantumHybridCognitive(n_qubits=2, n_classical=4)

    # Create test states
    quantum_state = np.array([1, 0, 0, 0], dtype=complex)
    classical_state = np.array([1, 0, 0, 0])

    # Compute coherence
    coherence = qhc.compute_cognitive_coherence(quantum_state, classical_state)

    # Test properties
    assert isinstance(coherence, float)
    assert 0 <= coherence <= 1  # Bounded
    assert np.isfinite(coherence)

def test_cognitive_evolution():
    """Test cognitive state evolution."""
    qhc = QuantumHybridCognitive(n_qubits=2, n_classical=4)

    # Initial states
    quantum_state = np.array([1, 0, 0, 0], dtype=complex)
    classical_state = np.array([1, 0, 0, 0])

    # Evolve states
    steps = 10
    evolution = qhc.evolve_cognitive_state(quantum_state, classical_state, steps)

    # Test evolution properties
    assert len(evolution) == steps + 1  # Including initial state
    for q_state, c_state in evolution:
        assert np.abs(np.linalg.norm(q_state) - 1.0) < 1e-10  # Quantum normalization
        assert np.abs(np.linalg.norm(c_state) - 1.0) < 1e-10  # Classical normalization

def test_hybrid_entropy():
    """Test hybrid entropy computation."""
    qhc = QuantumHybridCognitive(n_qubits=2, n_classical=4)

    # Test pure states
    quantum_state = np.array([1, 0, 0, 0], dtype=complex)
    classical_state = np.array([1, 0, 0, 0])

    # Compute entropy
    S = qhc.compute_hybrid_entropy(quantum_state, classical_state)

    # Test properties
    assert isinstance(S, float)
    assert S >= 0  # Non-negative
    assert np.isfinite(S)

def test_numerical_stability():
    """Test numerical stability of computations."""
    qhc = QuantumHybridCognitive(n_qubits=2, n_classical=4)

    # Test with near-zero states
    small_quantum = np.array([1e-8, 1e-8, 1-2e-8, 0], dtype=complex)
    small_quantum = small_quantum / np.linalg.norm(small_quantum)
    small_classical = np.array([1e-8, 1e-8, 1-2e-8, 0])
    small_classical = small_classical / np.linalg.norm(small_classical)

    # Ensure computations remain stable
    q_out, c_out = qhc.process_cognitive_input(small_quantum, small_classical)
    assert np.isfinite(q_out).all()
    assert np.isfinite(c_out).all()

def test_edge_cases():
    """Test edge cases and boundary conditions."""
    qhc = QuantumHybridCognitive(n_qubits=2, n_classical=4)

    # Test with maximally mixed states
    mixed_quantum = np.ones(4, dtype=complex) / 2
    mixed_classical = np.ones(4) / 2

    # Ensure operations remain valid
    q_out, c_out = qhc.process_cognitive_input(mixed_quantum, mixed_classical)
    assert np.isfinite(q_out).all()
    assert np.isfinite(c_out).all()

def test_quantum_classical_interaction():
    """Test quantum-classical interaction consistency."""
    qhc = QuantumHybridCognitive(n_qubits=2, n_classical=4)

    # Create test states
    quantum_state = np.array([1, 0, 0, 0], dtype=complex)
    classical_state = np.array([1, 0, 0, 0])

    # Test interaction
    q_out = qhc._quantum_classical_interaction(quantum_state, classical_state)
    assert len(q_out) == 4
    assert np.abs(np.linalg.norm(q_out) - 1.0) < 1e-10  # Normalization

def test_classical_evolution():
    """Test classical evolution with quantum feedback."""
    qhc = QuantumHybridCognitive(n_qubits=2, n_classical=4)

    # Create test states
    quantum_state = np.array([1, 0, 0, 0], dtype=complex)
    classical_state = np.array([1, 0, 0, 0])

    # Test evolution step
    c_out = qhc._classical_evolution_step(classical_state, quantum_state)
    assert len(c_out) == 4
    assert np.abs(np.linalg.norm(c_out) - 1.0) < 1e-10  # Normalization
