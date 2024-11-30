"""
Tests for quantum-classical interface implementation.
"""

import numpy as np
import pytest
from quantum_classical_interface import QuantumClassicalInterface

def test_quantum_measurement():
    """Test quantum measurement functionality."""
    interface = QuantumClassicalInterface(quantum_dim=2, classical_dim=2)

    # Prepare test quantum state
    quantum_state = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
    observable = np.array([[1, 0], [0, -1]])

    # Perform measurement
    expectation, post_state = interface.quantum_measurement(quantum_state, observable)

    # Verify expectation value is real
    assert np.isreal(expectation)
    # Verify post-measurement state is normalized
    assert np.abs(np.linalg.norm(post_state) - 1.0) < 1e-10

def test_classical_feedback():
    """Test classical feedback mechanism."""
    interface = QuantumClassicalInterface(quantum_dim=2, classical_dim=2)

    # Test inputs
    measurement = 0.5
    classical_state = np.array([1/np.sqrt(2), 1/np.sqrt(2)])

    # Apply feedback
    new_state = interface.classical_feedback(measurement, classical_state)

    # Verify normalization
    assert np.abs(np.linalg.norm(new_state) - 1.0) < 1e-10

def test_quantum_control():
    """Test quantum control operations."""
    interface = QuantumClassicalInterface(quantum_dim=2, classical_dim=2)

    # Prepare test states
    classical_state = np.array([1.0, 0.0])
    quantum_state = np.array([1/np.sqrt(2), 1/np.sqrt(2)])

    # Apply control
    controlled_state = interface.quantum_control(classical_state, quantum_state)

    # Verify unitarity
    assert np.abs(np.linalg.norm(controlled_state) - 1.0) < 1e-10

def test_hybrid_evolution():
    """Test hybrid quantum-classical evolution."""
    interface = QuantumClassicalInterface(quantum_dim=2, classical_dim=2)

    # Initial states
    quantum_state = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
    classical_state = np.array([1/np.sqrt(2), 1/np.sqrt(2)])

    # Evolve system
    final_quantum, final_classical = interface.hybrid_evolution(
        quantum_state, classical_state, time=1.0, dt=0.1)

    # Verify normalization
    assert np.abs(np.linalg.norm(final_quantum) - 1.0) < 1e-10
    assert np.abs(np.linalg.norm(final_classical) - 1.0) < 1e-10

def test_hybrid_entropy():
    """Test hybrid entropy calculation."""
    interface = QuantumClassicalInterface(quantum_dim=2, classical_dim=2)

    # Test states
    quantum_state = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
    classical_state = np.array([1/np.sqrt(2), 1/np.sqrt(2)])

    # Calculate entropy
    entropy = interface.compute_hybrid_entropy(quantum_state, classical_state)

    # Verify entropy is real and non-negative
    assert np.isreal(entropy)
    assert entropy >= 0

def test_quantum_classical_mapping():
    """Test quantum to classical and classical to quantum mappings."""
    interface = QuantumClassicalInterface(quantum_dim=2, classical_dim=2)

    # Test quantum to classical
    quantum_state = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
    classical_rep = interface.quantum_to_classical_mapping(quantum_state)
    assert np.abs(np.linalg.norm(classical_rep) - 1.0) < 1e-10

    # Test classical to quantum
    classical_state = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
    quantum_rep = interface.classical_to_quantum_mapping(classical_state)
    assert np.abs(np.linalg.norm(quantum_rep) - 1.0) < 1e-10

def test_hybrid_fidelity():
    """Test hybrid fidelity calculation."""
    interface = QuantumClassicalInterface(quantum_dim=2, classical_dim=2)

    # Test states
    quantum_state1 = np.array([1, 0])
    quantum_state2 = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
    classical_state1 = np.array([1, 0])
    classical_state2 = np.array([1/np.sqrt(2), 1/np.sqrt(2)])

    # Calculate fidelity
    fidelity = interface.hybrid_fidelity(
        quantum_state1, quantum_state2,
        classical_state1, classical_state2
    )

    # Verify fidelity is real and between 0 and 1
    assert np.isreal(fidelity)
    assert 0 <= fidelity <= 1

def test_error_handling():
    """Test error handling for invalid inputs."""
    interface = QuantumClassicalInterface(quantum_dim=2, classical_dim=2)

    # Test invalid quantum state
    with pytest.raises(ValueError):
        interface.quantum_measurement(
            np.array([1, 0, 0]),  # Wrong dimension
            np.eye(2)
        )

    # Test invalid classical state
    with pytest.raises(ValueError):
        interface.classical_feedback(
            0.5,
            np.array([1, 0, 0])  # Wrong dimension
        )

def test_edge_cases():
    """Test edge cases and boundary conditions."""
    interface = QuantumClassicalInterface(quantum_dim=2, classical_dim=2)

    # Test zero state handling
    zero_state = np.zeros(2)
    with pytest.raises(ValueError):
        interface.quantum_to_classical_mapping(zero_state)

    # Test small number handling
    small_state = np.array([1e-10, 1-1e-10])
    result = interface.quantum_measurement(small_state, np.eye(2))
    assert np.isfinite(result[0])
