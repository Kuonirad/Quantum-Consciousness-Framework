"""
Tests for quantum system implementation.
"""

import numpy as np
import pytest
from quantum_consciousness.core.quantum_system import QuantumSystem

def test_initialization():
    """Test quantum system initialization."""
    # Test valid initialization
    qs = QuantumSystem(n_qubits=3)
    assert qs.n_qubits == 3
    assert qs.dim == 8
    assert np.allclose(np.linalg.norm(qs.psi), 1.0)

    # Test invalid initialization
    with pytest.raises(ValueError):
        QuantumSystem(n_qubits=0)
    with pytest.raises(ValueError):
        QuantumSystem(n_qubits=-1)

def test_density_matrix_properties():
    """Test density matrix properties."""
    qs = QuantumSystem(n_qubits=2)
    rho = qs.rho

    # Test Hermiticity
    assert np.allclose(rho, rho.conj().T)

    # Test positive semi-definiteness
    eigenvals = np.linalg.eigvalsh(rho)
    assert np.all(eigenvals >= -1e-10)

    # Test trace normalization
    assert np.abs(np.trace(rho) - 1.0) < 1e-10

def test_hamiltonian_properties():
    """Test Hamiltonian properties."""
    qs = QuantumSystem(n_qubits=2)
    H = qs.hamiltonian

    # Test Hermiticity
    assert np.allclose(H, H.conj().T)

    # Test energy conservation
    psi_t = qs.psi
    energy_t0 = np.real(np.vdot(psi_t, H @ psi_t))

    # Evolve state
    evolution = qs.time_evolution(dt=0.1, steps=10)
    psi_t = evolution[-1] @ qs.psi
    energy_t1 = np.real(np.vdot(psi_t, H @ psi_t))

    assert np.abs(energy_t0 - energy_t1) < 1e-8

def test_phi_measure():
    """Test Φ measure properties."""
    qs = QuantumSystem(n_qubits=2)
    phi = qs.compute_phi_measure()

    # Test non-negativity
    assert phi >= 0

    # Test finite value
    assert np.isfinite(phi)

    # Test upper bound (theoretical maximum for 2 qubits)
    assert phi <= 4.0

def test_time_evolution():
    """Test quantum time evolution."""
    qs = QuantumSystem(n_qubits=2)
    dt = 0.1
    steps = 10
    evolution = qs.time_evolution(dt, steps)

    # Test number of evolution steps
    assert len(evolution) == steps

    # Test trace preservation
    for rho_t in evolution:
        assert np.abs(np.trace(rho_t) - 1.0) < 1e-10

        # Test Hermiticity preservation
        assert np.allclose(rho_t, rho_t.conj().T)

        # Test positive semi-definiteness
        eigenvals = np.linalg.eigvalsh(rho_t)
        assert np.all(eigenvals >= -1e-10)

def test_quantum_circuit():
    """Test quantum circuit application."""
    qs = QuantumSystem(n_qubits=3)
    initial_state = qs.psi.copy()

    # Apply circuit
    qs.apply_quantum_circuit()

    # Test state change
    assert not np.allclose(qs.psi, initial_state)

    # Test normalization
    assert np.abs(np.linalg.norm(qs.psi) - 1.0) < 1e-10

    # Test density matrix update
    assert np.allclose(qs.rho, np.outer(qs.psi, qs.psi.conj()))

def test_bures_distance():
    """Test Bures distance calculation."""
    qs1 = QuantumSystem(n_qubits=2)
    qs2 = QuantumSystem(n_qubits=2)

    # Test distance to self is zero
    assert np.abs(qs1.compute_bures_distance(qs1)) < 1e-10

    # Test symmetry
    d12 = qs1.compute_bures_distance(qs2)
    d21 = qs2.compute_bures_distance(qs1)
    assert np.abs(d12 - d21) < 1e-10

    # Test triangle inequality
    qs3 = QuantumSystem(n_qubits=2)
    d13 = qs1.compute_bures_distance(qs3)
    d23 = qs2.compute_bures_distance(qs3)
    assert d12 <= d13 + d23 + 1e-10

    # Test invalid input
    with pytest.raises(ValueError):
        qs1.compute_bures_distance(QuantumSystem(n_qubits=3).rho)

def test_edge_cases():
    """Test edge cases and numerical stability."""
    qs = QuantumSystem(n_qubits=2)

    # Test near-pure states
    qs.psi = np.array([1-1e-8, 1e-8, 1e-8, 1e-8]) / np.sqrt(1 + 3e-16)
    qs.rho = qs._density_matrix()

    # Ensure computations remain stable
    phi = qs.compute_phi_measure()
    assert np.isfinite(phi)
    assert phi >= 0

    # Test evolution stability
    evolution = qs.time_evolution(0.1, 10)
    for rho_t in evolution:
        assert np.abs(np.trace(rho_t) - 1.0) < 1e-8

def test_numerical_precision():
    """Test numerical precision and stability."""
    qs = QuantumSystem(n_qubits=2)

    # Test precision of superposition state
    assert np.abs(np.sum(np.abs(qs.psi)**2) - 1.0) < 1e-14

    # Test precision of density matrix
    assert np.abs(np.trace(qs.rho) - 1.0) < 1e-14

    # Test precision of Hamiltonian
    assert np.allclose(qs.hamiltonian, qs.hamiltonian.conj().T, atol=1e-14)

def test_fubini_study_metric():
    """Test Fubini-Study metric computation."""
    qs = QuantumSystem(n_qubits=2)

    # Create test tangent vectors
    v1 = np.array([0.1, -0.2, 0.3, -0.4]) / np.sqrt(0.3)
    v2 = np.array([-0.2, 0.3, -0.4, 0.1]) / np.sqrt(0.3)
    tangent_vectors = [v1, v2]

    # Compute metric
    G = qs.compute_fubini_study_metric(tangent_vectors)

    # Test metric properties
    assert G.shape == (2, 2)
    assert np.allclose(G, G.conj().T)  # Hermiticity
    eigenvals = np.linalg.eigvalsh(G)
    assert np.all(eigenvals >= -1e-10)  # Positive semi-definiteness

def test_parallel_transport():
    """Test parallel transport implementation."""
    qs = QuantumSystem(n_qubits=2)

    # Create test path
    theta = np.linspace(0, np.pi/2, 10)
    path = [np.array([np.cos(t), np.sin(t), 0, 0]) for t in theta]

    # Initial vector
    v0 = np.array([0, 1, 0, 0])

    # Compute parallel transport
    transported = qs.parallel_transport(v0, path)

    # Test properties
    assert len(transported) == len(path)
    for v in transported:
        assert np.abs(np.linalg.norm(v) - 1.0) < 1e-10  # Normalization

def test_geometric_phase():
    """Test geometric phase computation."""
    qs = QuantumSystem(n_qubits=2)

    # Create cyclic path
    theta = np.linspace(0, 2*np.pi, 20)
    cycle = [np.array([np.cos(t), np.sin(t), 0, 0]) for t in theta]

    # Compute geometric phase
    phase = qs.compute_geometric_phase(cycle)

    # Test properties
    assert np.abs(np.abs(phase) - 1.0) < 1e-10  # Unit modulus
    assert np.isfinite(phase)  # Finite value

def test_quantum_cup_product():
    """Test quantum cup product for cognitive architecture."""
    qs = QuantumSystem(n_qubits=2)
    dim = 2**2  # 2 qubits = 4 dimensional space

    # Create test states with correct dimensions
    perception = np.zeros(dim)
    perception[0] = 1.0  # |00⟩ state

    attention = np.zeros(dim)
    attention[1] = 1.0   # |01⟩ state

    memory = np.zeros(dim)
    memory[2] = 1.0      # |10⟩ state

    # Compute cup product
    result = qs.compute_quantum_cup_product(perception, attention, memory)

    # Test properties
    assert np.abs(np.linalg.norm(result) - 1.0) < 1e-10  # Normalization
    assert result.shape == (dim,)  # Correct dimension

def test_cognitive_fidelity():
    """Test cognitive fidelity computation."""
    qs = QuantumSystem(n_qubits=2)

    # Create test states
    state_a = np.array([1, 0, 0, 0]) / np.sqrt(1)
    state_b = np.array([1, 1, 0, 0]) / np.sqrt(2)

    # Compute fidelity
    fidelity = qs.compute_cognitive_fidelity(state_a, state_b)

    # Test properties
    assert 0 <= fidelity <= 1  # Bounded between 0 and 1
    assert np.isfinite(fidelity)  # Finite value
    assert np.abs(qs.compute_cognitive_fidelity(state_a, state_a) - 1.0) < 1e-10  # Self-fidelity

def test_cognitive_evolution():
    """Test cognitive state evolution."""
    qs = QuantumSystem(n_qubits=2)

    # Create test sequences
    initial_state = np.array([1, 0, 0, 0]) / np.sqrt(1)
    perception_seq = [np.array([np.cos(t), np.sin(t), 0, 0]) / np.sqrt(1)
                     for t in np.linspace(0, np.pi/2, 5)]
    attention_seq = [np.array([np.sin(t), np.cos(t), 0, 0]) / np.sqrt(1)
                    for t in np.linspace(0, np.pi/2, 5)]

    # Compute evolution
    evolution = qs.evolve_cognitive_state(initial_state, perception_seq, attention_seq)

    # Test properties
    assert len(evolution) == min(len(perception_seq), len(attention_seq)) + 1
    for state in evolution:
        assert np.abs(np.linalg.norm(state) - 1.0) < 1e-10  # Normalization
