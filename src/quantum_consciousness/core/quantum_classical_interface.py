"""
Quantum-classical interface implementation.
Handles interaction between quantum and classical systems.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Callable
from scipy.linalg import expm, sqrtm

class QuantumClassicalInterface:
    """Implements quantum-classical system interface."""

    def __init__(self, quantum_dim: int, classical_dim: int):
        """
        Initialize quantum-classical interface.

        Args:
            quantum_dim: Dimension of quantum subsystem
            classical_dim: Dimension of classical subsystem
        """
        self.quantum_dim = quantum_dim
        self.classical_dim = classical_dim
        self.epsilon = 1e-12

    def quantum_measurement(self, quantum_state: np.ndarray,
                          observable: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Perform quantum measurement
        ⟨A⟩ = ⟨ψ|A|ψ⟩

        Args:
            quantum_state: Quantum state vector
            observable: Quantum observable operator

        Returns:
            Tuple[float, np.ndarray]: Expectation value and post-measurement state
        """
        if quantum_state.shape[0] != self.quantum_dim:
            raise ValueError("Quantum state dimension mismatch")
        if observable.shape != (self.quantum_dim, self.quantum_dim):
            raise ValueError("Observable dimension mismatch")

        # Calculate expectation value
        expectation = np.real(np.vdot(quantum_state, observable @ quantum_state))

        # Project state
        eigenvals, eigenvecs = np.linalg.eigh(observable)
        probabilities = np.abs(eigenvecs.conj().T @ quantum_state)**2
        probabilities = probabilities / probabilities.sum()

        # Choose eigenstate based on probabilities
        outcome = np.random.choice(len(eigenvals), p=probabilities)
        post_state = eigenvecs[:, outcome]

        return expectation, post_state

    def classical_feedback(self, measurement: float,
                         classical_state: np.ndarray) -> np.ndarray:
        """
        Apply classical feedback based on quantum measurement
        x' = f(x, ⟨A⟩)

        Args:
            measurement: Quantum measurement result
            classical_state: Classical state vector

        Returns:
            numpy.ndarray: Updated classical state
        """
        if classical_state.shape[0] != self.classical_dim:
            raise ValueError("Classical state dimension mismatch")

        # Create feedback matrix
        F = np.eye(self.classical_dim) + measurement * np.random.randn(
            self.classical_dim, self.classical_dim)

        # Apply feedback
        new_classical_state = F @ classical_state

        return new_classical_state / np.linalg.norm(new_classical_state)

    def quantum_control(self, classical_state: np.ndarray,
                       quantum_state: np.ndarray) -> np.ndarray:
        """
        Apply classical control to quantum system
        |ψ'⟩ = U(x)|ψ⟩

        Args:
            classical_state: Classical control parameters
            quantum_state: Quantum state to control

        Returns:
            numpy.ndarray: Controlled quantum state
        """
        # Create control Hamiltonian
        H_control = self._create_control_hamiltonian(classical_state)

        # Create control unitary
        U = expm(-1j * H_control)

        # Apply control
        controlled_state = U @ quantum_state

        return controlled_state

    def _create_control_hamiltonian(self, classical_state: np.ndarray) -> np.ndarray:
        """Create control Hamiltonian from classical parameters."""
        # Generate Hermitian matrix
        H = np.zeros((self.quantum_dim, self.quantum_dim), dtype=complex)
        for i, param in enumerate(classical_state):
            if i >= self.quantum_dim:
                break
            H[i,i] = param

        return H + H.conj().T

    def hybrid_evolution(self, quantum_state: np.ndarray,
                        classical_state: np.ndarray,
                        time: float,
                        dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evolve hybrid quantum-classical system
        d|ψ⟩/dt = -iH(x)|ψ⟩
        dx/dt = f(x, ⟨ψ|A|ψ⟩)

        Args:
            quantum_state: Initial quantum state
            classical_state: Initial classical state
            time: Total evolution time
            dt: Time step

        Returns:
            Tuple[np.ndarray, np.ndarray]: Final quantum and classical states
        """
        steps = int(time / dt)
        current_quantum = quantum_state.copy()
        current_classical = classical_state.copy()

        for _ in range(steps):
            # Quantum evolution
            H = self._create_control_hamiltonian(current_classical)
            U = expm(-1j * H * dt)
            current_quantum = U @ current_quantum

            # Quantum measurement
            observable = np.diag(np.random.randn(self.quantum_dim))
            measurement, _ = self.quantum_measurement(current_quantum, observable)

            # Classical feedback
            current_classical = self.classical_feedback(measurement,
                                                     current_classical)

        return current_quantum, current_classical

    def compute_hybrid_entropy(self, quantum_state: np.ndarray,
                             classical_state: np.ndarray) -> float:
        """
        Calculate hybrid quantum-classical entropy
        S = S_q + S_c - I(Q:C)

        Args:
            quantum_state: Quantum state
            classical_state: Classical state

        Returns:
            float: Hybrid entropy
        """
        # Quantum entropy
        rho = np.outer(quantum_state, quantum_state.conj())
        eigenvals = np.linalg.eigvalsh(rho)
        eigenvals = eigenvals[eigenvals > self.epsilon]
        S_q = -np.sum(eigenvals * np.log2(eigenvals))

        # Classical entropy (using normalized probabilities)
        p = np.abs(classical_state)**2
        p = p[p > self.epsilon]
        S_c = -np.sum(p * np.log2(p))

        # Mutual information (simplified estimate)
        I_qc = np.abs(np.vdot(quantum_state, classical_state))**2

        return S_q + S_c - I_qc

    def quantum_to_classical_mapping(self, quantum_state: np.ndarray) -> np.ndarray:
        """
        Map quantum state to classical representation
        |ψ⟩ → x

        Args:
            quantum_state: Quantum state

        Returns:
            numpy.ndarray: Classical representation
        """
        if quantum_state.shape[0] != self.quantum_dim:
            raise ValueError("Quantum state dimension mismatch")
        if np.linalg.norm(quantum_state) < self.epsilon:
            raise ValueError("Quantum state has zero norm")

        # Calculate probabilities
        probs = np.abs(quantum_state)**2

        # Create classical representation
        classical_rep = np.zeros(self.classical_dim)
        for i in range(min(len(probs), self.classical_dim)):
            classical_rep[i] = probs[i]

        return classical_rep / np.linalg.norm(classical_rep)

    def classical_to_quantum_mapping(self, classical_state: np.ndarray) -> np.ndarray:
        """
        Map classical state to quantum representation
        x → |ψ⟩

        Args:
            classical_state: Classical state

        Returns:
            numpy.ndarray: Quantum representation
        """
        if classical_state.shape[0] != self.classical_dim:
            raise ValueError("Classical state dimension mismatch")
        if np.linalg.norm(classical_state) < self.epsilon:
            raise ValueError("Classical state has zero norm")

        # Create quantum state with amplitudes from classical state
        quantum_rep = np.zeros(self.quantum_dim, dtype=complex)
        for i in range(min(len(classical_state), self.quantum_dim)):
            quantum_rep[i] = np.sqrt(np.abs(classical_state[i]))

        return quantum_rep / np.linalg.norm(quantum_rep)

    def hybrid_fidelity(self, quantum_state1: np.ndarray,
                       quantum_state2: np.ndarray,
                       classical_state1: np.ndarray,
                       classical_state2: np.ndarray) -> float:
        """
        Calculate hybrid quantum-classical fidelity
        F = F_q * F_c

        Args:
            quantum_state1: First quantum state
            quantum_state2: Second quantum state
            classical_state1: First classical state
            classical_state2: Second classical state

        Returns:
            float: Hybrid fidelity
        """
        # Quantum fidelity
        F_q = np.abs(np.vdot(quantum_state1, quantum_state2))**2

        # Classical fidelity
        F_c = np.abs(np.vdot(classical_state1, classical_state2))**2

        return F_q * F_c
