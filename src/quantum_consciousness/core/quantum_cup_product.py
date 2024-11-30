"""
Quantum Cup Product implementation for cognitive architectures.
Integrates perception, attention, and memory through quantum operations.
"""

import numpy as np
from typing import Optional, Tuple, List, Union
from scipy.linalg import expm, logm
import qutip as qt

class QuantumCupProduct:
    """
    Implements quantum cup product for cognitive architectures:
    a ∗_q b ≈ (Perception ∪ Attention)β q^{⟨β,ω⟩} (Memory)
    """

    def __init__(self, dim: int):
        """
        Initialize quantum cup product system.

        Args:
            dim: Dimension of the quantum system

        Raises:
            ValueError: If dimension is not positive
        """
        if dim <= 0:
            raise ValueError("Dimension must be positive")

        self.dim = dim
        self.epsilon = 1e-12  # Numerical stability threshold
        self._initialize_operators()

    def _initialize_operators(self) -> None:
        """Initialize quantum operators for perception, attention, and memory."""
        # Perception operator (measurement-like)
        self.P = self._create_perception_operator()

        # Attention operator (selective focus)
        self.A = self._create_attention_operator()

        # Memory operator (storage and recall)
        self.M = self._create_memory_operator()

    def _create_perception_operator(self) -> np.ndarray:
        """
        Create perception operator using quantum measurement principles.

        Returns:
            numpy.ndarray: Perception operator
        """
        # Create basis states for perception
        P = np.zeros((self.dim, self.dim), dtype=complex)
        for i in range(self.dim):
            # Gaussian-like perception profile
            center = self.dim // 2
            width = self.dim / 4
            P[i,i] = np.exp(-(i - center)**2 / (2 * width**2))

        # Normalize
        P = P / np.trace(P)
        return P

    def _create_attention_operator(self) -> np.ndarray:
        """
        Create attention operator for selective focus.

        Returns:
            numpy.ndarray: Attention operator
        """
        # Create attention filter
        A = np.eye(self.dim, dtype=complex)

        # Add off-diagonal coherences
        for i in range(self.dim-1):
            A[i,i+1] = 0.1j
            A[i+1,i] = -0.1j

        # Ensure Hermiticity
        A = (A + A.conj().T) / 2
        return A

    def _create_memory_operator(self) -> np.ndarray:
        """
        Create memory operator for information storage.

        Returns:
            numpy.ndarray: Memory operator
        """
        # Create memory structure
        M = np.zeros((self.dim, self.dim), dtype=complex)

        # Add long-term storage elements
        for i in range(self.dim):
            decay = np.exp(-i / self.dim)  # Exponential decay
            M[i,i] = decay

        # Add associative connections
        for i in range(self.dim-1):
            M[i,i+1] = 0.05 * np.exp(-i / self.dim)
            M[i+1,i] = 0.05 * np.exp(-i / self.dim)

        return M

    def compute_cup_product(self,
                          state_a: np.ndarray,
                          state_b: np.ndarray,
                          beta: float = 1.0) -> np.ndarray:
        """
        Compute quantum cup product between two states.

        Args:
            state_a: First quantum state
            state_b: Second quantum state
            beta: Coupling strength parameter

        Returns:
            numpy.ndarray: Resulting quantum state

        Raises:
            ValueError: If state dimensions don't match
        """
        if state_a.shape != state_b.shape or len(state_a) != self.dim:
            raise ValueError("State dimensions must match system dimension")

        # Normalize input states
        state_a = state_a / np.linalg.norm(state_a)
        state_b = state_b / np.linalg.norm(state_b)

        # Compute perception-attention product
        PA = self.P @ self.A

        # Compute geometric coupling
        omega = np.vdot(state_a, state_b)
        q_factor = np.exp(beta * omega)

        # Apply quantum cup product
        result = q_factor * (PA @ state_a) * (self.M @ state_b)

        # Normalize result
        result = result / (np.linalg.norm(result) + self.epsilon)

        return result

    def compute_cognitive_fidelity(self,
                                 state_a: np.ndarray,
                                 state_b: np.ndarray) -> float:
        """
        Compute cognitive fidelity between two states.

        Args:
            state_a: First quantum state
            state_b: Second quantum state

        Returns:
            float: Cognitive fidelity measure
        """
        # Normalize states
        state_a = state_a / np.linalg.norm(state_a)
        state_b = state_b / np.linalg.norm(state_b)

        # Compute fidelity with cognitive operators
        F_p = np.abs(np.vdot(state_a, self.P @ state_b))**2  # Perception fidelity
        F_a = np.abs(np.vdot(state_a, self.A @ state_b))**2  # Attention fidelity
        F_m = np.abs(np.vdot(state_a, self.M @ state_b))**2  # Memory fidelity

        # Combine fidelities
        F_cog = (F_p + F_a + F_m) / 3

        return float(F_cog)

    def evolve_cognitive_state(self,
                             initial_state: np.ndarray,
                             steps: int,
                             dt: float = 0.1) -> List[np.ndarray]:
        """
        Evolve cognitive state through time.

        Args:
            initial_state: Initial quantum state
            steps: Number of evolution steps
            dt: Time step size

        Returns:
            List[np.ndarray]: Time evolution of quantum state
        """
        if len(initial_state) != self.dim:
            raise ValueError("Initial state dimension must match system dimension")

        state = initial_state / np.linalg.norm(initial_state)
        evolution = [state]

        # Cognitive Hamiltonian
        H = self.P @ self.A + self.M
        H = (H + H.conj().T) / 2  # Ensure Hermiticity

        # Time evolution
        U = expm(-1j * H * dt)  # Evolution operator

        for _ in range(steps):
            state = U @ state
            state = state / np.linalg.norm(state)
            evolution.append(state.copy())

        return evolution

    def compute_cognitive_entropy(self, state: np.ndarray) -> float:
        """
        Compute cognitive entropy of a quantum state.

        Args:
            state: Quantum state

        Returns:
            float: Cognitive entropy
        """
        # Create density matrix
        rho = np.outer(state, state.conj())

        # Apply cognitive operators
        rho_c = self.P @ rho @ self.A @ self.M

        # Ensure Hermiticity and normalization
        rho_c = (rho_c + rho_c.conj().T) / 2
        rho_c = rho_c / (np.trace(rho_c) + self.epsilon)

        # Compute eigenvalues
        eigenvals = np.linalg.eigvalsh(rho_c)
        eigenvals = eigenvals[eigenvals > self.epsilon]

        # Compute von Neumann entropy
        S = -np.sum(eigenvals * np.log2(eigenvals))

        return float(S)
