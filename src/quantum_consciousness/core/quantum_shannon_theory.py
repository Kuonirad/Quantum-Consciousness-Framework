"""
Quantum Shannon theory implementation for quantum information measures.
Implements von Neumann entropy, quantum mutual information, and relative entropy.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Callable
from scipy.linalg import logm, sqrtm

class QuantumShannonTheory:
    """Implements quantum information theory calculations."""

    def __init__(self, dim: int):
        """
        Initialize quantum information calculator.

        Args:
            dim: Dimension of the quantum system
        """
        self.dim = dim
        self.epsilon = 1e-12  # Small value for numerical stability

    def von_neumann_entropy(self, rho: np.ndarray) -> float:
        """
        Calculate von Neumann entropy
        S(ρ) = -Tr(ρ log₂ρ)

        Args:
            rho: Density matrix

        Returns:
            float: von Neumann entropy
        """
        eigenvals = np.linalg.eigvalsh(rho)
        # Remove small negative eigenvalues due to numerical errors
        eigenvals = eigenvals[eigenvals > self.epsilon]
        return -np.sum(eigenvals * np.log2(eigenvals))

    def quantum_relative_entropy(self, rho: np.ndarray, sigma: np.ndarray) -> float:
        """
        Calculate quantum relative entropy
        D(ρ||σ) = Tr(ρ(log₂ρ - log₂σ))

        Args:
            rho: First density matrix
            sigma: Second density matrix

        Returns:
            float: Quantum relative entropy
        """
        # Handle support issues
        support_rho = np.linalg.matrix_rank(rho)
        support_sigma = np.linalg.matrix_rank(sigma)

        if support_rho > support_sigma:
            return np.inf

        log_rho = logm(rho + self.epsilon * np.eye(self.dim))
        log_sigma = logm(sigma + self.epsilon * np.eye(self.dim))

        return np.real(np.trace(rho @ (log_rho - log_sigma))) / np.log(2)

    def quantum_mutual_information(self, rho_ab: np.ndarray) -> float:
        """
        Calculate quantum mutual information
        I(A:B) = S(ρᴬ) + S(ρᴮ) - S(ρᴬᴮ)

        Args:
            rho_ab: Joint density matrix

        Returns:
            float: Quantum mutual information
        """
        # Calculate reduced density matrices
        dim_a = int(np.sqrt(self.dim))
        dim_b = self.dim // dim_a

        rho_a = np.trace(rho_ab.reshape(dim_a, dim_b, dim_a, dim_b), axis1=1, axis2=3)
        rho_b = np.trace(rho_ab.reshape(dim_a, dim_b, dim_a, dim_b), axis1=0, axis2=2)

        # Calculate entropies
        S_a = self.von_neumann_entropy(rho_a)
        S_b = self.von_neumann_entropy(rho_b)
        S_ab = self.von_neumann_entropy(rho_ab)

        return S_a + S_b - S_ab

    def conditional_quantum_entropy(self, rho_ab: np.ndarray) -> float:
        """
        Calculate conditional quantum entropy
        S(A|B) = S(AB) - S(B)

        Args:
            rho_ab: Joint density matrix

        Returns:
            float: Conditional quantum entropy
        """
        # Calculate reduced density matrix for B
        dim_b = self.dim // 2
        rho_b = np.trace(rho_ab.reshape(2, dim_b, 2, dim_b), axis1=0, axis2=2)

        # Calculate entropies
        S_ab = self.von_neumann_entropy(rho_ab)
        S_b = self.von_neumann_entropy(rho_b)

        return S_ab - S_b

    def quantum_discord(self, rho_ab: np.ndarray) -> float:
        """
        Calculate quantum discord
        D(A|B) = I(A:B) - J(A:B)

        Args:
            rho_ab: Joint density matrix

        Returns:
            float: Quantum discord
        """
        # Calculate mutual information
        I_ab = self.quantum_mutual_information(rho_ab)

        # Calculate classical correlation (optimization over measurements)
        J_ab = self._optimize_classical_correlation(rho_ab)

        return I_ab - J_ab

    def _optimize_classical_correlation(self, rho_ab: np.ndarray) -> float:
        """
        Optimize classical correlation over measurements
        J(A:B) = max_{Π_b} [S(ρᴬ) - ∑_b p_b S(ρᴬ|b)]

        Args:
            rho_ab: Joint density matrix

        Returns:
            float: Optimized classical correlation
        """
        # Implement numerical optimization over projective measurements
        # This is a simplified version - full optimization would require more sophisticated methods
        dim_b = self.dim // 2
        max_correlation = -np.inf

        # Sample random measurements
        for _ in range(100):
            # Generate random unitary
            U = self._random_unitary(dim_b)

            # Create measurement operators
            Pi = [U @ np.diag([1 if i == j else 0 for j in range(dim_b)]) @ U.conj().T
                 for i in range(dim_b)]

            # Calculate classical correlation for this measurement
            correlation = self._classical_correlation_for_measurement(rho_ab, Pi)
            max_correlation = max(max_correlation, correlation)

        return max_correlation

    def _random_unitary(self, dim: int) -> np.ndarray:
        """Generate random unitary matrix."""
        X = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
        Q, R = np.linalg.qr(X)
        return Q

    def _classical_correlation_for_measurement(self, rho_ab: np.ndarray,
                                            Pi: List[np.ndarray]) -> float:
        """Calculate classical correlation for given measurement."""
        dim_a = self.dim // 2
        rho_a = np.trace(rho_ab.reshape(dim_a, 2, dim_a, 2), axis1=1, axis2=3)
        S_a = self.von_neumann_entropy(rho_a)

        # Calculate conditional entropy for measurement
        S_cond = 0
        for P in Pi:
            p_b = np.real(np.trace(rho_ab @ np.kron(np.eye(dim_a), P)))
            if p_b > self.epsilon:
                rho_cond = np.trace(rho_ab @ np.kron(np.eye(dim_a), P)) / p_b
                S_cond += p_b * self.von_neumann_entropy(rho_cond)

        return S_a - S_cond

    def quantum_channel_capacity(self, channel: Callable[[np.ndarray], np.ndarray]) -> float:
        """
        Calculate quantum channel capacity
        Q(Φ) = max_ρ [S(Φ(ρ)) - S_e(Φ, ρ)]

        Args:
            channel: Quantum channel function

        Returns:
            float: Quantum channel capacity
        """
        # Implement numerical optimization over input states
        max_capacity = -np.inf

        # Sample random pure states
        for _ in range(100):
            psi = np.random.randn(self.dim) + 1j * np.random.randn(self.dim)
            psi = psi / np.linalg.norm(psi)
            rho = np.outer(psi, psi.conj())

            # Calculate output entropy
            rho_out = channel(rho)
            S_out = self.von_neumann_entropy(rho_out)

            # Calculate entropy exchange
            S_e = self._entropy_exchange(channel, rho)

            capacity = S_out - S_e
            max_capacity = max(max_capacity, capacity)

        return max_capacity

    def _entropy_exchange(self, channel: Callable[[np.ndarray], np.ndarray],
                        rho: np.ndarray) -> float:
        """Calculate entropy exchange for channel."""
        # Create Choi matrix
        choi = np.zeros((self.dim**2, self.dim**2), dtype=complex)
        for i in range(self.dim):
            basis = np.zeros(self.dim)
            basis[i] = 1
            out = channel(np.outer(basis, basis.conj()))
            choi += np.kron(out, np.outer(basis, basis.conj()))

        return self.von_neumann_entropy(choi / self.dim)
