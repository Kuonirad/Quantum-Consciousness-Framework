"""
Quantum Information Geometry implementation.
Integrates Fubini-Study metric, quantum Fisher information, and Bures geometry.
"""

import numpy as np
from typing import Optional, Tuple, List, Union
from scipy.linalg import sqrtm, expm, logm

class QuantumInfoGeometry:
    """
    Implements quantum information geometric measures and metrics.
    Key metrics: Fubini-Study, quantum Fisher information, Bures distance.
    """

    def __init__(self, dim: int):
        """
        Initialize quantum information geometry system.

        Args:
            dim: Dimension of quantum system

        Raises:
            ValueError: If dimension is not positive
        """
        if dim <= 0:
            raise ValueError("Dimension must be positive")

        self.dim = dim
        self.epsilon = 1e-12  # Numerical stability threshold

    def fubini_study_metric(self,
                          psi: np.ndarray,
                          dpsi_i: np.ndarray,
                          dpsi_j: np.ndarray) -> complex:
        """
        Compute Fubini-Study metric: G_ij = Re(⟨∂_iψ|∂_jψ⟩ - ⟨∂_iψ|ψ⟩⟨ψ|∂_jψ⟩)

        Args:
            psi: Quantum state vector
            dpsi_i: Derivative of state in i direction
            dpsi_j: Derivative of state in j direction

        Returns:
            complex: Fubini-Study metric component
        """
        # Normalize state
        psi = psi / np.linalg.norm(psi)

        # Compute metric components
        term1 = np.vdot(dpsi_i, dpsi_j)
        term2 = np.vdot(dpsi_i, psi) * np.vdot(psi, dpsi_j)

        return term1 - term2

    def quantum_fisher_metric(self, rho: np.ndarray, A: np.ndarray) -> float:
        """
        Compute quantum Fisher information metric.
        F_Q(ρ, A) = 2 ∑[i,j] |⟨i|A|j⟩|²/(λᵢ+λⱼ)

        Args:
            rho: Density matrix
            A: Observable operator

        Returns:
            float: Quantum Fisher information
        """
        # Compute eigendecomposition
        eigenvals, eigenvecs = np.linalg.eigh(rho)
        eigenvals = np.maximum(eigenvals, self.epsilon)

        # Compute quantum Fisher information
        F_Q = 0.0
        for i in range(self.dim):
            for j in range(self.dim):
                if i != j:
                    denominator = eigenvals[i] + eigenvals[j]
                    if denominator > self.epsilon:
                        matrix_element = np.vdot(eigenvecs[:,i], A @ eigenvecs[:,j])
                        F_Q += np.abs(matrix_element)**2 * 2 / denominator

        return float(F_Q)

    def bures_metric(self, rho: np.ndarray, sigma: np.ndarray) -> float:
        """
        Compute Bures metric (infinitesimal distance).
        D_B(ρ, σ) = √(2-2√F(ρ, σ))

        Args:
            rho: First density matrix
            sigma: Second density matrix

        Returns:
            float: Bures distance
        """
        # Ensure Hermiticity
        rho = (rho + rho.conj().T) / 2
        sigma = (sigma + sigma.conj().T) / 2

        # Compute fidelity
        sqrt_rho = sqrtm(rho)
        fidelity = np.real(np.trace(sqrtm(sqrt_rho @ sigma @ sqrt_rho)))

        # Compute Bures distance
        return np.sqrt(max(0, 2 - 2 * np.sqrt(max(0, fidelity))))

    def parallel_transport(self,
                         psi: np.ndarray,
                         connection: np.ndarray) -> np.ndarray:
        """
        Implement parallel transport using quantum connection.

        Args:
            psi: Quantum state
            connection: Quantum connection (gauge field)

        Returns:
            numpy.ndarray: Transported state
        """
        # Normalize initial state
        psi = psi / np.linalg.norm(psi)

        # Apply parallel transport
        U = expm(-1j * connection)
        transported = U @ psi

        # Ensure normalization
        transported = transported / np.linalg.norm(transported)

        return transported

    def geometric_phase(self,
                       states: List[np.ndarray],
                       closed: bool = True) -> float:
        """
        Compute geometric (Berry) phase for a sequence of states.

        Args:
            states: List of quantum states forming a path
            closed: Whether the path is closed

        Returns:
            float: Geometric phase
        """
        phase = 0.0
        n_states = len(states)

        for i in range(n_states):
            # Get current and next state
            current = states[i]
            next_idx = (i + 1) % n_states
            next_state = states[next_idx]

            # Compute overlap
            overlap = np.vdot(current, next_state)


            # Accumulate phase
            phase += np.angle(overlap)

        if not closed:
            # Subtract dynamical phase for open paths
            total_overlap = np.vdot(states[0], states[-1])
            phase -= np.angle(total_overlap)

        return float(phase)

    def quantum_principal_bundle(self,
                               base_state: np.ndarray,
                               gauge_field: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Construct quantum principal bundle structure.

        Args:
            base_state: Reference quantum state
            gauge_field: Local gauge field

        Returns:
            Tuple[np.ndarray, np.ndarray]: (Horizontal lift, Vertical component)
        """
        # Normalize base state
        base_state = base_state / np.linalg.norm(base_state)

        # Compute projection operator
        P = np.outer(base_state, base_state.conj())

        # Horizontal component (gauge-covariant)
        H = (np.eye(self.dim) - P) @ gauge_field @ P

        # Vertical component (gauge transformation)
        V = P @ gauge_field @ P

        return H, V

    def information_metric_tensor(self,
                                rho: np.ndarray,
                                basis_operators: List[np.ndarray]) -> np.ndarray:
        """
        Compute quantum information metric tensor.

        Args:
            rho: Density matrix
            basis_operators: List of basis operators

        Returns:
            numpy.ndarray: Metric tensor
        """
        n_ops = len(basis_operators)
        g = np.zeros((n_ops, n_ops), dtype=complex)

        for i in range(n_ops):
            for j in range(n_ops):
                # Compute metric components
                g[i,j] = 0.5 * np.trace(
                    rho @ (basis_operators[i] @ basis_operators[j] +
                          basis_operators[j] @ basis_operators[i])
                )

        # Ensure Hermiticity
        g = (g + g.conj().T) / 2

        return g

    def symplectic_structure(self,
                           rho: np.ndarray,
                           basis_operators: List[np.ndarray]) -> np.ndarray:
        """
        Compute quantum symplectic structure.

        Args:
            rho: Density matrix
            basis_operators: List of basis operators

        Returns:
            numpy.ndarray: Symplectic form
        """
        n_ops = len(basis_operators)
        omega = np.zeros((n_ops, n_ops), dtype=complex)

        for i in range(n_ops):
            for j in range(n_ops):
                # Compute symplectic components
                omega[i,j] = -0.5j * np.trace(
                    rho @ (basis_operators[i] @ basis_operators[j] -
                          basis_operators[j] @ basis_operators[i])
                )

        return omega
