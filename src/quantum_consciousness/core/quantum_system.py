"""
Core quantum system implementation for the Quantum Consciousness Framework.
Implements a 10-qubit quantum system with integrated IIT measures and non-equilibrium dynamics.
"""

import numpy as np
from typing import Optional, Tuple, List, Union
import qutip as qt
import pennylane as qml
from scipy.linalg import logm, sqrtm
from .quantum_shannon_theory import QuantumShannonTheory

class QuantumSystem:
    """Implements a 10-qubit quantum system with consciousness measures."""

    def __init__(self, n_qubits: int = 10):
        """
        Initialize quantum system with n qubits.

        Args:
            n_qubits: Number of qubits (default: 10)

        Raises:
            ValueError: If n_qubits is not positive
        """
        if n_qubits <= 0:
            raise ValueError("Number of qubits must be positive")

        self.n_qubits = n_qubits
        self.dim = 2**n_qubits
        self.epsilon = 1e-12  # Numerical stability threshold
        self.psi = self._initialize_balanced_state()
        self.rho = self._density_matrix()
        self.hamiltonian = self._create_hamiltonian()

    def _initialize_balanced_state(self) -> np.ndarray:
        """
        Create balanced superposition state: ψ₀ = 1/√(2^n) * ∑[i=0 to 2^n-1] |i⟩

        Returns:
            numpy.ndarray: Initial quantum state vector
        """
        psi = np.ones(self.dim, dtype=complex) / np.sqrt(self.dim)
        return psi

    def _density_matrix(self) -> np.ndarray:
        """
        Compute density matrix ρ = |ψ⟩⟨ψ|

        Returns:
            numpy.ndarray: Density matrix

        Ensures:
            - Hermiticity: ρ = ρ†
            - Positive semi-definiteness
            - Trace normalization: Tr(ρ) = 1
        """
        rho = np.outer(self.psi, self.psi.conj())
        # Ensure Hermiticity
        rho = (rho + rho.conj().T) / 2
        # Ensure trace normalization
        rho = rho / np.trace(rho)
        return rho

    def _create_hamiltonian(self) -> np.ndarray:
        """
        Create system Hamiltonian with nearest-neighbor interactions and local fields
        H = ∑ᵢ J_i σˣᵢσˣᵢ₊₁ + ∑ᵢ hᵢσᶻᵢ

        Returns:
            numpy.ndarray: Hamiltonian matrix

        Ensures:
            - Hermiticity: H = H†
        """
        H = np.zeros((self.dim, self.dim), dtype=complex)

        # Coupling strength and local field strength
        J = 1.0  # Nearest-neighbor coupling
        h = 0.5  # Local field strength

        for i in range(self.n_qubits):
            # Local field terms
            sigma_z = qt.tensor([qt.qeye(2)] * i + [qt.sigmaz()] +
                              [qt.qeye(2)] * (self.n_qubits - i - 1))
            H += h * sigma_z.full()

            # Nearest-neighbor interactions
            if i < self.n_qubits - 1:
                sigma_x_i = qt.tensor([qt.qeye(2)] * i + [qt.sigmax()] +
                                    [qt.qeye(2)] * (self.n_qubits - i - 1))
                sigma_x_j = qt.tensor([qt.qeye(2)] * (i+1) + [qt.sigmax()] +
                                    [qt.qeye(2)] * (self.n_qubits - i - 2))
                H += J * (sigma_x_i.full() @ sigma_x_j.full())

        # Ensure Hermiticity
        H = (H + H.conj().T) / 2
        return H

    def compute_phi_measure(self) -> float:
        """
        Compute IIT's Φ measure using Quantum Fisher Information
        Φ ≈ F_Q(ρ, A) / (2 * ∑[i,j] (λᵢ+λⱼ))

        Returns:
            float: Integrated Information Φ

        Ensures:
            - Non-negativity: Φ ≥ 0
            - Finite value
        """
        # Compute eigendecomposition with numerical stability
        eigenvals, eigenvecs = np.linalg.eigh(self.rho)
        eigenvals = np.maximum(eigenvals, self.epsilon)  # Ensure positivity

        # Observable (use Hamiltonian as default observable)
        A = self.hamiltonian

        # Compute quantum Fisher information with numerical stability
        F_Q = 0.0
        for i in range(self.dim):
            for j in range(self.dim):
                if i != j:  # Skip diagonal terms to avoid numerical issues
                    denominator = eigenvals[i] + eigenvals[j]
                    if denominator > self.epsilon:
                        F_Q += (eigenvals[i] - eigenvals[j])**2 / denominator * \
                              np.abs(np.vdot(eigenvecs[:,i], A @ eigenvecs[:,j]))**2

        # Compute normalization with numerical stability
        norm_factor = 2 * sum(eigenvals[i] + eigenvals[j]
                            for i in range(self.dim)
                            for j in range(self.dim)
                            if eigenvals[i] + eigenvals[j] > self.epsilon)


        phi = F_Q / norm_factor if norm_factor > self.epsilon else 0.0
        return max(0.0, phi)  # Ensure non-negativity

    def time_evolution(self, dt: float, steps: int) -> List[np.ndarray]:
        """
        Implement non-equilibrium quantum dynamics
        ∂ρ/∂t = -i/ħ [H(t), ρ] + Λ(ρ, t)

        Args:
            dt: Time step
            steps: Number of evolution steps

        Returns:
            List[np.ndarray]: Time evolution of density matrix
        """
        hbar = 1.0  # Natural units
        evolution = []
        rho = self.rho.copy()

        for _ in range(steps):
            # Compute commutator [H, ρ]
            commutator = self.hamiltonian @ rho - rho @ self.hamiltonian

            # Lindblad term (decoherence)
            gamma = 0.01  # Decoherence rate
            lindblad = np.zeros_like(rho, dtype=complex)
            for i in range(self.n_qubits):
                sigma_z = qt.tensor([qt.qeye(2)] * i + [qt.sigmaz()] +
                                  [qt.qeye(2)] * (self.n_qubits - i - 1))
                sigma_z = sigma_z.full()
                lindblad += gamma * (sigma_z @ rho @ sigma_z - 0.5 * (sigma_z @ sigma_z @ rho + rho @ sigma_z @ sigma_z))

            # Update density matrix
            rho += dt * (-1j/hbar * commutator + lindblad)

            # Enforce Hermiticity and positivity
            rho = (rho + rho.conj().T) / 2
            eigvals, eigvecs = np.linalg.eigh(rho)
            eigvals = np.clip(eigvals, 0, None)
            rho = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T
            rho = rho / np.trace(rho)

            evolution.append(rho.copy())

        return evolution

    def apply_quantum_circuit(self) -> None:
        """
        Apply quantum circuit with Hadamard, CNOT chain, and Toffoli gates
        Circuit implements: Hⁿ ∘ CNOT_chain ∘ Toffoli_selected
        """
        dev = qml.device('default.qubit', wires=self.n_qubits)

        @qml.qnode(dev)
        def circuit():
            # Layer 1: Hadamard gates (creates superposition)
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)

            # Layer 2: CNOT chain (creates entanglement)
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i+1])
            qml.CNOT(wires=[self.n_qubits-1, 0])  # Complete the circle

            # Layer 3: Toffoli gates (implements three-qubit correlations)
            for i in range(0, self.n_qubits-2, 3):
                if i+2 < self.n_qubits:
                    qml.Toffoli(wires=[i, i+1, i+2])

            return qml.state()

        # Update quantum state and ensure normalization
        self.psi = circuit()
        self.psi = self.psi / np.linalg.norm(self.psi)
        self.rho = self._density_matrix()

    def compute_bures_distance(self, other_state: Union[np.ndarray, 'QuantumSystem']) -> float:
        """
        Compute Bures distance between current state and another state
        D_B(ρ, σ) = √(2-2√F(ρ, σ))

        Args:
            other_state: Another quantum state or system to compare with

        Returns:
            float: Bures distance

        Raises:
            ValueError: If dimensions don't match
        """
        if isinstance(other_state, QuantumSystem):
            other_rho = other_state.rho
        else:
            other_rho = other_state

        if other_rho.shape != self.rho.shape:
            raise ValueError("Quantum states must have same dimension")

        # Compute fidelity using matrix square root
        sqrt_rho = sqrtm(self.rho)
        fidelity = np.real(np.trace(sqrtm(sqrt_rho @ other_rho @ sqrt_rho)))

        # Compute Bures distance with numerical stability
        return np.sqrt(max(0, 2 - 2 * np.sqrt(max(0, fidelity))))

    def compute_quantum_cup_product(self, perception: np.ndarray, attention: np.ndarray, memory: np.ndarray,
                                  beta: float = 1.0, omega: float = 1.0) -> np.ndarray:
        """
        Compute quantum cup product for cognitive architecture integration
        a ∗_q b ≈ (Perception ∪ Attention)β q^{⟨β,ω⟩} (Memory)

        Args:
            perception: Quantum state representing perceptual input
            attention: Quantum state representing attentional filter
            memory: Quantum state representing memory state
            beta: Coupling strength parameter
            omega: Frequency parameter

        Returns:
            numpy.ndarray: Resulting quantum state after cognitive integration
        """
        # Ensure all states have correct dimensions
        if len(perception) != self.dim or len(attention) != self.dim or len(memory) != self.dim:
            raise ValueError(f"All states must have dimension {self.dim}")

        # Normalize input states
        perception = perception / np.linalg.norm(perception)
        attention = attention / np.linalg.norm(attention)
        memory = memory / np.linalg.norm(memory)

        # Combine perception, attention and memory
        result = perception + attention + memory

        # Apply global phase factor
        phase = np.exp(1j * beta * omega)
        result = phase * result

        # Normalize and ensure numerical stability
        norm = np.linalg.norm(result)
        if norm < self.epsilon:
            return np.zeros_like(result)
        return result / norm

    def compute_cognitive_fidelity(self, state_a: np.ndarray, state_b: np.ndarray) -> float:
        """
        Compute cognitive fidelity between quantum states
        F(ρ, σ) = |⟨ψ|φ⟩|²

        Args:
            state_a: First quantum state
            state_b: Second quantum state

        Returns:
            float: Cognitive fidelity measure
        """
        # Normalize states
        state_a = state_a / np.linalg.norm(state_a)
        state_b = state_b / np.linalg.norm(state_b)

        # Compute fidelity
        fidelity = np.abs(np.vdot(state_a, state_b))**2
        return float(fidelity)

    def evolve_cognitive_state(self, initial_state: np.ndarray,
                             perception_sequence: List[np.ndarray],
                             attention_sequence: List[np.ndarray],
                             steps: int = 10) -> List[np.ndarray]:
        """
        Evolve quantum cognitive state through time using cup product

        Args:
            initial_state: Initial cognitive state
            perception_sequence: Sequence of perceptual inputs
            attention_sequence: Sequence of attentional states
            steps: Number of evolution steps

        Returns:
            List[np.ndarray]: Evolution of cognitive state
        """
        evolution = [initial_state]
        current_state = initial_state.copy()

        for step in range(min(steps, len(perception_sequence), len(attention_sequence))):
            # Apply quantum cup product for cognitive evolution
            current_state = self.compute_quantum_cup_product(
                perception_sequence[step],
                attention_sequence[step],
                current_state
            )
            evolution.append(current_state.copy())

        return evolution

    def compute_fubini_study_metric(self, tangent_vectors: List[np.ndarray]) -> np.ndarray:
        n = len(tangent_vectors)
        G = np.zeros((n, n), dtype=complex)

        for i in range(n):
            for j in range(n):
                G[i,j] = np.vdot(tangent_vectors[i], tangent_vectors[j]) - \
                         np.vdot(tangent_vectors[i], self.psi) * \
                         np.vdot(self.psi, tangent_vectors[j])

        return np.real((G + G.conj().T) / 2)

    def parallel_transport(self, vector: np.ndarray, path: List[np.ndarray]) -> List[np.ndarray]:
        transported = [vector]
        current = vector.copy()

        for i in range(len(path)-1):
            inner1 = np.vdot(path[i], path[i+1])
            inner2 = np.vdot(path[i], current)

            current = current - path[i] * (np.vdot(path[i], current)) + \
                     path[i+1] * (inner2 * (1 - np.sqrt(1 - np.abs(inner1)**2)) / inner1)

            current = current / np.linalg.norm(current)
            transported.append(current)

        return transported

    def compute_geometric_phase(self, cycle: List[np.ndarray]) -> complex:
        phase = 1.0
        for i in range(len(cycle)-1):
            overlap = np.vdot(cycle[i], cycle[i+1])
            phase *= overlap / np.abs(overlap)

        overlap = np.vdot(cycle[-1], cycle[0])
        phase *= overlap / np.abs(overlap)

        return phase

    # ------------------------------------------------------------------
    # Information-theoretic helper methods used by visualization tools
    # ------------------------------------------------------------------

    def _partial_trace(self, rho: np.ndarray, keep: List[int]) -> np.ndarray:
        """Partial trace over qubits not in ``keep``."""
        if rho.shape != (self.dim, self.dim):
            raise ValueError("Density matrix has wrong dimension")

        if any(k >= self.n_qubits or k < 0 for k in keep):
            raise ValueError("Index out of range")

        keep_set = set(keep)
        trace_out = sorted(set(range(self.n_qubits)) - keep_set)

        dims = [2] * self.n_qubits
        tensor = rho.reshape(dims + dims)

        for idx in reversed(trace_out):
            tensor = np.trace(tensor, axis1=idx, axis2=idx + self.n_qubits)

        dim_keep = 2 ** len(keep)
        return tensor.reshape((dim_keep, dim_keep))

    def get_reduced_density_matrix(self, qubits: List[int]) -> np.ndarray:
        """Return the reduced state for the specified qubits."""
        return self._partial_trace(self.rho, qubits)

    def get_entropy(self, qubits: Optional[List[int]] = None) -> float:
        """Return the von Neumann entropy of the given subsystem."""
        if qubits is None:
            rho = self.rho
            dim = self.dim
        else:
            rho = self.get_reduced_density_matrix(qubits)
            dim = 2 ** len(qubits)

        qst = QuantumShannonTheory(dim)
        return float(qst.von_neumann_entropy(rho))

    def get_mutual_information(self, i: int, j: int) -> float:
        """Return the quantum mutual information between qubits ``i`` and ``j``."""
        rho_ij = self.get_reduced_density_matrix([i, j])
        qst = QuantumShannonTheory(4)
        return float(qst.quantum_mutual_information(rho_ij))
