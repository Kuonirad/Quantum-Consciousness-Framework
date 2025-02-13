import numpy as np
from typing import Optional, Union

class AdaptiveKuramoto:
    """
    A Kuramoto oscillator network with adaptive coupling and God Helmet perturbation support.
    
    Attributes:
        n (int): Number of oscillators.
        base_freqs (np.ndarray): Natural frequencies of oscillators (radians/sec).
        K (np.ndarray): Adaptive coupling matrix (n x n).
        phase (np.ndarray): Current phase of each oscillator (radians).
        plasticity_rate (float): Rate of coupling adjustment (Hebbian-like rule).
    """
    
    def __init__(
        self,
        num_oscillators: int = 100,
        base_freqs: Optional[np.ndarray] = None,
        coupling_strength: float = 0.1,
        plasticity_rate: float = 0.01
    ):
        self.n = num_oscillators
        self.base_freqs = base_freqs if base_freqs is not None else np.random.uniform(8, 12, num_oscillators)
        self.K = np.full((num_oscillators, num_oscillators), coupling_strength)
        np.fill_diagonal(self.K, 0)  # Remove self-coupling
        self.phase = np.random.uniform(0, 2 * np.pi, num_oscillators)
        self.plasticity_rate = plasticity_rate

    def apply_god_helmet_effect(self, target_indices: Union[list, np.ndarray], phase_shift: float = np.pi/4) -> None:
        """Apply phase perturbation to simulate Persinger's God Helmet effect."""
        self.phase[target_indices] += phase_shift
        self.phase %= 2 * np.pi

    def step(self, dt: float, external_force: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
        """Advance the simulation by one timestep."""
        phase_diff = self.phase[:, None] - self.phase  # Pairwise differences
        coupling_effect = np.sum(self.K * np.sin(phase_diff), axis=1) / self.n
        dphase_dt = self.base_freqs + coupling_effect
        
        if external_force is not None:
            dphase_dt += external_force
        
        self.phase += dphase_dt * dt
        self.phase %= 2 * np.pi
        
        # Update coupling matrix (Hebbian-like plasticity)
        phase_coherence = np.cos(phase_diff)
        self.K += self.plasticity_rate * phase_coherence * dt
        self.K = np.clip(self.K, 0, 1)  # Ensure bounded coupling
        
        return self.phase, self.K
