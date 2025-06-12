import numpy as np
from typing import Optional, Union

class AdaptiveKuramoto:
    """
    A Kuramoto oscillator network with adaptive coupling and God Helmet perturbation support.
    """
    
    def __init__(
        self,
        num_oscillators: int = 100,
        base_freqs: Optional[np.ndarray] = None,
        coupling_strength: float = 1000.0,
        plasticity_rate: float = 5.0
    ):
        self.n = num_oscillators
        # All oscillators at exactly 1 Hz for easier synchronization
        if base_freqs is None:
            self.base_freqs = np.ones(num_oscillators)
        else:
            self.base_freqs = base_freqs
            
        # Initialize with strong coupling
        self.K = np.full((num_oscillators, num_oscillators), coupling_strength)
        np.fill_diagonal(self.K, 0)
        # Initialize phases very close together
        self.phase = np.random.uniform(-np.pi/64, np.pi/64, num_oscillators)
        self.plasticity_rate = plasticity_rate
        
        # Debug info
        print(f"Initial frequency spread: {np.max(self.base_freqs) - np.min(self.base_freqs):.6f} Hz")
        print(f"Initial phase spread: {np.max(self.phase) - np.min(self.phase):.6f} rad")

    def apply_god_helmet_effect(self, target_indices: Union[list, np.ndarray], phase_shift: float = np.pi/4) -> None:
        """Apply phase perturbation to simulate Persinger's God Helmet effect."""
        self.phase[target_indices] = (self.phase[target_indices] + phase_shift) % (2 * np.pi)

    def step(self, dt: float, external_force: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
        """Advance the simulation by one timestep.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``phase`` is the array of oscillator phases after the update and
            ``coupling_matrix`` is the current adaptive coupling matrix.
        """
        # Correct phase difference calculation for attractive coupling
        phase_diff = self.phase[None, :] - self.phase[:, None]  # θj - θi
        
        # Strong coupling effect with proper normalization
        coupling_effect = np.sum(self.K * np.sin(phase_diff), axis=1) / self.n
        dphase_dt = 2 * np.pi * self.base_freqs + coupling_effect  # Convert Hz to rad/s
        
        if external_force is not None:
            dphase_dt += external_force
        
        # Use smaller substeps for numerical stability
        dt_small = dt / 10
        for _ in range(10):
            self.phase += dphase_dt * dt_small
            self.phase %= 2 * np.pi
        
        # Update coupling based on phase alignment
        phase_alignment = np.cos(phase_diff)
        plasticity = self.plasticity_rate * np.power(phase_alignment, 2) * dt  # Quadratic term
        self.K += plasticity
        self.K = np.clip(self.K, 0, 2000)  # Allow strong coupling
        
        return self.phase, self.K
