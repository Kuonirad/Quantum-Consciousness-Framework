"""
Kuramoto oscillator network implementation for resonance testing.
Implements coupled oscillator dynamics with frequency adaptation.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class OscillatorState:
    """State of Kuramoto oscillator network."""
    phases: np.ndarray  # Phase of each oscillator
    frequencies: np.ndarray  # Natural frequencies
    coupling_matrix: np.ndarray  # Coupling strengths between oscillators

class KuramotoNetwork:
    """Implements Kuramoto oscillator network for resonance analysis."""
    
    def __init__(self, 
                 n_oscillators: int,
                 coupling_strength: float = 1.0,
                 plasticity_rate: float = 0.1):
        """
        Initialize Kuramoto network.
        
        Args:
            n_oscillators: Number of oscillators
            coupling_strength: Base coupling strength
            plasticity_rate: Learning rate for coupling adaptation
        """
        self.n = n_oscillators
        self.K = coupling_strength
        self.eta = plasticity_rate
        
        # Initialize network state
        self.state = self._initialize_state()
        
    def _initialize_state(self) -> OscillatorState:
        """Initialize oscillator network state."""
        # Random initial phases
        phases = np.random.uniform(0, 2*np.pi, self.n)
        
        # Natural frequencies from normal distribution
        frequencies = np.random.normal(0, 1, self.n)
        
        # Initialize coupling matrix
        coupling = np.random.normal(0, 0.1, (self.n, self.n))
        coupling = (coupling + coupling.T) / 2  # Make symmetric
        
        return OscillatorState(phases, frequencies, coupling)
        
    def step(self, dt: float) -> None:
        """
        Evolve network state by one timestep.
        
        Args:
            dt: Time step size
        """
        # Compute phase differences
        phase_diffs = np.subtract.outer(self.state.phases, self.state.phases)
        
        # Compute coupling terms
        coupling_terms = self.state.coupling_matrix * np.sin(phase_diffs)
        
        # Update phases
        dphases = (self.state.frequencies + 
                  self.K * np.sum(coupling_terms, axis=1)) * dt
        self.state.phases += dphases
        
        # Apply coupling plasticity (Hebbian learning)
        phase_coherence = np.cos(phase_diffs)
        dcoupling = self.eta * phase_coherence * dt
        self.state.coupling_matrix += dcoupling
        
    def compute_order_parameter(self) -> complex:
        """
        Compute Kuramoto order parameter r*exp(iψ).
        
        Returns:
            Complex order parameter
        """
        return complex(np.mean(np.exp(1j * self.state.phases)))
        
    def compute_frequency_spectrum(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute frequency spectrum of oscillator phases.
        
        Returns:
            Tuple of (frequencies, amplitudes)
        """
        # Compute FFT of phase time series
        fft = np.fft.fft(np.exp(1j * self.state.phases))
        freqs = np.fft.fftfreq(len(fft))
        
        return freqs, np.abs(fft)
        
    def analyze_resonance(self, 
                         target_freq: float,
                         duration: float,
                         dt: float = 0.01) -> dict:
        """
        Analyze resonance characteristics near target frequency.
        
        Args:
            target_freq: Target frequency to analyze
            duration: Total simulation time
            dt: Time step size
            
        Returns:
            Dictionary containing resonance metrics
        """
        n_steps = int(duration / dt)
        order_params = np.zeros(n_steps, dtype=complex)
        
        # Run simulation
        for i in range(n_steps):
            self.step(dt)
            order_params[i] = self.compute_order_parameter()
            
        # Compute metrics
        mean_order = np.mean(np.abs(order_params))
        freq_response = np.fft.fft(order_params)
        peak_freq = np.abs(np.fft.fftfreq(len(freq_response)))[
            np.argmax(np.abs(freq_response))
        ]
        
        # Quality factor (Q) estimation
        power_spectrum = np.abs(freq_response)**2
        peak_idx = np.argmax(power_spectrum)
        half_power = power_spectrum[peak_idx] / 2
        
        # Find bandwidth
        left_idx = np.where(power_spectrum[:peak_idx] < half_power)[0][-1]
        right_idx = peak_idx + np.where(power_spectrum[peak_idx:] < half_power)[0][0]
        bandwidth = right_idx - left_idx
        Q_factor = peak_idx / bandwidth if bandwidth > 0 else float('inf')
        
        return {
            'mean_order': mean_order,
            'peak_frequency': peak_freq,
            'Q_factor': Q_factor,
            'bandwidth': bandwidth * np.fft.fftfreq(len(freq_response))[1],
            'final_coupling': np.mean(np.abs(self.state.coupling_matrix))
        }

class LoihiInterface:
    """Interface for Intel Loihi 3 neuromorphic chip integration."""
    
    def __init__(self, network: KuramotoNetwork):
        """
        Initialize Loihi interface.
        
        Args:
            network: KuramotoNetwork instance to interface with
        """
        self.network = network
        self.spike_encoder = self._create_spike_encoder()
        self.spike_decoder = self._create_spike_decoder()
        
    def _create_spike_encoder(self) -> nn.Module:
        """Create neural encoder for phase-to-spike conversion."""
        return nn.Sequential(
            nn.Linear(self.network.n, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.Sigmoid()
        )
        
    def _create_spike_decoder(self) -> nn.Module:
        """Create neural decoder for spike-to-phase conversion."""
        return nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.network.n),
            nn.Tanh()
        )
        
    def encode_phases(self) -> torch.Tensor:
        """Encode oscillator phases as spike patterns."""
        phases = torch.tensor(self.network.state.phases, dtype=torch.float32)
        return self.spike_encoder(phases.unsqueeze(0)).squeeze(0)
        
    def decode_spikes(self, spikes: torch.Tensor) -> np.ndarray:
        """Decode spike patterns back to phases."""
        phases = self.spike_decoder(spikes.unsqueeze(0)).squeeze(0)
        return phases.detach().numpy() * np.pi
