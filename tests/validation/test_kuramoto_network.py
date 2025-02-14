"""
Validation tests for Kuramoto oscillator network implementation.
"""

import numpy as np
import pytest
import torch
from src.quantum_consciousness.core.kuramoto_network import (
    KuramotoNetwork, LoihiInterface
)

def test_initialization():
    """Test proper initialization of Kuramoto network."""
    n_oscillators = 10
    network = KuramotoNetwork(n_oscillators)
    
    # Check dimensions
    assert len(network.state.phases) == n_oscillators
    assert len(network.state.frequencies) == n_oscillators
    assert network.state.coupling_matrix.shape == (n_oscillators, n_oscillators)
    
    # Check coupling matrix symmetry
    assert np.allclose(
        network.state.coupling_matrix,
        network.state.coupling_matrix.T
    )

def test_phase_evolution():
    """Test phase evolution and conservation laws."""
    network = KuramotoNetwork(n_oscillators=5)
    
    # Store initial state
    initial_phases = network.state.phases.copy()
    
    # Evolve system
    dt = 0.01
    for _ in range(100):
        network.step(dt)
        
    # Verify phases remain in [0, 2π]
    assert np.all(network.state.phases >= 0)
    assert np.all(network.state.phases <= 2*np.pi)
    
    # Verify phase differences have evolved
    assert not np.allclose(network.state.phases, initial_phases)

def test_order_parameter():
    """Test Kuramoto order parameter properties."""
    network = KuramotoNetwork(n_oscillators=10)
    
    # Compute order parameter
    r = network.compute_order_parameter()
    
    # Verify magnitude is in [0,1]
    assert np.abs(r) >= 0
    assert np.abs(r) <= 1
    
    # Set all phases equal
    network.state.phases[:] = np.pi/4
    r_aligned = network.compute_order_parameter()
    
    # Verify perfect alignment gives magnitude 1
    assert np.abs(r_aligned - 1.0) < 1e-10

def test_resonance_analysis():
    """Test resonance analysis functionality."""
    network = KuramotoNetwork(n_oscillators=5)
    
    # Analyze resonance
    metrics = network.analyze_resonance(
        target_freq=1.0,
        duration=10.0,
        dt=0.01
    )
    
    # Verify metric properties
    assert 0 <= metrics['mean_order'] <= 1
    assert metrics['Q_factor'] > 0
    assert metrics['bandwidth'] > 0
    assert metrics['final_coupling'] > 0

def test_loihi_interface():
    """Test Loihi interface spike encoding/decoding."""
    network = KuramotoNetwork(n_oscillators=5)
    interface = LoihiInterface(network)
    
    # Test encoding
    spikes = interface.encode_phases()
    assert isinstance(spikes, torch.Tensor)
    assert len(spikes.shape) == 1
    assert torch.all(spikes >= 0)
    assert torch.all(spikes <= 1)
    
    # Test decoding
    phases = interface.decode_spikes(spikes)
    assert isinstance(phases, np.ndarray)
    assert len(phases) == network.n
    assert np.all(np.abs(phases) <= np.pi)

def test_frequency_spectrum():
    """Test frequency spectrum computation."""
    network = KuramotoNetwork(n_oscillators=5)
    
    # Compute spectrum
    freqs, amps = network.compute_frequency_spectrum()
    
    # Verify basic properties
    assert len(freqs) == network.n
    assert len(amps) == network.n
    assert np.all(amps >= 0)
    
    # Verify Hermitian symmetry of FFT
    assert np.allclose(amps, np.flip(amps))
