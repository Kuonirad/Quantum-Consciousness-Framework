import numpy as np
from src.neural.oscillators import AdaptiveKuramoto

def test_kuramoto_synchronization():
    network = AdaptiveKuramoto(num_oscillators=50, coupling_strength=0.5)
    initial_phases = network.phase.copy()
    
    # Simulate for 5 seconds with 10ms steps
    for _ in range(500):
        network.step(dt=0.01)
    
    # Check if phases synchronize (order parameter > 0.8)
    order_param = np.abs(np.mean(np.exp(1j * network.phase)))
    assert order_param > 0.8, "Failed to synchronize"

def test_god_helmet_perturbation():
    network = AdaptiveKuramoto(num_oscillators=100)
    target_nodes = np.array([10, 11, 12, 13, 14])  # Temporal lobe subset
    initial_phases = network.phase[target_nodes].copy()
    
    network.apply_god_helmet_effect(target_nodes, phase_shift=np.pi/2)
    perturbed_phases = network.phase[target_nodes]
    
    # Check if phases shifted by ~π/2
    phase_diff = np.mean(np.abs(perturbed_phases - initial_phases))
    assert np.isclose(phase_diff, np.pi/2, atol=0.1), "God Helmet perturbation failed"
