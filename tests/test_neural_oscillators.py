import numpy as np
from src.neural.oscillators import AdaptiveKuramoto

def test_kuramoto_synchronization():
    # Parameters for identical oscillators
    network = AdaptiveKuramoto(
        num_oscillators=50,
        coupling_strength=1000.0,
        plasticity_rate=5.0
    )
    
    # Use smaller timestep
    dt = 0.0001  # 0.1ms timestep
    steps = 1000  # reduced from 200000 for faster test execution
    
    for step in range(steps):
        network.step(dt=dt)
    
    # Calculate final order parameter
    order_param = np.abs(np.mean(np.exp(1j * network.phase)))
    
    assert order_param > 0.8, f"Failed to synchronize: order parameter {order_param:.2f} below 0.8"

def test_god_helmet_perturbation():
    # Create a network with 100 oscillators
    network = AdaptiveKuramoto(num_oscillators=100)
    target_nodes = np.array([10, 11, 12, 13, 14])  # Temporal lobe subset
    initial_phases = network.phase[target_nodes].copy()
    
    # Apply God Helmet perturbation with a phase shift of π/2
    network.apply_god_helmet_effect(target_nodes, phase_shift=np.pi/2)
    perturbed_phases = network.phase[target_nodes]
    
    # Compute the minimal circular difference between phases
    phase_diff = np.angle(np.exp(1j * (perturbed_phases - initial_phases)))
    mean_diff = np.mean(np.abs(phase_diff))
    assert np.isclose(mean_diff, np.pi/2, atol=0.1), f"God Helmet perturbation failed: mean phase difference {mean_diff} is not close to {np.pi/2}"
