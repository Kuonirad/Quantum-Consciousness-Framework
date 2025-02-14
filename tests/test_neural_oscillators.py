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
    steps = 200000  # 20 seconds
    
    # Print intermediate order parameters
    for step in range(steps):
        network.step(dt=dt)
        if step % 20000 == 0:  # Every 2 seconds
            order_param = np.abs(np.mean(np.exp(1j * network.phase)))
            print(f"Step {step}, Time {step*dt:.2f}s: Order parameter = {order_param:.4f}")
            print(f"Mean coupling strength: {np.mean(network.K):.2f}")
            print(f"Phase spread: {np.max(network.phase) - np.min(network.phase):.4f} rad")
    
    # Calculate final order parameter
    order_param = np.abs(np.mean(np.exp(1j * network.phase)))
    print(f"\nFinal order parameter: {order_param:.4f}")
    print(f"Final mean coupling: {np.mean(network.K):.2f}")
    print(f"Final coupling range: [{np.min(network.K):.2f}, {np.max(network.K):.2f}]")
    print(f"Final phase spread: {np.max(network.phase) - np.min(network.phase):.4f} rad")
    
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
