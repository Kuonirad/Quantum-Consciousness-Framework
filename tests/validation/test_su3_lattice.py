"""
Validation tests for SU(3) lattice QCD implementation.
"""

import numpy as np
import pytest
from src.quantum_consciousness.core.su3_lattice import SU3LatticeQCD

def test_gell_mann_algebra():
    """Test Gell-Mann matrix algebra and commutation relations."""
    lattice = SU3LatticeQCD(lattice_size=4, spacing=0.1)
    generators = lattice._initialize_gell_mann_matrices()
    
    # Test hermiticity
    for g in generators:
        assert np.allclose(g, g.conj().T)
        
    # Test tracelessness
    for g in generators:
        assert np.abs(np.trace(g)) < 1e-10
        
    # Test normalization
    for g in generators:
        assert np.allclose(np.trace(g @ g), 2)
        
    # Test key commutation relations
    # [λ₁,λ₂] = 2iλ₃ etc.
    assert np.allclose(
        generators[0] @ generators[1] - generators[1] @ generators[0],
        2j * generators[2]
    )

def test_gauge_invariance():
    """Test gauge transformation invariance of Wilson loops."""
    lattice = SU3LatticeQCD(lattice_size=4, spacing=0.1)
    
    # Compute initial Wilson loop
    W1 = lattice.compute_wilson_loop(size=(2,2))
    
    # Apply random gauge transformation
    alpha = np.random.randn(8) * 0.1
    lattice.gauge_transform(alpha)
    
    # Compute transformed Wilson loop
    W2 = lattice.compute_wilson_loop(size=(2,2))
    
    # Verify gauge invariance
    assert np.abs(W1 - W2) < 1e-10

def test_unitarity_preservation():
    """Test preservation of unitarity under gauge transformations."""
    lattice = SU3LatticeQCD(lattice_size=4, spacing=0.1)
    
    # Verify initial unitarity
    assert lattice.verify_unitarity()
    
    # Apply series of gauge transformations
    for _ in range(10):
        alpha = np.random.randn(8) * 0.1
        lattice.gauge_transform(alpha)
        
        # Verify unitarity is preserved
        assert lattice.verify_unitarity()

def test_wilson_loop_properties():
    """Test basic properties of Wilson loops."""
    lattice = SU3LatticeQCD(lattice_size=4, spacing=0.1)
    
    # Test symmetry under exchange of R and T
    W_RT = lattice.compute_wilson_loop(size=(2,3))
    W_TR = lattice.compute_wilson_loop(size=(3,2))
    
    # Should be complex conjugates
    assert np.abs(W_RT - W_TR.conjugate()) < 1e-10
    
    # Test area law scaling
    W1 = np.abs(lattice.compute_wilson_loop(size=(1,1)))
    W2 = np.abs(lattice.compute_wilson_loop(size=(2,2)))
    
    # Larger loops should have smaller absolute values
    assert W2 < W1
