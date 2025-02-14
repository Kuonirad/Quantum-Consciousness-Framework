"""
Validation tests for geometric pattern analyzer.
"""

import numpy as np
import pytest
from src.quantum_consciousness.core.geometric_pattern_analyzer import (
    GeometricPatternAnalyzer, PatternMetrics
)

def create_test_pattern(pattern_type: str) -> np.ndarray:
    """Create test pattern of specified type."""
    size = 64
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    
    if pattern_type == 'symmetric':
        return np.sin(5*X**2 + 5*Y**2)
    elif pattern_type == 'random':
        return np.random.randn(size, size)
    elif pattern_type == 'fractal':
        pattern = np.zeros((size, size))
        for i in range(5):
            pattern += np.sin(2**i * X) * np.sin(2**i * Y)
        return pattern
    else:
        raise ValueError(f"Unknown pattern type: {pattern_type}")

def test_symmetry_detection():
    """Test symmetry score computation."""
    analyzer = GeometricPatternAnalyzer(resolution=64)
    
    # Test symmetric pattern
    symmetric = create_test_pattern('symmetric')
    sym_score = analyzer.compute_symmetry_score(symmetric)
    
    # Test random pattern
    random = create_test_pattern('random')
    rand_score = analyzer.compute_symmetry_score(random)
    
    # Symmetric pattern should have higher symmetry score
    assert sym_score > rand_score
    assert 0 <= sym_score <= 1
    assert 0 <= rand_score <= 1

def test_fractal_dimension():
    """Test fractal dimension computation."""
    analyzer = GeometricPatternAnalyzer(resolution=64)
    
    # Test fractal pattern
    fractal = create_test_pattern('fractal')
    dim = analyzer.compute_fractal_dimension(fractal)
    
    # Dimension should be between 1 and 2 for 2D pattern
    assert 1 <= dim <= 2

def test_pattern_complexity():
    """Test pattern complexity computation."""
    analyzer = GeometricPatternAnalyzer(resolution=64)
    
    # Test different patterns
    symmetric = create_test_pattern('symmetric')
    random = create_test_pattern('random')
    
    sym_complexity = analyzer.compute_pattern_complexity(symmetric)
    rand_complexity = analyzer.compute_pattern_complexity(random)
    
    # Random pattern should have higher complexity
    assert rand_complexity > sym_complexity
    assert 0 <= sym_complexity <= 1
    assert 0 <= rand_complexity <= 1

def test_pattern_comparison():
    """Test pattern comparison functionality."""
    analyzer = GeometricPatternAnalyzer(resolution=64)
    
    # Compare pattern with itself
    pattern = create_test_pattern('symmetric')
    self_similarity = analyzer.compare_patterns(pattern, pattern)
    
    # Compare different patterns
    other = create_test_pattern('random')
    diff_similarity = analyzer.compare_patterns(pattern, other)
    
    # Self-similarity should be 1, different patterns should be less similar
    assert np.abs(self_similarity - 1.0) < 1e-10
    assert diff_similarity < self_similarity

def test_comprehensive_analysis():
    """Test comprehensive pattern analysis."""
    analyzer = GeometricPatternAnalyzer(resolution=64)
    pattern = create_test_pattern('symmetric')
    
    metrics = analyzer.analyze_pattern(pattern)
    
    # Verify metric properties
    assert isinstance(metrics, PatternMetrics)
    assert 0 <= metrics.symmetry_score <= 1
    assert 0 <= metrics.complexity <= 1
    assert 1 <= metrics.fractal_dimension <= 2
    assert 0 <= metrics.correlation_strength <= 1
