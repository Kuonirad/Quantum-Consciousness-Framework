"""
Geometric pattern analyzer for crystallographic pattern quantification.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import signal, stats
from dataclasses import dataclass

@dataclass
class PatternMetrics:
    """Metrics for geometric pattern analysis."""
    symmetry_score: float  # Measure of pattern symmetry
    complexity: float  # Pattern complexity measure
    fractal_dimension: float  # Fractal dimension estimate
    correlation_strength: float  # Intent-pattern correlation
    
class GeometricPatternAnalyzer:
    """Analyzes geometric patterns in crystallographic data."""
    
    def __init__(self, resolution: int = 512):
        """
        Initialize pattern analyzer.
        
        Args:
            resolution: Image resolution for pattern analysis
        """
        self.resolution = resolution
        
    def compute_symmetry_score(self, pattern: np.ndarray) -> float:
        """
        Compute symmetry score of pattern.
        
        Args:
            pattern: 2D array representing pattern
            
        Returns:
            Symmetry score in [0,1]
        """
        # Compute rotational symmetry
        rot_scores = []
        center = pattern.shape[0] // 2
        
        for angle in range(30, 360, 30):
            rotated = signal.rotate(pattern, angle)
            similarity = np.corrcoef(
                pattern.flatten(),
                rotated.flatten()
            )[0,1]
            rot_scores.append(abs(similarity))
            
        # Compute reflection symmetry
        ref_scores = []
        for axis in range(0, 180, 45):
            reflected = signal.rotate(np.fliplr(pattern), axis)
            similarity = np.corrcoef(
                pattern.flatten(),
                reflected.flatten()
            )[0,1]
            ref_scores.append(abs(similarity))
            
        return np.mean(rot_scores + ref_scores)
        
    def compute_fractal_dimension(self, pattern: np.ndarray) -> float:
        """
        Compute fractal dimension using box-counting method.
        
        Args:
            pattern: 2D array representing pattern
            
        Returns:
            Estimated fractal dimension
        """
        # Convert to binary
        threshold = np.mean(pattern)
        binary = pattern > threshold
        
        # Box counting at different scales
        scales = np.logspace(0, 4, num=20, base=2, dtype=int)
        counts = []
        
        for scale in scales:
            if scale == 0:
                continue
            # Downsample pattern
            downsampled = signal.resample_poly(
                signal.resample_poly(binary, 1, scale, axis=0),
                1, scale, axis=1
            )
            counts.append(np.sum(downsampled > 0.5))
            
        # Compute dimension from log-log slope
        scales = scales[1:]  # Remove scale=0
        counts = np.array(counts)
        slope, _, _, _, _ = stats.linregress(
            np.log(1/scales),
            np.log(counts)
        )
        
        return slope
        
    def compute_pattern_complexity(self, pattern: np.ndarray) -> float:
        """
        Compute pattern complexity using entropy and gradient measures.
        
        Args:
            pattern: 2D array representing pattern
            
        Returns:
            Complexity score in [0,1]
        """
        # Compute gradient entropy
        gx, gy = np.gradient(pattern)
        grad_mag = np.sqrt(gx**2 + gy**2)
        grad_hist = np.histogram(grad_mag, bins=50)[0]
        grad_hist = grad_hist / np.sum(grad_hist)
        grad_entropy = -np.sum(
            grad_hist * np.log2(grad_hist + 1e-10)
        )
        
        # Compute spatial entropy
        spatial_hist = np.histogram(pattern, bins=50)[0]
        spatial_hist = spatial_hist / np.sum(spatial_hist)
        spatial_entropy = -np.sum(
            spatial_hist * np.log2(spatial_hist + 1e-10)
        )
        
        # Combine measures
        complexity = (grad_entropy + spatial_entropy) / 2
        return complexity / np.log2(50)  # Normalize by max entropy
        
    def analyze_pattern(self, pattern: np.ndarray) -> PatternMetrics:
        """
        Perform comprehensive pattern analysis.
        
        Args:
            pattern: 2D array representing pattern
            
        Returns:
            PatternMetrics object with analysis results
        """
        # Ensure pattern is properly sized
        pattern = signal.resample(
            signal.resample(pattern, self.resolution, axis=0),
            self.resolution, axis=1
        )
        
        # Compute metrics
        symmetry = self.compute_symmetry_score(pattern)
        complexity = self.compute_pattern_complexity(pattern)
        fractal_dim = self.compute_fractal_dimension(pattern)
        
        # Compute correlation strength (placeholder)
        correlation = np.mean([symmetry, complexity, fractal_dim])
        
        return PatternMetrics(
            symmetry_score=symmetry,
            complexity=complexity,
            fractal_dimension=fractal_dim,
            correlation_strength=correlation
        )
        
    def compare_patterns(self, 
                        pattern1: np.ndarray,
                        pattern2: np.ndarray) -> float:
        """
        Compare similarity between two patterns.
        
        Args:
            pattern1, pattern2: 2D arrays representing patterns
            
        Returns:
            Similarity score in [0,1]
        """
        # Compute metrics for both patterns
        metrics1 = self.analyze_pattern(pattern1)
        metrics2 = self.analyze_pattern(pattern2)
        
        # Compare metric vectors
        v1 = np.array([
            metrics1.symmetry_score,
            metrics1.complexity,
            metrics1.fractal_dimension
        ])
        v2 = np.array([
            metrics2.symmetry_score,
            metrics2.complexity,
            metrics2.fractal_dimension
        ])
        
        # Compute cosine similarity
        similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return float(similarity)  # Ensure return type is float
