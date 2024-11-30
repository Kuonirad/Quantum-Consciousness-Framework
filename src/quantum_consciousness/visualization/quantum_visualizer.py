"""
Quantum State Visualization Module
================================

Provides advanced visualization tools for quantum states, dynamics, and consciousness measures
using matplotlib, plotly, and custom OpenGL rendering.

Author: Kevin John Kull
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union, Tuple, List
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from ..core.quantum_system import QuantumSystem
from . import DEFAULT_SETTINGS

class QuantumVisualizer:
    """Handles visualization of quantum states and dynamics."""

    def __init__(self, settings: Optional[dict] = None):
        """
        Initialize visualizer with custom or default settings.

        Args:
            settings: Optional custom visualization settings
        """
        self.settings = DEFAULT_SETTINGS.copy()
        if settings:
            self.settings.update(settings)

        plt.style.use(self.settings['plot_style'])

        # Initialize AI visualization components
        self.state_encoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # 3D embedding
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.state_encoder.to(self.device)

    def plot_state_vector(self, state: np.ndarray, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Plot quantum state vector with AI-enhanced visualization.

        Args:
            state: Complex state vector
            ax: Optional matplotlib axes

        Returns:
            matplotlib.axes.Axes: Plot axes
        """
        if ax is None:
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')

        # Convert state to tensor
        state_tensor = torch.from_numpy(state).to(self.device)
        if len(state_tensor.shape) == 1:
            state_tensor = state_tensor.unsqueeze(0)

        # Generate AI-enhanced 3D embedding
        with torch.no_grad():
            embedding = self.state_encoder(state_tensor.abs()).cpu().numpy()

        # Extract quantum properties
        amplitudes = np.abs(state)
        phases = np.angle(state)

        # Create enhanced 3D scatter plot
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2],
                           c=phases, cmap=self.settings['phase_colormap'],
                           s=amplitudes * 500, alpha=0.6)

        # Add quantum probability streams
        for i in range(len(state)-1):
            ax.plot([embedding[i,0], embedding[i+1,0]],
                   [embedding[i,1], embedding[i+1,1]],
                   [embedding[i,2], embedding[i+1,2]],
                   alpha=min(amplitudes[i], amplitudes[i+1]),
                   color='cyan')

        # Customize visualization
        ax.set_xlabel('Quantum Dimension X')
        ax.set_ylabel('Quantum Dimension Y')
        ax.set_zlabel('Quantum Dimension Z')
        ax.set_title('AI-Enhanced Quantum State Visualization')

        # Add colorbar for phase information
        plt.colorbar(scatter, label='Phase')

        return ax

    def plot_density_matrix(self, rho: np.ndarray, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Plot density matrix as a 3D surface.

        Args:
            rho: Density matrix
            ax: Optional 3D axes

        Returns:
            matplotlib.axes.Axes: 3D plot axes
        """
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

        dim = rho.shape[0]
        x, y = np.meshgrid(range(dim), range(dim))

        # Plot real and imaginary parts
        ax.plot_surface(x, y, np.real(rho), cmap='viridis', alpha=0.6)
        ax.plot_surface(x, y, np.imag(rho), cmap='plasma', alpha=0.4)

        ax.set_xlabel('Row Index')
        ax.set_ylabel('Column Index')
        ax.set_zlabel('Value')
        ax.set_title('Density Matrix')

        return ax

    def animate_evolution(self, states: List[np.ndarray],
                         interval: int = 50) -> plt.animation.FuncAnimation:
        """
        Animate quantum state evolution.

        Args:
            states: List of state vectors or density matrices
            interval: Animation interval in milliseconds

        Returns:
            matplotlib.animation.FuncAnimation: Animation object
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        def update(frame):
            ax.clear()
            self.plot_state_vector(states[frame], ax)

        anim = plt.animation.FuncAnimation(fig, update, frames=len(states),
                                         interval=interval, blit=False)
        return anim

    def plot_entanglement_graph(self, system: QuantumSystem) -> go.Figure:
        """
        Create AI-enhanced interactive entanglement visualization.

        Args:
            system: QuantumSystem instance

        Returns:
            plotly.graph_objects.Figure: Interactive plot
        """
        n_qubits = system.n_qubits

        # Compute comprehensive entanglement metrics
        entanglement_matrix = np.zeros((n_qubits, n_qubits))
        mutual_info_matrix = np.zeros((n_qubits, n_qubits))

        for i in range(n_qubits):
            for j in range(i+1, n_qubits):
                # Compute reduced density matrix and entropy
                rho_ij = system.get_reduced_density_matrix([i, j])
                entanglement_matrix[i,j] = system.get_entropy()
                mutual_info_matrix[i,j] = system.get_mutual_information(i, j)
                entanglement_matrix[j,i] = entanglement_matrix[i,j]
                mutual_info_matrix[j,i] = mutual_info_matrix[i,j]

        # Use t-SNE for optimal layout
        positions = TSNE(n_components=2).fit_transform(entanglement_matrix)

        # Create interactive visualization
        fig = go.Figure()

        # Add entanglement connections
        for i in range(n_qubits):
            for j in range(i+1, n_qubits):
                if entanglement_matrix[i,j] > 0.1:  # Significant entanglement threshold
                    fig.add_trace(go.Scatter(
                        x=[positions[i,0], positions[j,0]],
                        y=[positions[i,1], positions[j,1]],
                        mode='lines',
                        line=dict(
                            width=entanglement_matrix[i,j] * 5,
                            color=f'rgba(100,200,255,{mutual_info_matrix[i,j]})'
                        ),
                        hoverinfo='text',
                        text=f'Entanglement: {entanglement_matrix[i,j]:.3f}'
                    ))

        # Add qubit nodes
        fig.add_trace(go.Scatter(
            x=positions[:,0],
            y=positions[:,1],
            mode='markers+text',
            marker=dict(
                size=20,
                color=np.diag(entanglement_matrix),
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Local Entropy')
            ),
            text=[f'Q{i}' for i in range(n_qubits)],
            textposition='top center'
        ))

        fig.update_layout(
            title='AI-Enhanced Quantum Entanglement Network',
            showlegend=False,
            hovermode='closest',
            xaxis_showgrid=False,
            yaxis_showgrid=False,
            xaxis_zeroline=False,
            yaxis_zeroline=False,
            xaxis_showticklabels=False,
            yaxis_showticklabels=False
        )

        return fig
