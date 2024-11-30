"""
OpenGL-based Quantum State Visualization
======================================

Provides hardware-accelerated 3D visualization of quantum states and dynamics
using OpenGL and modern shader techniques.
"""

import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from typing import Optional, Tuple, List
import math
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation
from ..core.quantum_system import QuantumSystem

class QuantumGLVisualizer:
    """OpenGL-based quantum state visualization with advanced shader effects."""

    def __init__(self, width: int = 800, height: int = 600):
        """Initialize OpenGL context, shaders, and AI components."""
        self.width = width
        self.height = height
        self.rotation = [0.0, 0.0, 0.0]

        # Initialize AI components
        self.quantum_renderer = nn.Sequential(
            nn.Linear(4, 64),  # 4D input: 3D position + time
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)   # RGB output
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.quantum_renderer.to(self.device)

        # Initialize GLUT and create window
        glutInit()
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
        glutInitWindowSize(width, height)
        glutCreateWindow(b"Quantum State Visualization")

        # Load and compile shaders
        self.shader_program = self.create_shader_program()

        # Create texture for interference pattern
        self.interference_texture = self.create_interference_texture()

        # Initialize uniforms
        self.time_location = glGetUniformLocation(self.shader_program, "time")
        self.quantum_state_location = glGetUniformLocation(self.shader_program, "quantum_state")
        self.interference_pattern_location = glGetUniformLocation(self.shader_program, "interference_pattern")

        # Enable features
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)

        # Set up enhanced lighting
        glLightfv(GL_LIGHT0, GL_POSITION, [1, 1, 1, 0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.3, 0.3, 0.4, 1])  # Enhanced ambient
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.9, 0.9, 1.0, 1])  # Enhanced diffuse
        glLightfv(GL_LIGHT0, GL_SPECULAR, [1.0, 1.0, 1.0, 1]) # Added specular

        self.setup_camera()

    def setup_camera(self):
        """Set up the camera and perspective."""
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, self.width/self.height, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

    def draw_bloch_sphere(self):
        """Draw the Bloch sphere with coordinate axes."""
        # Draw sphere
        glPushMatrix()
        glColor3f(0.8, 0.8, 0.8)
        glutWireSphere(1.0, 32, 32)

        # Draw axes
        glBegin(GL_LINES)
        # X axis (red)
        glColor3f(1, 0, 0)
        glVertex3f(-1.2, 0, 0)
        glVertex3f(1.2, 0, 0)
        # Y axis (green)
        glColor3f(0, 1, 0)
        glVertex3f(0, -1.2, 0)
        glVertex3f(0, 1.2, 0)
        # Z axis (blue)
        glColor3f(0, 0, 1)
        glVertex3f(0, 0, -1.2)
        glVertex3f(0, 0, 1.2)
        glEnd()

        glPopMatrix()

    def draw_state_vector(self, theta: float, phi: float):
        """Draw quantum state vector with AI-enhanced visualization effects."""
        x = math.sin(theta) * math.cos(phi)
        y = math.sin(theta) * math.sin(phi)
        z = math.cos(theta)

        # Generate AI-enhanced colors
        time_param = glutGet(GLUT_ELAPSED_TIME) / 1000.0
        position = torch.tensor([x, y, z, time_param], device=self.device)
        with torch.no_grad():
            color = self.quantum_renderer(position).cpu().numpy()

        glPushMatrix()
        glColor3f(*color)  # Use AI-generated color

        # Draw enhanced state vector with glow effect
        glLineWidth(3.0)
        glBegin(GL_LINES)
        glVertex3f(0, 0, 0)
        glVertex3f(x, y, z)
        glEnd()

        # Draw quantum probability cloud
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        num_particles = 50
        for i in range(num_particles):
            # Generate particle position with quantum uncertainty
            angle = 2 * np.pi * i / num_particles
            radius = 0.1 * np.random.rayleigh(0.5)
            px = x + radius * np.cos(angle)
            py = y + radius * np.sin(angle)
            pz = z + radius * np.sin(2*angle)

            glPushMatrix()
            glTranslatef(px, py, pz)
            glColor4f(*color, 0.3)  # Semi-transparent
            glutSolidSphere(0.02, 8, 8)
            glPopMatrix()

        glDisable(GL_BLEND)
        glPopMatrix()

    def draw_density_matrix(self, rho: np.ndarray):
        """Visualize density matrix with AI-enhanced 3D effects."""
        dim = rho.shape[0]
        scale = 1.0 / dim

        glPushMatrix()
        glTranslatef(-0.5, -0.5, 0)

        # Enable blending for quantum interference effects
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Draw enhanced surface with quantum interference patterns
        glBegin(GL_QUADS)
        for i in range(dim-1):
            for j in range(dim-1):
                # Get values and generate AI-enhanced visualization
                v1 = abs(rho[i,j])
                v2 = abs(rho[i,j+1])
                v3 = abs(rho[i+1,j+1])
                v4 = abs(rho[i+1,j])

                # Generate AI-enhanced colors for quantum interference
                time_param = glutGet(GLUT_ELAPSED_TIME) / 1000.0
                position = torch.tensor([i*scale, j*scale, v1, time_param], device=self.device)
                with torch.no_grad():
                    color = self.quantum_renderer(position).cpu().numpy()

                # Draw quantum-enhanced quad with interference effects
                phase = np.angle(rho[i,j]) / (2*np.pi) + 0.5
                glColor4f(*color, 0.7 + 0.3*np.sin(phase*np.pi))

                # Add interference patterns
                interference = 0.1 * np.sin(10*phase + time_param)

                glVertex3f(i*scale, j*scale, v1 + interference)
                glVertex3f(i*scale, (j+1)*scale, v2 + interference)
                glVertex3f((i+1)*scale, (j+1)*scale, v3 + interference)
                glVertex3f((i+1)*scale, j*scale, v4 + interference)
        glEnd()

        glDisable(GL_BLEND)
        glPopMatrix()

    def display(self, state: Optional[np.ndarray] = None):
        """Main display function with shader-based quantum effects."""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glTranslatef(0, 0, -5)

        # Use shader program
        glUseProgram(self.shader_program)

        # Update uniforms
        current_time = glutGet(GLUT_ELAPSED_TIME) / 1000.0
        glUniform1f(self.time_location, current_time)

        # Apply rotation
        glRotatef(self.rotation[0], 1, 0, 0)
        glRotatef(self.rotation[1], 0, 1, 0)
        glRotatef(self.rotation[2], 0, 0, 1)

        # Bind interference texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.interference_texture)
        glUniform1i(self.interference_pattern_location, 0)

        # Draw quantum visualization with enhanced effects
        self.draw_bloch_sphere()
        if state is not None:
            if state.ndim == 1:  # State vector
                theta = np.arccos(state[0])
                phi = np.angle(state[1])
                glUniform3f(self.quantum_state_location,
                           math.sin(theta) * math.cos(phi),
                           math.sin(theta) * math.sin(phi),
                           math.cos(theta))
                self.draw_state_vector(theta, phi)
            elif state.ndim == 2:  # Density matrix
                self.draw_density_matrix(state)

        # Disable shader program
        glUseProgram(0)
        glutSwapBuffers()

    def rotate(self, dx: float, dy: float, dz: float):
        """Update rotation angles."""
        self.rotation[0] += dx
        self.rotation[1] += dy
        self.rotation[2] += dz
        glutPostRedisplay()

    def run(self):
        """Start the visualization loop."""
        glutMainLoop()

    def create_shader_program(self) -> int:
        """Create and compile shader program."""
        # Read shader sources
        with open('visualization/shaders/quantum_vertex.glsl', 'r') as f:
            vertex_source = f.read()
        with open('visualization/shaders/quantum_fragment.glsl', 'r') as f:
            fragment_source = f.read()

        # Create shaders
        vertex_shader = glCreateShader(GL_VERTEX_SHADER)
        fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)

        # Set shader source
        glShaderSource(vertex_shader, vertex_source)
        glShaderSource(fragment_shader, fragment_source)

        # Compile shaders
        glCompileShader(vertex_shader)
        glCompileShader(fragment_shader)

        # Create program
        program = glCreateProgram()
        glAttachShader(program, vertex_shader)
        glAttachShader(program, fragment_shader)
        glLinkProgram(program)

        # Clean up
        glDeleteShader(vertex_shader)
        glDeleteShader(fragment_shader)

        return program

    def create_interference_texture(self) -> int:
        """Create texture for quantum interference patterns."""
        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)

        # Generate AI-based interference pattern
        size = 512
        pattern = np.zeros((size, size, 3), dtype=np.float32)
        for i in range(size):
            for j in range(size):
                x = i / size * 2 - 1
                y = j / size * 2 - 1
                with torch.no_grad():
                    pos = torch.tensor([x, y, 0.0, 0.0], device=self.device)
                    pattern[i, j] = self.quantum_renderer(pos).cpu().numpy()

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, size, size, 0,
                    GL_RGB, GL_FLOAT, pattern)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        return texture
