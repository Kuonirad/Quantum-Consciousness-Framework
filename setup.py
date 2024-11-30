from setuptools import setup, find_packages

setup(
    name="quantum-consciousness-framework",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "qutip>=4.7.0",
        "matplotlib>=3.5.0",
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "pydantic>=2.0.0",
        "pennylane>=0.30.0",  # Quantum ML and simulation
        "torch>=2.0.0",       # Deep learning integration
        "networkx>=2.6.0",    # Graph theory and topology
        "sympy>=1.9.0",       # Symbolic mathematics
        "pytest>=7.0.0",      # Testing framework
        "hypothesis>=6.0.0",  # Property-based testing
        "mypy>=1.0.0",       # Static type checking
        "black>=22.0.0",     # Code formatting
        "sphinx>=4.0.0",     # Documentation generation
        "nbsphinx>=0.8.0",   # Jupyter notebook documentation
        "ipykernel>=6.0.0",  # Jupyter kernel
        "plotly>=5.0.0",     # Interactive visualizations
        "scikit-learn>=1.0.0"  # Machine learning utilities
    ],
    extras_require={
        'dev': [
            'pytest-cov>=2.12.0',
            'pylint>=2.8.0',
            'flake8>=3.9.0',
            'pre-commit>=2.15.0'
        ]
    },
    author="Kevin John Kull",
    author_email="kevinkull.kk@gmail.com",
    description="Advanced quantum simulation framework integrating consciousness studies",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
    ],
    keywords="quantum computing, consciousness, simulation, quantum mechanics, topology",
    project_urls={
        "Documentation": "https://quantum-consciousness-framework.readthedocs.io",
        "Source": "https://github.com/Kuonirad/Quantum-Consciousness-Framework",
        "Bug Reports": "https://github.com/Kuonirad/Quantum-Consciousness-Framework/issues",
    }
)
