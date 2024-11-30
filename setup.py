from setuptools import setup, find_packages

setup(
    name="quantum-consciousness-framework",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "qutip",
        "matplotlib",
        "fastapi",
        "uvicorn",
        "pydantic==1.10.0"
    ],
    author="Devin AI",
    author_email="devin@cognition-labs.com",
    description="Advanced quantum simulation framework integrating consciousness studies",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
)
