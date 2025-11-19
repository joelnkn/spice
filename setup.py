"""Setup script for SPICE project."""

from setuptools import setup, find_packages
from pathlib import Path

# Read requirements from requirements.txt if it exists
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    requirements = [
        line.strip() 
        for line in requirements_file.read_text().splitlines() 
        if line.strip() and not line.startswith("#")
    ]
else:
    requirements = []

setup(
    name="spice",
    version="0.1.0",
    description="Synthetic Polyglot Injection for Cross-lingual Evaluation",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
)
