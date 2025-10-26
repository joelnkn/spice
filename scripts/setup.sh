#!/bin/bash
# Setup script for synthetic language generation environment

echo "Setting up synthetic language generation environment..."

# Install dependencies
pip install -r requirements.txt

# Setup directories
mkdir -p synthetic/data/raw
mkdir -p synthetic/data/processed
mkdir -p synthetic/outputs

echo "Setup complete!"

