#!/bin/bash
# Setup script for synthetic language generation environment

echo "Setting up synthetic language generation environment..."

# Check for Python 3.10+
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "!! python3 not found. Install Python 3.10+ and re-run."
  exit 1
fi

# Upgrade pip and wheel to ensure proper package installations
echo "==> Upgrading pip and wheel..."
"$PYTHON_BIN" -m pip install --upgrade pip wheel

# Submodules
if [ -d ".git" ]; then
  echo "==> Initializing submodules..."
  git submodule update --init --recursive
else
  echo "!! Not a git repo (no .git). Skipping submodule init."
fi

# Install dependencies
echo "==> Installing main requirements..."
pip install -r requirements.txt

# Install Conglanger dependencies if present
CONGL_DIR="third_party/conglanger"
if [ -d "$CONGL_DIR" ] && [ -f "$CONGL_DIR/requirements.txt" ]; then
  echo "==> Installing Conglanger requirements..."
  pip install -r "$CONGL_DIR/requirements.txt"
else
  echo "==> Conglanger requirements not found (ok if not added yet)."
fi

# Setup directories
echo "==> Setting up directories..."
mkdir -p synthetic/data/raw
mkdir -p synthetic/data/processed
mkdir -p synthetic/output

# 6) .env scaffold ------------------------------------------------------------
if [ -f ".env.example" ] && [ ! -f ".env" ]; then
  echo "==> Creating .env from .env.example (edit it to add API keys)..."
  cp .env.example .env
elif [ ! -f ".env" ]; then
  echo "==> No .env.example found; create .env and set at least one of:"
  echo "    OPENAI_API_KEY=..., GOOGLE_API_KEY=..., TOGETHER_API_KEY=..."
fi

echo "Setup complete!"

