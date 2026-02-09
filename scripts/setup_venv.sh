#!/usr/bin/env bash
# Setup virtual environment for Battery Thermal Surrogate
# Works on Linux and macOS
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_DIR/.venv"

echo "================================================"
echo "Battery Thermal Surrogate - Environment Setup"
echo "================================================"

# Create virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
else
    echo "Virtual environment already exists at $VENV_DIR"
fi

# Activate
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r "$PROJECT_DIR/requirements.txt"

# Install package in development mode
echo "Installing package in development mode..."
pip install -e "$PROJECT_DIR"

# Verify installation
echo ""
echo "Verifying installation..."
python -c "
import torch
import numpy as np
import h5py
import matplotlib
print(f'PyTorch:    {torch.__version__}')
print(f'NumPy:      {np.__version__}')
print(f'h5py:       {h5py.__version__}')
print(f'Matplotlib: {matplotlib.__version__}')
if torch.cuda.is_available():
    print(f'Device:     CUDA ({torch.cuda.get_device_name(0)})')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print(f'Device:     MPS (Apple Silicon)')
else:
    print(f'Device:     CPU')
"

echo ""
echo "Setup complete! Activate with: source .venv/bin/activate"
