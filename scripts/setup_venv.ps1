# Setup virtual environment for Battery Thermal Surrogate
# Works on Windows PowerShell
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir
$VenvDir = Join-Path $ProjectDir ".venv"

Write-Host "================================================"
Write-Host "Battery Thermal Surrogate - Environment Setup"
Write-Host "================================================"

# Create virtual environment
if (-Not (Test-Path $VenvDir)) {
    Write-Host "Creating virtual environment..."
    python -m venv $VenvDir
} else {
    Write-Host "Virtual environment already exists at $VenvDir"
}

# Activate
Write-Host "Activating virtual environment..."
& "$VenvDir\Scripts\Activate.ps1"

# Upgrade pip
Write-Host "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
Write-Host "Installing dependencies..."
pip install -r (Join-Path $ProjectDir "requirements.txt")

# Install package in development mode
Write-Host "Installing package in development mode..."
pip install -e $ProjectDir

# Verify installation
Write-Host ""
Write-Host "Verifying installation..."
python -c @"
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
"@

Write-Host ""
Write-Host "Setup complete! Activate with: .\.venv\Scripts\Activate.ps1"
