# Physics-Informed ML Surrogate for 2D Battery Thermal Management

A neural network surrogate model that replaces traditional finite-difference PDE solvers for 2D battery thermal management, achieving 100-1000x speedup with physics-informed training.

## Problem Statement

- **Input**: 2D temperature field at time t (128x128 grid) + material properties, heat generation, boundary conditions
- **Output**: Predicted temperature field at time t+dt
- **Physics**: rho*cp * dT/dt = div(k*grad(T)) + q(x,y,t) with convection boundary conditions
- **Key Innovation**: Hybrid data-physics training (not just black-box ML)

## Installation

```bash
# Create virtual environment
python -m venv .venv

# Activate (Linux/macOS)
source .venv/bin/activate

# Activate (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Quick Start

### 1. Generate Training Data

```bash
# Quick test (small dataset)
python scripts/generate_data.py --quick_test

# Full dataset (2000 trajectories)
python scripts/generate_data.py --config configs/generate.yaml
```

### 2. Train Model

```bash
# Debug mode (small dataset, few epochs)
python scripts/train.py --config configs/train.yaml --debug

# Full training
python scripts/train.py --config configs/train.yaml

# Resume from checkpoint
python scripts/train.py --config configs/train.yaml --resume checkpoints/latest.pt
```

### 3. Evaluate Model

```bash
python scripts/evaluate.py --model checkpoints/best_model.pt
```

### 4. Run Tests

```bash
pytest tests/ -v
```

## Project Structure

```
battery_surrogate/
├── src/
│   ├── physics/          # FD solver, materials, validation
│   ├── models/           # PC-U-Net, CNN, POD architectures
│   ├── training/         # Trainer, losses, GradNorm
│   ├── data_utils/       # Dataset, generator, preprocessing
│   ├── evaluation/       # Visualization utilities
│   └── utils/            # Device, config, logging
├── configs/              # YAML configuration files
├── scripts/              # CLI scripts (train, evaluate, generate)
├── notebooks/            # Jupyter notebooks for exploration
├── tests/                # Unit and integration tests
└── data/                 # Generated datasets (HDF5)
```

## Model Architecture

The primary model is a **Physics-Conditioned U-Net (PC-U-Net)**:
- 9 input channels: [T, material masks (3), k, q, h, SDFs (2)]
- Physics conditioning blocks at each encoder/decoder level
- MC dropout for uncertainty quantification
- Stability constraint layer (output clipping)

### Training Strategy

3-phase training with adaptive loss balancing (GradNorm):
1. **Phase 1** (50 epochs): Data-only pretraining
2. **Phase 2** (100 epochs): Physics-informed (data + PDE + BC losses)
3. **Phase 3** (50 epochs): Consistency fine-tuning (all losses)

## Cross-Platform Support

- Auto-detects CUDA (NVIDIA GPU), MPS (Apple Silicon), or CPU
- Pure Python with pip-installable dependencies only
- Works on Windows, macOS, Linux, and Google Colab

## License

MIT
