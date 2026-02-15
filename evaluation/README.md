# Evaluation Framework

Publication-grade evaluation for battery thermal ML models.

## Quick Start

```bash
# Train all baselines (~4-5 hours)
make train-all

# Run full evaluation
make evaluate

# Compare all models
make compare
```

## Directory Structure

```
evaluation/
├── baselines/              # SOTA baseline implementations
│   ├── fno.py             # Fourier Neural Operator (Li et al., ICLR 2021)
│   ├── deeponet.py        # Deep Operator Network (Lu et al., Nature 2021)
│   └── simple_cnn.py      # Data-only U-Net baseline
├── metrics/               # Honest error metrics
│   └── honest_metrics.py  # NRMSE, MAE (no DC offset tricks)
├── tests/                 # OOD and long rollout tests
│   ├── ood_tests.py       # Out-of-distribution generalization
│   └── long_rollouts.py   # 100+ step stability
└── scripts/               # Main executable scripts
    ├── train_baseline.py      # Train FNO/DeepONet/SimpleCNN
    ├── compare_all.py         # Compare all models
    └── run_full_evaluation.py # Run all tests
```

## Usage

### 1. Train Baselines

```bash
# FNO (~2 hours)
python evaluation/scripts/train_baseline.py --model fno --epochs 100

# DeepONet (~2 hours)
python evaluation/scripts/train_baseline.py --model deeponet --epochs 100

# SimpleCNN (~45 min)
python evaluation/scripts/train_baseline.py --model simple-cnn --epochs 50
```

### 2. Run OOD Tests

```bash
python evaluation/scripts/run_full_evaluation.py --ood-only
```

Tests 7 out-of-distribution cases:
- Low/high thermal conductivity
- Extreme heat generation
- Low/high convection
- Extreme ambient temperature
- Multiple OOD parameters

### 3. Run Long Rollouts

```bash
python evaluation/scripts/run_full_evaluation.py --rollouts-only
```

Tests 100-step stability for:
- PC-U-Net (physics-informed)
- FNO (frequency domain)
- DeepONet (operator learning)

### 4. Compare All Models

```bash
python evaluation/scripts/compare_all.py \
    --test-data data/test \
    --checkpoints checkpoints \
    --output results/comparison.txt
```

Outputs comparison table:
```
Model          MAE (K)   NRMSE   Inference   Params
───────────────────────────────────────────────────
PC-U-Net       0.046     0.07%   3.6ms       2.1M
SimpleCNN      0.089     0.14%   3.2ms       1.8M
FNO            0.???     ?.??%   ?.?ms       ???K
DeepONet       0.???     ?.??%   ?.?ms       ???K
```

### 5. Run Full Evaluation

```bash
python evaluation/scripts/run_full_evaluation.py --all
```

Runs everything:
- OOD tests (7 cases)
- 100-step rollouts (3 models)
- Saves results to `results/`

## API Usage

### Import Baselines

```python
from evaluation.baselines import FNO2d, DeepONet, SimpleCNN

# Initialize
fno = FNO2d(modes1=12, modes2=12, width=64, n_layers=4)
deeponet = DeepONet(branch_input_dim=128*128+6, hidden_dim=128)
simple_cnn = SimpleCNN(base_channels=64, depth=4)
```

### Compute Metrics

```python
from evaluation.metrics import compute_metrics, print_metrics

metrics = compute_metrics(predictions, targets)
print_metrics(metrics, title="Test Set")
```

### Run OOD Test

```python
from evaluation.tests import run_ood_test

result = run_ood_test(
    model=my_model,
    params_dict={'k_cell': 0.3, 'q0': 6e6, ...},
    n_steps=10,
    model_type='pc-unet'
)
print(result['metrics'])
```

### Run Long Rollout

```python
from evaluation.tests import run_long_rollout

result = run_long_rollout(
    model=my_model,
    initial_state=T_init,
    params_dict={'k_cell': 2.0, ...},
    n_steps=100,
    model_type='fno'
)
print(f"Error growth: {result['error_growth']:.2f}×")
```

## Key Features

### Honest Metrics
- **NRMSE**: Normalized by temperature RANGE (not magnitude)
- **MAE**: Absolute error in Kelvin
- **RMSE**: Root mean squared error
- **Exposes DC offset tricks** in relative L2 error

### OOD Tests
- Tests generalization beyond training range
- 7 carefully chosen test cases
- Reports degradation factors

### Long Rollouts
- 100-200 step stability testing
- Autoregressive error accumulation
- Compares against PDE solver ground truth

### SOTA Baselines
- **FNO**: Frequency domain, SOTA for PDEs
- **DeepONet**: Operator learning, universal approximation
- **SimpleCNN**: Data-only, no physics

## For Publication

Use this framework to generate:

✅ Comparison tables (Table 1)
✅ OOD generalization analysis (Table 2)
✅ Long rollout stability (Figure 5)
✅ Honest error metrics (Results section)

**No more overclaimed results. No more misleading metrics.**

## Requirements

- PyTorch >= 1.12
- NumPy
- tqdm
- matplotlib (optional, for plotting)

## Citation

If you use this evaluation framework, cite:

```bibtex
@software{battery_thermal_eval,
  title={Publication-Grade Evaluation for Physics-Informed Neural Networks},
  author={Your Name},
  year={2024},
  url={https://github.com/...}
}
```
