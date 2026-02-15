"""
Out-of-distribution tests for generalization assessment.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

import torch
import numpy as np
from data.pde_solver import simulate_battery_thermal_2d
from evaluation.metrics import compute_metrics


# Training ranges
TRAIN_RANGES = {
    'k_cell': (0.5, 5.0),
    'k_coolant': (0.1, 1.0),
    'q0': (1e6, 5e6),
    'freq': (0.1, 2.0),
    'h': (10.0, 100.0),
    'T_amb': (280.0, 320.0)
}

# OOD test cases
OOD_TEST_CASES = [
    {
        'name': 'Low thermal conductivity',
        'params': {'k_cell': 0.3, 'k_coolant': 0.5, 'q0': 3e6, 'freq': 1.0, 'h': 50.0, 'T_amb': 300.0},
        'why': 'k_cell=0.3 < training min (0.5)'
    },
    {
        'name': 'High thermal conductivity',
        'params': {'k_cell': 6.0, 'k_coolant': 0.5, 'q0': 3e6, 'freq': 1.0, 'h': 50.0, 'T_amb': 300.0},
        'why': 'k_cell=6.0 > training max (5.0)'
    },
    {
        'name': 'Extreme heat generation',
        'params': {'k_cell': 2.0, 'k_coolant': 0.5, 'q0': 6e6, 'freq': 1.0, 'h': 50.0, 'T_amb': 300.0},
        'why': 'q0=6e6 > training max (5e6)'
    },
    {
        'name': 'Low convection',
        'params': {'k_cell': 2.0, 'k_coolant': 0.5, 'q0': 3e6, 'freq': 1.0, 'h': 5.0, 'T_amb': 300.0},
        'why': 'h=5.0 < training min (10.0)'
    },
    {
        'name': 'High convection',
        'params': {'k_cell': 2.0, 'k_coolant': 0.5, 'q0': 3e6, 'freq': 1.0, 'h': 150.0, 'T_amb': 300.0},
        'why': 'h=150.0 > training max (100.0)'
    },
    {
        'name': 'Extreme cold ambient',
        'params': {'k_cell': 2.0, 'k_coolant': 0.5, 'q0': 3e6, 'freq': 1.0, 'h': 50.0, 'T_amb': 250.0},
        'why': 'T_amb=250K < training min (280K)'
    },
    {
        'name': 'Multiple OOD parameters',
        'params': {'k_cell': 0.3, 'k_coolant': 0.5, 'q0': 6e6, 'freq': 0.05, 'h': 150.0, 'T_amb': 250.0},
        'why': 'k_cell, q0, freq, h, T_amb all OOD'
    }
]


def run_ood_test(model, params_dict, n_steps=10, dt=0.001, nx=128, ny=128, model_type='pc-unet', device='cuda'):
    """
    Run OOD test: generate ground truth, compare to model.

    Args:
        model: Neural network model
        params_dict: Physics parameters
        n_steps: Number of steps
        model_type: 'pc-unet', 'fno', 'deeponet', 'simple-cnn'
        device: 'cuda' or 'cpu'

    Returns:
        dict with predictions, targets, metrics
    """
    # Generate ground truth
    gt_trajectory = simulate_battery_thermal_2d(
        k_cell=params_dict['k_cell'],
        k_coolant=params_dict['k_coolant'],
        q0=params_dict['q0'],
        freq=params_dict['freq'],
        h=params_dict['h'],
        T_amb=params_dict['T_amb'],
        dt=dt,
        nx=nx,
        ny=ny,
        n_steps=n_steps,
        save_every=1
    )

    # Prepare inputs
    params_tensor = torch.tensor([
        params_dict['k_cell'],
        params_dict['k_coolant'],
        params_dict['q0'],
        params_dict['freq'],
        params_dict['h'],
        params_dict['T_amb']
    ], dtype=torch.float32).unsqueeze(0).to(device)

    k_map = torch.from_numpy(gt_trajectory['k_maps'][0]).unsqueeze(0).unsqueeze(0).float().to(device)
    q_map = torch.from_numpy(gt_trajectory['q_maps'][0]).unsqueeze(0).unsqueeze(0).float().to(device)
    sdf_cell = torch.from_numpy(gt_trajectory['sdf_cell']).unsqueeze(0).unsqueeze(0).float().to(device)
    sdf_cool = torch.from_numpy(gt_trajectory['sdf_coolant']).unsqueeze(0).unsqueeze(0).float().to(device)

    # Rollout
    predictions = []
    T_curr = torch.from_numpy(gt_trajectory['T'][0]).unsqueeze(0).unsqueeze(0).float().to(device)

    model.eval()
    with torch.no_grad():
        for step in range(n_steps - 1):
            if model_type == 'pc-unet':
                pred = model(T_curr, params_tensor, k_map, q_map, sdf_cell, sdf_cool)
            elif model_type == 'simple-cnn':
                pred = model(T_curr, params_tensor)
            elif model_type == 'fno':
                B, _, H, W = T_curr.shape
                params_spatial = params_tensor.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
                model_input = torch.cat([T_curr, params_spatial], dim=1)
                pred = model(model_input)
            elif model_type == 'deeponet':
                pred = model(params_tensor, T_curr)
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            predictions.append(pred.cpu().numpy()[0, 0])
            T_curr = pred

    predictions = np.array(predictions)
    targets = gt_trajectory['T'][1:n_steps]

    # Metrics
    metrics = compute_metrics(torch.from_numpy(predictions), torch.from_numpy(targets))

    return {
        'predictions': predictions,
        'targets': targets,
        'metrics': metrics,
        'trajectory': gt_trajectory
    }
