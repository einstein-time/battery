"""
Long-term rollout stability testing (100+ steps).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

import torch
import numpy as np
import time
from data.pde_solver import simulate_battery_thermal_2d


def run_long_rollout(model, initial_state, params_dict, n_steps=100, dt=0.001, model_type='pc-unet', device='cuda'):
    """
    Perform long autoregressive rollout.

    Args:
        model: Neural network model
        initial_state: Initial temperature field (H, W) numpy array
        params_dict: Physics parameters
        n_steps: Number of rollout steps
        model_type: 'pc-unet', 'fno', 'deeponet', 'simple-cnn'
        device: 'cuda' or 'cpu'

    Returns:
        dict with predictions, errors, ground truth
    """
    print(f"\n🔄 Running {n_steps}-step rollout ({n_steps*dt:.3f}s physical time)...")

    # Generate ground truth
    print("  1/3: Generating ground truth...", end=' ', flush=True)
    start_time = time.time()
    gt_trajectory = simulate_battery_thermal_2d(
        k_cell=params_dict['k_cell'],
        k_coolant=params_dict['k_coolant'],
        q0=params_dict['q0'],
        freq=params_dict['freq'],
        h=params_dict['h'],
        T_amb=params_dict['T_amb'],
        dt=dt,
        nx=128,
        ny=128,
        n_steps=n_steps + 1,
        save_every=1
    )
    pde_time = time.time() - start_time
    print(f"Done ({pde_time:.1f}s)")

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
    print("  2/3: Running model rollout...", end=' ', flush=True)
    start_time = time.time()
    predictions = []
    errors = []
    T_curr = torch.from_numpy(initial_state).unsqueeze(0).unsqueeze(0).float().to(device)

    model.eval()
    with torch.no_grad():
        for step in range(n_steps):
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

            pred_np = pred.cpu().numpy()[0, 0]
            predictions.append(pred_np)

            gt = gt_trajectory['T'][step + 1]
            error = np.abs(pred_np - gt).mean()
            errors.append(error)

            T_curr = pred

    rollout_time = time.time() - start_time
    print(f"Done ({rollout_time:.1f}s)")

    # Statistics
    predictions = np.array(predictions)
    errors = np.array(errors)
    ground_truth = gt_trajectory['T'][1:n_steps+1]

    initial_error = errors[0]
    final_error = errors[-1]
    max_error = errors.max()
    error_growth = final_error / initial_error if initial_error > 0 else float('inf')

    print(f"  3/3: Analysis...")
    print(f"\n  📊 Rollout Statistics:")
    print(f"     Initial error:  {initial_error:.4f} K")
    print(f"     Final error:    {final_error:.4f} K")
    print(f"     Max error:      {max_error:.4f} K")
    print(f"     Error growth:   {error_growth:.2f}×")
    print(f"     PDE time:       {pde_time:.1f}s ({pde_time/n_steps*1000:.1f}ms/step)")
    print(f"     Model time:     {rollout_time:.1f}s ({rollout_time/n_steps*1000:.1f}ms/step)")

    return {
        'predictions': predictions,
        'ground_truth': ground_truth,
        'errors': errors,
        'initial_error': initial_error,
        'final_error': final_error,
        'max_error': max_error,
        'error_growth': error_growth,
        'pde_time': pde_time,
        'rollout_time': rollout_time
    }
