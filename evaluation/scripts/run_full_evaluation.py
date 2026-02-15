#!/usr/bin/env python3
"""
Run full evaluation: OOD tests, long rollouts, baselines comparison.

Usage:
    python run_full_evaluation.py --all
    python run_full_evaluation.py --ood-only
    python run_full_evaluation.py --rollouts-only
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
from tqdm.auto import tqdm

from model.architecture import PhysicsConditionedUNet
from baselines import FNO2d, DeepONet, SimpleCNN
from tests import run_ood_test, run_long_rollout, OOD_TEST_CASES
from data.pde_solver import simulate_battery_thermal_2d
from metrics import print_metrics


def run_ood_evaluation(device='cuda'):
    """Run all OOD tests."""
    print("\n" + "="*80)
    print("  OUT-OF-DISTRIBUTION TESTS")
    print("="*80)

    # Load PC-U-Net
    pc_unet = PhysicsConditionedUNet(
        in_channels=1, out_channels=1, base_channels=64, n_params=6, depth=4
    ).to(device)
    checkpoint = torch.load('checkpoints/best_model.pt', map_location=device)
    pc_unet.load_state_dict(checkpoint['model_state_dict'])

    ood_results = []
    for case in tqdm(OOD_TEST_CASES, desc='Running OOD tests'):
        print(f"\n{'='*80}")
        print(f"  {case['name']}")
        print(f"  {case['why']}")
        print(f"{'='*80}")

        result = run_ood_test(pc_unet, case['params'], n_steps=10, model_type='pc-unet', device=device)
        result['case'] = case
        ood_results.append(result)

        print_metrics(result['metrics'], case['name'])

    # Summary
    print("\n" + "="*80)
    print("  OOD SUMMARY")
    print("="*80)
    print(f"\n{'Test Case':<35} {'MAE (K)':>10} {'NRMSE':>10} {'Status':>15}")
    print("-" * 80)

    for result in ood_results:
        metrics = result['metrics']
        mae = metrics['mae_K']
        nrmse = metrics['nrmse']

        if mae < 0.15:
            status = '✓ Good'
        elif mae < 0.5:
            status = '⚠ Degraded'
        else:
            status = '✗ Failed'

        print(f"{result['case']['name']:<35} {mae:>10.4f} {nrmse:>10.4f} {status:>15}")

    return ood_results


def run_rollout_evaluation(device='cuda'):
    """Run 100-step rollouts for all models."""
    print("\n" + "="*80)
    print("  100-STEP ROLLOUT COMPARISON")
    print("="*80)

    test_params = {
        'k_cell': 2.0,
        'k_coolant': 0.5,
        'q0': 3e6,
        'freq': 1.0,
        'h': 50.0,
        'T_amb': 300.0
    }

    # Generate initial state
    initial_traj = simulate_battery_thermal_2d(
        k_cell=test_params['k_cell'],
        k_coolant=test_params['k_coolant'],
        q0=test_params['q0'],
        freq=test_params['freq'],
        h=test_params['h'],
        T_amb=test_params['T_amb'],
        dt=0.001,
        nx=128,
        ny=128,
        n_steps=2,
        save_every=1
    )
    initial_state = initial_traj['T'][0]

    # Load models
    models = {}

    # PC-U-Net
    pc_unet = PhysicsConditionedUNet(
        in_channels=1, out_channels=1, base_channels=64, n_params=6, depth=4
    ).to(device)
    checkpoint = torch.load('checkpoints/best_model.pt', map_location=device)
    pc_unet.load_state_dict(checkpoint['model_state_dict'])
    models['PC-U-Net'] = (pc_unet, 'pc-unet')

    # FNO
    fno = FNO2d(
        modes1=12, modes2=12, width=64, n_layers=4, in_channels=7, out_channels=1
    ).to(device)
    checkpoint = torch.load('checkpoints/fno_best.pt', map_location=device)
    fno.load_state_dict(checkpoint['model_state_dict'])
    models['FNO'] = (fno, 'fno')

    # DeepONet
    deeponet = DeepONet(
        branch_input_dim=128*128 + 6, trunk_input_dim=2, hidden_dim=128, latent_dim=128
    ).to(device)
    checkpoint = torch.load('checkpoints/deeponet_best.pt', map_location=device)
    deeponet.load_state_dict(checkpoint['model_state_dict'])
    models['DeepONet'] = (deeponet, 'deeponet')

    # Run rollouts
    rollout_results = {}
    for model_name, (model, model_type) in models.items():
        print(f"\n{'='*80}")
        print(f"  {model_name}")
        print(f"{'='*80}")

        rollout = run_long_rollout(
            model, initial_state, test_params, n_steps=100, model_type=model_type, device=device
        )
        rollout_results[model_name] = rollout

    # Summary
    print("\n" + "="*80)
    print("  ROLLOUT SUMMARY")
    print("="*80)
    print(f"\n{'Model':<15} {'Initial (K)':>12} {'Final (K)':>12} {'Growth':>10} {'Status':>10}")
    print("-" * 80)

    for name, rollout in rollout_results.items():
        initial = rollout['initial_error']
        final = rollout['final_error']
        growth = rollout['error_growth']
        status = '✓' if growth < 3 else '⚠' if growth < 10 else '✗'

        print(f"{name:<15} {initial:>12.4f} {final:>12.4f} {growth:>10.2f}× {status:>10}")

    return rollout_results


def main():
    parser = argparse.ArgumentParser(description='Run full evaluation')
    parser.add_argument('--all', action='store_true', help='Run all evaluations')
    parser.add_argument('--ood-only', action='store_true', help='Run OOD tests only')
    parser.add_argument('--rollouts-only', action='store_true', help='Run rollouts only')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if args.all or args.ood_only:
        run_ood_evaluation(device)

    if args.all or args.rollouts_only:
        run_rollout_evaluation(device)

    print("\n✓ Evaluation complete!")


if __name__ == '__main__':
    main()
