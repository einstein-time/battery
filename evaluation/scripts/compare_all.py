#!/usr/bin/env python3
"""
Compare all models: PC-U-Net, SimpleCNN, FNO, DeepONet.

Usage:
    python compare_all.py --test-data data/test --output results/comparison.txt
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
import time
import numpy as np
from tqdm.auto import tqdm

from model.architecture import PhysicsConditionedUNet
from baselines import FNO2d, DeepONet, SimpleCNN
from data.dataset import BatteryThermalDataset
from metrics import compute_metrics


def evaluate_model(model, test_loader, model_type, device):
    """Evaluate model on test set."""
    model.eval()

    all_preds = []
    all_targets = []
    inference_times = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f'Evaluating {model_type}'):
            T_curr = batch['T_curr'].to(device)
            T_next = batch['T_next'].to(device)
            params = batch['params'].to(device)

            start_time = time.time()

            if model_type == 'pc-unet':
                k_map = batch['k_map'].to(device)
                q_map = batch['q_map'].to(device)
                sdf_cell = batch['sdf_cell'].to(device)
                sdf_cool = batch['sdf_cool'].to(device)
                pred = model(T_curr, params, k_map, q_map, sdf_cell, sdf_cool)
            elif model_type == 'simple-cnn':
                pred = model(T_curr, params)
            elif model_type == 'fno':
                B, _, H, W = T_curr.shape
                params_spatial = params.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
                model_input = torch.cat([T_curr, params_spatial], dim=1)
                pred = model(model_input)
            elif model_type == 'deeponet':
                pred = model(params, T_curr)

            inference_times.append(time.time() - start_time)

            all_preds.append(pred.cpu())
            all_targets.append(T_next.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    metrics = compute_metrics(all_preds, all_targets)
    metrics['inference_time_ms'] = np.mean(inference_times) * 1000

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Compare all models')
    parser.add_argument('--test-data', type=str, default='data/test', help='Test data directory')
    parser.add_argument('--checkpoints', type=str, default='checkpoints', help='Checkpoints directory')
    parser.add_argument('--output', type=str, default='results/comparison.txt', help='Output file')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load test data
    print("Loading test dataset...")
    test_dataset = BatteryThermalDataset(
        data_dir=args.test_data,
        n_samples=50,
        trajectory_length=10
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    print(f"Test set: {len(test_dataset)} samples\n")

    # Load all models
    models = {}

    # PC-U-Net
    print("Loading PC-U-Net...")
    pc_unet = PhysicsConditionedUNet(
        in_channels=1, out_channels=1, base_channels=64, n_params=6, depth=4
    ).to(device)
    checkpoint = torch.load(f'{args.checkpoints}/best_model.pt', map_location=device)
    pc_unet.load_state_dict(checkpoint['model_state_dict'])
    models['PC-U-Net'] = (pc_unet, 'pc-unet')

    # SimpleCNN
    print("Loading SimpleCNN...")
    simple_cnn = SimpleCNN(
        in_channels=1, out_channels=1, base_channels=64, n_params=6, depth=4
    ).to(device)
    checkpoint = torch.load(f'{args.checkpoints}/simple_baseline.pt', map_location=device)
    simple_cnn.load_state_dict(checkpoint['model_state_dict'])
    models['SimpleCNN'] = (simple_cnn, 'simple-cnn')

    # FNO
    print("Loading FNO...")
    fno = FNO2d(
        modes1=12, modes2=12, width=64, n_layers=4, in_channels=7, out_channels=1
    ).to(device)
    checkpoint = torch.load(f'{args.checkpoints}/fno_best.pt', map_location=device)
    fno.load_state_dict(checkpoint['model_state_dict'])
    models['FNO'] = (fno, 'fno')

    # DeepONet
    print("Loading DeepONet...")
    deeponet = DeepONet(
        branch_input_dim=128*128 + 6, trunk_input_dim=2, hidden_dim=128, latent_dim=128
    ).to(device)
    checkpoint = torch.load(f'{args.checkpoints}/deeponet_best.pt', map_location=device)
    deeponet.load_state_dict(checkpoint['model_state_dict'])
    models['DeepONet'] = (deeponet, 'deeponet')

    print("\n" + "="*100)
    print("  COMPREHENSIVE MODEL COMPARISON")
    print("="*100)

    # Evaluate all
    results = {}
    for name, (model, model_type) in models.items():
        print(f"\nEvaluating {name}...")
        results[name] = evaluate_model(model, test_loader, model_type, device)

    # Print comparison table
    print(f"\n{'Model':<15} {'MAE (K)':>10} {'NRMSE':>10} {'Inference':>12} {'Params':>12}")
    print("-" * 100)

    model_params = {
        'PC-U-Net': sum(p.numel() for p in pc_unet.parameters()),
        'SimpleCNN': sum(p.numel() for p in simple_cnn.parameters()),
        'FNO': sum(p.numel() for p in fno.parameters()),
        'DeepONet': sum(p.numel() for p in deeponet.parameters())
    }

    for name in ['PC-U-Net', 'SimpleCNN', 'FNO', 'DeepONet']:
        metrics = results[name]
        print(f"{name:<15} "
              f"{metrics['mae_K']:>10.4f} "
              f"{metrics['nrmse']:>10.4f} "
              f"{metrics['inference_time_ms']:>10.2f}ms "
              f"{model_params[name]:>12,}")

    # Save to file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write("="*100 + "\n")
        f.write("  COMPREHENSIVE MODEL COMPARISON\n")
        f.write("="*100 + "\n\n")

        for name in ['PC-U-Net', 'SimpleCNN', 'FNO', 'DeepONet']:
            metrics = results[name]
            f.write(f"\n{name}:\n")
            f.write(f"  MAE:        {metrics['mae_K']:.4f} K\n")
            f.write(f"  RMSE:       {metrics['rmse_K']:.4f} K\n")
            f.write(f"  NRMSE:      {metrics['nrmse']:.4f} ({metrics['nrmse']*100:.2f}%)\n")
            f.write(f"  Inference:  {metrics['inference_time_ms']:.2f} ms\n")
            f.write(f"  Parameters: {model_params[name]:,}\n")

    print(f"\n✓ Saved results to: {output_path}")


if __name__ == '__main__':
    main()
