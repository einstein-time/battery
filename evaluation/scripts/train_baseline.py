#!/usr/bin/env python3
"""
Train baseline models (FNO, DeepONet, SimpleCNN).

Usage:
    python train_baseline.py --model fno --epochs 100
    python train_baseline.py --model deeponet --epochs 100
    python train_baseline.py --model simple-cnn --epochs 50
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from baselines import FNO2d, DeepONet, SimpleCNN
from data.dataset import BatteryThermalDataset


def train_model(model, model_type, train_loader, val_loader, n_epochs, device, save_path):
    """Train baseline model."""

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(n_epochs):
        # Train
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{n_epochs}', leave=False):
            T_curr = batch['T_curr'].to(device)
            T_next = batch['T_next'].to(device)
            params = batch['params'].to(device)

            optimizer.zero_grad()

            if model_type == 'fno':
                B, _, H, W = T_curr.shape
                params_spatial = params.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
                model_input = torch.cat([T_curr, params_spatial], dim=1)
                pred = model(model_input)
            elif model_type == 'deeponet':
                pred = model(params, T_curr)
            elif model_type == 'simple-cnn':
                pred = model(T_curr, params)
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            loss = F.mse_loss(pred, T_next)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validate
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                T_curr = batch['T_curr'].to(device)
                T_next = batch['T_next'].to(device)
                params = batch['params'].to(device)

                if model_type == 'fno':
                    B, _, H, W = T_curr.shape
                    params_spatial = params.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
                    model_input = torch.cat([T_curr, params_spatial], dim=1)
                    pred = model(model_input)
                elif model_type == 'deeponet':
                    pred = model(params, T_curr)
                elif model_type == 'simple-cnn':
                    pred = model(T_curr, params)

                loss = F.mse_loss(pred, T_next)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss
            }, save_path)

        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Train={train_loss:.6f}, Val={val_loss:.6f}, Best={best_val_loss:.6f}")

    print(f"\n✓ Training complete. Best val loss: {best_val_loss:.6f}")
    print(f"✓ Saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Train baseline models')
    parser.add_argument('--model', type=str, required=True, choices=['fno', 'deeponet', 'simple-cnn'],
                        help='Model to train')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    parser.add_argument('--output-dir', type=str, default='checkpoints', help='Output directory')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load datasets
    print("Loading datasets...")
    train_dataset = BatteryThermalDataset(
        data_dir=f'{args.data_dir}/train',
        n_samples=200,
        trajectory_length=10
    )
    val_dataset = BatteryThermalDataset(
        data_dir=f'{args.data_dir}/val',
        n_samples=50,
        trajectory_length=10
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=32, shuffle=False
    )

    # Initialize model
    print(f"Initializing {args.model.upper()}...")
    if args.model == 'fno':
        model = FNO2d(
            modes1=12, modes2=12, width=64, n_layers=4,
            in_channels=7, out_channels=1
        ).to(device)
        save_path = f'{args.output_dir}/fno_best.pt'
    elif args.model == 'deeponet':
        model = DeepONet(
            branch_input_dim=128*128 + 6,
            trunk_input_dim=2,
            hidden_dim=128,
            latent_dim=128
        ).to(device)
        save_path = f'{args.output_dir}/deeponet_best.pt'
    elif args.model == 'simple-cnn':
        model = SimpleCNN(
            in_channels=1, out_channels=1,
            base_channels=64, n_params=6, depth=4
        ).to(device)
        save_path = f'{args.output_dir}/simple_baseline.pt'

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Train
    print(f"\nTraining {args.model.upper()} for {args.epochs} epochs...")
    train_model(model, args.model, train_loader, val_loader, args.epochs, device, save_path)


if __name__ == '__main__':
    main()
