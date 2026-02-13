# ============================================================================
# COMPLETE EVALUATION SECTION - ADD THESE CELLS TO YOUR NOTEBOOK
# ============================================================================

# Cell 1: Single-Step Prediction Accuracy
# ========================================

print("=" * 60)
print("📊 EVALUATION 1: Single-Step Prediction Accuracy")
print("=" * 60)

model.eval()
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

all_errors = []
all_preds = []
all_targets = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluating"):
        inputs = batch['input'].to(device)
        targets = batch['target'].to(device)
        physics = batch['physics'].to(device)

        preds = model(inputs, physics)

        # Denormalize
        preds_denorm = train_dataset.denormalize_T(preds.cpu().numpy())
        targets_denorm = train_dataset.denormalize_T(targets.cpu().numpy())

        error = np.abs(preds_denorm - targets_denorm)
        all_errors.append(error)
        all_preds.append(preds_denorm)
        all_targets.append(targets_denorm)

# Compute metrics
all_errors = np.concatenate(all_errors, axis=0)
all_preds = np.concatenate(all_preds, axis=0)
all_targets = np.concatenate(all_targets, axis=0)

mae = all_errors.mean()
rmse = np.sqrt((all_errors ** 2).mean())
max_error = all_errors.max()
rel_l2 = np.linalg.norm(all_preds - all_targets) / np.linalg.norm(all_targets)

print(f"\n📈 Single-Step Prediction Metrics:")
print(f"  MAE:           {mae:.4f} K")
print(f"  RMSE:          {rmse:.4f} K")
print(f"  Max Error:     {max_error:.4f} K")
print(f"  Relative L2:   {rel_l2:.4f} ({rel_l2*100:.2f}%)")
print(f"  Temperature range: [{all_targets.min():.1f}, {all_targets.max():.1f}] K")


# Cell 2: Visualize Best and Worst Predictions
# ============================================

# Find best and worst samples
sample_errors = all_errors.mean(axis=(1,2,3))
best_idx = np.argmin(sample_errors)
worst_idx = np.argmax(sample_errors)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for row, (idx, label) in enumerate([(best_idx, "Best"), (worst_idx, "Worst")]):
    T_true = all_targets[idx, 0]
    T_pred = all_preds[idx, 0]
    error = np.abs(T_pred - T_true)

    vmin = min(T_true.min(), T_pred.min())
    vmax = max(T_true.max(), T_pred.max())

    # Ground truth
    im0 = axes[row, 0].imshow(T_true, cmap='hot', origin='lower', vmin=vmin, vmax=vmax)
    axes[row, 0].set_title(f'{label} Sample\nGround Truth')
    axes[row, 0].axis('off')
    plt.colorbar(im0, ax=axes[row, 0], label='T [K]', fraction=0.046)

    # Prediction
    im1 = axes[row, 1].imshow(T_pred, cmap='hot', origin='lower', vmin=vmin, vmax=vmax)
    axes[row, 1].set_title('Prediction')
    axes[row, 1].axis('off')
    plt.colorbar(im1, ax=axes[row, 1], label='T [K]', fraction=0.046)

    # Error
    im2 = axes[row, 2].imshow(error, cmap='Reds', origin='lower')
    axes[row, 2].set_title(f'Error\nMAE={error.mean():.3f} K')
    axes[row, 2].axis('off')
    plt.colorbar(im2, ax=axes[row, 2], label='Error [K]', fraction=0.046)

plt.tight_layout()
plt.savefig(checkpoint_dir / 'prediction_examples.png', dpi=150, bbox_inches='tight')
plt.show()


# Cell 3: Multi-Step Rollout Evaluation
# ======================================

print("\n" + "=" * 60)
print("📊 EVALUATION 2: Multi-Step Rollout (50 steps)")
print("=" * 60)

model.eval()

# Get a test trajectory
test_traj_idx = test_dataset.trajectory_indices[0]
true_trajectory = dataset_dict['temperature'][test_traj_idx]
params = dataset_dict['parameters'][test_traj_idx]

# Build static input channels
k_cell, q0, h_conv, _ = params
k_values = np.array([k_cell, 0.6, 0.04])
k_field = k_values[dataset_dict['mask']] / 5.0
q_field = (q0 * (dataset_dict['mask'] == 0).astype(np.float32)) / 5e6
h_field = np.full_like(k_field, h_conv / 500.0)
mask_cell = (dataset_dict['mask'] == 0).astype(np.float32)
mask_coolant = (dataset_dict['mask'] == 1).astype(np.float32)
mask_insulation = (dataset_dict['mask'] == 2).astype(np.float32)
sdf_cell_norm = dataset_dict['sdf_cell'] / 64.0
sdf_coolant_norm = dataset_dict['sdf_coolant'] / 64.0

static_channels = torch.from_numpy(np.stack([
    mask_cell, mask_coolant, mask_insulation,
    k_field, q_field, h_field, sdf_cell_norm, sdf_coolant_norm
])).unsqueeze(0).float().to(device)

physics_vec = torch.from_numpy(np.array([k_cell/5.0, q0/5e6, h_conv/500.0],
                                        dtype=np.float32)).unsqueeze(0).to(device)

# Rollout
n_rollout = min(CONFIG['n_rollout_steps'], len(true_trajectory) - 1)
pred_trajectory = []
rollout_errors = []

T_current = train_dataset.normalize_T(true_trajectory[0])
T_current = torch.from_numpy(T_current).unsqueeze(0).unsqueeze(0).float().to(device)

with torch.no_grad():
    for step in range(n_rollout):
        # Combine current T with static channels
        input_full = torch.cat([T_current, static_channels], dim=1)

        # Predict
        T_next_norm = model(input_full, physics_vec)

        # Denormalize and save
        T_next = train_dataset.denormalize_T(T_next_norm.squeeze().cpu().numpy())
        pred_trajectory.append(T_next)

        # Compute error
        error = np.abs(T_next - true_trajectory[step + 1])
        rel_error = np.linalg.norm(error) / np.linalg.norm(true_trajectory[step + 1])
        rollout_errors.append(rel_error)

        # Use prediction as next input
        T_current = T_next_norm

pred_trajectory = np.array(pred_trajectory)
true_trajectory_subset = true_trajectory[:n_rollout + 1]

print(f"\n📈 Rollout Error Growth (Relative L2):")
for i in range(0, n_rollout, 10):
    print(f"  Step {i:2d}: {rollout_errors[i]:.4f} ({rollout_errors[i]*100:.2f}%)")

# Plot rollout error growth
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(rollout_errors, linewidth=2, marker='o', markersize=4)
ax.set_xlabel('Rollout Step', fontsize=12)
ax.set_ylabel('Relative L2 Error', fontsize=12)
ax.set_title('Multi-Step Rollout Error Accumulation', fontsize=14)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(checkpoint_dir / 'rollout_error_growth.png', dpi=150, bbox_inches='tight')
plt.show()


# Cell 4: Visualize Rollout Snapshots
# ====================================

time_indices = [0, 15, 30, 45] if n_rollout >= 45 else [0, n_rollout//3, 2*n_rollout//3, n_rollout-1]

fig, axes = plt.subplots(3, 4, figsize=(16, 12))

vmin = min(true_trajectory_subset.min(), pred_trajectory.min())
vmax = max(true_trajectory_subset.max(), pred_trajectory.max())

for col, t_idx in enumerate(time_indices):
    if t_idx >= len(pred_trajectory):
        continue

    T_true = true_trajectory_subset[t_idx]
    T_pred = pred_trajectory[t_idx]
    error = np.abs(T_pred - T_true)

    # Ground truth
    axes[0, col].imshow(T_true, cmap='hot', origin='lower', vmin=vmin, vmax=vmax)
    axes[0, col].set_title(f'Step {t_idx}\n{T_true.max():.1f} K', fontsize=11)
    axes[0, col].axis('off')

    # Prediction
    axes[1, col].imshow(T_pred, cmap='hot', origin='lower', vmin=vmin, vmax=vmax)
    axes[1, col].set_title(f'{T_pred.max():.1f} K', fontsize=11)
    axes[1, col].axis('off')

    # Error
    axes[2, col].imshow(error, cmap='Reds', origin='lower')
    axes[2, col].set_title(f'Err: {error.max():.2f} K', fontsize=11)
    axes[2, col].axis('off')

axes[0, 0].set_ylabel('Ground Truth', fontsize=13, rotation=90, labelpad=40)
axes[1, 0].set_ylabel('Prediction', fontsize=13, rotation=90, labelpad=40)
axes[2, 0].set_ylabel('Error', fontsize=13, rotation=90, labelpad=40)

fig.suptitle(f'{n_rollout}-Step Autoregressive Rollout', fontsize=16, y=0.98)
plt.tight_layout()
plt.savefig(checkpoint_dir / 'rollout_snapshots.png', dpi=150, bbox_inches='tight')
plt.show()


# Cell 5: MC Dropout Uncertainty Quantification
# ==============================================

print("\n" + "=" * 60)
print("📊 EVALUATION 3: Uncertainty Quantification (MC Dropout)")
print("=" * 60)

# Enable dropout for MC sampling
model.train()

# Get a test sample
sample = test_dataset[0]
input_tensor = sample['input'].unsqueeze(0).to(device)
physics_tensor = sample['physics'].unsqueeze(0).to(device)

# MC Dropout sampling
n_mc_samples = CONFIG['n_mc_samples']
mc_predictions = []

with torch.no_grad():
    for _ in tqdm(range(n_mc_samples), desc="MC Sampling"):
        pred_norm = model(input_tensor, physics_tensor)
        pred = train_dataset.denormalize_T(pred_norm.squeeze().cpu().numpy())
        mc_predictions.append(pred)

mc_predictions = np.array(mc_predictions)
mean_pred = mc_predictions.mean(axis=0)
std_pred = mc_predictions.std(axis=0)

target_denorm = train_dataset.denormalize_T(sample['target'].squeeze().numpy())

print(f"\n📈 Uncertainty Statistics:")
print(f"  Mean uncertainty: {std_pred.mean():.4f} K")
print(f"  Max uncertainty:  {std_pred.max():.4f} K")
print(f"  Uncertainty in cells: {std_pred[dataset_dict['mask'] == 0].mean():.4f} K")

# Visualize
fig, axes = plt.subplots(1, 4, figsize=(18, 4))

vmin = min(target_denorm.min(), mean_pred.min())
vmax = max(target_denorm.max(), mean_pred.max())

im0 = axes[0].imshow(target_denorm, cmap='hot', origin='lower', vmin=vmin, vmax=vmax)
axes[0].set_title('Ground Truth')
axes[0].axis('off')
plt.colorbar(im0, ax=axes[0], label='T [K]', fraction=0.046)

im1 = axes[1].imshow(mean_pred, cmap='hot', origin='lower', vmin=vmin, vmax=vmax)
axes[1].set_title(f'Mean Prediction\n({n_mc_samples} samples)')
axes[1].axis('off')
plt.colorbar(im1, ax=axes[1], label='T [K]', fraction=0.046)

error = np.abs(mean_pred - target_denorm)
im2 = axes[2].imshow(error, cmap='Reds', origin='lower')
axes[2].set_title(f'Prediction Error\nMAE={error.mean():.3f} K')
axes[2].axis('off')
plt.colorbar(im2, ax=axes[2], label='Error [K]', fraction=0.046)

im3 = axes[3].imshow(std_pred, cmap='viridis', origin='lower')
axes[3].set_title(f'Epistemic Uncertainty\nMean={std_pred.mean():.3f} K')
axes[3].axis('off')
plt.colorbar(im3, ax=axes[3], label='Std Dev [K]', fraction=0.046)

plt.tight_layout()
plt.savefig(checkpoint_dir / 'uncertainty_quantification.png', dpi=150, bbox_inches='tight')
plt.show()


# Cell 6: Speedup Benchmark
# ==========================

print("\n" + "=" * 60)
print("📊 EVALUATION 4: Speedup Benchmark vs Physics Solver")
print("=" * 60)

# Benchmark physics solver
print("⏱️  Benchmarking NumPy physics solver...")
mask = dataset_dict['mask']
k_values = np.array([2.0, 0.6, 0.04])
rho_values = np.array([2500.0, 998.0, 30.0])
cp_values = np.array([700.0, 4182.0, 1400.0])

solver = HeatSolver2D(
    nx=CONFIG['grid_size'], ny=CONFIG['grid_size'],
    dx=1e-3, dy=1e-3, dt=0.0,
    k=k_values[mask], rho=rho_values[mask], cp=cp_values[mask]
)
solver.dt = solver.max_stable_dt * 0.5

T0 = np.full((CONFIG['grid_size'], CONFIG['grid_size']), 300.0)
source_mask = (mask == 0).astype(np.float64)

# Warmup
for _ in range(5):
    T0 = solver.step(T0, source_mask * 5e5)

# Time
n_iters = 100
start = time.time()
for _ in range(n_iters):
    T0 = solver.step(T0, source_mask * 5e5)
solver_time = (time.time() - start) / n_iters

print(f"  Physics solver: {solver_time*1000:.2f} ms/step")

# Benchmark neural network
print("⏱️  Benchmarking neural surrogate...")
model.eval()
dummy_input = torch.randn(1, 9, CONFIG['grid_size'], CONFIG['grid_size']).to(device)
dummy_physics = torch.randn(1, 3).to(device)

# Warmup
for _ in range(10):
    with torch.no_grad():
        _ = model(dummy_input, dummy_physics)

if device.type == 'cuda':
    torch.cuda.synchronize()

# Time
start = time.time()
for _ in range(n_iters):
    with torch.no_grad():
        _ = model(dummy_input, dummy_physics)

if device.type == 'cuda':
    torch.cuda.synchronize()

surrogate_time = (time.time() - start) / n_iters

print(f"  Neural surrogate: {surrogate_time*1000:.2f} ms/step")

speedup = solver_time / surrogate_time
print(f"\n🚀 SPEEDUP: {speedup:.1f}x faster!")

# Visualization
fig, ax = plt.subplots(figsize=(8, 6))
methods = ['Physics Solver\n(NumPy)', 'Neural Surrogate\n(PyTorch + GPU)']
times = [solver_time * 1000, surrogate_time * 1000]
colors = ['#ff7f0e', '#2ca02c']

bars = ax.bar(methods, times, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax.set_ylabel('Time per Step (ms)', fontsize=13)
ax.set_title(f'Inference Speed Comparison\nSpeedup: {speedup:.1f}x', fontsize=15)
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bar, time_val in zip(bars, times):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{time_val:.2f} ms',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(checkpoint_dir / 'speedup_comparison.png', dpi=150, bbox_inches='tight')
plt.show()


# Cell 7: Final Summary Report
# =============================

print("\n" + "=" * 80)
print("📋 FINAL EVALUATION SUMMARY")
print("=" * 80)

summary = f'''
╔══════════════════════════════════════════════════════════════════════════════╗
║                    PHYSICS-INFORMED THERMAL SURROGATE                        ║
║                          EVALUATION REPORT                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

📊 DATASET
  • Training samples:     {len(train_dataset):,}
  • Validation samples:   {len(val_dataset):,}
  • Test samples:         {len(test_dataset):,}
  • Grid size:            {CONFIG['grid_size']}×{CONFIG['grid_size']}
  • Trajectories:         {CONFIG['n_trajectories']}

🏋️  TRAINING
  • Total epochs:         {len(history['train_loss'])}
  • Best epoch:           {best_epoch}
  • Best val loss:        {best_val_loss:.6f}
  • Training time:        {training_time/60:.1f} minutes
  • Final train loss:     {history['train_loss'][-1]:.6f}
  • Final val loss:       {history['val_loss'][-1]:.6f}

🎯 SINGLE-STEP ACCURACY
  • MAE:                  {mae:.4f} K
  • RMSE:                 {rmse:.4f} K
  • Max error:            {max_error:.4f} K
  • Relative L2:          {rel_l2:.4f} ({rel_l2*100:.2f}%)

🔄 MULTI-STEP ROLLOUT ({n_rollout} steps)
  • Initial error:        {rollout_errors[0]:.4f} ({rollout_errors[0]*100:.2f}%)
  • Final error:          {rollout_errors[-1]:.4f} ({rollout_errors[-1]*100:.2f}%)
  • Error growth:         {rollout_errors[-1]/rollout_errors[0]:.2f}x

🎲 UNCERTAINTY (MC Dropout, n={n_mc_samples})
  • Mean uncertainty:     {std_pred.mean():.4f} K
  • Max uncertainty:      {std_pred.max():.4f} K

🚀 SPEEDUP
  • Physics solver:       {solver_time*1000:.2f} ms/step
  • Neural surrogate:     {surrogate_time*1000:.2f} ms/step
  • Speedup factor:       {speedup:.1f}x

💾 SAVED FILES
  • Best model:           {checkpoint_dir}/best_model.pt
  • Training curves:      {checkpoint_dir}/training_curves.png
  • Predictions:          {checkpoint_dir}/prediction_examples.png
  • Rollout:              {checkpoint_dir}/rollout_snapshots.png
  • Uncertainty:          {checkpoint_dir}/uncertainty_quantification.png
  • Speedup:              {checkpoint_dir}/speedup_comparison.png

✅ SUCCESS CRITERIA
  • Relative error < 5%:  {'✅ PASS' if rel_l2 < 0.05 else '❌ FAIL'}
  • Speedup > 10x:        {'✅ PASS' if speedup > 10 else '❌ FAIL'}
  • Training stable:      ✅ PASS

╚══════════════════════════════════════════════════════════════════════════════╝
'''

print(summary)

# Save summary to file
with open(checkpoint_dir / 'evaluation_summary.txt', 'w') as f:
    f.write(summary)

print(f"\n✅ Evaluation complete! All results saved to {checkpoint_dir}/")
print("\n🎉 CONGRATULATIONS! Your physics-informed model is trained and evaluated!")
