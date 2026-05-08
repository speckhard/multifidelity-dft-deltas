"""Training loop for MegaPaiNN surrogate (Phase 2: Active Learning).

Simplified training for absolute property prediction (formation energy + bandgap).
Uses MSELoss on StandardScaler-normalized targets. Metrics reported in original
units via inverse-transform.

Usage:
  python -m modelling.gnn.train_surrogate
  python -m modelling.gnn.train_surrogate +wandb_group="sweep_v1" training.batch_size=64
"""

import os
import sys

import hydra
import numpy as np
import torch
import torch.nn as nn
import wandb
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import mean_absolute_error, r2_score
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch_geometric.loader import DataLoader

# --- Import Fix ---
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from modelling.gnn.mega_painn_model import MegaPaiNN
from modelling.gnn.train_pipeline import EarlyStopping


def inverse_transform(scaler, values_np):
    """Inverse-transform 1D array using a fitted StandardScaler."""
    return scaler.inverse_transform(values_np.reshape(-1, 1)).flatten()


def compute_metrics(preds_scaled, targets_scaled, scaler, prefix):
    """Compute MAE, RMSE, R2 in original units.

    Args:
        preds_scaled: predictions in scaled space (numpy)
        targets_scaled: targets in scaled space (numpy)
        scaler: fitted StandardScaler for inverse transform
        prefix: metric name prefix (e.g. 'form', 'gap')

    Returns:
        dict of metrics
    """
    preds_orig = inverse_transform(scaler, preds_scaled)
    targets_orig = inverse_transform(scaler, targets_scaled)

    mae = mean_absolute_error(targets_orig, preds_orig)
    rmse = np.sqrt(np.mean((preds_orig - targets_orig) ** 2))
    r2 = r2_score(targets_orig, preds_orig) if len(targets_orig) > 1 else 0.0

    return {
        f'MAE_{prefix}': mae,
        f'RMSE_{prefix}': rmse,
        f'R2_{prefix}': r2,
    }


def train_step(model, loader, optimizer, w_form, w_gap, device):
    """One training epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    total_graphs = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        out = model(data)

        loss_form = nn.functional.mse_loss(out['e_form'], data.y_form_scaled)
        loss_gap = nn.functional.mse_loss(out['e_gap'], data.y_gap_scaled)
        loss = w_form * loss_form + w_gap * loss_gap

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * data.num_graphs
        total_graphs += data.num_graphs

    return total_loss / total_graphs


@torch.no_grad()
def evaluate(model, loader, form_scaler, gap_scaler, w_form, w_gap, device):
    """Evaluate model. Returns metrics dict with original-unit MAE/RMSE/R2."""
    model.eval()

    all_form_pred = []
    all_form_targ = []
    all_gap_pred = []
    all_gap_targ = []
    total_loss = 0.0
    total_graphs = 0

    for data in loader:
        data = data.to(device)
        out = model(data)

        loss_form = nn.functional.mse_loss(out['e_form'], data.y_form_scaled)
        loss_gap = nn.functional.mse_loss(out['e_gap'], data.y_gap_scaled)
        loss = w_form * loss_form + w_gap * loss_gap

        total_loss += loss.item() * data.num_graphs
        total_graphs += data.num_graphs

        all_form_pred.append(out['e_form'].cpu().numpy())
        all_form_targ.append(data.y_form_scaled.cpu().numpy())
        all_gap_pred.append(out['e_gap'].cpu().numpy())
        all_gap_targ.append(data.y_gap_scaled.cpu().numpy())

    form_pred = np.concatenate(all_form_pred)
    form_targ = np.concatenate(all_form_targ)
    gap_pred = np.concatenate(all_gap_pred)
    gap_targ = np.concatenate(all_gap_targ)

    metrics = {'loss': total_loss / total_graphs}
    metrics.update(compute_metrics(form_pred, form_targ, form_scaler, 'form'))
    metrics.update(compute_metrics(gap_pred, gap_targ, gap_scaler, 'gap'))

    return metrics


def scale_targets(graphs, form_scaler, gap_scaler):
    """Apply fitted scalers to graph targets. Adds y_form_scaled, y_gap_scaled."""
    for g in graphs:
        g.y_form_scaled = torch.tensor(
            form_scaler.transform(g.y_form.numpy().reshape(-1, 1)).flatten(),
            dtype=torch.float)
        g.y_gap_scaled = torch.tensor(
            gap_scaler.transform(g.y_gap.numpy().reshape(-1, 1)).flatten(),
            dtype=torch.float)
    return graphs


@hydra.main(version_base=None, config_path="conf", config_name="mega_painn_config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Seed
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # --- Data ---
    if not os.path.exists(cfg.data_file):
        print(f"Error: {cfg.data_file} not found.")
        return

    print(f"Loading data from {cfg.data_file}...")
    loaded = torch.load(cfg.data_file, weights_only=False)
    all_graphs = loaded['graphs']
    form_scaler = loaded['y_form_scaler']
    gap_scaler = loaded['y_gap_scaler']

    # Split
    train_graphs = [g for g in all_graphs if g.split == 'train']
    val_graphs = [g for g in all_graphs if g.split == 'val']
    test_graphs = [g for g in all_graphs if g.split == 'test']
    print(f"Split: Train={len(train_graphs)}, Val={len(val_graphs)}, "
          f"Test={len(test_graphs)}")

    # Scale targets
    train_graphs = scale_targets(train_graphs, form_scaler, gap_scaler)
    val_graphs = scale_targets(val_graphs, form_scaler, gap_scaler)
    test_graphs = scale_targets(test_graphs, form_scaler, gap_scaler)

    train_loader = DataLoader(
        train_graphs, batch_size=cfg.training.batch_size, shuffle=True)
    val_loader = DataLoader(
        val_graphs, batch_size=cfg.training.batch_size, shuffle=False)
    test_loader = DataLoader(
        test_graphs, batch_size=cfg.training.batch_size, shuffle=False)

    # --- Model ---
    model = MegaPaiNN(
        num_layers=cfg.model.num_layers,
        hidden_dim=cfg.model.hidden_dim,
        num_precision_settings=cfg.model.num_precision_settings,
        max_z=cfg.model.max_z,
        num_rbf=cfg.model.num_rbf,
        cutoff=cfg.model.cutoff,
        embedding_dim=cfg.model.embedding_dim,
        use_film=cfg.model.use_film,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # --- Optimizer + Scheduler ---
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    warmup_epochs = cfg.training.warmup_epochs
    total_epochs = cfg.training.epochs

    scheduler_warmup = LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0,
        total_iters=warmup_epochs)
    scheduler_decay = CosineAnnealingLR(
        optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-6)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[scheduler_warmup, scheduler_decay],
        milestones=[warmup_epochs])

    # --- WandB ---
    wandb.init(
        project=cfg.wandb_project,
        group=cfg.get("wandb_group", None),
        config=OmegaConf.to_container(cfg, resolve=True),
        name=cfg.job_name,
        dir=os.getcwd(),
    )
    wandb.run.summary["total_parameters"] = total_params

    # --- Training ---
    w_form = cfg.weights.formation_energy
    w_gap = cfg.weights.bandgap
    early_stopper = EarlyStopping(
        patience=cfg.training.early_stopping_patience)

    best_val_loss = float('inf')

    header = f"{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>10} | " \
             f"{'MAE Form':>10} | {'MAE Gap':>10} | {'R2 Form':>8} | {'R2 Gap':>8}"
    print(header)
    print("-" * len(header))

    for epoch in range(1, total_epochs + 1):
        train_loss = train_step(
            model, train_loader, optimizer, w_form, w_gap, device)
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1 or epoch == total_epochs:
            val_m = evaluate(
                model, val_loader, form_scaler, gap_scaler,
                w_form, w_gap, device)

            print(f"{epoch:5d} | {train_loss:10.6f} | {val_m['loss']:10.6f} | "
                  f"{val_m['MAE_form']:10.4f} | {val_m['MAE_gap']:10.4f} | "
                  f"{val_m['R2_form']:8.4f} | {val_m['R2_gap']:8.4f}")

            lr = optimizer.param_groups[0]['lr']
            wandb.log({
                "epoch": epoch,
                "Train/Loss": train_loss,
                "Train/LR": lr,
                "Val/Loss": val_m['loss'],
                "Val/MAE_FormEnergy": val_m['MAE_form'],
                "Val/MAE_Bandgap": val_m['MAE_gap'],
                "Val/RMSE_FormEnergy": val_m['RMSE_form'],
                "Val/RMSE_Bandgap": val_m['RMSE_gap'],
                "Val/R2_FormEnergy": val_m['R2_form'],
                "Val/R2_Bandgap": val_m['R2_gap'],
            }, step=epoch)

            # Checkpoint
            if val_m['loss'] < best_val_loss:
                best_val_loss = val_m['loss']
                torch.save(model.state_dict(), 'best_surrogate.pth')
                wandb.save('best_surrogate.pth')

            torch.save(model.state_dict(), 'last_surrogate.pth')

            # Early stopping on val loss
            early_stopper(val_m['loss'])
            if early_stopper.early_stop:
                print(f"Early stopping at epoch {epoch}")
                break

    # --- Final Test ---
    model.load_state_dict(torch.load('best_surrogate.pth', weights_only=True))
    test_m = evaluate(
        model, test_loader, form_scaler, gap_scaler, w_form, w_gap, device)
    print("\n--- Test Results (best model) ---")
    for k, v in test_m.items():
        print(f"  {k}: {v:.6f}")
        wandb.run.summary[f"Test/{k}"] = v

    wandb.finish()


if __name__ == "__main__":
    main()
