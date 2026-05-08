"""
Training pipeline for FiLM-PaiNN with Fidelity-Aware Denoising.

Adds a data-driven noise calibration step and an auxiliary denoising loss
(Noisy Nodes / Denoising Score Matching) to regularize the backbone.

Reuses evaluate(), loss classes, plotting, and metrics from train_pipeline.py.
"""

import torch
import pickle
import hydra
from omegaconf import DictConfig, OmegaConf
from torch_geometric.loader import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error as mae_score
import numpy as np
import os
import sys
import wandb
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from collections import defaultdict

# --- Import Fix ---
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from modelling.gnn.delta_painn_model import DeltaPaiNN
from modelling.gnn.train_pipeline import (
    log_model_stats,
    log_element_embeddings,
    plot_history,
    plot_log_regression,
    plot_uncertainty_calibration,
    EarlyStopping,
    save_history,
    GaussianNLLLoss,
    SMAPELoss,
    evaluate,
    _get_geo_weights,
)


# --- Fidelity-Aware Noise Calibration ---

def calibrate_noise_schedule(loader, sigma_low, sigma_high):
    """Build a data-driven fidelity map by scanning the training set.

    Groups data by unique precision_settings vectors. For each group,
    computes the MAE of delta_total_energy_per_atom. Maps higher error
    (worse fidelity) to higher noise sigma.

    Args:
        loader: DataLoader over training graphs.
        sigma_low: noise sigma for highest-fidelity group.
        sigma_high: noise sigma for lowest-fidelity group.

    Returns:
        fidelity_map: dict mapping precision_settings tuple -> sigma float.
    """
    group_errors = defaultdict(list)

    for data in loader:
        prec = data.precision_settings  # [B, 12]
        targets = data.delta_total_energy_per_atom  # [B]

        for i in range(prec.size(0)):
            key = tuple(prec[i].tolist())
            group_errors[key].append(abs(targets[i].item()))

    # Compute MAE per group
    group_mae = {}
    for key, errors in group_errors.items():
        group_mae[key] = np.mean(errors)

    mae_values = list(group_mae.values())
    min_mae = min(mae_values)
    max_mae = max(mae_values)

    fidelity_map = {}
    for key, mae_val in group_mae.items():
        # Edge case: single group or all groups have same MAE
        if max_mae - min_mae < 1e-12:
            score = 0.0
        else:
            score = (mae_val - min_mae) / (max_mae - min_mae)

        sigma = sigma_low + score * (sigma_high - sigma_low)
        fidelity_map[key] = sigma

    return fidelity_map


def get_per_atom_sigma(precision_settings, batch, fidelity_map,
                       fallback_sigma=None):
    """Look up per-atom noise sigma from the fidelity map.

    Args:
        precision_settings: [B, 12] per-graph precision vectors.
        batch: [N] atom-to-graph assignment.
        fidelity_map: dict from calibrate_noise_schedule.
        fallback_sigma: sigma for unseen settings (default: midpoint).

    Returns:
        sigma_atom: [N, 1] per-atom noise standard deviation.
    """
    B = precision_settings.size(0)
    device = precision_settings.device

    if fallback_sigma is None:
        sigmas = list(fidelity_map.values())
        fallback_sigma = np.mean(sigmas) if sigmas else 0.05

    sigma_graph = torch.zeros(B, device=device)
    for i in range(B):
        key = tuple(precision_settings[i].tolist())
        sigma_graph[i] = fidelity_map.get(key, fallback_sigma)

    sigma_atom = sigma_graph[batch].unsqueeze(-1)  # [N, 1]
    return sigma_atom


# --- Training Step with Denoising ---

def denoise_train_step(model, loader, optimizer, criterion, weights,
                       device, cfg, fidelity_map):
    """Training step with fidelity-aware denoising auxiliary loss.

    For each batch:
    1. Look up per-atom noise sigma from fidelity_map
    2. Inject noise: x_noisy = x + epsilon, epsilon ~ N(0, sigma^2 I)
    3. Forward pass on noisy positions
    4. Loss = main_loss + denoising_weight * MSE(vector_output, -epsilon)
    """
    model.train()
    total_loss = 0
    w_vol, w_lat = _get_geo_weights(weights)

    denoising_weight = cfg.denoising_weight

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        # 1. Noise injection
        sigma_atom = get_per_atom_sigma(
            data.precision_settings, data.batch, fidelity_map
        )  # [N, 1]

        epsilon = torch.randn_like(data.x) * sigma_atom  # [N, 3]

        # Clone positions before perturbing (safety)
        data.x = data.x.clone() + epsilon

        # 2. Forward pass
        target_delta_vol = data.delta_final_volume_per_atom.view(-1, 1)
        target_delta_lat = data.delta_lattice_params.view(-1, 6)

        pred_pos, pred_energy, pred_gap, pred_geo = model(data)
        geo_mean, geo_logvar = pred_geo

        # 3. Main losses
        loss_pos = criterion(
            pred_pos, data.delta_relaxed_atom_positions, is_position=True)
        loss_energy = criterion(
            pred_energy, data.delta_total_energy_per_atom, is_position=False)
        loss_gap = criterion(
            pred_gap, data.delta_homo_lumo_gap, is_position=False)
        loss_vol = criterion(
            (geo_mean[:, 0:1], geo_logvar[:, 0:1]),
            target_delta_vol, is_position=False)
        loss_lat = criterion(
            (geo_mean[:, 1:], geo_logvar[:, 1:]),
            target_delta_lat, is_position=False)

        loss_main = (weights.delta_r * loss_pos +
                     weights.delta_e * loss_energy +
                     weights.delta_gap * loss_gap +
                     w_vol * loss_vol +
                     w_lat * loss_lat)

        # 4. Denoising auxiliary loss: predict -epsilon from vector output
        loss_denoise = F.mse_loss(model._vector_output, -epsilon)

        # 5. Total loss
        total_loss_item = loss_main + denoising_weight * loss_denoise

        total_loss_item.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += total_loss_item.item() * data.num_graphs

    return total_loss / len(loader.dataset)


# --- Main ---

@hydra.main(version_base=None, config_path="conf",
            config_name="painn_denoising_config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Seed for reproducibility
    seed = cfg.get("seed", None)
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"Random seed set to {seed}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    wandb.init(
        project=cfg.get("wandb_project", "painn-delta-denoising"),
        group=cfg.get("wandb_group", None),
        config=OmegaConf.to_container(cfg, resolve=True),
        name=cfg.job_name,
        dir=os.getcwd()
    )

    if not os.path.exists(cfg.data_file):
        print(f"Error: {cfg.data_file} not found.")
        return

    print(f"Loading data from {cfg.data_file}...")
    loaded_dict = torch.load(cfg.data_file, weights_only=False)
    all_graphs = loaded_dict['graphs']

    # Loss setup
    loss_type = cfg.training.get("loss_function", "nll").lower()
    print(f"--- CONFIGURING LOSS FUNCTION: {loss_type.upper()} ---")

    if loss_type == "smape":
        criterion = SMAPELoss()
        loss_label = "sMAPE"
    elif loss_type == "nll":
        criterion = GaussianNLLLoss()
        loss_label = "NLL"
    else:
        raise ValueError(f"Unknown loss function: {loss_type}")

    train_graphs = [d for d in all_graphs if d.split == 'train']
    val_graphs = [d for d in all_graphs if d.split == 'val']
    test_graphs = [d for d in all_graphs if d.split == 'test']

    print(f"Dataset Split: Train={len(train_graphs)}, "
          f"Val={len(val_graphs)}, Test={len(test_graphs)}")

    if len(val_graphs) == 0:
        print("WARNING: No validation data found. Fallback to split.")
        total_train = len(train_graphs)
        split_idx = int(total_train * 0.9)
        val_graphs = train_graphs[split_idx:]
        train_graphs = train_graphs[:split_idx]

    train_loader = DataLoader(
        train_graphs, batch_size=cfg.training.batch_size, shuffle=True)
    val_loader = DataLoader(
        val_graphs, batch_size=cfg.training.batch_size, shuffle=False)
    test_loader = DataLoader(
        test_graphs, batch_size=cfg.training.batch_size, shuffle=False)

    # --- Calibrate noise schedule ---
    print("--- Calibrating fidelity-aware noise schedule ---")
    fidelity_map = calibrate_noise_schedule(
        train_loader,
        sigma_low=cfg.noise_scales.low,
        sigma_high=cfg.noise_scales.high,
    )
    print(f"Found {len(fidelity_map)} unique fidelity groups.")
    for key, sigma in sorted(fidelity_map.items(), key=lambda x: x[1]):
        print(f"  sigma={sigma:.4f}")

    # Log fidelity map to WandB
    wandb.run.summary["num_fidelity_groups"] = len(fidelity_map)
    wandb.run.summary["fidelity_sigma_range"] = [
        min(fidelity_map.values()), max(fidelity_map.values())
    ]

    # --- Model ---
    model = DeltaPaiNN(
        num_layers=cfg.model.num_layers,
        hidden_dim=cfg.model.hidden_dim,
        num_cheap_dft_inputs=cfg.model.num_cheap_dft_inputs,
        num_precision_settings=cfg.model.num_precision_settings,
        num_geo_inputs=cfg.model.num_geo_inputs,
        max_z=cfg.model.max_z,
        num_rbf=cfg.model.num_rbf,
        cutoff=cfg.model.cutoff,
        use_film=cfg.model.get("use_film", True),
    ).to(device)

    log_model_stats(model)

    # --- Optimizer & Scheduler ---
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=1e-5)

    warmup_epochs = 10
    total_epochs = cfg.training.epochs
    scheduler_warmup = LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0,
        total_iters=warmup_epochs)
    scheduler_decay = CosineAnnealingLR(
        optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-6)
    scheduler = SequentialLR(
        optimizer, schedulers=[scheduler_warmup, scheduler_decay],
        milestones=[warmup_epochs])

    # --- History tracking ---
    metrics_names = [
        "MAE_delta_energy", "MAE_delta_gap", "MAE_delta_positions",
        "MAE_delta_geo_total", "MAE_delta_volume",
        "MAE_delta_a", "MAE_delta_b", "MAE_delta_c",
        "MAE_delta_alpha", "MAE_delta_beta", "MAE_delta_gamma",
    ]

    history = {'train_loss': []}
    for m in metrics_names:
        history[f'Train_{m}'] = []
        history[f'Val_{m}'] = []
    history['Train_Loss'] = []
    history['Val_Loss'] = []

    best_val_loss = float('inf')
    best_val_mae_e = float('inf')
    best_val_smape_e = float('inf')

    early_stopper = EarlyStopping(patience=10, min_delta=1e-5)

    print(f"{'Epoch':>5} | {'Step_Loss':>9} | {'Val_Loss':>9} | "
          f"{'Val_MAE_E':>9} | {'Val_sMAPE_E':>12}")
    print("-" * 60)

    # --- Training loop ---
    for epoch in range(1, cfg.training.epochs + 1):
        train_step_loss = denoise_train_step(
            model, train_loader, optimizer, criterion,
            cfg.weights, device, cfg, fidelity_map
        )
        history['train_loss'].append(train_step_loss)
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1 or epoch == cfg.training.epochs:
            val_m, val_arrays = evaluate(
                model, val_loader, criterion, cfg.weights)
            train_m, _ = evaluate(
                model, train_loader, criterion, cfg.weights)

            current_val_loss = val_m['Val_Loss']

            history['Val_Loss'].append((epoch, current_val_loss))
            history['Train_Loss'].append((epoch, train_m['Val_Loss']))

            for k in metrics_names:
                history[f'Val_{k}'].append((epoch, val_m[k]))
                history[f'Train_{k}'].append((epoch, train_m[k]))

            print(f"{epoch:05d} | {train_step_loss:.4f}    | "
                  f"{current_val_loss:.4f}    | "
                  f"{val_m['MAE_delta_energy']:.4f}      | "
                  f"{val_m['sMAPE_delta_energy']:.2f}%")

            current_lr = optimizer.param_groups[0]['lr']

            log_dict = {
                "epoch": epoch,
                f"Train/Loss_{loss_label}": train_step_loss,
                "Train/Learning_Rate": current_lr,
                f"Train/Eval_Loss_{loss_label}": train_m['Val_Loss'],
                f"Val/Loss_{loss_label}": current_val_loss,
                "Train/MAE_Energy": train_m['MAE_delta_energy'],
                "Train/sMAPE_Energy": train_m['sMAPE_delta_energy'],
                "Train/MagAcc_Energy": train_m['MagAcc_delta_energy'],
                "Val/MAE_Energy": val_m['MAE_delta_energy'],
                "Val/sMAPE_Energy": val_m['sMAPE_delta_energy'],
                "Val/MagAcc_Energy": val_m['MagAcc_delta_energy'],
                "Train/MAE_Gap": train_m['MAE_delta_gap'],
                "Train/sMAPE_Gap": train_m['sMAPE_delta_gap'],
                "Train/MagAcc_Gap": train_m['MagAcc_delta_gap'],
                "Val/MAE_Gap": val_m['MAE_delta_gap'],
                "Val/sMAPE_Gap": val_m['sMAPE_delta_gap'],
                "Val/MagAcc_Gap": val_m['MagAcc_delta_gap'],
                "Train/MAE_Pos": train_m['MAE_delta_positions'],
                "Train/sMAPE_Pos": train_m['sMAPE_delta_positions'],
                "Train/MagAcc_Pos": train_m['MagAcc_delta_positions'],
                "Val/MAE_Pos": val_m['MAE_delta_positions'],
                "Val/sMAPE_Pos": val_m['sMAPE_delta_positions'],
                "Val/MagAcc_Pos": val_m['MagAcc_delta_positions'],
                "Val/Geo/Total_MAE": val_m['MAE_delta_geo_total'],
            }

            geo_vars = [
                "delta_volume", "delta_a", "delta_b", "delta_c",
                "delta_alpha", "delta_beta", "delta_gamma",
            ]
            for key in geo_vars:
                log_dict[f"Train/Geo/MAE/{key}"] = train_m[f"MAE_{key}"]
                log_dict[f"Val/Geo/MAE/{key}"] = val_m[f"MAE_{key}"]
                log_dict[f"Train/Geo/sMAPE/{key}"] = train_m[f"sMAPE_{key}"]
                log_dict[f"Val/Geo/sMAPE/{key}"] = val_m[f"sMAPE_{key}"]
                log_dict[f"Train/Geo/MagAcc/{key}"] = train_m[f"MagAcc_{key}"]
                log_dict[f"Val/Geo/MagAcc/{key}"] = val_m[f"MagAcc_{key}"]

            wandb.log(log_dict, step=epoch)

            # Checkpointing
            if val_m['MAE_delta_energy'] < best_val_mae_e:
                best_val_mae_e = val_m['MAE_delta_energy']
                torch.save(model.state_dict(), 'best_delta_model.pth')
                wandb.save('best_delta_model.pth')

            if val_m['sMAPE_delta_energy'] < best_val_smape_e:
                best_val_smape_e = val_m['sMAPE_delta_energy']
                torch.save(model.state_dict(), 'best_smape_model.pth')
                wandb.save('best_smape_model.pth')

            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                torch.save(model.state_dict(),
                           f'best_{loss_type}_loss_model.pth')

            torch.save(model.state_dict(), 'last_model.pth')
            wandb.save('last_model.pth')

            # Save fidelity map alongside model checkpoint
            torch.save(fidelity_map, 'fidelity_map.pt')

            plot_history(history, save_path="training_progress.png",
                         loss_label=loss_label)
            plot_log_regression(val_arrays,
                                save_path="regression_scatter.png")
            plot_uncertainty_calibration(
                val_arrays, save_path="calibration_scatter.png")

            wandb.log({
                "learning_curves": wandb.Image("training_progress.png"),
                "regression_scatter": wandb.Image("regression_scatter.png"),
            }, step=epoch)

            save_history(history, filename="training_history.pkl")
            wandb.save("training_history.pkl")

            # Early stopping
            early_stopper(val_m['sMAPE_delta_energy'])
            if early_stopper.early_stop:
                print(f"Early stopping triggered at epoch {epoch}!")
                break

    log_element_embeddings(model)
    wandb.finish()


if __name__ == "__main__":
    main()
