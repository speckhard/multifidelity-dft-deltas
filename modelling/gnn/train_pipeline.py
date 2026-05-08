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
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

# --- Safe Import ---
try:
    from mendeleev import element
except ImportError:
    element = None

# --- Import Fix ---
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from modelling.gnn.delta_egnn_model import DeltaGNN
from modelling.gnn.settings_only_egnn_model import SettingsOnlyGNN
from modelling.gnn.delta_egnn_attention_model import DeltaAttentionGNN

# --- Helpers ---

def log_model_stats(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_size_bytes = total_params * 4
    buffer_size_bytes = sum(b.numel() * 4 for b in model.buffers())
    total_size_mb = (param_size_bytes + buffer_size_bytes) / (1024 ** 2)

    print(f"Total Params: {total_params:,}")
    print(f"Trainable Params: {trainable_params:,}")
    print(f"Static Model Size: {total_size_mb:.2f} MB")
    
    wandb.run.summary["total_parameters"] = total_params
    wandb.run.summary["trainable_parameters"] = trainable_params
    wandb.run.summary["model_size_mb"] = total_size_mb

def get_z_to_symbol_map(max_z=100):
    if element is None: return {}
    z_to_symbol = {}
    for z in range(1, max_z + 1):
        try:
            el = element(z)
            z_to_symbol[z] = el.symbol
        except: continue
    return z_to_symbol

Z_TO_SYMBOL = get_z_to_symbol_map(max_z=118)

def log_element_embeddings(model):
    print("Logging element embeddings to WandB...")
    embedding_matrix = model.z_embedding.weight.detach().cpu().numpy()
    data = []
    columns = ["Z", "Symbol", "Group", "Period"] + [f"dim_{i}" for i in range(embedding_matrix.shape[1])]
    for z in range(1, len(embedding_matrix)):
        if z in Z_TO_SYMBOL:
            symbol = Z_TO_SYMBOL[z]
            try:
                el = element(z)
                group = el.group_id if el.group_id else -1
                period = el.period
            except: group = -1; period = -1
            vector = embedding_matrix[z].tolist()
            row = [z, symbol, group, period] + vector
            data.append(row)
    table = wandb.Table(columns=columns, data=data)
    wandb.log({"element_embeddings": table})

def plot_history(history, save_path, loss_label="Loss"):
    """Plot learning curves."""
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    
    def plot_metric(ax_idx, metric_key, title, y_label, log_scale=False):
        ax = axes[ax_idx]
        if metric_key == 'train_loss':
            ax.plot(epochs, history['train_loss'], label='Step Loss', color='navy', alpha=0.3)
            if 'Val_Loss' in history and history['Val_Loss']:
                eval_epochs = [x[0] for x in history['Val_Loss']]
                val_vals = [x[1] for x in history['Val_Loss']]
                ax.plot(eval_epochs, val_vals, label=f'Val {loss_label}', color='orange', linewidth=2)
            if 'Train_Loss' in history and history['Train_Loss']:
                eval_epochs = [x[0] for x in history['Train_Loss']]
                train_vals = [x[1] for x in history['Train_Loss']]
                ax.plot(eval_epochs, train_vals, label=f'Train {loss_label}', color='blue', linestyle='--')
        else:
            val_key = f"Val_{metric_key}"
            train_key = f"Train_{metric_key}"
            if val_key in history and history[val_key]:
                eval_epochs = [x[0] for x in history[val_key]]
                val_vals = [x[1] for x in history[val_key]]
                ax.plot(eval_epochs, val_vals, label='Validation', linewidth=2, color='orange')
                if train_key in history and history[train_key]:
                    train_vals = [x[1] for x in history[train_key]]
                    ax.plot(eval_epochs, train_vals, label='Train (Full)', linestyle='--', alpha=0.7, color='blue')
                    
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel('Epochs')
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.3)
        if log_scale: ax.set_yscale('log')
        ax.legend(fontsize='small')

    plot_metric(0, 'train_loss', f'Optimization Loss ({loss_label})', 'Loss')
    plot_metric(1, 'MAE_delta_energy', 'Delta Energy', r'MAE (eV/atom)', log_scale=True)
    plot_metric(2, 'MAE_delta_gap', 'Delta Gap', 'MAE (eV)', log_scale=True)
    plot_metric(3, 'MAE_delta_positions', 'Delta Positions', r'MAE ($\AA$)')
    plot_metric(4, 'MAE_delta_geo_total', 'Total Geometry MAE', 'MAE (Mixed)')
    plot_metric(5, 'MAE_delta_volume', 'Delta Volume', r'MAE ($\AA^3$/atom)')
    plot_metric(6, 'MAE_delta_a', 'Delta Lattice a', r'MAE ($\AA$)')
    plot_metric(7, 'MAE_delta_b', 'Delta Lattice b', r'MAE ($\AA$)')
    plot_metric(8, 'MAE_delta_c', 'Delta Lattice c', r'MAE ($\AA$)')
    plot_metric(9, 'MAE_delta_alpha', 'Delta Alpha', 'MAE (Degrees)')
    plot_metric(10, 'MAE_delta_beta', 'Delta Beta', 'MAE (Degrees)')
    plot_metric(11, 'MAE_delta_gamma', 'Delta Gamma', 'MAE (Degrees)')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_log_regression(arrays, save_path):
    epsilon = 1e-7
    keys = [('delta_energy', 'Delta Energy'), ('delta_gap', 'Delta Gap'), 
            ('delta_positions', 'Delta Positions'), ('delta_volume', 'Delta Volume'), 
            ('delta_a', 'Delta a'), ('delta_b', 'Delta b'), ('delta_c', 'Delta c'),
            ('delta_alpha', 'Delta Alpha'), ('delta_beta', 'Delta Beta'), ('delta_gamma', 'Delta Gamma')]
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()

    for idx, (key, title) in enumerate(keys):
        ax = axes[idx]
        if key not in arrays: continue
        pred = np.array(arrays[key]['pred']).flatten()
        targ = np.array(arrays[key]['targ']).flatten()
        if len(pred) == 0: continue
        log_pred = np.log10(np.abs(pred) + epsilon)
        log_targ = np.log10(np.abs(targ) + epsilon)
        ax.scatter(log_targ, log_pred, alpha=0.3, s=5, c='navy', label='Data')
        min_val = min(log_targ.min(), log_pred.min())
        max_val = max(log_targ.max(), log_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='Ideal')
        ax.set_title(f"{title}: Prediction Fit", fontweight='bold')
        ax.set_xlabel(r'$Log_{10}(|Target|)$')
        ax.set_ylabel(r'$Log_{10}(|Pred|)$')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize='small')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_uncertainty_calibration(arrays, save_path):
    epsilon = 1e-7
    keys = [('delta_energy', 'Delta Energy'), ('delta_gap', 'Delta Gap'), 
            ('delta_positions', 'Delta Positions'), ('delta_volume', 'Delta Volume'), 
            ('delta_a', 'Delta a'), ('delta_b', 'Delta b'), ('delta_c', 'Delta c'),
            ('delta_alpha', 'Delta Alpha'), ('delta_beta', 'Delta Beta'), ('delta_gamma', 'Delta Gamma')]
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()

    for idx, (key, title) in enumerate(keys):
        ax = axes[idx]
        if key not in arrays: continue
        pred = np.array(arrays[key]['pred']).flatten()
        targ = np.array(arrays[key]['targ']).flatten()
        sigma = np.array(arrays[key]['sigma']).flatten() 
        if len(pred) == 0: continue
        abs_error = np.abs(pred - targ)
        log_sigma = np.log10(sigma + epsilon)
        log_error = np.log10(abs_error + epsilon)
        ax.scatter(log_sigma, log_error, alpha=0.3, s=5, c='teal', label='Data')
        min_val = min(log_sigma.min(), log_error.min())
        max_val = max(log_sigma.max(), log_error.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='Ideal Calibration')
        ax.set_title(f"{title}: Uncertainty Calibration", fontweight='bold')
        ax.set_xlabel(r'$Log_{10}(\sigma_{pred})$')
        ax.set_ylabel(r'$Log_{10}(|Error|)$')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize='small')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, current_metric):
        if self.best_score is None:
            self.best_score = current_metric
        elif current_metric > self.best_score - self.min_delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = current_metric
            self.counter = 0

def save_history(history, filename="training_history.pkl"):
    with open(filename, 'wb') as f:
        pickle.dump(history, f)
    print(f"History saved to {filename}")

# --- Metrics Helpers ---
def calculate_smape(pred, target, epsilon=1e-7):
    numerator = torch.abs(pred - target)
    denominator = torch.abs(pred) + torch.abs(target) + epsilon
    smape = 100.0 * 2.0 * numerator / denominator
    return torch.mean(smape).item()

def calculate_rmslae(pred, target, epsilon=1e-4):
    """RMSLAE using log10(|x| + ε). Matches RMSLAELoss training loss and RF scorer."""
    abs_pred = torch.abs(pred)
    abs_target = torch.abs(target)
    log_pred = torch.log10(abs_pred + epsilon)
    log_target = torch.log10(abs_target + epsilon)
    mse_log = F.mse_loss(log_pred, log_target)
    return torch.sqrt(mse_log).item()

def calculate_magnitude_accuracy(pred, target):
    boundaries = torch.tensor([1e-3, 1e-2, 1e-1, 1.0, 10.0], device=pred.device)
    abs_pred = torch.abs(pred)
    abs_target = torch.abs(target)
    pred_bins = torch.bucketize(abs_pred, boundaries)
    target_bins = torch.bucketize(abs_target, boundaries)
    correct = (pred_bins == target_bins).float()
    return torch.mean(correct).item()

# --- Loss Functions ---
class SMAPELoss(nn.Module):
    def __init__(self, epsilon=1e-4):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, preds, target, is_position=False):
        if isinstance(preds, tuple):
            mean, _ = preds
        else:
            mean = preds
        numerator = torch.abs(mean - target)
        denominator = torch.abs(mean) + torch.abs(target) + self.epsilon
        loss = 2.0 * numerator / denominator
        return 100.0 * loss.mean()

class AsinhL1Loss(nn.Module):
    """Asinh-transformed L1 loss: L1(asinh(x/scale), asinh(y/scale))."""
    def __init__(self, scale=1e-5):
        super().__init__()
        self.scale = scale

    def forward(self, preds, target, is_position=False):
        if isinstance(preds, tuple):
            mean = preds[0]
        else:
            mean = preds
        loss = torch.abs(torch.asinh(mean / self.scale) - torch.asinh(target / self.scale))
        return loss.mean()

class RMSLAELossLn(nn.Module):
    """Root Mean Squared Log Absolute Error (natural log variant, legacy)."""
    def __init__(self, epsilon=1e-4):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, preds, target, is_position=False):
        if isinstance(preds, tuple):
            mean = preds[0]
        else:
            mean = preds
        log_pred = torch.log(torch.abs(mean) + self.epsilon)
        log_target = torch.log(torch.abs(target) + self.epsilon)
        return torch.sqrt(F.mse_loss(log_pred, log_target))

class RMSLAELoss(nn.Module):
    """Root Mean Squared Log10 Absolute Error: sqrt(mean((log10(|pred|+ε) - log10(|targ|+ε))²))."""
    def __init__(self, epsilon=1e-4):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, preds, target, is_position=False):
        if isinstance(preds, tuple):
            mean = preds[0]
        else:
            mean = preds
        log_pred = torch.log10(torch.abs(mean) + self.epsilon)
        log_target = torch.log10(torch.abs(target) + self.epsilon)
        return torch.sqrt(F.mse_loss(log_pred, log_target))

class GaussianNLLLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, target, is_position=False):
        mean, log_var = preds
        log_var = torch.clamp(log_var, min=-20, max=20)
        if is_position:
            loss = torch.exp(-log_var) * (target - mean)**2 + log_var
            return 0.5 * loss.mean()
        else:
            loss = torch.exp(-log_var) * (target - mean)**2 + log_var
            return 0.5 * loss.mean()

# --- Training Logic ---
def _get_geo_weights(weights):
    """Get separate volume and lattice weights, backward compatible with delta_geo."""
    w_vol = getattr(weights, 'delta_vol', getattr(weights, 'delta_geo', 1.0))
    w_lat = getattr(weights, 'delta_lat', getattr(weights, 'delta_geo', 1.0))
    return w_vol, w_lat


def train_step(model, loader, optimizer, criterion, weights, device):
    model.train()
    total_loss = 0
    w_vol, w_lat = _get_geo_weights(weights)

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        target_delta_vol = data.delta_final_volume_per_atom.view(-1, 1)
        target_delta_lat = data.delta_lattice_params.view(-1, 6)

        pred_delta_positions, pred_delta_energy, pred_delta_gap, pred_delta_geo = model(data)
        geo_mean, geo_logvar = pred_delta_geo

        loss_delta_positions = criterion(pred_delta_positions, data.delta_relaxed_atom_positions, is_position=True)
        loss_delta_energy = criterion(pred_delta_energy, data.delta_total_energy_per_atom, is_position=False)
        loss_delta_gap = criterion(pred_delta_gap, data.delta_homo_lumo_gap, is_position=False)
        loss_delta_vol = criterion(
            (geo_mean[:, 0:1], geo_logvar[:, 0:1]), target_delta_vol, is_position=False)
        loss_delta_lat = criterion(
            (geo_mean[:, 1:], geo_logvar[:, 1:]), target_delta_lat, is_position=False)

        total_loss_item = (weights.delta_r * loss_delta_positions +
                           weights.delta_e * loss_delta_energy +
                           weights.delta_gap * loss_delta_gap +
                           w_vol * loss_delta_vol +
                           w_lat * loss_delta_lat)

        total_loss_item.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += total_loss_item.item() * data.num_graphs
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, criterion, weights):
    model.eval()
    device = next(model.parameters()).device
    w_vol, w_lat = _get_geo_weights(weights)

    arrays = {
        'delta_energy': {'pred': [], 'targ': [], 'sigma': []},
        'delta_gap': {'pred': [], 'targ': [], 'sigma': []},
        'delta_positions': {'pred': [], 'targ': [], 'sigma': []}, 
        'delta_volume': {'pred': [], 'targ': [], 'sigma': []},
        'delta_a': {'pred': [], 'targ': [], 'sigma': []},
        'delta_b': {'pred': [], 'targ': [], 'sigma': []},
        'delta_c': {'pred': [], 'targ': [], 'sigma': []},
        'delta_alpha': {'pred': [], 'targ': [], 'sigma': []},
        'delta_beta': {'pred': [], 'targ': [], 'sigma': []},
        'delta_gamma': {'pred': [], 'targ': [], 'sigma': []},
    }

    total_mae_pos = 0.0
    total_mae_geo = 0.0
    total_mae_geo_dims = np.zeros(7) 
    
    total_smape_geo_dims = np.zeros(7)
    total_smape_delta_energy = 0.0
    total_smape_delta_gap = 0.0
    total_smape_delta_positions = 0.0

    total_rmslae_geo_dims = np.zeros(7)
    total_rmslae_delta_energy = 0.0
    total_rmslae_delta_gap = 0.0
    total_rmslae_delta_positions = 0.0
    
    total_acc_geo_dims = np.zeros(7)
    total_acc_delta_energy = 0.0
    total_acc_delta_gap = 0.0
    total_acc_delta_positions = 0.0
    
    total_val_loss = 0.0
    total_atoms = 0
    total_graphs = 0
    epsilon = 1e-7  # Strict epsilon for validation metrics (training loss uses 1e-4)

    geo_keys = ["delta_volume", "delta_a", "delta_b", "delta_c", "delta_alpha", "delta_beta", "delta_gamma"]

    for data in loader:
        data = data.to(device)
        target_delta_vol = data.delta_final_volume_per_atom.view(-1, 1)
        target_delta_lat = data.delta_lattice_params.view(-1, 6)
        target_delta_geo = torch.cat([target_delta_vol, target_delta_lat], dim=1)

        pred_delta_positions, pred_delta_energy, pred_delta_gap, pred_delta_geo = model(data)
        
        pos_mean, pos_logvar = pred_delta_positions
        eng_mean, eng_logvar = pred_delta_energy
        gap_mean, gap_logvar = pred_delta_gap
        geo_mean, geo_logvar = pred_delta_geo

        loss_delta_positions = criterion(pred_delta_positions, data.delta_relaxed_atom_positions, is_position=True)
        loss_delta_energy = criterion(pred_delta_energy, data.delta_total_energy_per_atom, is_position=False)
        loss_delta_gap = criterion(pred_delta_gap, data.delta_homo_lumo_gap, is_position=False)
        loss_delta_vol = criterion(
            (geo_mean[:, 0:1], geo_logvar[:, 0:1]), target_delta_vol, is_position=False)
        loss_delta_lat = criterion(
            (geo_mean[:, 1:], geo_logvar[:, 1:]), target_delta_lat, is_position=False)

        batch_loss = (weights.delta_r * loss_delta_positions +
                      weights.delta_e * loss_delta_energy +
                      weights.delta_gap * loss_delta_gap +
                      w_vol * loss_delta_vol +
                      w_lat * loss_delta_lat)
        total_val_loss += batch_loss.item() * data.num_graphs

        # --- Metrics ---
        # Energy
        arrays['delta_energy']['pred'].append(eng_mean.cpu().numpy())
        arrays['delta_energy']['targ'].append(data.delta_total_energy_per_atom.cpu().numpy())
        arrays['delta_energy']['sigma'].append(torch.exp(0.5 * eng_logvar).cpu().numpy())
        
        total_smape_delta_energy += calculate_smape(eng_mean, data.delta_total_energy_per_atom, epsilon) * data.num_graphs
        total_rmslae_delta_energy += calculate_rmslae(eng_mean, data.delta_total_energy_per_atom) * data.num_graphs
        total_acc_delta_energy += calculate_magnitude_accuracy(eng_mean, data.delta_total_energy_per_atom) * data.num_graphs
        
        # Gap
        arrays['delta_gap']['pred'].append(gap_mean.cpu().numpy())
        arrays['delta_gap']['targ'].append(data.delta_homo_lumo_gap.cpu().numpy())
        arrays['delta_gap']['sigma'].append(torch.exp(0.5 * gap_logvar).cpu().numpy())
        
        total_smape_delta_gap += calculate_smape(gap_mean, data.delta_homo_lumo_gap, epsilon) * data.num_graphs
        total_rmslae_delta_gap += calculate_rmslae(gap_mean, data.delta_homo_lumo_gap) * data.num_graphs
        total_acc_delta_gap += calculate_magnitude_accuracy(gap_mean, data.delta_homo_lumo_gap) * data.num_graphs

        # Positions
        p_r_norm = torch.norm(pos_mean, dim=1) 
        t_r_norm = torch.norm(data.delta_relaxed_atom_positions, dim=1)
        
        arrays['delta_positions']['pred'].append(p_r_norm.detach().cpu().numpy())
        arrays['delta_positions']['targ'].append(t_r_norm.detach().cpu().numpy())
        arrays['delta_positions']['sigma'].append(torch.exp(0.5 * pos_logvar).detach().cpu().numpy())

        diff_pos = (pos_mean - data.delta_relaxed_atom_positions).cpu().numpy()
        total_mae_pos += np.sum(np.linalg.norm(diff_pos, axis=1))
        
        p_flat = pos_mean.view(-1); t_flat = data.delta_relaxed_atom_positions.view(-1)
        total_smape_delta_positions += calculate_smape(p_flat, t_flat, epsilon) * data.num_graphs
        total_rmslae_delta_positions += calculate_rmslae(p_flat, t_flat) * data.num_graphs
        total_acc_delta_positions += calculate_magnitude_accuracy(p_flat, t_flat) * data.num_graphs

        total_atoms += data.x.shape[0]

        # Geometry
        total_mae_geo += F.l1_loss(geo_mean, target_delta_geo, reduction='sum').item()
        diff_geo = torch.abs(geo_mean - target_delta_geo)
        batch_geo_sum = torch.sum(diff_geo, dim=0).cpu().numpy()
        total_mae_geo_dims += batch_geo_sum
        
        for i, key in enumerate(geo_keys):
            p = geo_mean[:, i]
            t = target_delta_geo[:, i]
            s = torch.exp(0.5 * geo_logvar[:, i])
            
            total_smape_geo_dims[i] += calculate_smape(p, t, epsilon) * data.num_graphs
            total_rmslae_geo_dims[i] += calculate_rmslae(p, t) * data.num_graphs
            total_acc_geo_dims[i] += calculate_magnitude_accuracy(p, t) * data.num_graphs
            
            arrays[key]['pred'].append(p.detach().cpu().numpy())
            arrays[key]['targ'].append(t.detach().cpu().numpy())
            arrays[key]['sigma'].append(s.detach().cpu().numpy())

        total_graphs += data.num_graphs

    metrics = {}
    metrics["Val_Loss"] = total_val_loss / total_graphs

    for k in arrays:
        if len(arrays[k]['pred']) > 0:
            arrays[k]['pred'] = np.concatenate(arrays[k]['pred'])
            arrays[k]['targ'] = np.concatenate(arrays[k]['targ'])
            arrays[k]['sigma'] = np.concatenate(arrays[k]['sigma'])
        else:
            arrays[k]['pred'] = np.array([])
            arrays[k]['targ'] = np.array([])
            arrays[k]['sigma'] = np.array([])

    # Guard against NaN predictions (training instability at high LR).
    # Return inf metrics so early stopping triggers gracefully.
    has_nan = any(np.any(np.isnan(arrays[k]['pred'])) for k in arrays if len(arrays[k]['pred']) > 0)
    if has_nan:
        inf = float('inf')
        metrics["Val_Loss"] = inf
        metrics["MAE_delta_energy"] = inf
        metrics["MAE_delta_gap"] = inf
        metrics["MAE_delta_positions"] = inf
        metrics["MAE_delta_geo_total"] = inf
        for suffix in ["delta_energy", "delta_gap", "delta_positions"]:
            metrics[f"sMAPE_{suffix}"] = inf
            metrics[f"RMSLAE_{suffix}"] = inf
            metrics[f"MagAcc_{suffix}"] = 0.0
        geo_keys = ["delta_volume", "delta_a", "delta_b", "delta_c", "delta_alpha", "delta_beta", "delta_gamma"]
        for key in geo_keys:
            metrics[f"MAE_{key}"] = inf
            metrics[f"sMAPE_{key}"] = inf
            metrics[f"RMSLAE_{key}"] = inf
            metrics[f"MagAcc_{key}"] = 0.0
        return metrics, arrays

    if len(arrays['delta_energy']['pred']) > 0:
        metrics["MAE_delta_energy"] = mae_score(arrays['delta_energy']['targ'], arrays['delta_energy']['pred'])
        metrics["MAE_delta_gap"] = mae_score(arrays['delta_gap']['targ'], arrays['delta_gap']['pred'])
    else:
        metrics["MAE_delta_energy"] = 0.0
        metrics["MAE_delta_gap"] = 0.0

    metrics["MAE_delta_positions"] = total_mae_pos / total_atoms if total_atoms > 0 else 0.0
    metrics["MAE_delta_geo_total"] = total_mae_geo / total_graphs if total_graphs > 0 else 0.0

    div = total_graphs if total_graphs > 0 else 1.0
    
    metrics["sMAPE_delta_energy"] = total_smape_delta_energy / div
    metrics["RMSLAE_delta_energy"] = total_rmslae_delta_energy / div
    metrics["MagAcc_delta_energy"] = total_acc_delta_energy / div
    
    metrics["sMAPE_delta_gap"] = total_smape_delta_gap / div
    metrics["RMSLAE_delta_gap"] = total_rmslae_delta_gap / div
    metrics["MagAcc_delta_gap"] = total_acc_delta_gap / div
    
    metrics["sMAPE_delta_positions"] = total_smape_delta_positions / div
    metrics["RMSLAE_delta_positions"] = total_rmslae_delta_positions / div
    metrics["MagAcc_delta_positions"] = total_acc_delta_positions / div
    
    geo_mae_avg = total_mae_geo_dims / div
    geo_smape_avg = total_smape_geo_dims / div
    geo_rmslae_avg = total_rmslae_geo_dims / div
    geo_acc_avg = total_acc_geo_dims / div

    for i, key in enumerate(geo_keys):
        metrics[f"MAE_{key}"] = geo_mae_avg[i]
        metrics[f"sMAPE_{key}"] = geo_smape_avg[i]
        metrics[f"RMSLAE_{key}"] = geo_rmslae_avg[i]
        metrics[f"MagAcc_{key}"] = geo_acc_avg[i]
        
    return metrics, arrays

@hydra.main(version_base=None, config_path="conf", config_name="config")
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
        project=cfg.get("wandb_project", "egnn-delta-learning"),
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

    loss_type = cfg.training.get("loss_function", "nll").lower()
    print(f"--- CONFIGURING LOSS FUNCTION: {loss_type.upper()} ---")
    
    smape_eps = cfg.training.get("smape_epsilon", 1e-4)
    asinh_scale = cfg.training.get("asinh_scale", 1e-5)

    if loss_type == "smape":
        criterion = SMAPELoss(epsilon=smape_eps)
        loss_label = "sMAPE"
    elif loss_type == "nll":
        criterion = GaussianNLLLoss()
        loss_label = "NLL"
    elif loss_type == "asinh_l1":
        criterion = AsinhL1Loss(scale=asinh_scale)
        loss_label = "AsinhL1"
    elif loss_type == "rmsle":
        criterion = RMSLAELossLn(epsilon=1e-4)
        loss_label = "RMSLAE_ln"
    elif loss_type == "rmslae":
        criterion = RMSLAELoss(epsilon=1e-4)
        loss_label = "RMSLAE"
    else:
        raise ValueError(f"Unknown loss function: {loss_type}")

    train_graphs = [data for data in all_graphs if data.split == 'train']
    val_graphs   = [data for data in all_graphs if data.split == 'val']
    test_graphs  = [data for data in all_graphs if data.split == 'test']

    print(f"Dataset Split: Train={len(train_graphs)}, Val={len(val_graphs)}, Test={len(test_graphs)}")

    if len(val_graphs) == 0:
        print("WARNING: No validation data found. Fallback to split.")
        total_train = len(train_graphs)
        split_idx = int(total_train * 0.9)
        val_graphs = train_graphs[split_idx:]
        train_graphs = train_graphs[:split_idx]

    # Learning curve support: subsample training data
    train_fraction = cfg.training.get("train_fraction", 1.0)
    if train_fraction < 1.0:
        n_full = len(train_graphs)
        n_subset = max(1, int(n_full * train_fraction))
        # Deterministic shuffle with seed, then take first n_subset
        rng = np.random.RandomState(seed if seed is not None else 42)
        indices = rng.permutation(n_full)[:n_subset]
        train_graphs = [train_graphs[i] for i in sorted(indices)]
        print(f"Learning curve: using {n_subset}/{n_full} training graphs "
              f"({train_fraction*100:.0f}%)")

    train_loader = DataLoader(train_graphs, batch_size=cfg.training.batch_size, shuffle=True)
    val_loader   = DataLoader(val_graphs, batch_size=cfg.training.batch_size, shuffle=False)
    test_loader  = DataLoader(test_graphs, batch_size=cfg.training.batch_size, shuffle=False)

    model_type = cfg.get("model_type", "delta")

    if model_type == "settings_only":
        print("--- Using SettingsOnlyGNN (no cheap DFT inputs) ---")
        model = SettingsOnlyGNN(
            num_layers=cfg.model.num_layers,
            hidden_features=cfg.model.hidden_features,
            num_precision_settings=cfg.model.num_precision_settings,
            max_z=cfg.model.max_z
        ).to(device)
    elif model_type == "painn":
        from modelling.gnn.delta_painn_model import DeltaPaiNN
        print("--- Using DeltaPaiNN (FiLM-conditioned PaiNN) ---")
        model = DeltaPaiNN(
            num_layers=cfg.model.num_layers,
            hidden_dim=cfg.model.get("hidden_dim", cfg.model.get("hidden_features", 128)),
            num_cheap_dft_inputs=cfg.model.num_cheap_dft_inputs,
            num_precision_settings=cfg.model.num_precision_settings,
            num_geo_inputs=cfg.model.num_geo_inputs,
            max_z=cfg.model.max_z,
            num_rbf=cfg.model.get("num_rbf", 20),
            cutoff=cfg.model.get("cutoff", 5.0),
            use_film=cfg.model.get("use_film", True),
        ).to(device)
    elif model_type == "painn_attention":
        from modelling.gnn.delta_painn_attention_model import DeltaPaiNNAttention
        print("--- Using DeltaPaiNNAttention (PaiNN + edge attention + FiLM) ---")
        model = DeltaPaiNNAttention(
            num_layers=cfg.model.num_layers,
            hidden_dim=cfg.model.get("hidden_dim", cfg.model.get("hidden_features", 128)),
            num_cheap_dft_inputs=cfg.model.num_cheap_dft_inputs,
            num_precision_settings=cfg.model.num_precision_settings,
            num_geo_inputs=cfg.model.num_geo_inputs,
            max_z=cfg.model.max_z,
            num_rbf=cfg.model.get("num_rbf", 20),
            cutoff=cfg.model.get("cutoff", 5.0),
            use_film=cfg.model.get("use_film", True),
        ).to(device)
    elif model_type == "attention_film":
        print("--- Using DeltaAttentionGNN + FiLM ---")
        model = DeltaAttentionGNN(
            num_layers=cfg.model.num_layers,
            hidden_features=cfg.model.hidden_features,
            num_cheap_dft_inputs=cfg.model.num_cheap_dft_inputs,
            num_precision_settings=cfg.model.num_precision_settings,
            num_geo_inputs=cfg.model.num_geo_inputs,
            max_z=cfg.model.max_z,
            use_film=True,
        ).to(device)
    elif model_type == "attention":
        print("--- Using DeltaAttentionGNN (attention message passing + pooling) ---")
        model = DeltaAttentionGNN(
            num_layers=cfg.model.num_layers,
            hidden_features=cfg.model.hidden_features,
            num_cheap_dft_inputs=cfg.model.num_cheap_dft_inputs,
            num_precision_settings=cfg.model.num_precision_settings,
            num_geo_inputs=cfg.model.num_geo_inputs,
            max_z=cfg.model.max_z
        ).to(device)
    else:
        use_film = cfg.model.get("use_film", False)
        print(f"--- Using DeltaGNN (with cheap DFT inputs, FiLM={use_film}) ---")
        model = DeltaGNN(
            num_layers=cfg.model.num_layers,
            hidden_features=cfg.model.hidden_features,
            num_cheap_dft_inputs=cfg.model.num_cheap_dft_inputs,
            num_precision_settings=cfg.model.num_precision_settings,
            num_geo_inputs=cfg.model.num_geo_inputs,
            max_z=cfg.model.max_z,
            use_film=use_film,
        ).to(device)
    
    log_model_stats(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)

    warmup_epochs = 10 
    total_epochs = cfg.training.epochs
    
    scheduler_warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
    scheduler_decay = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_decay], milestones=[warmup_epochs])

    metrics_names = [
        # MAE (existing)
        "MAE_delta_energy", "MAE_delta_gap", "MAE_delta_positions",
        "MAE_delta_geo_total",
        "MAE_delta_volume", "MAE_delta_a", "MAE_delta_b", "MAE_delta_c",
        "MAE_delta_alpha", "MAE_delta_beta", "MAE_delta_gamma",
        # sMAPE
        "sMAPE_delta_energy", "sMAPE_delta_gap", "sMAPE_delta_positions",
        "sMAPE_delta_volume",
        # MagAcc
        "MagAcc_delta_energy", "MagAcc_delta_gap", "MagAcc_delta_positions",
        "MagAcc_delta_volume",
        # RMSLAE
        "RMSLAE_delta_energy", "RMSLAE_delta_gap", "RMSLAE_delta_positions",
        "RMSLAE_delta_volume",
    ]
    
    history = {'train_loss': []}
    for m in metrics_names:
        history[f'Train_{m}'] = []
        history[f'Val_{m}'] = []
    history['Train_Loss'] = [] 
    history['Val_Loss'] = []   

    early_stop_metric = cfg.training.get("early_stop_metric", "Val_Loss")
    print(f"Early stopping metric: {early_stop_metric}")

    best_val_loss = float('inf')
    best_val_mae_e = float('inf')
    best_val_smape_e = float('inf')
    best_val_target_smape = float('inf')     

    early_stopper = EarlyStopping(patience=10, min_delta=1e-5)

    print(f"{'Epoch':>5} | {'Step_Loss':>9} | {'Val_Loss':>9} | {'Val_MAE_E':>9} | {'Val_sMAPE_E':>12}")
    print("-" * 155)

    for epoch in range(1, cfg.training.epochs + 1):
        train_step_loss = train_step(model, train_loader, optimizer, criterion, cfg.weights, device)
        history['train_loss'].append(train_step_loss)
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1 or epoch == cfg.training.epochs:
            val_m, val_arrays = evaluate(model, val_loader, criterion, cfg.weights)
            train_m, _ = evaluate(model, train_loader, criterion, cfg.weights)
            
            current_val_loss = val_m['Val_Loss']
            
            history['Val_Loss'].append((epoch, current_val_loss))
            history['Train_Loss'].append((epoch, train_m['Val_Loss']))
            
            for k in metrics_names:
                history[f'Val_{k}'].append((epoch, val_m[k]))
                history[f'Train_{k}'].append((epoch, train_m[k]))

            print(f"{epoch:05d} | {train_step_loss:.4f}    | {current_val_loss:.4f}    | "
                  f"{val_m['MAE_delta_energy']:.4f}      | {val_m['sMAPE_delta_energy']:.2f}%")

            current_lr = optimizer.param_groups[0]['lr']
            
            log_dict = {
                "epoch": epoch,
                f"Train/Loss_{loss_label}": train_step_loss,
                "Train/Learning_Rate": current_lr,
                
                f"Train/Eval_Loss_{loss_label}": train_m['Val_Loss'],
                f"Val/Loss_{loss_label}": current_val_loss,

                # Energy
                "Train/MAE_Energy": train_m['MAE_delta_energy'],
                "Train/sMAPE_Energy": train_m['sMAPE_delta_energy'],
                "Train/MagAcc_Energy": train_m['MagAcc_delta_energy'],
                "Val/MAE_Energy": val_m['MAE_delta_energy'],
                "Val/sMAPE_Energy": val_m['sMAPE_delta_energy'],
                "Val/MagAcc_Energy": val_m['MagAcc_delta_energy'],

                # Band Gap
                "Train/MAE_Gap": train_m['MAE_delta_gap'],
                "Train/sMAPE_Gap": train_m['sMAPE_delta_gap'],
                "Train/MagAcc_Gap": train_m['MagAcc_delta_gap'],
                "Val/MAE_Gap": val_m['MAE_delta_gap'],
                "Val/sMAPE_Gap": val_m['sMAPE_delta_gap'],
                "Val/MagAcc_Gap": val_m['MagAcc_delta_gap'],

                # Positions
                "Train/MAE_Pos": train_m['MAE_delta_positions'],
                "Train/sMAPE_Pos": train_m['sMAPE_delta_positions'],
                "Train/MagAcc_Pos": train_m['MagAcc_delta_positions'],
                "Val/MAE_Pos": val_m['MAE_delta_positions'],
                "Val/sMAPE_Pos": val_m['sMAPE_delta_positions'],
                "Val/MagAcc_Pos": val_m['MagAcc_delta_positions'],
                
                "Val/Geo/Total_MAE": val_m['MAE_delta_geo_total'],
            }
            
            # Geometry Metrics
            geo_vars = ["delta_volume", "delta_a", "delta_b", "delta_c", "delta_alpha", "delta_beta", "delta_gamma"]
            for key in geo_vars:
                # MAE
                log_dict[f"Train/Geo/MAE/{key}"] = train_m[f"MAE_{key}"]
                log_dict[f"Val/Geo/MAE/{key}"] = val_m[f"MAE_{key}"]
                # sMAPE
                log_dict[f"Train/Geo/sMAPE/{key}"] = train_m[f"sMAPE_{key}"]
                log_dict[f"Val/Geo/sMAPE/{key}"] = val_m[f"sMAPE_{key}"]
                # MagAcc
                log_dict[f"Train/Geo/MagAcc/{key}"] = train_m[f"MagAcc_{key}"]
                log_dict[f"Val/Geo/MagAcc/{key}"] = val_m[f"MagAcc_{key}"]

            wandb.log(log_dict, step=epoch)

            # --- Checkpointing ---
            if val_m['MAE_delta_energy'] < best_val_mae_e:
                best_val_mae_e = val_m['MAE_delta_energy']
                torch.save(model.state_dict(), 'best_delta_model.pth')
                wandb.save('best_delta_model.pth')

            if val_m['sMAPE_delta_energy'] < best_val_smape_e:
                best_val_smape_e = val_m['sMAPE_delta_energy']
                torch.save(model.state_dict(), 'best_smape_model.pth')
                wandb.save('best_smape_model.pth')

            if val_m[early_stop_metric] < best_val_target_smape:
                best_val_target_smape = val_m[early_stop_metric]
                torch.save(model.state_dict(), 'best_target_smape_model.pth')
                wandb.save('best_target_smape_model.pth')

            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                torch.save(model.state_dict(), f'best_{loss_type}_loss_model.pth')
            
            torch.save(model.state_dict(), 'last_model.pth')
            wandb.save('last_model.pth')
          
            plot_history(history, save_path="training_progress.png", loss_label=loss_label)
            plot_log_regression(val_arrays, save_path="regression_scatter.png") 
            plot_uncertainty_calibration(val_arrays, save_path="calibration_scatter.png")

            wandb.log({
                "learning_curves": wandb.Image("training_progress.png"),
                "regression_scatter": wandb.Image("regression_scatter.png")
            }, step=epoch)

            save_history(history, filename="training_history.pkl")
            wandb.save("training_history.pkl")
        
            # Early Stopping (Monitors configurable target sMAPE)
            early_stopper(val_m[early_stop_metric])
            
            if early_stopper.early_stop:
                print(f"Early stopping triggered at epoch {epoch}!")
                break

    log_element_embeddings(model)
    wandb.finish()

if __name__ == "__main__":
    main()
