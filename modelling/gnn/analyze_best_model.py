import os
import sys
import torch
import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
from absl import app, flags

# --- Import your model ---
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from modelling.gnn.delta_egnn_model import DeltaGNN

# --- FLAGS ---
FLAGS = flags.FLAGS
flags.DEFINE_string('wandb_project', 'egnn-delta-learning', 'WandB Project Name')
flags.DEFINE_string('wandb_group', 'sweep_v2_new_metrics', 'WandB Group to analyze')
flags.DEFINE_string('output_dir', './', 'Directory to save outputs')
flags.DEFINE_string('data_file', '/u/dansp/egnn/parsed_results/final_egnn_data.pt', 'Path to input data')
flags.DEFINE_boolean('force_cpu', False, 'Force CPU usage')
# NEW: Allow selecting by different metrics
flags.DEFINE_string('sort_metric', 'Val/MAE_Energy', 'Metric to select best model (e.g. Val/MAE_Energy, Val/Loss_NLL)')
flags.DEFINE_string('sort_mode', 'min', 'min or max')


class EmbeddingVisualizer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        
    def get_element_data(self):
        """Extracts weights and matches them to periodic table info."""
        # [Max_Z, Hidden_Dim] -> numpy
        weights = self.model.z_embedding.weight.detach().cpu().numpy()
        
        valid_indices = []
        vectors = []
        meta = [] # Stores (Symbol, Group, Period)
        
        # Iterate Z=1 up to max_z (skip 0/padding if unused)
        for z in range(1, weights.shape[0]):
            # Skip if weight is all zeros (likely unused)
            if np.allclose(weights[z], 0):
                continue
                
            try:
                if element:
                    el = element(z)
                    sym = el.symbol
                    grp = el.group_id if el.group_id is not None else -1
                    prd = el.period
                else:
                    # Fallback if mendeleev missing
                    sym = str(z)
                    grp = 0
                    prd = 0
            except:
                continue
                
            valid_indices.append(z)
            vectors.append(weights[z])
            meta.append({'symbol': sym, 'group': grp, 'period': prd, 'z': z})
            
        return np.array(vectors), meta

    def plot(self, save_dir):
        vectors, meta_list = self.get_element_data()
        
        if len(vectors) < 3:
            print("Not enough elements to plot PCA.")
            return

        # 1. PCA Reduction
        pca = PCA(n_components=2)
        coords = pca.fit_transform(vectors) # [N_elements, 2]
        
        # 2. Extract Groups for coloring
        groups = [m['group'] for m in meta_list]
        symbols = [m['symbol'] for m in meta_list]
        
        # 3. Plot
        plt.figure(figsize=(12, 10))
        
        # Use a discrete colormap (tab20 is good for distinct groups)
        scatter = plt.scatter(coords[:, 0], coords[:, 1], 
                              c=groups, cmap='tab20', s=100, edgecolors='k', alpha=0.8)
        
        # Annotate
        for i, txt in enumerate(symbols):
            plt.annotate(txt, (coords[i, 0], coords[i, 1]), 
                         xytext=(5, 5), textcoords='offset points', fontsize=9)
            
        plt.title(f"Learned Atom Embeddings (PCA) - Colored by Group\nExpl. Var: {pca.explained_variance_ratio_.sum():.2f}")
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2f})")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2f})")
        plt.colorbar(scatter, label='Group Number')
        plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(save_dir, "atom_embeddings_pca.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Saved Embedding Plot to: {save_path}")



class ModelAnalyzer:
    def __init__(self, project_name, group_name, output_dir):
        self.api = wandb.Api()
        self.project = f"dtts/{project_name}"
        self.group = group_name
        self.output_dir = output_dir
        
    def get_best_run(self, metric, mode):
        print(f"Scanning runs in {self.project} (Group: {self.group})...")
        runs = self.api.runs(self.project, filters={"group": self.group})
        best_run = None
        # Initialize best_metric to infinity (for min) or -infinity (for max)
        best_metric = float('inf') if mode == "min" else float('-inf')
        
        found_any = False
        for run in runs:
            # Skip crashes
            if metric not in run.summary: continue
            
            val = run.summary[metric]
            found_any = True
            
            if mode == "min":
                if val < best_metric:
                    best_metric = val
                    best_run = run
            else:
                if val > best_metric:
                    best_metric = val
                    best_run = run
                    
        if best_run:
            print(f"Found Best Run: {best_run.name} ({best_run.id})")
            print(f"Selected by {metric}: {best_metric}")
            return best_run
        
        if not found_any:
            print(f"WARNING: No runs found with metric '{metric}'. Check exact spelling in WandB!")
        raise ValueError(f"No valid runs found in group {self.group}")

    def download_checkpoint(self, run, filename="best_delta_model.pth"):
        files = [f.name for f in run.files()]
        if filename not in files:
            print(f"Warning: {filename} not found. Available: {files}")
            return None
        print(f"Downloading {filename} to {self.output_dir}...")
        run.file(filename).download(root=self.output_dir, replace=True)
        return os.path.join(self.output_dir, filename)

    def load_model(self, checkpoint_path, config, device):
        model = DeltaGNN(
            num_layers=config['model']['num_layers'],
            hidden_features=config['model']['hidden_features'],
            num_cheap_dft_inputs=config['model']['num_cheap_dft_inputs'],
            num_precision_settings=config['model']['num_precision_settings'],
            num_geo_inputs=config['model']['num_geo_inputs'],
            max_z=config['model']['max_z']
        ).to(device)
        # Weights_only=False for older PyG compatibility
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False))
        model.eval()
        return model

class Evaluator:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def calculate_mag_acc_batch(self, pred, target):
        """Helper to calc mag acc on tensors before moving to numpy"""
        boundaries = torch.tensor([1e-3, 1e-2, 1e-1, 1.0, 10.0], device=self.device)
        abs_pred = torch.abs(pred)
        abs_target = torch.abs(target)
        pred_bins = torch.bucketize(abs_pred, boundaries)
        target_bins = torch.bucketize(abs_target, boundaries)
        correct = (pred_bins == target_bins).float()
        return correct # Returns tensor [Batch]

    @torch.no_grad()
    def predict_dataset(self, loader):
        # 1. Define all targets we want to track
        keys = [
            'delta_energy', 'delta_gap', 'delta_positions', 
            'delta_volume', 'delta_a', 'delta_b', 'delta_c', 
            'delta_alpha', 'delta_beta', 'delta_gamma'
        ]
        
        results = {k: {'pred': [], 'true': [], 'sigma': [], 'mag_acc': []} for k in keys}
        
        for data in tqdm(loader, desc="Inference"):
            data = data.to(self.device)
            p_pos, p_eng, p_gap, p_geo = self.model(data)
            
            # Unpack Tuples (Mean, LogVar)
            mu_eng, logvar_eng = p_eng
            mu_gap, logvar_gap = p_gap
            mu_geo, logvar_geo = p_geo   # [B, 7]
            mu_pos, logvar_pos = p_pos   # [N, 3] -> Norm -> [N]
            
            # --- Energy ---
            results['delta_energy']['pred'].append(mu_eng.cpu().numpy())
            results['delta_energy']['true'].append(data.delta_total_energy_per_atom.cpu().numpy())
            results['delta_energy']['sigma'].append(torch.exp(0.5 * logvar_eng).cpu().numpy())
            results['delta_energy']['mag_acc'].append(
                self.calculate_mag_acc_batch(mu_eng, data.delta_total_energy_per_atom).cpu().numpy()
            )

            # --- Gap ---
            results['delta_gap']['pred'].append(mu_gap.cpu().numpy())
            results['delta_gap']['true'].append(data.delta_homo_lumo_gap.cpu().numpy())
            results['delta_gap']['sigma'].append(torch.exp(0.5 * logvar_gap).cpu().numpy())
            results['delta_gap']['mag_acc'].append(
                self.calculate_mag_acc_batch(mu_gap, data.delta_homo_lumo_gap).cpu().numpy()
            )
            
            # --- Positions (Norm) ---
            # We compare Magnitude of prediction vs Magnitude of target vector
            p_pos_norm = torch.norm(mu_pos, dim=1)
            t_pos_norm = torch.norm(data.delta_relaxed_atom_positions, dim=1)
            s_pos = torch.exp(0.5 * logvar_pos).squeeze() # Simplify assuming isotropic sigma for now

            results['delta_positions']['pred'].append(p_pos_norm.cpu().numpy())
            results['delta_positions']['true'].append(t_pos_norm.cpu().numpy())
            results['delta_positions']['sigma'].append(s_pos.cpu().numpy())
            results['delta_positions']['mag_acc'].append(
                self.calculate_mag_acc_batch(p_pos_norm, t_pos_norm).cpu().numpy()
            )

            # --- Geometry Components ---
            # Indices: 0:Vol, 1:a, 2:b, 3:c, 4:alpha, 5:beta, 6:gamma
            # Targets: delta_final_volume_per_atom (1) + delta_lattice_params (6)
            
            # Reconstruct full target vector for easy indexing
            t_vol = data.delta_final_volume_per_atom.view(-1, 1)
            t_lat = data.delta_lattice_params.view(-1, 6)
            t_geo = torch.cat([t_vol, t_lat], dim=1) # [B, 7]
            
            geo_names = ['delta_volume', 'delta_a', 'delta_b', 'delta_c', 'delta_alpha', 'delta_beta', 'delta_gamma']
            
            for i, name in enumerate(geo_names):
                p = mu_geo[:, i]
                t = t_geo[:, i]
                s = torch.exp(0.5 * logvar_geo[:, i])
                
                results[name]['pred'].append(p.cpu().numpy())
                results[name]['true'].append(t.cpu().numpy())
                results[name]['sigma'].append(s.cpu().numpy())
                results[name]['mag_acc'].append(
                    self.calculate_mag_acc_batch(p, t).cpu().numpy()
                )

        # Flatten all
        for k in results:
            for sub_k in results[k]:
                results[k][sub_k] = np.concatenate(results[k][sub_k]).flatten()
                
        return results

def calculate_metrics(results_dict):
    metrics = {}
    epsilon = 1e-7
    
    for target, data in results_dict.items():
        y_true = data['true']
        y_pred = data['pred']
        mag_acc = data['mag_acc']
        
        # 1. Basic MAE
        mae = mean_absolute_error(y_true, y_pred)
        
        # 2. sMAPE
        numerator = np.abs(y_pred - y_true)
        denominator = np.abs(y_pred) + np.abs(y_true) + epsilon
        smape = 100.0 * 2.0 * np.mean(numerator / denominator)
        
        # 3. Magnitude Accuracy
        acc = np.mean(mag_acc)
        
        # 4. Context Stats
        t_mean = np.mean(y_true)
        t_std = np.std(y_true)
        
        metrics[f"{target}_MAE"] = mae
        metrics[f"{target}_sMAPE"] = smape
        metrics[f"{target}_MagAcc"] = acc
        metrics[f"{target}_TrueMean"] = t_mean
        metrics[f"{target}_TrueStd"] = t_std
        
    return metrics

def plot_regression_and_calib(results_dict, split_name, save_dir):
    """Generates plots for key metrics"""
    # Key metrics to plot
    plot_keys = ['delta_energy', 'delta_gap', 'delta_positions', 'delta_volume']
    titles = ['Energy', 'Gap', 'Positions', 'Volume']
    
    # 1. Regression (Log-Log)
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    epsilon = 1e-7
    
    for ax, key, title in zip(axes, plot_keys, titles):
        d = results_dict[key]
        y_t, y_p = np.abs(d['true'])+epsilon, np.abs(d['pred'])+epsilon
        
        ax.scatter(np.log10(y_t), np.log10(y_p), alpha=0.1, s=2, c='navy')
        low, high = min(np.log10(y_t).min(), np.log10(y_p).min()), max(np.log10(y_t).max(), np.log10(y_p).max())
        ax.plot([low, high], [low, high], 'r--', alpha=0.8)
        ax.set_title(f"{title} - {split_name}")
        ax.set_xlabel("Log10 |Target|")
        ax.set_ylabel("Log10 |Pred|")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{split_name}_regression.png"), dpi=150)
    plt.close()

    # 2. Calibration
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    for ax, key, title in zip(axes, plot_keys, titles):
        d = results_dict[key]
        sigma = d['sigma'] + epsilon
        error = np.abs(d['true'] - d['pred']) + epsilon
        
        ax.scatter(np.log10(sigma), np.log10(error), alpha=0.1, s=2, c='teal')
        low, high = min(np.log10(sigma).min(), np.log10(error).min()), max(np.log10(sigma).max(), np.log10(error).max())
        ax.plot([low, high], [low, high], 'r--', alpha=0.8)
        ax.set_title(f"{title} Calib - {split_name}")
        ax.set_xlabel("Log10 Sigma")
        ax.set_ylabel("Log10 |Error|")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{split_name}_calibration.png"), dpi=150)
    plt.close()

def main(argv):
    # 1. Setup
    device = "cpu" if FLAGS.force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Analysis Worker Started ---")
    print(f"Output Dir: {FLAGS.output_dir}")
    print(f"Selection Metric: {FLAGS.sort_metric} ({FLAGS.sort_mode})")
    
    # 2. WandB Interaction
    analyzer = ModelAnalyzer(FLAGS.wandb_project, FLAGS.wandb_group, FLAGS.output_dir)
    # Pass the metric flag here
    best_run = analyzer.get_best_run(metric=FLAGS.sort_metric, mode=FLAGS.sort_mode)
    
    ckpt_path = analyzer.download_checkpoint(best_run)
    
    # 3. Data Loading
    print(f"Loading data: {FLAGS.data_file}")
    # weights_only=False fix for PyG Data objects
    all_data = torch.load(FLAGS.data_file, weights_only=False)['graphs']
    
    loaders = {
        'Train': DataLoader([d for d in all_data if d.split == 'train'], batch_size=64, shuffle=False),
        'Val': DataLoader([d for d in all_data if d.split == 'val'], batch_size=64, shuffle=False),
        'Test': DataLoader([d for d in all_data if d.split == 'test'], batch_size=64, shuffle=False)
    }

    # 4. Inference
    model = analyzer.load_model(ckpt_path, best_run.config, device)
    evaluator = Evaluator(model, device)
    
    all_metrics = []
    
    for split_name, loader in loaders.items():
        if len(loader) == 0:
            print(f"Skipping {split_name} (empty)")
            continue
            
        print(f"Evaluating {split_name}...")
        res = evaluator.predict_dataset(loader)
        
        # Calculate Metrics
        met = calculate_metrics(res)
        met['Split'] = split_name
        all_metrics.append(met)
        
        # Plots
        plot_regression_and_calib(res, split_name, FLAGS.output_dir)

    # 5. Saving Results
    print(f"\n--- Compiling Final CSV ---")
    df = pd.DataFrame(all_metrics)

    # --- 6. Visualize Embeddings (NEW) ---
    print("\n--- Visualizing Learned Embeddings ---")
    viz = EmbeddingVisualizer(model, device)
    viz.plot(FLAGS.output_dir)

    # 1. Save FULL file with all columns (keep this safe)
    # Reorder columns: Split first, then alphabetical
    all_cols = ['Split'] + sorted([c for c in df.columns if c != 'Split'])
    df_full = df[all_cols]
    
    csv_path = os.path.join(FLAGS.output_dir, "final_metrics.csv")
    df_full.to_csv(csv_path, index=False)
    
    # 2. Print PREVIEW of specific targets only
    # We construct a list of exactly what you want to see
    preview_targets = ['delta_energy', 'delta_gap', 'delta_volume']
    preview_metrics = ['MAE', 'sMAPE', 'MagAcc']
    
    preview_cols = ['Split']
    for t in preview_targets:
        for m in preview_metrics:
            key = f"{t}_{m}"
            if key in df.columns:
                preview_cols.append(key)

    print("\nPreview (Energy, Gap, Volume):")
    # Using to_string() ensures it prints the whole table without hiding columns
    print(df[preview_cols].to_string(index=False)) 
    
    print(f"\nSaved Full Metrics (including positions/lattice) to: {csv_path}")

if __name__ == "__main__":
    app.run(main)
