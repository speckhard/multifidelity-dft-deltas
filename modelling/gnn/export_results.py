"""Export GNN results to file."""
import os
import sys
import torch
import numpy as np
import pandas as pd
import pickle
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from absl import app, flags
import wandb

# --- Import your model ---
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from modelling.gnn.delta_egnn_model import DeltaGNN

FLAGS = flags.FLAGS
flags.DEFINE_string('wandb_project', 'egnn-delta-learning', 'Project Name')
flags.DEFINE_string('wandb_group', 'sweep_v2_new_metrics', 'Group Name')
flags.DEFINE_string('data_file', '/u/dansp/egnn/parsed_results/final_egnn_data.pt', 'Path to .pt file')
flags.DEFINE_string('output_file', 'thesis_data_export.pkl', 'Output filename')
flags.DEFINE_string('metric', 'Val/MAE_Energy', 'Metric to pick best model')

def get_best_run(project, group, metric):
    api = wandb.Api()
    path = f"dtts/{project}"
    runs = api.runs(path, filters={"group": group})
    best_run, best_metric = None, float('inf')
    
    print(f"Scanning runs in {path} (Group: {group})...")
    for run in runs:
        if metric in run.summary:
            val = run.summary[metric]
            if val < best_metric:
                best_metric = val
                best_run = run
    
    if best_run:
        print(f"Best Run: {best_run.name} ({metric}={best_metric})")
        return best_run
    raise ValueError("No valid runs found.")

@torch.no_grad()
def run_inference(model, loader, device):
    """Runs model and collects everything into a dict of lists."""
    results = {
        'delta_energy_pred': [], 'delta_energy_true': [],
        'delta_volume_pred': [], 'delta_volume_true': [],
        'spacegroup': []
        # Add other targets here if you need them (gap, a, b, c...)
    }
    
    for data in tqdm(loader, desc="Inference"):
        data = data.to(device)
        _, p_eng, _, p_geo = model(data)
        
        # Energy
        results['delta_energy_pred'].append(p_eng[0].cpu().numpy()) # Mean
        results['delta_energy_true'].append(data.delta_total_energy_per_atom.cpu().numpy())
        
        # Volume (Index 0 of geo head)
        results['delta_volume_pred'].append(p_geo[0][:, 0].cpu().numpy())
        results['delta_volume_true'].append(data.delta_final_volume_per_atom.cpu().numpy())

        # Spacegroup (Handle missing)
        if hasattr(data, 'spacegroup'):
            results['spacegroup'].append(data.spacegroup.cpu().numpy())
        else:
            results['spacegroup'].append(np.zeros(data.num_graphs))

    # Flatten
    flat_data = {}
    for k, v in results.items():
        flat_data[k] = np.concatenate(v).flatten()
        
    return pd.DataFrame(flat_data)

def main(argv):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Get Model
    best_run = get_best_run(FLAGS.wandb_project, FLAGS.wandb_group, FLAGS.metric)
    best_run.file("best_delta_model.pth").download(root='./', replace=True)
    
    # 2. Load Data
    print("Loading Dataset...")
    raw_data = torch.load(FLAGS.data_file, weights_only=False)['graphs']
    splits = {
        'Train': [d for d in raw_data if d.split == 'train'],
        'Val':   [d for d in raw_data if d.split == 'val'],
        'Test':  [d for d in raw_data if d.split == 'test']
    }
    
    # 3. Load Model Structure
    cfg = best_run.config
    model = DeltaGNN(
        num_layers=cfg['model']['num_layers'],
        hidden_features=cfg['model']['hidden_features'],
        num_cheap_dft_inputs=cfg['model']['num_cheap_dft_inputs'],
        num_precision_settings=cfg['model']['num_precision_settings'],
        num_geo_inputs=cfg['model']['num_geo_inputs'],
        max_z=cfg['model']['max_z']
    ).to(device)
    model.load_state_dict(torch.load('best_delta_model.pth', map_location=device, weights_only=False))
    model.eval()

    # 4. Export Data
    export_dict = {}
    
    # A. Element Embeddings (Z=1 to 100)
    print("Exporting Embeddings...")
    export_dict['embeddings'] = model.z_embedding.weight.detach().cpu().numpy()
    
    # B. Inference Results
    for name, data_list in splits.items():
        if len(data_list) == 0: continue
        print(f"Running Inference on {name}...")
        loader = DataLoader(data_list, batch_size=64, shuffle=False)
        df = run_inference(model, loader, device)
        export_dict[f'df_{name}'] = df

    # 5. Save
    print(f"Saving to {FLAGS.output_file}...")
    with open(FLAGS.output_file, 'wb') as f:
        pickle.dump(export_dict, f)
    print("Done! You can now scp this file to your laptop.")

if __name__ == "__main__":
    app.run(main)