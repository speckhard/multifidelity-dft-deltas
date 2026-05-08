#!/usr/bin/env python3.12
"""Unified evaluation of ALL GNN checkpoints + EGNN CSVs.

Auto-discovers all completed runs. Evaluates both best_delta_model.pth (MAE-best)
and best_smape_model.pth (sMAPE energy-best) on train/val/test.

Computes both epsilon variants for sMAPE and RMSLAE:
  - sMAPE_1e-4, sMAPE_1e-7, RMSLAE_1e-4, RMSLAE_1e-7
Plus MAE and MagAcc (epsilon-independent).

Usage:
    python unified_eval.py --group st_painn         # single-target painn only
    python unified_eval.py --group st_egnn           # single-target egnn only
    python unified_eval.py --group st_egnn_film      # single-target egnn_film only
    python unified_eval.py --group st_attention       # single-target attention only
    python unified_eval.py --group mt_painn          # multi-target DeltaPaiNN
    python unified_eval.py --group mt_egnn           # multi-target DeltaEGNN (CSVs)
    python unified_eval.py                            # everything (slow)
"""

import argparse
import csv
import re
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error as mae_score

# --- Project imports ---
ROOT = Path("/u/dansp/egnn/errorbar_modelling")
sys.path.insert(0, str(ROOT))

from modelling.gnn.delta_egnn_model import DeltaGNN
from modelling.gnn.delta_egnn_attention_model import DeltaAttentionGNN

# ── Constants ──

EPS_VARIANTS = [1e-4, 1e-7]

DATA_FILE = Path(
    "/u/dansp/egnn/relaxation_data_with_kpoints/data/"
    "delta_combined_relaxations_23_51__25_1_2026_egnn_data.pt"
)
SWEEP_DIR = Path("/u/dansp/egnn/delta_painn/single_target_sweep_2026_03_10")
EGNN_CSV_DIR = Path(
    "/u/dansp/egnn/egnn_sweep_results/comparison_runs/comparison_15_02_2026_21_46"
)
MT_PAINN_ABLATION_DIR = Path("/u/dansp/egnn/delta_painn/ablation/sweep_feb_25_2026")
MT_PAINN_LOSS_DIR = Path("/u/dansp/egnn/delta_painn/loss_sweep")

CKPT_MAE = "best_delta_model.pth"
CKPT_SMAPE = "best_smape_model.pth"

ST_MODELS = ["painn", "egnn", "egnn_film", "attention"]
ST_TARGETS = ["energy", "bandgap", "volume", "lattice", "all_geo"]
ST_LOSSES = ["smape", "asinh_l1", "rmslae"]

MODEL_TYPE_MAP = {
    "painn": "painn",
    "egnn": "delta",
    "egnn_film": "delta",
    "attention": "attention",
}

MULTI_EGNN_CSVS = OrderedDict([
    ("MT_egnn_nll_top1", "egnn_nll_top1"),
    ("MT_egnn_nll_ens5", "egnn_nll_ensemble5"),
    ("MT_egnn_smape_top1", "egnn_smape_top1"),
    ("MT_egnn_smape_ens5", "egnn_smape_ensemble5"),
    ("MT_egnn_attention_top1", "egnn_attention_top1"),
    ("MT_egnn_attention_ens5", "egnn_attention_ensemble5"),
    ("MT_egnn_no_cheap_dft_top1", "egnn_no_cheap_dft_top1"),
    ("MT_egnn_no_cheap_dft_ens5", "egnn_no_cheap_dft_ensemble5"),
    ("MT_egnn_no_lattice_top1", "egnn_no_lattice_top1"),
    ("MT_egnn_no_lattice_ens5", "egnn_no_lattice_ensemble5"),
    ("MT_egnn_no_lattice_smape_top1", "egnn_no_lattice_smape_top1"),
    ("MT_egnn_no_lattice_smape_ens5", "egnn_no_lattice_smape_ensemble5"),
])

# ── Metric functions ──


def calculate_smape_np(pred, target, epsilon):
    num = np.abs(pred - target)
    den = np.abs(pred) + np.abs(target) + epsilon
    return float(np.mean(100.0 * 2.0 * num / den))


def calculate_rmslae_np(pred, target, epsilon):
    log_pred = np.log(np.abs(pred) + epsilon)
    log_targ = np.log(np.abs(target) + epsilon)
    return float(np.sqrt(np.mean((log_pred - log_targ) ** 2)))


def calculate_magacc_np(pred, target):
    boundaries = np.array([1e-3, 1e-2, 1e-1, 1.0, 10.0])
    pred_bins = np.digitize(np.abs(pred), boundaries)
    targ_bins = np.digitize(np.abs(target), boundaries)
    return float(np.mean(pred_bins == targ_bins))


def compute_all_metrics(pred, target):
    m = {
        "MAE": float(mae_score(target, pred)),
        "MagAcc": calculate_magacc_np(pred, target),
    }
    for eps in EPS_VARIANTS:
        tag = f"1e{int(np.log10(eps))}"
        m[f"sMAPE_{tag}"] = calculate_smape_np(pred, target, eps)
        m[f"RMSLAE_{tag}"] = calculate_rmslae_np(pred, target, eps)
    return m


# ── Discovery ──


def discover_single_target_runs(sweep_dir, model_filter=None):
    """Auto-discover all completed single-target sweep runs."""
    models = [model_filter] if model_filter else ST_MODELS
    runs = []
    for model_name in models:
        model_type = MODEL_TYPE_MAP[model_name]
        for target in ST_TARGETS:
            for loss in ST_LOSSES:
                loss_dir = sweep_dir / model_name / target / loss
                if not loss_dir.is_dir():
                    continue
                for hp_dir in sorted(loss_dir.iterdir()):
                    if not hp_dir.is_dir():
                        continue
                    m = re.match(r"lr([\d.e-]+)_h(\d+)", hp_dir.name)
                    if not m:
                        continue
                    lr, hdim = m.group(1), m.group(2)
                    ckpts = []
                    for tag, fname in [("mae", CKPT_MAE), ("smape", CKPT_SMAPE)]:
                        if (hp_dir / fname).exists():
                            ckpts.append((tag, fname))
                    if not ckpts:
                        continue
                    runs.append({
                        "label": f"ST_{model_name}_{target}_{loss}_lr{lr}_h{hdim}",
                        "model_type": model_type,
                        "model_name": model_name,
                        "hdim": hdim,
                        "use_film": model_name in ("painn", "egnn_film"),
                        "run_dir": hp_dir,
                        "checkpoints": ckpts,
                    })
    return runs


def discover_mt_painn_runs():
    """Auto-discover all multi-target DeltaPaiNN runs (ablation + loss sweep)."""
    runs = []

    # Ablation: ablation_{film|nofilm}_{denoise|nodenoise}_s{seed}
    if MT_PAINN_ABLATION_DIR.is_dir():
        for d in sorted(MT_PAINN_ABLATION_DIR.iterdir()):
            if not d.is_dir() or not d.name.startswith("ablation_"):
                continue
            ckpts = []
            for tag, fname in [("mae", CKPT_MAE), ("smape", CKPT_SMAPE)]:
                if (d / fname).exists():
                    ckpts.append((tag, fname))
            if not ckpts:
                continue
            use_film = "film_" in d.name and "nofilm" not in d.name
            runs.append({
                "label": f"MT_painn_ablation_{d.name.replace('ablation_', '')}",
                "run_dir": d,
                "use_film": use_film,
                "checkpoints": ckpts,
            })

    # Loss sweep: {loss}/seed_{seed}
    if MT_PAINN_LOSS_DIR.is_dir():
        for loss_dir in sorted(MT_PAINN_LOSS_DIR.iterdir()):
            if not loss_dir.is_dir():
                continue
            loss_name = loss_dir.name
            for seed_dir in sorted(loss_dir.iterdir()):
                if not seed_dir.is_dir() or "wandb" in seed_dir.name:
                    continue
                ckpts = []
                for tag, fname in [("mae", CKPT_MAE), ("smape", CKPT_SMAPE)]:
                    if (seed_dir / fname).exists():
                        ckpts.append((tag, fname))
                if not ckpts:
                    continue
                runs.append({
                    "label": f"MT_painn_loss_{loss_name}_{seed_dir.name}",
                    "run_dir": seed_dir,
                    "use_film": True,
                    "checkpoints": ckpts,
                })

    return runs


# ── Model construction ──


def build_model(model_type, model_name, hdim, num_layers=3, use_film=False, device="cpu"):
    hdim = int(hdim)
    common = dict(
        num_layers=int(num_layers),
        num_cheap_dft_inputs=12,
        num_precision_settings=12,
        num_geo_inputs=7,
        max_z=120,
    )
    if model_type == "painn":
        from modelling.gnn.delta_painn_model import DeltaPaiNN
        return DeltaPaiNN(
            hidden_dim=hdim, num_rbf=20, cutoff=5.0,
            use_film=use_film, **common,
        ).to(device)
    elif model_type == "attention":
        return DeltaAttentionGNN(hidden_features=hdim, **common).to(device)
    else:
        return DeltaGNN(hidden_features=hdim, use_film=use_film, **common).to(device)


# ── Inference ──


def run_inference(model, loader, device):
    """Run model on loader, return dict of arrays {target: {pred, targ, sigma}}."""
    model.eval()
    geo_keys = ["delta_volume", "delta_a", "delta_b", "delta_c",
                "delta_alpha", "delta_beta", "delta_gamma"]
    arrays = {k: {"pred": [], "targ": [], "sigma": []}
              for k in ["delta_energy", "delta_gap"] + geo_keys}
    arrays["delta_positions"] = {"pred": [], "targ": []}

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            target_vol = data.delta_final_volume_per_atom.view(-1, 1)
            target_lat = data.delta_lattice_params.view(-1, 6)
            target_geo = torch.cat([target_vol, target_lat], dim=1)

            pred_pos, pred_eng, pred_gap, pred_geo = model(data)
            eng_mean, eng_logvar = pred_eng
            gap_mean, gap_logvar = pred_gap
            pos_mean, pos_logvar = pred_pos
            geo_mean, geo_logvar = pred_geo

            arrays["delta_energy"]["pred"].append(eng_mean.cpu().numpy())
            arrays["delta_energy"]["targ"].append(data.delta_total_energy_per_atom.cpu().numpy())
            arrays["delta_energy"]["sigma"].append(torch.exp(0.5 * eng_logvar).cpu().numpy())
            arrays["delta_gap"]["pred"].append(gap_mean.cpu().numpy())
            arrays["delta_gap"]["targ"].append(data.delta_homo_lumo_gap.cpu().numpy())
            arrays["delta_gap"]["sigma"].append(torch.exp(0.5 * gap_logvar).cpu().numpy())

            p_norm = torch.norm(pos_mean, dim=1).cpu().numpy()
            t_norm = torch.norm(data.delta_relaxed_atom_positions, dim=1).cpu().numpy()
            arrays["delta_positions"]["pred"].append(p_norm)
            arrays["delta_positions"]["targ"].append(t_norm)

            for i, key in enumerate(geo_keys):
                arrays[key]["pred"].append(geo_mean[:, i].cpu().numpy())
                arrays[key]["targ"].append(target_geo[:, i].cpu().numpy())
                arrays[key]["sigma"].append(torch.exp(0.5 * geo_logvar[:, i]).cpu().numpy())

    for k in arrays:
        for field in arrays[k]:
            arrays[k][field] = np.concatenate(arrays[k][field])
    return arrays


def arrays_to_metrics(arrays):
    metrics = {}
    mapping = {
        "energy": "delta_energy", "gap": "delta_gap", "volume": "delta_volume",
        "a": "delta_a", "b": "delta_b", "c": "delta_c",
        "alpha": "delta_alpha", "beta": "delta_beta", "gamma": "delta_gamma",
    }
    for short, key in mapping.items():
        if key not in arrays or len(arrays[key]["pred"]) == 0:
            continue
        m = compute_all_metrics(arrays[key]["pred"], arrays[key]["targ"])
        for metric_name, val in m.items():
            metrics[f"{metric_name}_{short}"] = val

    geo_keys = ["delta_volume", "delta_a", "delta_b", "delta_c",
                "delta_alpha", "delta_beta", "delta_gamma"]
    total_abs = 0.0
    n = 0
    for gk in geo_keys:
        if gk in arrays and len(arrays[gk]["pred"]) > 0:
            total_abs += np.sum(np.abs(arrays[gk]["pred"] - arrays[gk]["targ"]))
            n = len(arrays[gk]["pred"])
    if n > 0:
        metrics["MAE_geo_total"] = total_abs / n
    return metrics


def save_predictions_csv(arrays, path):
    n = len(arrays["delta_energy"]["pred"])
    rows = []
    for i in range(n):
        row = {}
        for key in arrays:
            if key == "delta_positions":
                continue
            short = key.replace("delta_", "")
            row[f"{short}_pred"] = arrays[key]["pred"][i]
            row[f"{short}_true"] = arrays[key]["targ"][i]
            if "sigma" in arrays[key] and len(arrays[key]["sigma"]) > 0:
                row[f"{short}_sigma"] = arrays[key]["sigma"][i]
        rows.append(row)
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def load_egnn_csv(csv_path):
    data = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for col in row:
                data.setdefault(col, []).append(float(row[col]))
    for col in data:
        data[col] = np.array(data[col])

    arrays = {}
    for key in ["delta_energy", "delta_gap", "delta_volume", "delta_a", "delta_b",
                "delta_c", "delta_alpha", "delta_beta", "delta_gamma"]:
        pred_col = f"{key}_pred"
        true_col = f"{key}_true"
        if pred_col not in data:
            continue
        entry = {"pred": data[pred_col], "targ": data[true_col]}
        sigma_col = f"{key}_sigma"
        entry["sigma"] = data.get(sigma_col, np.array([]))
        if isinstance(entry["sigma"], list):
            entry["sigma"] = np.array(entry["sigma"])
        arrays[key] = entry
    return arrays


def _write_summary(rows, path):
    if not rows:
        print("No results to write.")
        return
    all_cols = set()
    for r in rows:
        all_cols.update(r.keys())
    fixed = ["model", "checkpoint", "split"]
    metric_cols = sorted(all_cols - set(fixed))
    fieldnames = fixed + metric_cols
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"\nSummary saved to {path} ({len(rows)} rows)")


# ── Evaluation loops ──


def eval_egnn_csvs(out_dir, pred_dir):
    """Part 1: Multi-target EGNN from existing CSVs. No GPU needed."""
    rows = []
    print("=" * 60)
    print("Multi-target EGNN baselines (from CSVs)")
    print("=" * 60)
    for label, basename in MULTI_EGNN_CSVS.items():
        for split in ["train", "val", "test"]:
            csv_path = EGNN_CSV_DIR / f"{basename}_{split}.csv"
            if not csv_path.exists():
                print(f"  SKIP {label}/{split}")
                continue
            print(f"  {label} / {split} ... ", end="", flush=True)
            arrays = load_egnn_csv(csv_path)
            metrics = arrays_to_metrics(arrays)
            metrics["model"] = label
            metrics["checkpoint"] = "mae"
            metrics["split"] = split
            rows.append(metrics)
            save_predictions_csv(arrays, pred_dir / f"{label}__ckpt_mae_{split}.csv")
            print("done")
    return rows


def eval_checkpoint_runs(runs, splits, device, pred_dir, group_name):
    """Evaluate a list of checkpoint-based runs across all data splits."""
    rows = []
    print(f"\n{'=' * 60}")
    print(f"{group_name} ({len(runs)} models)")
    print("=" * 60)

    for i, run in enumerate(runs):
        label = run["label"]
        for ckpt_tag, ckpt_file in run["checkpoints"]:
            ckpt_path = run["run_dir"] / ckpt_file
            if not ckpt_path.exists():
                continue

            print(f"  [{i+1}/{len(runs)}] {label} / ckpt_{ckpt_tag}")

            # Load state dict first to detect FiLM from checkpoint keys
            state = torch.load(ckpt_path, map_location=device, weights_only=True)
            ckpt_has_film = any("film" in k for k in state.keys())

            # Build model
            if "model_type" in run:
                model = build_model(
                    run["model_type"], run.get("model_name", ""),
                    run["hdim"], device=device,
                    use_film=ckpt_has_film,
                )
            else:
                # Multi-target PaiNN
                model = build_model(
                    "painn", "painn", 128, num_layers=3,
                    use_film=ckpt_has_film, device=device,
                )

            model.load_state_dict(state)

            for split_name, loader in splits.items():
                print(f"    {split_name} ... ", end="", flush=True)
                arrays = run_inference(model, loader, device)
                metrics = arrays_to_metrics(arrays)
                metrics["model"] = label
                metrics["checkpoint"] = ckpt_tag
                metrics["split"] = split_name
                rows.append(metrics)
                save_predictions_csv(
                    arrays, pred_dir / f"{label}__ckpt_{ckpt_tag}_{split_name}.csv")
                print("done")

            del model
            torch.cuda.empty_cache()

    return rows


# ── Main ──


def main():
    parser = argparse.ArgumentParser(description="Unified GNN checkpoint evaluation")
    parser.add_argument("--group", type=str, default=None,
                        help="Eval group: st_painn, st_egnn, st_egnn_film, st_attention, "
                             "mt_painn, mt_egnn, or omit for all")
    parser.add_argument("--output_dir", type=str,
                        default="/u/dansp/egnn/delta_painn/single_target_sweep_2026_03_10/unified_eval",
                        help="Directory for output CSVs")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)

    group = args.group
    summary_rows = []

    # ── EGNN CSVs (no GPU) ──
    if group is None or group == "mt_egnn":
        summary_rows.extend(eval_egnn_csvs(out_dir, pred_dir))

    # ── GPU-based evaluations ──
    needs_gpu = group is None or group.startswith("st_") or group == "mt_painn"
    splits = None

    if needs_gpu:
        print(f"\nLoading data from {DATA_FILE} ...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")

        from torch_geometric.loader import DataLoader
        loaded = torch.load(DATA_FILE, weights_only=False)
        all_graphs = loaded["graphs"]
        splits = OrderedDict()
        for split_name in ["train", "val", "test"]:
            graphs = [g for g in all_graphs if g.split == split_name]
            splits[split_name] = DataLoader(graphs, batch_size=32, shuffle=False)
            print(f"  {split_name}: {len(graphs)} graphs")

    # ── Single-target sweep ──
    st_groups = {
        "st_painn": "painn", "st_egnn": "egnn",
        "st_egnn_film": "egnn_film", "st_attention": "attention",
    }

    if group is None:
        # All single-target
        runs = discover_single_target_runs(SWEEP_DIR)
        summary_rows.extend(eval_checkpoint_runs(
            runs, splits, device, pred_dir, "Single-target sweep (all)"))
    elif group in st_groups:
        model_filter = st_groups[group]
        runs = discover_single_target_runs(SWEEP_DIR, model_filter=model_filter)
        summary_rows.extend(eval_checkpoint_runs(
            runs, splits, device, pred_dir, f"Single-target {model_filter}"))

    # ── Multi-target PaiNN ──
    if group is None or group == "mt_painn":
        mt_runs = discover_mt_painn_runs()
        summary_rows.extend(eval_checkpoint_runs(
            mt_runs, splits, device, pred_dir, "Multi-target DeltaPaiNN"))

    # ── Write summary ──
    suffix = f"_{group}" if group else ""
    summary_path = out_dir / f"summary_metrics{suffix}.csv"
    _write_summary(summary_rows, summary_path)

    print(f"\nDone. Results in {out_dir}/")
    print(f"  {summary_path.name}  — {len(summary_rows)} rows")
    print(f"  predictions/  — per-sample CSVs")


if __name__ == "__main__":
    main()
