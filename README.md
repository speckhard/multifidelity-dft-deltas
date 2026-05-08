# Multifidelity DFT Deltas

Code for an ICML 2026 workshop submission on **uncertainty-aware delta-learning across DFT fidelities**. We train graph neural networks (PaiNN, EGNN, and attention variants) and tree-based baselines (random forest, XGBoost) to predict the *delta* between a cheap, low-precision density-functional-theory calculation and an expensive, high-precision reference, and we quantify epistemic uncertainty over that delta via deep ensembles.

The goal is a model that, given a cheap DFT result and its precision settings, returns both the corrected high-precision target and a calibrated error bar — letting practitioners decide whether the cheap calculation is trustworthy or whether the expensive reference is needed.

## Repository layout

| Path | Contents |
|---|---|
| `modelling/gnn/` | PaiNN, EGNN, and attention-augmented variants; Hydra configs in `conf/`; training pipeline (`train_pipeline.py`), denoising / surrogate variants, sweep launchers, evaluation utilities. |
| `modelling/rf/` | Random-forest and XGBoost baselines: `rf_trainer.py`, dataset configs (`rf_config*.py`), DOS-similarity feature pipeline, retrain / export scripts. |
| `feat_eng/` | Descriptor construction, valence-electron lookups, categorical encoding. |
| `parsing/` | Ingestion from FHI-aims (via FHI-vibes) and exciting; delta-dataset construction; DOS-fingerprint similarity. |
| `tests/` | Pytest suite for each of the above; small fixtures under `tests/data/`. |

## Installation

Python 3.10+ is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`torch` and `torch-geometric` may need a CUDA-specific install — see the [PyTorch](https://pytorch.org/get-started/locally/) and [PyG](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) instructions for your platform.

## Running tests

From the repository root:

```bash
pytest tests/
```

The model and pipeline tests are self-contained (they construct synthetic ASE databases via `tmp_path`). The parser tests exercise small DFT-output trees bundled under `tests/data/`.

## Training

GNN models are configured with [Hydra](https://hydra.cc/). Configs live in `modelling/gnn/conf/` (one per architecture variant); the data path and architecture/training hyperparameters are all overridable from the command line.

```bash
# Train the delta-EGNN with the default config:
python -m modelling.gnn.train_pipeline data_file=/path/to/delta_dataset.pt

# Switch to PaiNN with a different sweep tag:
python -m modelling.gnn.train_pipeline \
    --config-name painn_attention_config \
    data_file=/path/to/delta_dataset.pt \
    +wandb_group=my_sweep
```

Random-forest baselines use absl flags:

```bash
python modelling/rf/rf_trainer.py \
    --data_file=/path/to/delta_dataset.csv \
    --output_dir=runs/rf \
    --target_key=energy \
    --metric_key=mae
```

## Datasets

The training datasets (delta DFT calculations from FHI-aims and exciting sweeps over precision settings, basis-set tiers, and k-point densities) are not redistributed in this repo due to size. The `parsing/` module documents how the delta dataset is constructed from raw DFT output trees, and `parsing/output_parser_vibes.py` / `parsing/output_parser_exciting_geo_opt.py` show the fields extracted per calculation.

Reach out to the authors for access to the curated `.pt` and `.csv` artifacts used in the paper experiments.

## License

MIT — see [LICENSE](LICENSE).
