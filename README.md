# Multifidelity DFT Deltas

Code accompanying our ICML 2026 workshop submission on uncertainty-aware delta-learning for multifidelity density-functional theory (DFT) calculations. We train graph neural networks (GNNs) and random forests to predict the *delta* between a cheap, low-precision DFT calculation and an expensive, high-precision reference, and quantify epistemic uncertainty over the delta.

> Paper title and arXiv link will be added here once the submission is public.

## Repository layout

```
modelling/
  gnn/          PaiNN, EGNN, and attention variants for delta-learning;
                training pipelines, sweep launchers, evaluation utilities.
  rf/           Random-forest and XGBoost baselines (trainer, configs,
                exporters, DOS-similarity feature pipeline).
feat_eng/       Feature engineering: descriptor construction, valence
                lookups, categorical encoding.
parsing/        Data ingestion from FHI-aims (vibes), exciting, and
                derived delta-dataset construction.
tests/          Pytest suite for each of the above; fixtures under
                tests/data/.
```

## Installation

We recommend a fresh Python 3.10+ environment.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`torch` and `torch-geometric` may need a CUDA-specific install; see the [PyTorch](https://pytorch.org/get-started/locally/) and [PyG](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) docs.

## Running tests

From the repository root:

```bash
pytest tests/
```

## Reproducing key results

The primary training entry point for the delta-GNN models is:

```bash
python -m modelling.gnn.train_pipeline --config-path conf --config-name config
```

Hydra configs for each model variant live in `modelling/gnn/conf/`. Random-forest baselines:

```bash
python modelling/rf/rf_trainer.py
```

## Citation

```bibtex
@inproceedings{speckhard2026multifidelity,
  title = {Uncertainty-aware delta-learning for multifidelity DFT},
  author = {Speckhard, Daniel T. and others},
  booktitle = {ICML 2026 Workshop on \ldots},
  year = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
