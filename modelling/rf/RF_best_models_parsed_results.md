## RF Sweep Results — All 15 jobs COMPLETED successfully

### Single-target models (best results per target)

| Target | Metric | Test MAE | Test sMAPE | Test MagAcc |
|--------|--------|----------|------------|-------------|
| **energy** | all 3 identical | 9.14 | **9.70%** | **95.8%** |
| **bandgap** | all 3 identical | **0.053** | 30.5% | 88.4% |
| **volume (asinh/smape)** | asinh & smape | 1.87 | **28.2%** | **88.8%** |
| **volume (mae)** | mae | **1.11** | 39.9% | 83.6% |

For energy and bandgap, all 3 optimization metrics converged to the **same best model** — very robust. For volume, asinh and smape agree (better sMAPE/MagAcc) while mae finds a different model (lower MAE but much worse sMAPE).

### Multi-target: geometry (7 outputs) — best metric = smape

Angles are hard to predict (sMAPE 70-87%); lengths and volume are reasonable:

| Column | smape MAE | smape sMAPE | asinh sMAPE | mae sMAPE |
|--------|-----------|-------------|-------------|-----------|
| volume | 0.60 | 33.6% | 33.1% | 36.6% |
| a_len | 0.16 | 37.6% | 37.9% | 47.5% |
| b_len | 0.31 | 37.1% | 37.8% | 44.8% |
| c_len | 0.92 | 41.5% | 41.9% | 49.9% |
| alpha | 1.02 | **71.0%** | 84.7% | 90.2% |
| beta | 1.24 | **70.9%** | 87.8% | 91.8% |
| gamma | 1.33 | **72.7%** | 87.4% | 91.4% |

The smape metric is clearly best for angles (71% vs 85-91%).

### Multi-target: all_scalar (9 outputs) — worse than single-target

The all_scalar models degrade compared to dedicated single-target models:
- Energy sMAPE: 21.2% (all_scalar_smape) vs **9.7%** (single-target) — **2.2x worse**
- Bandgap sMAPE: 34.4% (all_scalar_asinh) vs **30.5%** (single-target) — worse
- Volume sMAPE: 32.9% (all_scalar_smape) vs **28.2%** (single-target) — worse

### Key takeaways

1. **Single-target RF models clearly beat multi-target ones.** Joint prediction of all 9 targets in one RF hurts every individual target.
2. **sMAPE optimization metric** is the best choice for the geometry group (big wins on angles). For energy/bandgap/volume the metric choice doesn't matter much (same models selected).
3. **Angles are very hard** (sMAPE 70-87% even with best metric) — this is likely a fundamental limitation of tabular features for angular predictions.
4. The CV results CSVs are saved if you want to do deeper analysis on the hyperparameter landscape.

**Recommended next step:** Use the single-target models (energy, bandgap, volume optimized with smape/asinh) plus the geometry model (optimized with smape) as your RF baselines for comparison against the EGNN.
