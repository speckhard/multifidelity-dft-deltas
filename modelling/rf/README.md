# Random-forest feature sets

The FHI-aims ablation uses these inputs in the maintained data schema:

| Feature group | RF-Full | RF-PreCalc | RF-Minimal |
|---|---:|---:|---:|
| Atomic descriptors: 10 properties × 4 statistics | 40 | 40 | 40 |
| Valence-electron and basis-function statistics | 8 | 8 | 8 |
| Input cell volume, lengths and angles | 7 | 7 | 7 |
| `binary_precision` and `k_point_density` | 2 | 2 | 2 |
| Elemental-solid descriptors and their deltas | 32 | 32 | 0 |
| Cheap compound energy outputs | 12 | 0 | 0 |
| Relaxed volume per atom | 1 | 0 | 0 |
| **Total** | **102** | **89** | **57** |

The atomic properties are `EA_half`, `IP_half`, `EA_delta`, `IP_delta`,
`HOMO`, `LUMO`, `rs`, `rp`, `rd`, and `rf`. Statistics are minimum, maximum,
mean, and mean absolute deviation. Valence-electron and basis-function
counts each use the same four statistics.

“Monomer” means an elemental solid here. Its descriptors are the four
statistics of energy per atom, volume per atom, bandstructure gap, and
gamma-point gap (16 features), plus their `delta_*` counterparts (16).
PreCalc assumes these elemental-solid calculations have been precomputed;
it does not require a candidate compound DFT calculation.

**Input crystal geometry remains included in all three sets.** PreCalc
removes the cheap compound energy outputs and `final_volume_per_atom`.
Minimal additionally removes every column whose name contains `monomer`.

## Code and exact column names

- [feature_contracts.py](feature_contracts.py) exports ordered
  `FHI_FULL_FEATURES`, `FHI_PRECALC_FEATURES`, `FHI_MINIMAL_FEATURES`, and
  `FHI_FEATURE_SETS`. These lists are ported from the original
  [errorbar_modelling feature contracts](https://github.com/speckhard/errorbar_modelling/blob/active_learning/modelling/rf/feature_contracts.py),
  with Minimal derived using the original loader's monomer exclusion.
- [rf_config.py](rf_config.py) defines common exclusions and `FEATURE_SETS`.
- [data_loader.py](data_loader.py) applies the supplied exclusions, removes
  non-monomer `delta_*` columns, keeps numeric columns, and zero-fills missing
  values. [rf_trainer.py](rf_trainer.py) forwards the selected ablation.
- [output_parser_vibes.py](../../parsing/output_parser_vibes.py) constructs
  the descriptors; [create_delta_dataset.py](../../parsing/create_delta_dataset.py)
  constructs the delta features and targets.

Counts above refer to the maintained schema, not every possible CSV. The
legacy loader uses exclusions rather than an allowlist, so extra numeric
columns can change the counts. For example, its exclusion list also covers
the legacy `electronic_free_energy` column, which is absent from the
102-feature list. Check the returned `X.columns` or the saved estimator's
`feature_names_in_` against `FHI_FEATURE_SETS` for an exact benchmark.

The 57-feature Minimal count follows from the implemented filtering. These
definitions do not authenticate the inputs of every historical paper
checkpoint or claim that new training reproduces its reported scores.

## Training

```bash
python modelling/rf/rf_trainer.py \
    --data_file=/path/to/delta_dataset.csv \
    --output_dir=runs/rf_precalc \
    --target_key=energy \
    --metric_key=mae \
    --feature_set=precalc
```

Use `full`, `precalc`, or `minimal` for the FHI config. Configs without
`FEATURE_SETS` (including exciting energy and DOS similarity) support
`full` only; requesting another regime raises an error.

```bash
python -m pytest tests/test_rf_feature_sets.py tests/test_rf_exciting.py \
    tests/rf_config_dos_similarity_test.py
```
