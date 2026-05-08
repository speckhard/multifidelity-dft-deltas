import ase.db
import ase.build
import numpy as np
import ast
import re
from pathlib import Path
from absl import flags
from absl import app
import os
import sys
import random
import pandas as pd  # Added for CSV export
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from parsing.reference_selectors import SELECTORS, get_selector  # noqa: E402

# Configuration
INPUT_DB_PATH = 'relaxations.db'
OUTPUT_DB_PATH = 'delta_learning.db'
OUTPUT_CSV_PATH = 'delta_learning.csv'   # New path for CSV
REFERENCE_PRECISION = 11

# ---------------------------------------------------------------------------
# Per-dataset key sets.
#
# Each selector (see parsing/reference_selectors.py) is paired here with a
# dict describing which raw scalars become `*_per_atom`, which fields should
# have `delta_*` computed, and which are arrays (need special parse / rotate
# handling). New datasets: add a new entry AND register a selector.
# ---------------------------------------------------------------------------

AIMS_KEYS_TO_CONVERT_TO_PER_ATOM = [
    'total_energy', 'aims_free_energy', 'final_volume',
    'sum_eigenvalues', 'xc_energy_correction', 'xc_potential_correction',
    'free_atom_electrostatic_energy', 'hartree_energy_correction', 'entropy_correction',
    'total_energy_T0', 'kinetic_energy', 'electrostatic_energy', 'multipole_correction',
]

AIMS_KEYS_TO_DELTA = [
    # Converted Energy Components
    'total_energy_per_atom', 'aims_free_energy_per_atom',
    'sum_eigenvalues_per_atom', 'xc_energy_correction_per_atom', 'xc_potential_correction_per_atom',
    'free_atom_electrostatic_energy_per_atom', 'hartree_energy_correction_per_atom',
    'entropy_correction_per_atom', 'total_energy_T0_per_atom', 'kinetic_energy_per_atom',
    'electrostatic_energy_per_atom', 'multipole_correction_per_atom',

    # Existing Per-Atom Keys
    'electronic_free_energy_per_atom',

    # Electronic Structure
    'vbm', 'cbm', 'homo_lumo_gap', 'chemical_potential',
    'final_volume_per_atom',
    # Geometry - Scalars
    'relaxed_a_len', 'relaxed_b_len', 'relaxed_c_len',
    'relaxed_alpha_angle', 'relaxed_beta_angle', 'relaxed_gamma_angle',
    # Relaxed Geometry - Arrays (Stored as strings)
    'relaxed_atom_positions', 'relaxed_cell',

    # --- Monomer Statistical Features ---
    'max_monomer_total_energy_per_atom', 'min_monomer_total_energy_per_atom',
    'mean_monomer_total_energy_per_atom', 'mad_monomer_total_energy_per_atom',
    'max_monomer_volume_per_atom', 'min_monomer_volume_per_atom',
    'mean_monomer_volume_per_atom', 'mad_monomer_volume_per_atom',
    'max_monomer_bandstructure_gap', 'min_monomer_bandstructure_gap',
    'mean_monomer_bandstructure_gap', 'mad_monomer_bandstructure_gap',
    'max_monomer_gamma_gap', 'min_monomer_gamma_gap',
    'mean_monomer_gamma_gap', 'mad_monomer_gamma_gap',

    # Basis set features
    'min_basis_functions', 'max_basis_functions', 'mean_basis_functions',
    'mad_basis_functions',
]

AIMS_ARRAY_KEYS = ('relaxed_atom_positions', 'relaxed_cell')


EXCITING_KEYS_TO_CONVERT_TO_PER_ATOM = [
    # INFO.OUT energy decomposition (10 terms, in Hartree)
    'total_energy', 'kinetic_energy', 'exchange_energy', 'correlation_energy',
    'hartree_energy', 'electron_nuclear_energy', 'nuclear_nuclear_energy',
    'xc_potential_energy', 'madelung_energy', 'core_electron_kinetic_energy',
    # Volume (exciting parser uses `relaxed_volume`, not aims' `final_volume`)
    'relaxed_volume',
]

EXCITING_KEYS_TO_DELTA = [
    # Per-atom energy components
    'total_energy_per_atom', 'kinetic_energy_per_atom',
    'exchange_energy_per_atom', 'correlation_energy_per_atom',
    'hartree_energy_per_atom', 'electron_nuclear_energy_per_atom',
    'nuclear_nuclear_energy_per_atom', 'xc_potential_energy_per_atom',
    'madelung_energy_per_atom', 'core_electron_kinetic_energy_per_atom',

    # Fermi / DOS (not per-atom; Hartree and states/Ha/cell respectively)
    'fermi_energy', 'dos_at_fermi',

    # Band gaps from two sources (eigval sparse grid + bandstructure dense path)
    'band_gap_scf_eV', 'band_gap_bands_eV',

    # Geometry - scalars
    'relaxed_volume_per_atom',
    'relaxed_a_len', 'relaxed_b_len', 'relaxed_c_len',
    'relaxed_alpha_angle', 'relaxed_beta_angle', 'relaxed_gamma_angle',

    # Geometry - arrays (exciting parser: `relaxed_lattice`, not `relaxed_cell`)
    'relaxed_atom_positions', 'relaxed_lattice',
]

EXCITING_ARRAY_KEYS = ('relaxed_atom_positions', 'relaxed_lattice')

EXCITING_ENERGY_KEYS_HA = frozenset({
    'total_energy', 'kinetic_energy', 'exchange_energy', 'correlation_energy',
    'hartree_energy', 'electron_nuclear_energy', 'nuclear_nuclear_energy',
    'xc_potential_energy', 'madelung_energy', 'core_electron_kinetic_energy',
    'fermi_energy',
})


# Registry: selector name -> key-set bundle.
#
# `precision_key` names the DB column that uniquely identifies a
# calculation's precision under the selector's convention. It's used to
# dedup cheap rows per ICSD: if the same (ICSD, precision_key value) shows
# up more than once (e.g., same structure computed in two AFLOW batch
# subdirs), only the first match is paired against the reference. Mirrors
# the first-match-wins policy already used for the reference row.
KEY_SETS = {
    'aims': {
        'convert_to_per_atom': AIMS_KEYS_TO_CONVERT_TO_PER_ATOM,
        'delta': AIMS_KEYS_TO_DELTA,
        'array_keys': AIMS_ARRAY_KEYS,
        'precision_key': 'binary_precision',
        'energy_keys_ha': frozenset(),
    },
    'exciting': {
        'convert_to_per_atom': EXCITING_KEYS_TO_CONVERT_TO_PER_ATOM,
        'delta': EXCITING_KEYS_TO_DELTA,
        'array_keys': EXCITING_ARRAY_KEYS,
        'precision_key': 'APWprecision_input',
        'energy_keys_ha': EXCITING_ENERGY_KEYS_HA,
    },
}

# Backwards-compat aliases — some external code may import these by name.
KEYS_TO_CONVERT_TO_PER_ATOM = AIMS_KEYS_TO_CONVERT_TO_PER_ATOM
KEYS_TO_DELTA = AIMS_KEYS_TO_DELTA

FLAGS = flags.FLAGS
flags.DEFINE_string('input_db_path', INPUT_DB_PATH, 'Path to input DB.')
flags.DEFINE_string('output_db_path', OUTPUT_DB_PATH, 'Path to save output DB.')
flags.DEFINE_string('output_csv_path', OUTPUT_CSV_PATH, 'Path to save output CSV.')
flags.DEFINE_float('train_split_ratio', 0.8, 'Ratio of data to use for training.')
flags.DEFINE_float('val_split_ratio', 0.1, 'Ratio of data to use for training.')
flags.DEFINE_integer('random_seed', 42, 'Seed for reproducible splitting.')
flags.DEFINE_enum(
    'reference_selector', 'aims', sorted(SELECTORS.keys()),
    'Which dataset convention to use when picking the per-ICSD reference row.',
)
flags.DEFINE_float(
    'max_scf_delta_eV_per_atom', -1.0,
    'If > 0, drop rows whose final-iteration |ΔE_total|/N_atoms exceeds '
    'this (eV/atom). Reads INFO.OUT for each row. Default -1.0 disables '
    'the filter. For exciting runs pass 1e-4 to drop numerically '
    'diverged SCF runs that still returned a clean exit code.',
)


_ABS_DE_HA_RE = re.compile(
    r"Absolute change in total energy\s*(?:\([^)]*\))?\s*:\s*"
    r"([-+0-9.eE]+)"
)
_HA_TO_EV = 27.211386245988


def _last_scf_delta_eV_per_atom(info_path, num_atoms):
    """Parse INFO.OUT and return |last-iter ΔE_total| / N in eV/atom.

    Returns None on any failure (missing file, no match, zero atoms).
    Caller decides whether to drop the row.
    """
    if num_atoms is None or num_atoms <= 0:
        return None
    try:
        text = Path(info_path).read_text()
    except (OSError, TypeError):
        return None
    matches = _ABS_DE_HA_RE.findall(text)
    if not matches:
        return None
    try:
        dE_Ha = float(matches[-1])
    except ValueError:
        return None
    return abs(dE_Ha) * _HA_TO_EV / float(num_atoms)


def _convert_energy_keys_ha_to_ev(data_dict, energy_keys_ha):
    """Convert energy-related keys and their per-atom variants from Hartree to eV in-place."""
    for key in energy_keys_ha:
        if key in data_dict and data_dict[key] is not None:
            try:
                data_dict[key] = float(data_dict[key]) * _HA_TO_EV
            except (ValueError, TypeError):
                pass
        pa_key = f'{key}_per_atom'
        if pa_key in data_dict and data_dict[pa_key] is not None:
            try:
                data_dict[pa_key] = float(data_dict[pa_key]) * _HA_TO_EV
            except (ValueError, TypeError):
                pass


def parse_array_string(array_str):
    """Parses a string representation of a numpy array.

    Also accepts raw list/tuple/ndarray inputs (exciting row.data path) so
    callers don't have to care which storage shape the ASE DB used.
    """
    if array_str is None:
        return None
    if isinstance(array_str, np.ndarray):
        return array_str
    if isinstance(array_str, (list, tuple)):
        try:
            return np.asarray(array_str)
        except (TypeError, ValueError):
            return None
    try:
        return np.array(ast.literal_eval(array_str))
    except (ValueError, SyntaxError, TypeError):
        try:
            clean_str = array_str.replace('[', '').replace(']', '').strip()
            return np.fromstring(clean_str, sep=',')
        except Exception:
            pass
        try:
            clean_str = array_str.replace('[', '').replace(']', '').strip()
            return np.fromstring(clean_str, sep=' ')
        except Exception:
            return None

def format_array_string(array_obj):
    """Formats a numpy array back to the string format used in the DB."""
    return np.array2string(
        array_obj, separator=',',
        formatter={'float_kind': lambda x: "%.16f" % x}
    ).replace('\n', '')

def get_row_value(row, key, kv_dict=None):
    """Robustly retrieve a value from an ASE DB row.

    Reads (in priority order) the row attribute, the passed-in workspace
    dict (if provided), `row.key_value_pairs`, then `row.data`. The last
    lookup matters for the exciting flow, where arrays and numeric fields
    that don't fit in `key_value_pairs` are routed to `data` by the DB
    writer (see `nomad_parsing/exciting_to_ase_db.py::_split_kvp_and_data`).
    """
    if hasattr(row, key):
        return getattr(row, key)
    if kv_dict is not None and key in kv_dict:
        return kv_dict.get(key)
    if hasattr(row, 'key_value_pairs'):
        val = row.key_value_pairs.get(key)
        if val is not None:
            return val
    data = getattr(row, 'data', None)
    if isinstance(data, dict) and key in data:
        return data.get(key)
    return None


def _merge_row_arrays_into_kvp(row, data_dict, array_keys):
    """Fold array-valued fields from `row.data` into `data_dict` as strings.

    The existing delta pipeline treats relaxed geometry arrays as stringified
    numpy arrays stored alongside the scalar kvps. The aims DB writer already
    stores them that way; the exciting DB writer routes arrays to `row.data`
    as Python lists (or numpy arrays, after ase.db round-trip). This helper
    pulls those back into `data_dict` in the expected string format so the
    downstream delta logic stays uniform across datasets.
    """
    row_data = getattr(row, 'data', None)
    if not isinstance(row_data, dict):
        return
    for key in array_keys:
        if key in data_dict and data_dict.get(key) is not None:
            continue  # already in kvp as string (aims case)
        val = row_data.get(key)
        if val is None:
            continue
        arr = np.asarray(val)
        try:
            data_dict[key] = format_array_string(arr.astype(float))
        except (TypeError, ValueError):
            continue

def export_db_to_csv(db_path, csv_path):
    """Reads the generated ASE DB and exports all KV pairs to CSV."""
    print(f"Exporting database to CSV: {csv_path}")
    if not os.path.exists(db_path):
        print("Database not found, skipping CSV export.")
        return

    db = ase.db.connect(db_path)
    data_list = []

    for row in tqdm(db.select(), desc="Generating CSV"):
        # Start with the key_value_pairs dictionary
        row_data = row.key_value_pairs.copy()
        
        # Add standard ASE identifiers
        row_data['id'] = row.id
        row_data['formula'] = row.formula
        
        # CRITICAL: Add atomic numbers so the position arrays are meaningful
        # We convert numpy array to list/string for CSV compatibility
        row_data['atomic_numbers'] = list(row.numbers)
        
        data_list.append(row_data)

    if data_list:
        df = pd.DataFrame(data_list)
        df.to_csv(csv_path, index=False)
        print(f"CSV saved with {len(df)} rows.")
    else:
        print("No data found to export.")

def create_delta_dataset(
        input_db_path, output_db_path, csv_output_path, train_ratio,
        val_ratio=0.1, seed=42, reference_selector='aims',
        max_scf_delta_eV_per_atom=None):
    """Build delta-learning DB + CSV.

    Args:
        max_scf_delta_eV_per_atom: if not None and > 0, drops rows whose
            final SCF |ΔE_total|/N_atoms exceeds the threshold (eV/atom).
            Reads INFO.OUT per row. Designed for exciting where some
            `status=ok` rows have numerically-diverged SCF.
    """
    # --- PRE-CHECKS ---
    selector_fn = get_selector(reference_selector)
    if reference_selector not in KEY_SETS:
        raise ValueError(
            f"No KEY_SETS entry for selector {reference_selector!r}. "
            f"Known: {sorted(KEY_SETS)}"
        )
    key_set = KEY_SETS[reference_selector]
    keys_to_per_atom = key_set['convert_to_per_atom']
    keys_to_delta = key_set['delta']
    array_keys = tuple(key_set['array_keys'])
    precision_key = key_set.get('precision_key')
    energy_keys_ha = key_set.get('energy_keys_ha', frozenset())
    print(f"Using reference selector: {reference_selector!r}")
    print(f"  per-atom keys: {len(keys_to_per_atom)}, "
          f"delta keys: {len(keys_to_delta)}, "
          f"array keys: {array_keys}, "
          f"precision_key: {precision_key!r}")
    if max_scf_delta_eV_per_atom and max_scf_delta_eV_per_atom > 0:
        print(f"  SCF filter: drop rows with |ΔE_final|/N > "
              f"{max_scf_delta_eV_per_atom} eV/atom")
    if not os.path.exists(input_db_path):
        print(f"Error: Input database {input_db_path} not found.")
        return

    if os.path.exists(output_db_path):
        print(f"Warning: Output database {output_db_path} already exists. Overwriting.")
        try:
            os.remove(output_db_path)
        except OSError as e:
             print(f"Error removing existing output DB: {e}")
             return

    # --- PHASE 1: READ INPUT ---
    # We use 'with' here so the input file is strictly closed after loading data
    print(f"Reading input database: {input_db_path}")
    grouped_data = {}
    n_dropped_scf = 0
    scf_filter_enabled = (
        max_scf_delta_eV_per_atom is not None
        and max_scf_delta_eV_per_atom > 0
    )

    with ase.db.connect(input_db_path) as src_db:
        # Note: src_db.select() loads Row objects into memory.
        # They persist in 'grouped_data' even after src_db closes.
        for row in tqdm(src_db.select(), desc="Grouping data"):
            if not hasattr(row, 'ICSD_number'):
                continue
            if scf_filter_enabled:
                info_path = get_row_value(row, 'path')
                num_atoms = get_row_value(row, 'num_atoms')
                if info_path is not None:
                    info_path = str(info_path) + '/INFO.OUT'
                scf_d = _last_scf_delta_eV_per_atom(info_path, num_atoms)
                if scf_d is None or scf_d > max_scf_delta_eV_per_atom:
                    n_dropped_scf += 1
                    continue
            icsd = row.ICSD_number
            if icsd not in grouped_data:
                grouped_data[icsd] = []
            grouped_data[icsd].append(row)

    if scf_filter_enabled:
        print(f"SCF filter dropped {n_dropped_scf} rows (>{max_scf_delta_eV_per_atom} eV/atom).")

    # Input DB connection is now CLOSED.

    # --- PHASE 2: CALCULATE SPLITS ---
    unique_icsds = list(grouped_data.keys())
    print(f"Found {len(unique_icsds)} unique ICSD entries.")

    # --- PHASE 2: CALCULATE SPLITS ---
    unique_icsds = list(grouped_data.keys())
    random.seed(seed)
    random.shuffle(unique_icsds)

    total_icsds = len(unique_icsds)
    n_train = int(total_icsds * train_ratio)
    n_val = int(total_icsds * val_ratio)
    
    train_icsds = set(unique_icsds[:n_train])
    val_icsds = set(unique_icsds[n_train : n_train + n_val])
    test_icsds = set(unique_icsds[n_train + n_val:])

    print(f"Split: Train={len(train_icsds)}, Val={len(val_icsds)}, Test={len(test_icsds)}")

    # ... (writing loop) ...
    for icsd, rows in tqdm(grouped_data.items(), desc="Processing Deltas"):
        if icsd in train_icsds:
            split_label = 'train'
        elif icsd in val_icsds:
            split_label = 'val'
        else:
            split_label = 'test'

    # --- PHASE 3: PROCESS & WRITE OUTPUT ---
    count_saved = 0
    count_discarded = 0
    count_cheap_deduped = 0

    # Open the new database safely
    with ase.db.connect(output_db_path) as new_db:
        for icsd, rows in tqdm(grouped_data.items(), desc="Processing Deltas"):

            # Per-ICSD set of already-seen cheap precision values, used
            # for first-match-wins cheap-row dedup (mirrors the ref-row
            # policy). `None` values bypass the dedup — they can't be
            # compared reliably.
            seen_cheap_precisions = set()

            # --- CORRECT LOGIC ---
            if icsd in train_icsds:
                split_label = 'train'
            elif icsd in val_icsds:
                split_label = 'val'
            else:
                split_label = 'test'

            # Get the reference calculation data via the configured selector.
            ref_row = None
            for row in rows:
                if selector_fn(row):
                    ref_row = row
                    break

            if ref_row is None:
                count_discarded += len(rows)
                continue

            # --- PREPARE REFERENCE DATA ---
            ref_atoms_obj = ref_row.toatoms()
            N_ref = len(ref_atoms_obj)
            ref_data_dict = ref_row.key_value_pairs.copy()
            _merge_row_arrays_into_kvp(ref_row, ref_data_dict, array_keys)

            all_keys_needed = list(keys_to_delta) + list(keys_to_per_atom) + list(array_keys)
            for key in all_keys_needed:
                val = get_row_value(ref_row, key, ref_data_dict)
                if val is not None:
                    ref_data_dict[key] = val

            for key in keys_to_per_atom:
                val = ref_data_dict.get(key, None)
                if val is not None and N_ref > 0:
                    try:
                        ref_data_dict[f'{key}_per_atom'] = float(val) / N_ref
                    except (ValueError, TypeError):
                        continue

            _convert_energy_keys_ha_to_ev(ref_data_dict, energy_keys_ha)

            ref_arrays = {}
            for key in array_keys:
                ref_arrays[key] = parse_array_string(ref_data_dict.get(key))


            # Add delta information for each row
            for row in rows:
                if row.id == ref_row.id:
                    continue

                # De-dup cheap rows by (ICSD, precision_key value). Two
                # rows with the same (icsd, precision) on disk are AFLOW
                # batch dupes (e.g., aflow_binaries_exciting_0_999 vs
                # _1000_1999). Keep only the first, discard the rest —
                # mirrors the ref-row first-match policy.
                if precision_key is not None:
                    prec_val = get_row_value(row, precision_key,
                                             row.key_value_pairs)
                    if prec_val is not None:
                        try:
                            prec_tag = round(float(prec_val), 4)
                        except (TypeError, ValueError):
                            prec_tag = prec_val
                        if prec_tag in seen_cheap_precisions:
                            count_cheap_deduped += 1
                            continue
                        seen_cheap_precisions.add(prec_tag)

                data_dict = row.key_value_pairs.copy()
                _merge_row_arrays_into_kvp(row, data_dict, array_keys)
                atoms_obj = row.toatoms()
                N_curr = len(atoms_obj)

                # ADD THE SPLIT LABEL
                data_dict['split'] = split_label

                # Convert to per-atom
                for key in keys_to_per_atom:
                    val = get_row_value(row, key, data_dict)
                    if val is not None and N_curr > 0:
                        try:
                            data_dict[f'{key}_per_atom'] = float(val) / N_curr
                        except (ValueError, TypeError):
                            continue

                _convert_energy_keys_ha_to_ev(data_dict, energy_keys_ha)

                # Calculate Deltas
                for key in keys_to_delta:
                    val_curr = data_dict.get(key)
                    if val_curr is None:
                         val_curr = get_row_value(row, key, data_dict)

                    val_ref = ref_data_dict.get(key, None)
                    delta_key_name = f'delta_{key}'

                    if val_curr is not None and val_ref is not None:
                        # Array Types
                        if key in array_keys:
                            arr_curr = parse_array_string(val_curr)
                            arr_ref = ref_arrays.get(key) 
                            
                            if arr_curr is not None and arr_ref is not None:
                                try:
                                    if arr_curr.shape == arr_ref.shape:
                                        if key == 'relaxed_atom_positions':
                                            tmp_ref = ref_atoms_obj.copy()
                                            tmp_ref.set_positions(arr_ref)
                                            tmp_curr = atoms_obj.copy()
                                            tmp_curr.set_positions(arr_curr)
                                            ase.build.minimize_rotation_and_translation(target=tmp_ref, atoms=tmp_curr)
                                            delta_arr = tmp_curr.get_positions() - tmp_ref.get_positions()
                                        else:
                                            delta_arr = arr_curr - arr_ref

                                        data_dict[delta_key_name] = format_array_string(delta_arr)
                                except Exception:
                                    pass
                        # Scalar Types
                        else:
                            try:
                                delta_val = float(val_curr) - float(val_ref)
                                data_dict[delta_key_name] = delta_val
                            except (ValueError, TypeError):
                                pass
                
                # Write to DB safely
                new_db.write(atoms_obj, key_value_pairs=data_dict)
                count_saved += 1
    
    # Output DB connection is now CLOSED automatically.

    print("-" * 30)
    print(f"Processing Complete.")
    print(f"Rows saved to {output_db_path}: {count_saved}")
    if count_cheap_deduped:
        print(f"Cheap-side batch dupes collapsed: {count_cheap_deduped} "
              f"(first-match wins per (ICSD, {precision_key}))")
    
    # --- PHASE 4: EXPORT TO CSV ---
    # Now it is safe to open the DB again for reading
    export_db_to_csv(output_db_path, csv_output_path)


def main(argv):
    max_scf = (FLAGS.max_scf_delta_eV_per_atom
               if FLAGS.max_scf_delta_eV_per_atom > 0 else None)
    create_delta_dataset(
        FLAGS.input_db_path,
        FLAGS.output_db_path,
        FLAGS.output_csv_path,
        FLAGS.train_split_ratio,
        FLAGS.val_split_ratio,
        FLAGS.random_seed,
        reference_selector=FLAGS.reference_selector,
        max_scf_delta_eV_per_atom=max_scf,
    )


if __name__ == '__main__':
    app.run(main)
