"""Parse output files from DFT simulations.

Scripts in this file are used to parse data output
and save it to a json like structure. We then save
each json like object to an ASE database and a CSV file.

Eventually, we would like to use the aims parser."""
import ase.io
import os
from numpy.linalg import norm
import logging
from absl import flags
from absl import app
import ase.calculators.calculator
import ase.calculators.aims
import sys
import glob
import csv
import numpy as np
import pickle
import mendeleev
# import mendeleev  # This had issues with numpy compatability.
import pandas as pd
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import parsing.json_parser as js

# Mapping from FHI-aims basis-set name to integer tier index, inlined here
# from the historical ``data_gen.gen_data.GenData`` class. The clean repo
# omits that data-generation package, so we keep only this small constant
# (the only thing the parser ever needed from it).
_TIER_MAP: dict[str, int | None] = {
    "standard": None,
    "default": None,
    "minimal": 0,
    "tier1": 1,
    "tier2": 2,
    "tier3": 3,
}
from pathlib import Path


PBE_FEATURES_CSV = (
    'modelling/data/descriptor_aims_data/really_tight_full_cut20_pbesol.csv')

AIMS_MONOMERS_CSV = 'parsing/data/json_parser/aims_monomers.csv'
BASIS_DICT_PKL = 'modelling/data/aims_basis_function_dict.pickle'

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'input_paths_txt_file', 'None',
    'Where to find list of paths to parse.')
flags.DEFINE_string(
    'save_directory',
    'None',
    'Where to save the files/folders.')
flags.DEFINE_boolean(
    'bandstructure_calculations',
    'False',
    'Where to save the files/folders.')
flags.DEFINE_string('atomic_data_path', './atomic_data.csv', 'Path to the CSV with atomic features.')

flags.DEFINE_string('monomers_data_path', './aims_monomers.csv', 'Path to the CSV with elemental solid features.')

flags.DEFINE_string('basis_dict_path', BASIS_DICT_PKL, 'Path to the pickle file with basis function counts.')


class OutputParser():
    """Parses data output."""
    def __init__(
            self, input_paths_txt_file, save_directory,
            bandstructure_calculations=False, atomic_data_path=PBE_FEATURES_CSV,
            monomers_data_path=AIMS_MONOMERS_CSV,
            basis_dict_path=BASIS_DICT_PKL):
        """Constructor

        Args:
        path_txt_file: (str) of paths where DFT output files
            can be found.
        csv_filename: (str) where to save data to a csv file.
        paths_to_resubmit: (str) path where to save simulation
            paths that didn't exit nicely so they can be
            resubmited.
        paths_misbehaving: path to files where we are not sure what went wrong
            but they didn't simulate correctly.
        paths_increase_charge_mix:
        paths_decrease_charge_mix:
        """
        # Normally a file containing paths on new lines is given
        # and not a list of paths.
        self.path_list = self.get_path_list(input_paths_txt_file)
        # self.atoms_obj_list = atoms_obj_list
        self.logger = logging.getLogger(__name__)
        self.bandstructure_calculations = bandstructure_calculations
        self.tier_map = _TIER_MAP

        self.save_directory = Path(save_directory)
        # CSV filename of where to save parsed data.
        self.parsing_time_and_day_date = get_time_and_day()

        # Define a list of paths that we shoudl resubmit
        # to a longer queue since they expired during calculation.
        self.paths_to_resubmit = self.save_directory / ('paths_to_resubmit_' + self.parsing_time_and_day_date + '.txt')

        self.paths_misbehaving = self.save_directory / ('paths_misbehaving_' + self.parsing_time_and_day_date + '.txt')
        # Paths where the SCF didn't converge. For these we want
        # to submit with larger charge mix param.
        self.paths_increase_charge_mix = self.save_directory / ('paths_increase_charge_mix_' + self.parsing_time_and_day_date + '.txt')
        # Many of these paths I am not yet using and they are copied over from exciting' output parser.
        self.paths_decrease_charge_mix = self.save_directory / ('paths_decrease_charge_mix_' + self.parsing_time_and_day_date + '.txt')
        self.paths_out_of_npl =  self.save_directory / ('paths_out_of_npl_' + self.parsing_time_and_day_date + '.txt')
        self.paths_increase_optimizer = self.save_directory / ('paths_increase_optimizer_' + self.parsing_time_and_day_date + '.txt')
        self.paths_diag_fail = self.save_directory / ('paths_diag_fail_' + self.parsing_time_and_day_date + '.txt')
        self.paths_mt_overlap = self.save_directory / ('paths_mt_overlap_' + self.parsing_time_and_day_date + '.txt')
        self.paths_ndirac_limit = self.save_directory / ('paths_ndirac_limit_' + self.parsing_time_and_day_date + '.txt')
        self.paths_species_xml_missing = self.save_directory / ('paths_species_xml_missing_' + self.parsing_time_and_day_date + '.txt')
        self.paths_scalar_to_integer = self.save_directory / ('paths_scalar_to_integer_' + self.parsing_time_and_day_date + '.txt')
        self.paths_ongoing = self.save_directory / ('paths_ongoing_' + self.parsing_time_and_day_date + '.txt')
        self.paths_gs_not_started = self.save_directory / ('paths_gs_not_started_' + self.parsing_time_and_day_date + '.txt')
        self.paths_geo_opt_not_started = self.save_directory / ('paths_geo_opt_not_started_' + self.parsing_time_and_day_date + '.txt')
        self.paths_expired = self.save_directory / ('paths_expired_' + self.parsing_time_and_day_date + '.txt')
        self.paths_resubmit_from_scratch = self.save_directory / ('paths_resubmit_from_scratch_' + self.parsing_time_and_day_date + '.txt')
        self.paths_to_resubmit = self.save_directory / ('paths_to_resubmit_' + self.parsing_time_and_day_date + '.txt')
        self.paths_out_of_memory = self.save_directory / ('paths_out_of_memory_' + self.parsing_time_and_day_date + '.txt')
        self.paths_symmetry_issues = self.save_directory / ('paths_symmetry_issues_' + self.parsing_time_and_day_date + '.txt')
        self.paths_finished_correctly = self.save_directory / ('paths_finished_correctly_' + self.parsing_time_and_day_date + '.txt')
        self.csv_filename = self.save_directory / ('parsed_data_csv_' + self.parsing_time_and_day_date + '.csv')
        self.db_path = self.save_directory / ('relaxations_' + self.parsing_time_and_day_date + '.db')

        # --- CSV SETUP ---
        # Initialize the Counter UID
        self.uid_counter = 0

        # Define features and statistics
        self.atomic_feature_keys = [
            'EA_half', 'IP_half', 'EA_delta', 'IP_delta', 
            'HOMO', 'LUMO', 'rs', 'rp', 'rd', 'rf'
        ]
        self.monomer_feature_keys = [
            'total_energy_per_atom', 'volume_per_atom', 
            'bandstructure_gap', 'gamma_gap'
        ]
        self.stats_suffixes = ['max', 'min', 'mean', 'mad']

        # Define CSV filename and columns
        # NOTE: 'free_energy' is a reserved key in ASE DB, so we rename it to 'aims_free_energy'
        self.csv_columns = [
            'uid', # New Unique Identifier
            'compound_name', 'ICSD_number','chem_formula',
            # 
            'k_point_density', 'rel_setting', 
            'basis_size', 'num_setting', 'functional', 'relaxation',
            # Energy
            'total_energy', 'aims_free_energy', 
            'total_energy_per_atom', 
            # Detailed Energy Components
            'sum_eigenvalues', 'xc_energy_correction', 'xc_potential_correction', 
            'free_atom_electrostatic_energy', 'hartree_energy_correction', 'entropy_correction', 
            'total_energy_T0', 'kinetic_energy', 'electrostatic_energy', 'multipole_correction', 
            'sum_eigenvalues_per_atom', 'total_energy_T0_per_atom', 'electronic_free_energy_per_atom', 
            'vbm', 'cbm', 'homo_lumo_gap', 'chemical_potential',
            # Original Geometry
            'original_volume', 
            'original_atom_positions', 'original_cell', 
            'original_a_len', 'original_b_len', 'original_c_len', 
            'original_alpha_angle', 'original_beta_angle', 'original_gamma_angle',
            # Relaxed Geometry
            'final_volume', 
            'relaxed_atom_positions', 'relaxed_cell',
            'relaxed_a_len', 'relaxed_b_len', 'relaxed_c_len', 
            'relaxed_alpha_angle', 'relaxed_beta_angle', 'relaxed_gamma_angle',
            # Metadata
            'time_day', 'path', 'binary_precision'
        ]

        # --- NEW: Add Statistical Feature Columns ---
        # e.g., max_atomic_EA_half, min_atomic_EA_half, ...
        for feature in self.atomic_feature_keys:
            for stat in self.stats_suffixes:
                self.csv_columns.append(f'{stat}_atomic_{feature}')
        # --- Add Statistical Monomer Feature Columns ---
        for feature in self.monomer_feature_keys:
            for stat in self.stats_suffixes:
                self.csv_columns.append(f'{stat}_monomer_{feature}')
        # --- NEW: Add Valence and Basis Function Columns ---
        for stat in self.stats_suffixes:
            self.csv_columns.append(f'{stat}_valence_electrons')
            self.csv_columns.append(f'{stat}_basis_functions')
        # --- Load Atomic Features Data ---
        try:
            self.atomic_df = pd.read_csv(atomic_data_path)
            # Ensure column names are stripped of whitespace
            self.atomic_df.columns = self.atomic_df.columns.str.strip()
            print(f"Loaded atomic features from {atomic_data_path}")
        except FileNotFoundError:
            raise ValueError('Cant find the atomic csv')

        # --- Load Monomer (Elemental Solid) Features Data ---
        try:
            self.monomers_df = pd.read_csv(monomers_data_path)
            self.monomers_df.columns = self.monomers_df.columns.str.strip()
            print(f"Loaded monomers features from {monomers_data_path}")
        except FileNotFoundError:
            raise ValueError('Cant find the monomers csv')
        # Load Basis Function Dictionary Pickle
        try:
            with open(basis_dict_path, "rb") as f:
                self.basis_dict = pickle.load(f)
            print(f"Loaded basis function dict from {basis_dict_path}")
        except (FileNotFoundError, pickle.UnpicklingError) as e:
            print(f"Warning: Could not load basis dict pickle. Basis features will be empty. Error: {e}")
            self.basis_dict = None

        # Initialize a simple cache for mendeleev to avoid repeated API calls
        self.valence_cache = {}
        # Create CSV file and write header immediately
        try:
            with open(self.csv_filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(self.csv_columns)
            print(f"CSV initialized at: {self.csv_filename}")
        except Exception as e:
            logging.error(f"Failed to initialize CSV file: {e}")

    def get_path_list(self, paths_txt_file):
        """Convert .txt file with paths in each newline to list.

        Args:
        path_txt_file: (string) path to a txt file where there
            is a path to a different simulation on each newline.

        Returns:
        path_list: (list) list of path strings to each simulation
            script.
        """
        with open(paths_txt_file, "r") as txt_file:
            # Strip new line chars (\n) from each line.
            path_list = [line.rstrip('\n') for line in txt_file]
        return path_list


    # --- NEW: Valence Electron Logic ---
    def get_valence_features(self, atoms_obj):
        """Calculates statistics for valence electrons of the elements."""
        features_dict = {}
        if atoms_obj is None:
            return features_dict
        
        valence_values = []
        # Iterate over every atom in the object (e.g. GaAs -> 1 Ga, 1 As)
        for z in atoms_obj.numbers:
            z = int(z)
            if z not in self.valence_cache:
                try:
                    # Use mendeleev
                    self.valence_cache[z] = mendeleev.element(z).nvalence()
                except Exception as e:
                    self.logger.error(f"Mendeleev lookup failed for Z={z}: {e}")
                    self.valence_cache[z] = None
    
            if self.valence_cache[z] is not None:
                valence_values.append(self.valence_cache[z])
            else:
                raise ValueError('Coulndt find valence number for number {z}')

        vals = np.array(valence_values)
        if len(vals) > 0:
            features_dict['max_valence_electrons'] = np.max(vals)
            features_dict['min_valence_electrons'] = np.min(vals)
            features_dict['mean_valence_electrons'] = np.mean(vals)
            features_dict['mad_valence_electrons'] = np.mean(np.abs(vals - np.mean(vals)))
        else:
            for stat in self.stats_suffixes:
                features_dict[f'{stat}_valence_electrons'] = None
        
        return features_dict

    # --- NEW: Basis Function Logic ---
    def get_basis_features(self, atoms_obj, binary_precision):
        """Calculates basis function statistics based on precision and atoms."""
        features_dict = {}
        if atoms_obj is None or self.basis_dict is None or binary_precision is None:
            print(f'object {atoms_obj} basis dict {self.basis_dict} or binary precision {binary_precision} is nONe')
            for stat in self.stats_suffixes:
                features_dict[f'{stat}_basis_functions'] = None
            return features_dict

        # Get the settings used for this precision (Logic ported from your script)
        settings = get_aims_basis_set_size(int(binary_precision))
        numerical_setting = settings[0] # e.g., 'really_tight'
        basis_set_size = settings[1]    # e.g., 2 or 'standard'
        print(f'basis_set_size {basis_set_size}')
        print(f'numerical_setting {numerical_setting}')
        basis_counts = []
        
        for z in atoms_obj.numbers:
            z = int(z)
            # The pickle keys are (numerical_setting, atomic_number, basis_set_size)
            key = (numerical_setting, z, basis_set_size)
            
            try:
                count = self.basis_dict.get(key)
                if count is not None:
                    basis_counts.append(count)
                else:
                    # Fallback or error handling if key missing in pickle
                    pass 
            except Exception:
                pass

        vals = np.array(basis_counts)
        if len(vals) > 0:
            features_dict['max_basis_functions'] = np.max(vals)
            features_dict['min_basis_functions'] = np.min(vals)
            # Mean basis functions per atom (same as sum(basis) / num_atoms)
            features_dict['mean_basis_functions'] = np.mean(vals)
            features_dict['mad_basis_functions'] = np.mean(np.abs(vals - np.mean(vals)))
        else:
            for stat in self.stats_suffixes:
                features_dict[f'{stat}_basis_functions'] = None
        
        return features_dict


    def get_monomer_features(self, atoms_obj, precision_level, k_point_density):
        """
        Calculates statistical features for elemental solid (monomer) properties.
        Filters by: category='monomers_unrelaxed', functional='pbe', 
        matching precision_level and k_point_density.
        """
        features_dict = {}
        if atoms_obj is None or self.monomers_df.empty:
            return features_dict

        unique_numbers = list(set(atoms_obj.numbers))
        values_map = {k: [] for k in self.monomer_feature_keys}

        for z in unique_numbers:
            # Filter DF for this element and specific settings
            print(f'atom number {z}')
            subset = self.monomers_df[
                (self.monomers_df['category'] == 'monomers_unrelaxed') &
                (self.monomers_df['functional'] == 'pbe') &
                (self.monomers_df['relativistic_setting'] == 'atomic_zora') &
                (self.monomers_df['precision_level'] == int(precision_level)) &
                (self.monomers_df['k_point_density'] == int(k_point_density)) &
                (self.monomers_df['min_atom_num'] == int(z))
            ]
            
            if not subset.empty:
                row = subset.iloc[0]
                print(subset)
                print(len(subset['num_atoms']))
                assert len(subset['num_atoms']) == 1
                # Extract values
                try:
                    num_atoms = float(row['num_atoms'])
                    total_energy = float(row['total_energy'])
                    print(f'total_energy {total_energy}')
                    volume = float(row['volume'])
                    
                    if num_atoms > 0:
                        values_map['total_energy_per_atom'].append(total_energy / num_atoms)
                        values_map['volume_per_atom'].append(volume / num_atoms)
                    
                    values_map['bandstructure_gap'].append(float(row['bandstructure_gap']))
                    values_map['gamma_gap'].append(float(row['gamma_gap']))
                except (ValueError, KeyError):
                    continue

        # Compute Statistics
        for key in self.monomer_feature_keys:
            vals = np.array(values_map[key])
            if len(vals) > 0:
                features_dict[f'max_monomer_{key}'] = np.max(vals)
                features_dict[f'min_monomer_{key}'] = np.min(vals)
                features_dict[f'mean_monomer_{key}'] = np.mean(vals)
                features_dict[f'mad_monomer_{key}'] = np.mean(np.abs(vals - np.mean(vals)))
            else:
                # Fill Nones if no data found for ANY element in the binary
                for stat in self.stats_suffixes:
                    features_dict[f'{stat}_monomer_{key}'] = None

        return features_dict


    def get_atomic_features(self, atoms_obj):
        """
        Calculates statistical features (max, min, mean, MAD) for atomic properties.
        Statistics are calculated based on the unique elements present.
        """
        features_dict = {}
        
        if atoms_obj is None or self.atomic_df.empty:
            return features_dict

        # Get unique atomic numbers present (e.g., Ge2O3 -> {32, 8})
        unique_numbers = list(set(atoms_obj.numbers))
        
        # Collect values for each feature
        feature_values = {key: [] for key in self.atomic_feature_keys}
        
        for z in unique_numbers:
            row = self.atomic_df[self.atomic_df['Atomic number'] == z]
            if not row.empty:
                for key in self.atomic_feature_keys:
                    # Get value if column exists
                    if key in row.columns:
                        feature_values[key].append(float(row[key].values[0]))
        
        # Calculate statistics
        for key in self.atomic_feature_keys:
            vals = np.array(feature_values[key])
            
            if len(vals) > 0:
                _max = np.max(vals)
                _min = np.min(vals)
                _mean = np.mean(vals)
                # Mean Absolute Deviation: mean(|x - mean|)
                _mad = np.mean(np.abs(vals - _mean))
                
                features_dict[f'max_atomic_{key}'] = _max
                features_dict[f'min_atomic_{key}'] = _min
                features_dict[f'mean_atomic_{key}'] = _mean
                features_dict[f'mad_atomic_{key}'] = _mad
            else:
                # If data missing for elements, fill None
                for stat in self.stats_suffixes:
                    features_dict[f'{stat}_atomic_{key}'] = None
                
        return features_dict


    def parse_aims_out(self, path, relaxation=True):
        """Attach a calculator to the atoms object."""
        label_path = path
        if relaxation:
            label_path += '/relaxation/calculation/'
        aims_out_path = os.path.join(label_path, 'aims.out')
        aims_dict = ase.io.aims.read_aims_results(aims_out_path)
        # print(aims_dict.keys())
        return aims_dict

    def parse_detailed_energies(self, path, relaxation=True):
        """Parse detailed energy components from the bottom of aims.out.
        
        Reads the file backwards to find the final energy components and 
        orbital data (HOMO, LUMO, Gap) in eV.
        """
        label_path = path
        if relaxation:
            label_path += '/relaxation/calculation/'
        aims_out_path = os.path.join(label_path, 'aims.out')

        energy_dict = {}
        if not os.path.exists(aims_out_path):
            return energy_dict

        # Map string in file to dictionary key
        # Ordered by specificity (longer matches first) to handle "per atom" conflicts
        map_text_to_key = [
            ('Sum of eigenvalues per atom', 'sum_eigenvalues_per_atom'),
            ('Sum of eigenvalues', 'sum_eigenvalues'),
            ('XC energy correction', 'xc_energy_correction'),
            ('XC potential correction', 'xc_potential_correction'),
            ('Free-atom electrostatic energy', 'free_atom_electrostatic_energy'),
            ('Hartree energy correction', 'hartree_energy_correction'),
            ('Entropy correction', 'entropy_correction'),
            ('Total energy (T->0) per atom', 'total_energy_T0_per_atom'),
            ('Total energy, T -> 0', 'total_energy_T0'),
            ('Kinetic energy', 'kinetic_energy'),
            ('Electrostatic energy', 'electrostatic_energy'),
            ('error in Hartree potential', 'multipole_correction'), # Matches the line with the value
            ('Electronic free energy per atom', 'electronic_free_energy_per_atom'),
            # ('Electronic free energy', 'electronic_free_energy'), # Duplicate of free_energy
            ('Highest occupied state (VBM)', 'vbm'),
            ('Lowest unoccupied state (CBM)', 'cbm'),
            ('ESTIMATED overall HOMO-LUMO gap', 'homo_lumo_gap'),
            ('Chemical Potential', 'chemical_potential')
        ]

        try:
            with open(aims_out_path, 'r') as f:
                # Read all lines
                lines = f.readlines()
                
                # Iterate backwards
                for line in reversed(lines):
                    if 'eV' in line:
                        for text, key in map_text_to_key:
                            if text in line:
                                # Only set if not already found (we want the last occurrence in the file, which is the first we see backwards)
                                if key not in energy_dict:
                                    try:
                                        parts = line.split()
                                        # value is usually immediately before 'eV'
                                        ev_index = parts.index('eV')
                                        val = float(parts[ev_index-1])
                                        energy_dict[key] = val
                                    except (ValueError, IndexError):
                                        pass
                                # Break inner loop once we matched the text for this line
                                break 
                    
                    # Stop condition: if we hit the start of a new SCF cycle, stop to prevent reading previous steps
                    if "Begin self-consistency iteration" in line:
                        break
                        
        except Exception as e:
            self.logger.error(f"Error parsing detailed energies in {aims_out_path}: {e}")
            
        return energy_dict

    def get_volume(self, atoms_object):
        """Get the volume of an ASE atoms object."""
        return atoms_object.get_volume()

    def get_num_atoms(self, atoms_object):
        """Use ASE to count # atoms in ASE Atoms Obj."""
        return atoms_object.get_number_of_atoms()

    def read_geometry(self, path, relaxation=False):
        """Collect data from geometry file.

        Args:
        path: to folder containing geometry.in file.
        """
        # Check if path to geometry.in file exists
        if relaxation:
            geometry_file = path + '/relaxation/geometry.in.next_step'
        else:
            geometry_file = path + '/geometry.in'
        if os.path.exists(geometry_file):
            atoms_object = ase.io.aims.read_aims(
                geometry_file)
        else:
            if relaxation:
                print(
                    f'file {path}/relaxation/geometry.in.next_step does not exist.')
                self.logger.error(
                    f'file {path}/relaxation/geometry.in.next_step does not exist.')
            else:
                print(f'file {path}/geometry.in does not exist')
                self.logger.error('geometry input not found')
            # sys.exit('geometry input not found.')
            return None

        return atoms_object

    def read_ase_params(self, path):
        """Read ASE Parameters used in the simulation."""
        ase_param_file = path + 'parameters.ase'
        if os.path.exists(path):
            params = ase.calculators.calculator.Parameters.read(
                ase_param_file)
        else:
            logging.error(
                'ASE Parameters file not found in %s', ase_param_file)
        return params

    def get_gamma_gap(self, calc):
        """Get gap at gamma point."""
        # Store occupany values for each eigenvalue.
        occupation = calc.get_occupations()  # Read occupation at Gamma
        # If the occupation of the last level (index -1 in python)
        # is non-zero that means even the highest energy levels are partially
        # occupied. In this case we don't have a energy gap at the gamma point.
        # since all levels (even the most energetic) are partially occupied.
        # In this case we set gamma_gap to -1 meaning there is no gap.
        # This is becaues all energy levels are filled.
        if occupation[-1] != 0:
            gamma_gap = -1
        else:
            # DTS: Otherwise the gamma gap is the eigenvalue
            # (energy level) of the last occupation level -
            # the eigen value (energy level) of the lowest
            # non occupied level.
            # Get eigennvalues
            eigen = self.get_eigenvalues(calc)
            gamma_gap = -(
                eigen[occupation != 0][-1] - eigen[occupation == 0][0])

        return gamma_gap

    def get_bandgap_data(self, calc):
        """"Get bandgap data from aims calculations."""
        if calc.name != 'aims':
            self.logging.error(
                'This method has not yet been developed for other codes.')
        # DTS: Initalize homo lumo gap to zero.
        HOMO_LUMO_gap = 0
        # Store the occupatancy values for each eigenvalue.
        # occupation = calc.get_occupations()  # Read occupation at Gamma point.
        # Initialize the bandgap to None.
        gap_bandstructure = None
        # DTS: we open out the aims.out file to read data.
        with open(calc.label + 'aims.out', 'r') as aims_output:
            # DTS: we go line by line in the output.out file.
            for line in aims_output.readlines():
                # Grab the aims version.
                if line.rfind('          Version ') != -1:
                    # Take last value of the line, the version.
                    aims_version = split_line(line)[-1]
                # Grab the estimated overall HOMO-LUMO gap.   
                if line.rfind('ESTIMATED overall HOMO-LUMO gap:') != -1:
                    HOMO_LUMO_gap = float(split_line(line)[4])  # VBM CBM Gap
                # Look for line showing energy differences between bands.
                if line.rfind('| Energy difference      :') != -1:
                    # DTS: if the last value of the occupation vector
                    # is non-zero then all energy levels are filled.
                    # This means we don't have a bandgap.
                    if occupation[-1] != 0:
                        gap_bandstructure = -1
                    else:
                        # DTS: Grab the bandstructure gap
                        if split_line(line)[-2] != '****************':
                            gap_bandstructure = float(split_line(line)[-2])

        if gap_bandstructure is None:
            # Check to see if all KS states are occupied.
            if occupation[-1] != 0:
                # If this is true then all states
                # are occupied and there is no orbital
                # that is unoccupied -> LUMO = 0.
                LUMO = 0
                HOMO = -1
            else:
                gap_bandstructure = self.get_band_file_data(calc)
        return HOMO_LUMO_gap, gap_bandstructure

    def get_band_file_data(self, calc):
        """Take a deep look into bandstructure files to get bandgap."""
        band_files = glob.glob(calc.label+'band*')
        gap_bandstructure = 1.0e6
        HOMO = -1.0e6
        LUMO = 1.0e6
        # Go through each band file one at a time.
        # Update the HOMO if we find a larger HOMO.
        # Update LUMO if we find a smaller LUMO.
        for band in band_files:
            # Previously laodtxt was not defined, so I assumed it's np.loadtext.
            data = np.loadtxt(band)[:, 4:]
            HOMO = max(
                HOMO,
                data[:, np.arange(1, len(data[0, :]), 2)][
                    data[:, np.arange(0, len(data[0, :]), 2)] != 0].max())
            if sum(data[:, np.arange(0, len(data[0, :]), 2)] == 0) != 0:
                LUMO = min(LUMO, data[:, np.arange(1, len(data[0, :]), 2)][
                    data[:, np.arange(0, len(data[0, :]), 2)] == 0].min())

        gap_bandstructure = LUMO-HOMO
        return gap_bandstructure

    @staticmethod
    def parse_path(path, ICSD_number=True, expansion=False,
                   bandstructure_calculation=False):
        """Parse data from file path.

        This method takes in a path and splits
        the path based on forward slashes. It works
        backwards (right to left) and assigns values
        to what settings were used for data contained
        in the path.

        Args:
        path: (str) full pathname to folder where calc output is stored.

        Returns:
        settings_dict: (dict) dictionary that contains data to be saved.
        """
        # Split the path name based on forward slashes
        if bandstructure_calculation is True:
            # Then we need to remove one folder from the path since it's too
            # long.
            path = os.path.dirname(path)
        list_of_settings = path.split('/')
        setting_dict = {}
        try:
            if ICSD_number:
                setting_dict['compound_name'] = list_of_settings[-2].split('_')[0]
                setting_dict['ICSD_number'] = int(list_of_settings[-2].split('_')[-1])
            else:
                setting_dict['compound_name'] = list_of_settings[-2]
            setting_dict['k_point_density'] = int(list_of_settings[-3])
            setting_dict['rel_setting'] = list_of_settings[-4]
            setting_dict['basis_size'] = list_of_settings[-5]
            setting_dict['num_setting'] = list_of_settings[-6]
            setting_dict['functional'] = list_of_settings[-7]
            if expansion:
                setting_dict['expansion'] = list_of_settings[-8]
        except IndexError:
            # Static method doesn't can't access private member
            # variable.
            print('path: %s, is not properly formatted', path)
            sys.exit('path not formated correctly')
        return setting_dict

    def gather_all_path_data(self, submission_path):
        """Parse all data contained in path.

        Go to the path folder and get data from the
        files living there.

        Args:
        path: (str) path to the submission script that was
            used to submit simulation.

        Returns:
        data_dict: (dict) containing data that was parsed from
            files living in the path.
        ase_atoms_obj: (ASE Atoms Object) contains structural data
            used in the simulation. Useful for storing in an ASE db.        
        """
        # If we are running an aims bandstructure calc we are not performing
        # a relaxation.
        relaxation = not self.bandstructure_calculations
        # Save the time and day for later use to know when
        # a row was added to a database/csv.
        time_and_day = get_time_and_day()
        # Get information related to settings from the
        # submission script path.
        data_dict = self.parse_path(
            submission_path, bandstructure_calculation=self.bandstructure_calculations)
        data_dict['relaxation'] = relaxation
        # Ok, now remove the last part of the path since
        # the /submission_XY.sh is not useful. Let's take
        # the parent directory.
        parent_path = os.path.dirname(submission_path)
        # Check if the simulation was even started.
        if self.check_if_aims_out_exists(
            parent_path,
            bandstructure_calculation=self.bandstructure_calculations):
            # Then check if this is an issue related to symmetry.
            # Check for djob.err file
            most_recent_error_path = OutputParser.get_most_recent_djob_err(
                parent_path)
            if most_recent_error_path is None:
                self.add_out_of_npl_path(submission_path)
                print(f'path: {submission_path} added as out of npl file.')
            elif self.check_for_symmetry_issue(
                    parent_path, most_recent_error_path):
                print(f'path: {submission_path} added as symmetry issue.')
                self.add_symmetry_issue_path(submission_path)
            else:
                print(f'path: {submission_path} added as misbehaving. No aims.out')
                self.add_misbehaving_path(submission_path)

            return None, None, None

        # Check if sim/relaxation finished, if not if time expired.
        if relaxation:
            (
                calc_finished_bool, calc_expired_bool,
                out_of_npl_bool, scf_bool) = self.check_relaxation_finished(
                    parent_path)
        else:
            (
                calc_finished_bool, calc_expired_bool,
                out_of_npl_bool, scf_bool) = self.check_calc_finished(
                    parent_path)
        
        if (
                calc_finished_bool is False and
                calc_expired_bool is True and
                scf_bool is False):
            # Comment this out since on HLRN we run for 24 hrs
            # for all non-light runs.
            # queue = self.get_queue(submission_path)
            # if queue == 'general':
            self.add_increase_charge_mix_path(submission_path)
            print(f'path: {submission_path} expired due to a time limit')
            # else:
            #     self.add_expired_path(submission_path)
            return None, None, None
        elif calc_finished_bool is False and out_of_npl_bool is True:
            # This means the job was cancelled due to being out of NPL.
            # This is a job we want to resubmit.
            self.add_expired_path(submission_path)
            print(f'path: {submission_path} out of npl since it was cancelled.')
            return None, None, None
        elif scf_bool is True:
            # We used to check if calc was finishd but this tells us nothing
            # SCF can not converge and have a nice day will still be printed.
            self.add_decrease_charge_mix_path(
                submission_path)
            print(
                f'path: {submission_path} calc '
                'finished but scf did not converge.')
            return None, None, None
            # If calc didn't finish and time didn't expire, then
            # we add it to a list of misbehaving paths.

        elif calc_finished_bool is True:
            # We should be able to parse the files.
            aims_dict = self.parse_aims_out(
                path=parent_path, relaxation=relaxation)
            # print(f'aims_dict is {aims_dict}')
            # Get Energy.
            if 'total_energy' in aims_dict:
                data_dict['total_energy'] = aims_dict['total_energy']
            elif 'energy' in aims_dict:
                data_dict['total_energy'] = aims_dict['energy']
            else:
                print('Cannot find the energy or total energy')
                self.add_misbehaving_path(submission_path)
                raise ValueError('Cannot find the energy or total energy')
            
            # Rename free_energy to avoid ASE DB reserved key conflict
            data_dict['aims_free_energy'] = aims_dict['free_energy']
            
            # --- NEW: Parse Detailed Energies ---
            detailed_energies = self.parse_detailed_energies(parent_path, relaxation=relaxation)
            data_dict.update(detailed_energies)

            if self.bandstructure_calculations:
                # TODO: need to edit this. Then add the information about the band gap:
                pass
            
            # # Add time/day when this row of data was grabbed.
            data_dict['time_day'] = time_and_day
            # # Add the path from which data was taken.
            data_dict['path'] = submission_path
            data_dict['binary_precision'] = js.get_aims_precision_level(
                    data_dict['basis_size'], data_dict['num_setting'])

            # --- Parse Original (Input) Geometry ---
            og_ase_atoms_obj = self.read_geometry(
                    parent_path, relaxation=False)

            if og_ase_atoms_obj is not None:
                # --- CALCULATE STATISTICAL ATOMIC FEATURES ---
                atomic_feats = self.get_atomic_features(og_ase_atoms_obj)
                data_dict.update(atomic_feats)

                # --- NEW: CALCULATE STATISTICAL MONOMER FEATURES ---
                # Uses binary_precision and k_point_density from data_dict
                monomer_feats = self.get_monomer_features(
                    og_ase_atoms_obj, 
                    precision_level=data_dict.get('binary_precision', 0),
                    k_point_density=data_dict.get('k_point_density', 0)
                )
                data_dict.update(monomer_feats)

                # 4. NEW: Valence Electron Features
                valence_feats = self.get_valence_features(og_ase_atoms_obj)
                data_dict.update(valence_feats)

                # 5. NEW: Basis Function Features
                basis_feats = self.get_basis_features(
                    og_ase_atoms_obj,
                    binary_precision=data_dict.get('binary_precision')
                )
                data_dict.update(basis_feats)

                data_dict['original_volume'] = self.get_volume(og_ase_atoms_obj)
                
                # Calc energy per atom using original number of atoms
                data_dict['total_energy_per_atom'] = data_dict['total_energy']/len(
                    og_ase_atoms_obj.numbers)
                
                # Original Position Matrix
                data_dict['original_atom_positions'] = np.array2string(
                    og_ase_atoms_obj.get_positions(), separator=',',
                    formatter={'float_kind': lambda x: "%.16f" % x}).replace('\n', '')

                # Original Cell Matrix
                data_dict['original_cell'] = np.array2string(
                    og_ase_atoms_obj.get_cell(), separator=',',
                    formatter={'float_kind': lambda x: "%.16f" % x}).replace('\n', '')
                
                # Updated to use cellpar() to avoid DeprecationWarning
                cell_lengths_angles_og = og_ase_atoms_obj.cell.cellpar()
                data_dict['original_a_len'] = cell_lengths_angles_og[0]
                data_dict['original_b_len'] = cell_lengths_angles_og[1]
                data_dict['original_c_len'] = cell_lengths_angles_og[2]
                data_dict['original_alpha_angle'] = cell_lengths_angles_og[3]
                data_dict['original_beta_angle'] = cell_lengths_angles_og[4]
                data_dict['original_gamma_angle'] = cell_lengths_angles_og[5]
            else:
                # Set Nones if file missing
                data_dict['original_volume'] = None
                data_dict['total_energy_per_atom'] = None
                data_dict['original_atom_positions'] = None
                data_dict['original_cell'] = None
                data_dict['original_a_len'] = None
                data_dict['original_b_len'] = None
                data_dict['original_c_len'] = None
                data_dict['original_alpha_angle'] = None
                data_dict['original_beta_angle'] = None
                data_dict['original_gamma_angle'] = None


            # --- Parse Final (Relaxed) Geometry ---
            relaxed_ase_atoms_obj = self.read_geometry(
                parent_path, relaxation=relaxation)
            
            if relaxed_ase_atoms_obj is not None:
                data_dict['final_volume'] = self.get_volume(relaxed_ase_atoms_obj)
                data_dict['chem_formula'] = relaxed_ase_atoms_obj.get_chemical_formula()
                
                # Updated to use cellpar() to avoid DeprecationWarning
                cell_lengths_angles = relaxed_ase_atoms_obj.cell.cellpar()
                data_dict['relaxed_a_len'] = cell_lengths_angles[0]
                data_dict['relaxed_b_len'] = cell_lengths_angles[1]
                data_dict['relaxed_c_len'] = cell_lengths_angles[2]
                data_dict['relaxed_alpha_angle'] = cell_lengths_angles[3]
                data_dict['relaxed_beta_angle'] = cell_lengths_angles[4]
                data_dict['relaxed_gamma_angle'] = cell_lengths_angles[5]

                # Relaxed Position Matrix
                data_dict['relaxed_atom_positions'] = np.array2string(
                    relaxed_ase_atoms_obj.get_positions(), separator=',',
                    formatter={'float_kind': lambda x: "%.16f" % x}).replace('\n', '')
                
                # Relaxed Cell Matrix
                data_dict['relaxed_cell'] = np.array2string(
                    relaxed_ase_atoms_obj.get_cell(), separator=',',
                    formatter={'float_kind': lambda x: "%.16f" % x}).replace('\n', '')

            else:
                data_dict['final_volume'] = None
                data_dict['chem_formula'] = None
                data_dict['relaxed_a_len'] = None
                data_dict['relaxed_b_len'] = None
                data_dict['relaxed_c_len'] = None
                data_dict['relaxed_alpha_angle'] = None
                data_dict['relaxed_beta_angle'] = None
                data_dict['relaxed_gamma_angle'] = None
                data_dict['relaxed_atom_positions'] = None
                data_dict['relaxed_cell'] = None

            
            # --- Update UID Counter ---
            # We increment this here because we have successfully gathered data
            self.uid_counter += 1
            data_dict['uid'] = self.uid_counter
            
            self.add_finished_correctly_path(submission_path)
            
            # Return all three
            return data_dict, og_ase_atoms_obj, relaxed_ase_atoms_obj

        else:
            # This catches jobs that didn't finish, didn't expire, and didn't match
            # specific error patterns (e.g., immediate crashes/segfaults).
            self.add_misbehaving_path(submission_path)
            print(f'path: {submission_path} fell through logic gaps (likely crashed silently).')
            return None, None, None
s

    def write_to_csv(self, data_dict):
        """Appends a row of parsed data to the CSV file."""
        # Check for missing keys and fill with None
        csv_row = []
        for col in self.csv_columns:
            if col not in data_dict:
                # self.logger.warning(f"Key {col} missing for UID {data_dict.get('uid', 'Unknown')}")
                csv_row.append(None)
            else:
                csv_row.append(data_dict[col])

        try:
            with open(self.csv_filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(csv_row)
        except Exception as e:
            self.logger.error(f"Failed to write to CSV: {e}")

    def check_if_aims_out_exists(
            self, submission_path, bandstructure_calculation=False):
        """Check if the simulation never ran."""
        if bandstructure_calculation is True:
            aims_out_path = os.path.join(
                submission_path, 'aims.out')
            # print(f'aims_out_path: {aims_out_path}')
        else:
            aims_out_path = submission_path + '/relaxation/calculation/aims.out'
        
        if os.path.isfile(aims_out_path):
            return False
        else:
            return True

    def get_queue(self, submission_path):
        """Get queue information from the submission path script.

        Args:
        submissions_path: (str) simulation submission script.

        Returns:
        queue: (str) which queue the script was submitted to -
            general/short."""
        queue = 'None'
        with open(submission_path) as f:
            script_text = f.read()
            if '#SBATCH --partition=short' in script_text:
                queue = 'short'
            elif '#SBATCH --partition=general' in script_text:
                queue = 'general'
            else:
                self.logger.error(
                    'Unable to find queue in sub'
                    'script %s' % submission_path)
                print('Unable to find queue info %s' % submission_path)
        return queue

    def add_expired_path(self, submission_path):
        """Add a path name where a simulation expired to due to being out of time.

        Args:
        path: (str) path name to submission script so that it
            can be resubmitted.
        """
        with open(
                self.paths_to_resubmit, 'a') as fo:
            fo.writelines(submission_path + '\n')

    def add_increase_charge_mix_path(self, submission_path):
        """Add a path name for sim where SCF didnt converge.
        Args:
        path: (str) path name to submission script so that it
            can be resubmitted.
        """
        with open(
                self.paths_increase_charge_mix, 'a') as fo:
            fo.writelines(submission_path + '\n')

    def add_finished_correctly_path(self, submission_path):
        """Add a path name where the job finished correctly.

        Args:
        path: (str) path name to submission script so that it
            can be resubmitted.
        """
        with open(
                self.paths_finished_correctly, 'a') as fo:
            fo.writelines(submission_path + '\n')  

    def add_out_of_npl_path(self, submission_path):
        """Add a path name for sim where sim didn't run due to out of npl.
        Args:
        path: (str) path name to submission script so that it
            can be resubmitted.
        """
        with open(
                self.paths_out_of_npl, 'a') as fo:
            fo.writelines(submission_path + '\n')

    def add_symmetry_issue_path(self, submission_path):
        """Add a path name for sim where a symmetry issue was seen.
        Args:
        path: (str) path name to submission script so that it
            can be resubmitted.
        """
        with open(
                self.paths_symmetry_issues, 'a') as fo:
            fo.writelines(submission_path + '\n')

    def add_decrease_charge_mix_path(self, submission_path):
        """Add a path name for sim where sim took longer than 24h.
        Args:
        path: (str) path name to submission script so that it
            can be resubmitted.
        """
        with open(
                self.paths_decrease_charge_mix, 'a') as fo:
            fo.writelines(submission_path + '\n')

    def add_misbehaving_path(self, submission_path):
        """Add a path name where a simulation didn't end nicely.

        Args:
        path: (str) path name to submission script so that it
            can be resubmitted.
        """
        with open(
                self.paths_misbehaving, 'a') as fo:
            fo.writelines(submission_path + '\n')

    @staticmethod
    def check_for_symmetry_issue(parent_path, most_recent_error_path):
        """Check the most recent error path for a symmetry issue.

        Args:
            most_recent_error_path: (str) path to the most recent error file.

        Returns:
            True if there was a symmetry issue in the error file path file.
        """
        with open(
                os.path.join(parent_path, most_recent_error_path), 'r') as fin:
            for line in reversed(list(fin)):
                if "std_cell = dataset['std_lattice']" in line:
                    return True
        return False

    @staticmethod
    def check_relaxation_finished(parent_path):
        """Check if simulation exited nicely.

        We look line by line (from bottom of aims.out)
        if 'have a nice day' is there. If it is
        we return True. We continue looking for 50 lines
        and return False if we haven't found it by that
        point.

        Args:
            path: (str) path to folder containing aims.out.

        Returns:
            calc_finished_bool: (bool) True if calc exited
            nicely.
            expired_time_bool: (bool) True if sim ran out of time.
            scf_bool: (bool) True if the sim's SCF cycle didn't
                converge.
        """
        aims_output_file = parent_path + '/relaxation/calculation/aims.out'
        relaxation_file = parent_path + '/relaxation.log'
        line_num = 0
        calc_finished_bool = False
        scf_bool = False
        out_of_npl_bool = False
        # with open(aims_output_file, 'r') as fd:
        for line in reversed(list(open(aims_output_file))):
            # if "Have a nice day." in line:
            #     calc_finished_bool = True
            #     break
            if "scf_solver: SCF cycle not converged." in line:
                scf_bool = True
                break
            line_num += 1
            if line_num > 50:
                break

        line_num = 0
        for line in reversed(list(open(relaxation_file))):
            if 'Relaxation converged.' in line:
                calc_finished_bool = True
                break
            line_num += 1
            if line_num > 20:
                break

        expired_time_bool = False
        if (calc_finished_bool or scf_bool) is not True:
            # Then we need to take a look at the error jobs.
            # Call it with OutputParser namespace since
            # this method is static method and the method
            # we want to call is also a static method.
            expired_time_bool, out_of_npl_bool = OutputParser.check_sim_time_lim(
                parent_path)
        return calc_finished_bool, expired_time_bool, out_of_npl_bool, scf_bool


    @staticmethod
    def check_calc_finished(parent_path):
        """Check if simulation exited nicely.

        We look line by line (from bottom of aims.out)
        if 'have a nice day' is there. If it is
        we return True. We continue looking for 50 lines
        and return False if we haven't found it by that
        point.

        Args:
            path: (str) path to folder containing aims.out.

        Returns:
            calc_finished_bool: (bool) True if calc exited
            nicely.
            expired_time_bool: (bool) True if sim ran out of time.
            scf_bool: (bool) True if the sim's SCF cycle didn't
                converge.
        """
        aims_output_file = os.path.join(parent_path, 'aims.out')
        line_num = 0
        calc_finished_bool = False
        scf_bool = False
        out_of_npl_bool = False
        # with open(aims_output_file, 'r') as fd:
        for line in reversed(list(open(aims_output_file))):
            if "Have a nice day." in line:
                calc_finished_bool = True
                break
            if "scf_solver: SCF cycle not converged." in line:
                scf_bool = True
                break
            line_num += 1
            if line_num > 50:
                break

        expired_time_bool = False
        if (calc_finished_bool or scf_bool) is not True:
            # Then we need to take a look at the error jobs.
            # Call it with OutputParser namespace since
            # this method is static method and the method
            # we want to call is also a static method.
            expired_time_bool, out_of_npl_bool = OutputParser.check_sim_time_lim(
                parent_path)
        return calc_finished_bool, expired_time_bool, out_of_npl_bool, scf_bool

    @staticmethod
    def get_most_recent_djob_err(path):
        """Get the most recent job error path for a simulation.

        Args:
        path: (str) string to the submission script path.

        Returns:
        djob_path: (str) string to the most recent djob error path.
        """
        # Get list of files in the folder where djob.err would
        # live (aims.out as well).
        file_list = os.listdir(path)
        # Now we need to check for djob.err type files.
        djob_err_list = [x for x in file_list if 'djob.err.' in x]
        # Check if the list is empty
        if not djob_err_list:
            return None
        # Now we need to choose the most recent djob err file. Let's
        # look at which has the largest integer after splitting the
        # file name by . and looking at last value. We first
        # list of integers of name of Jobs.
        djob_int_list = [x.split('.')[-1] for x in djob_err_list]
        # Now find the index with the largest integer.
        most_recent_job_num = djob_int_list.index(max(djob_int_list))
        # Use the index to get the djob err path name.
        most_recent_djob_err_path = djob_err_list[most_recent_job_num]

        return most_recent_djob_err_path

    @staticmethod
    def check_sim_time_lim(path):
        """If time expired on sim, it returns True.

        If the sim didn't have a Have a nice day in aims.out
        then this method should be called to determine
        if the sim simply ran out of time because it was submitted
        to a queue that let it run only for a short amount of time.

        First we check for any file names with /djob.err.* in our folder.
        We do so by first getting a list of all files in our folder.
        Then looking if there's a match with the type. Then we choose
        the djob.err with the largest #. We look in this djob.err
        for markers CALCELLED and TIME LIMIT EXPIRED. If we find
        them we return true.

        Args:
        path: (str) path to folder containing aims.out.

        Returns:
        expired_time_bool: (bool) True if the sim ran out of time.
            False otherwise.
        """

        # Now go through each line in the most recent djob.err.*
        # file in the folder and see if we can spot the marker.
        first_marker = 'CANCELLED'
        second_marker = 'DUE TO TIME LIMIT'
        # By default the bool we return is False since we
        # haven't seen markers.
        expired_time_bool = False
        out_of_npl_bool = False
        # Get the path to most recent djob error.
        most_recent_djob_err = OutputParser.get_most_recent_djob_err(path)
        if most_recent_djob_err is not None:
            # Print path of most recent djob error.
            print('Most Recent Djob Err: %s' % most_recent_djob_err)
            for line in reversed(list(open(path + '/' + most_recent_djob_err))):
                print(line)
                if first_marker in line and second_marker not in line:
                    out_of_npl_bool = True
                if first_marker in line and second_marker in line:
                    expired_time_bool = True
                    break
        return expired_time_bool, out_of_npl_bool


    def write_all_path_data(self):
        """Write all data from path list to an ASE database.

        Connects to the database specified in self.db_path.
        For each path, it parses the data and writes a new row containing:
        - The final (relaxed) Atoms object.
        - All scalar data (energy, volume, parameters) in key_value_pairs.
        - The original (unrelaxed) Atoms object in the 'data' field.
        """
        # Connect to the ASE database
        with ase.db.connect(self.db_path) as db:
            print(f"Connected to database: {self.db_path}")

            for path in self.path_list:
                try:
                    # Get data stored in files in the path folder.
                    # Capture all three return values.
                    (
                        data_dict, 
                        og_ase_atoms_obj, 
                        relaxed_ase_atoms_obj
                    ) = self.gather_all_path_data(path)

                    # If parsing failed, data_dict or the atoms objects will be None.
                    # The error is already logged by gather_all_path_data.
                    if data_dict is None or relaxed_ase_atoms_obj is None or og_ase_atoms_obj is None:
                        print(f"Skipping path (no valid data): {path}")
                        continue
                    
                    # Write to CSV
                    self.write_to_csv(data_dict)

                    # Write the data to the database
                    # The main 'atoms' entry is the final relaxed structure
                    # All scalar metadata is saved in key_value_pairs
                    # The original structure is saved in the 'data' blob
                    # The UID is now included in data_dict
                    db.write(
                        atoms=og_ase_atoms_obj,
                        key_value_pairs=data_dict,
                        data={'relaxed_structure': relaxed_ase_atoms_obj}
                    )
                    
                    print(f"Successfully added to DB: {path}")

                except Exception as e:
                    print(f"This path: {path} had issues")
                    print(f"Error: {e}")
                    self.logger.error(
                        f"This path: {path} gives this error: {e}")


def split_line(lines):
    """Split input line"""
    # Strip() removes leading and trailing whitespace
    # Then we split on the whitespace between words.
    # Store as a numpy array.
    line_array = np.array(lines.strip().split(' '))
    # Remove any elements that might be empty strings
    # in the array.
    line_vals = line_array[line_array != '']
    # Return the line values.
    return line_vals


def get_time_and_day():
    """Return the time and day now as string.

    Returns:
    time_day_str: (str) time and day in a string.
    """
    now = datetime.now()

    # dd/mm/YY H:M:S
    dt_string = now.strftime("%H_%M_%S__%d_%m_%Y")
    return dt_string


def convert_basis_set_2_int(basis_set):
    if basis_set == 'minimal': num_int = 0
    elif basis_set == 'tier1': num_int = 1
    elif basis_set == 'standard': num_int = 2
    elif basis_set == 'tier2': num_int = 3
    else: num_int = None # Handle errors gracefully
    return num_int

def convert_num_setting_2_int(numerical_setting):
    if numerical_setting == 'light': num_int = 1
    elif numerical_setting == 'tight': num_int = 2
    elif numerical_setting == 'really_tight': num_int = 3
    else: num_int = None
    return num_int

def get_aims_basis_set_size(binary_precision):
    """Maps integer precision to (numerical_setting, basis_set_size)."""
    if binary_precision == 11:
        basis_set_size = 2
        numerical_setting = 'really_tight'
    elif binary_precision == 10:
        basis_set_size = 1
        numerical_setting = 'really_tight'
    elif binary_precision == 9:
        basis_set_size = 'standard'
        numerical_setting = 'really_tight'
    elif binary_precision == 8:
        basis_set_size = 0
        numerical_setting = 'really_tight'
    elif binary_precision == 7:
        basis_set_size = 2
        numerical_setting = 'tight'
    elif binary_precision == 6:
        basis_set_size = 1
        numerical_setting = 'tight'
    elif binary_precision == 5:
        basis_set_size = 'standard'
        numerical_setting = 'tight'
    elif binary_precision == 4:
        basis_set_size = 0
        numerical_setting = 'tight'
    elif binary_precision == 3:
        basis_set_size = 2
        numerical_setting = 'light'
    elif binary_precision == 2:
        basis_set_size = 1
        numerical_setting = 'light'
    elif binary_precision == 1:
        basis_set_size = 'standard'
        numerical_setting = 'light'
    elif binary_precision == 0:
        basis_set_size = 0
        numerical_setting = 'light'
    else:
        return [None, None] 
    return [numerical_setting, basis_set_size]


def main(argv):
    """Main fxn to allow us to call this method directly."""
    input_paths_txt_file = FLAGS.input_paths_txt_file
    if input_paths_txt_file == 'None':
        sys.exit('input_path_txt_files is None / not specified')
    print(f'paths txt file is {input_paths_txt_file}')
    save_directory = FLAGS.save_directory
    if save_directory == 'None':
        sys.exit('save_directoryname is None / not specified')
    parse_obj = OutputParser(
        input_paths_txt_file=input_paths_txt_file,
        save_directory=save_directory,
        bandstructure_calculations=FLAGS.bandstructure_calculations,
        atomic_data_path=FLAGS.atomic_data_path,
        monomers_data_path=FLAGS.monomers_data_path,
        basis_dict_path=FLAGS.basis_dict_path)
    # Now submit all jobs.
    parse_obj.write_all_path_data()


if __name__ == '__main__':
    app.run(main)