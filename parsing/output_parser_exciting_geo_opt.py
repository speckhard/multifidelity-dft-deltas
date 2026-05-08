"""Parse output files from DFT simulations.

Scripts in this file are used to parse data output
and save it to a json like structure. We then save
each json like object to an ASE database.

Eventually, we would like to use the aims parser."""
import os
import sys
import csv
import logging
from pathlib import Path
from datetime import datetime

from absl import flags
from absl import app

from excitingtools.exciting_dict_parsers.groundstate_parser import parse_info_out
from excitingtools.input.structure import ExcitingStructure

from excitingtools.exciting_dict_parsers.input_parser import parse_structure
from excitingtools.structure.ase_utilities import exciting_structure_to_ase

import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


FLAGS = flags.FLAGS
flags.DEFINE_string(
    'input_paths_txt_file', 'None',
    'Where to find list of paths to parse.')
flags.DEFINE_string(
    'save_directory',
    'None',
    'Where to save the files/folders.')


class OutputParserExciting():
    """Parses data output."""
    def __init__(
            self, input_paths_txt_file, save_directory):
        """Constructor

        Args:
        input_path_txt_file: (str) of paths to be parsed where DFT output files
            can be found.
        save_directory: (str) name of directory where to save parsed CSV, and
            list of paths to be resubmitted or that had issues with parsing.
        """
        # Normally a file containing paths on new lines is given
        # and not a list of paths.
        self.path_list = self.get_path_list(input_paths_txt_file)
        # self.atoms_obj_list = atoms_obj_list
        self.logger = logging.getLogger(__name__)

        # Name of columns for header in csv.
        self.csv_columns = [
            'compound_name',
            'APW_precision',
            'k_point_density',
            'basis_size',
            'functional',
            'status',
            'path',
            'num_empty_states',
            'total_energy',
            'k_point_total',
            'total_num_los',
            'total_comp_time',
            'chem_formula',
            # 'species_files',
            # 'hartree_energy',
            'num_apw_functions',
            'rgkmax',
            'rmt_scaling',
            'band_gap',
            'unit_cell_volume',
            'parsing_time_day',
            'relaxed_atom_positions',
            'relaxed_cell',
            'relaxed_volume',
            'relaxed_a_len',
            'relaxed_b_len',
            'relaxed_c_len',
            'relaxed_alpha_angle',
            'relaxed_beta_angle',
            'relaxed_gamma_angle'
]
        self.save_directory = Path(save_directory)
        # CSV filename of where to save parsed data.
        self.parsing_time_and_day_date = get_time_and_day()
        self.csv_filename = self.save_directory / ('parsed_csv_' + self.parsing_time_and_day_date + '.csv')
        # Define a list of paths that we shoudl resubmit
        # to a longer queue since they expired during calculation.
        self.paths_to_resubmit = self.save_directory / ('paths_to_resubmit_' + self.parsing_time_and_day_date + '.txt')

        self.paths_misbehaving = self.save_directory / ('paths_misbehaving_' + self.parsing_time_and_day_date + '.txt')
        # Paths where the SCF didn't converge. For these we want
        # to submit with larger charge mix param.
        self.paths_increase_charge_mix = self.save_directory / ('paths_increase_charge_mix_' + self.parsing_time_and_day_date + '.txt')
        # Paths where the simulation ran out of time after 24 hours
        # in the general queue.
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
        self.paths_out_of_memory = self.save_directory / ('paths_out_of_memory_' + self.parsing_time_and_day_date + '.txt')
        self.paths_finished_correclty = self.save_directory / ('paths_finished_correctly_' + self.parsing_time_and_day_date + '.txt')

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

    def create_header(self):
        """Create csv with header."""
        try:
            with open(self.csv_filename, 'w') as csvfile:
                writer = csv.DictWriter(
                    csvfile, fieldnames=self.csv_columns)
                # Write header.
                writer.writeheader()
        except IOError:
            print("I/O error")
            self.logger.error("I/O error writing header to csv file.")
            sys.exit('I/O issues writing header to csv file.')


    @staticmethod
    def parse_path(parent_path):
        """Parse data from file path.

        Ex. path:
        'tests/exciting/sven_regen/old_species_path/GGA_PBE/'
        'precision_0_3/rmt_scaling_0_95/8/Be4S4')
        """
        # Split the path name based on forward slashes
        list_of_settings = parent_path.split('/')
        setting_dict = {}
        try:
            setting_dict['compound_name'] = list_of_settings[-2]
            setting_dict['k_point_density'] = int(list_of_settings[-3])
            rmt_scaling_string = list_of_settings[-4]
            setting_dict['rmt_scaling'] = float(
                rmt_scaling_string.split('_')[-2] + '.' + rmt_scaling_string.split(
                    '_')[-1])
            apw_precision_string = list_of_settings[-5]
            setting_dict['APW_precision'] = float(
                apw_precision_string.split('_')[-2] + '.' + apw_precision_string.split(
                    '_')[-1])
            setting_dict['functional'] = list_of_settings[-6]

        except IndexError:
            # Static method doesn't can't access private member
            # variable.
            print('path: %s, is not properly formatted', parent_path)
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
        """
        # Save the time and day for later use to know when
        # a row was added to a database/csv.
        time_and_day = get_time_and_day()
        # Get information related to settings from the
        # submission script path.
        data_dict = self.parse_path(submission_path)
        # Add time/day when this row of data was grabbed.
        data_dict['status'] = 'unknown'
        data_dict['parsing_time_day'] = time_and_day
        # Add the path from which data was taken.
        data_dict['path'] = submission_path
        # Ok, now remove the last part of the path since
        # the /submission_XY.sh is not useful. Let's take
        # the parent directory.
        parent_path = os.path.dirname(submission_path)
        # Check if the simulation was even started.
        if self.check_if_exciting_info_out_does_not_exist(parent_path):
            # Check for djob.err file
            most_recent_error_path = OutputParserExciting.get_most_recent_djob_err(
                parent_path)
            if most_recent_error_path is None:
                self.add_out_of_npl_path(submission_path)
                print(f'path: {submission_path} added as out of npl file.')
            
            else:
                species_xml_missing_bool, scalar_to_integer_bool = OutputParserExciting.check_no_info_out_err_file(
                    parent_path)
                if species_xml_missing_bool is True:
                    self.add_species_xml_missing_path(submission_path)
                    print(f'path: {submission_path} failed due to missing species file error.')
                    data_dict['status'] = 'missing_species_file'
                elif scalar_to_integer_bool is True:
                    self.add_scalar_to_integer_path(submission_path)
                    print(f'path: {submission_path} failed due to scalar to integer issue.')
                    data_dict['status'] = 'scalar_to_int'

                elif OutputParserExciting.check_state_out_read_issue(submission_path):
                    # Then we need to add this path to resubmit_from_scratch list:
                    self.add_resubmit_from_scratch_path(submission_path)
                    data_dict['status'] = 'state_out_does_not_exist'
                else:
                    print(f'path: {submission_path} added as misbehaving.')
                    self.add_misbehaving_path(submission_path)
                    data_dict['status'] = 'no_exciting_info_out_misbehaving'
            return data_dict

        # Check if sim finished, if not if time expired.
        (
            calc_finished_bool, geo_opt_finished_bool,
            geo_opt_started_bool, gs_started, expired_time_bool,
            out_of_npl_bool, diag_fail_bool, mt_overlap_bool, calc_ongoing_bool,
            ndirac_limit_bool, oom_bool, total_comp_time) = self.check_calc_finished(
                parent_path)
        
        data_dict['total_comp_time'] = total_comp_time

        # Check if the geo opt file exists if geo_opt_finished_bool is True
        # Check for the opt file, if it doesnt exist than the maximum force
        # target was achieved at the initial configuration.
        geo_opt_path_bool = False
        if geo_opt_finished_bool is True:
            geo_opt_path = os.path.join(
                    parent_path, 'geometry_opt.xml')
            geo_opt_path_bool = os.path.isfile(geo_opt_path)
            if geo_opt_path_bool is False:
                geo_opt_path = os.path.join(
                    parent_path, 'geometry.xml')
                geo_opt_path_bool = os.path.isfile(geo_opt_path)

        if diag_fail_bool is True:
            self.add_diag_fail_path(submission_path)
            print(f'path: {submission_path} failed due to diagonalization error.')
            data_dict['status'] = 'diagonalization_error'

            return data_dict

        if oom_bool is True:
            self.add_oom_path(submission_path)
            print(f'path: {submission_path} failed due to being out of memory.') 
            data_dict['status'] = 'out_of_memory'

            return data_dict            
        if mt_overlap_bool is True:
            self.add_mt_overlap_path(submission_path)
            print(f'path: {submission_path} failed due to muffin tin overlap issue.') 
            data_dict['status'] = 'muffin_tin_overlap'

            return data_dict

        if ndirac_limit_bool is True:
            self.add_ndirac_limit_path(submission_path)
            print(f'path: {submission_path} failed due to running into ndiract limit.') 
            data_dict['status'] = 'ndirac_limit'
            return data_dict

        if expired_time_bool is True:
            # The simulation ran out of time.
            if gs_started is False:
                self.add_gs_not_started_path(submission_path)
                data_dict['status'] = 'ground_state_not_started'

            elif geo_opt_started_bool is False:
                self.add_geo_opt_not_started_path(submission_path)
                data_dict['status'] = 'geometry_optimization_not_started'

            elif geo_opt_finished_bool is False:
                self.add_expired_time_path(submission_path)
                data_dict['status'] = 'geometry_optimization_not_finished'

            else:
                print(f'path: {submission_path} added as misbehaving.')
                self.add_misbehaving_path(submission_path)
                data_dict['status'] = 'expired_unknown_logic_misbehaving'
            return data_dict
        
        elif out_of_npl_bool is True:
            self.add_out_of_npl_path(submission_path)
            data_dict['status'] = 'out_of_computing_credits'
            return data_dict
        elif calc_ongoing_bool is True:
            self.add_ongoing_path(submission_path)
            data_dict['status'] = 'ongoing_path'
            return data_dict
        elif calc_finished_bool is True and geo_opt_path_bool is False:
            print(f'path: {submission_path} says calc finished but there is no'
                  f' geo opt path: {geo_opt_path_bool}.')
            data_dict['status'] = 'finished_but_no_geo_opt_path_misbehaving'
            self.add_misbehaving_path(submission_path)
            return data_dict
        elif calc_finished_bool is True:
            data_dict['status'] = 'calc_finished_before_parsing'
            results_dict = parse_info_out(os.path.join(
                parent_path, 'INFO.OUT'))
            ase_atoms_obj_opt = OutputParserExciting.get_opt_ase_object(geo_opt_path)
            # print(f'results dict is {results_dict}')
            # print(os.path.join(
            #     parent_path, 'INFO.OUT'))
            # print(results_dict)
            if results_dict["scl"]:
                final_scl_iteration = list(results_dict["scl"].keys())[-1]

                data_dict['total_energy'] = float(results_dict["scl"][final_scl_iteration][
                        "Total energy"])
                # Exciting won't output the gap if during the first scf cycles it notices
                # that there is a metal.
                if "Estimated fundamental gap" in results_dict["scl"][final_scl_iteration]:
                    data_dict['band_gap'] = float(results_dict["scl"][final_scl_iteration][
                            "Estimated fundamental gap"])
                else:
                    data_dict['band_gap'] = 'not_found'
            else:
                data_dict['band_gap'] = 'not_found'
                data_dict['total_energy'] = 'not_found'


            data_dict['unit_cell_volume'] = float(
                results_dict['initialization']['Unit cell volume'])
            data_dict['k_point_total'] = float(results_dict['initialization'][
                'Total number of k-points'])
            data_dict['num_apw_functions'] = float(results_dict['initialization'][
                'APW functions'])
            data_dict['num_empty_states'] = float(results_dict['initialization'][
                'Number of empty states'])
            data_dict['total_num_los'] = float(results_dict['initialization'][
                'Total number of local-orbitals'])
            data_dict['rgkmax'] = float(results_dict[
                'initialization']['R^MT_min * |G+k|_max (rgkmax)'])
            
            data_dict['relaxed_atom_positions'] = np.array2string(
                ase_atoms_obj_opt.get_positions(), separator=',',
                formatter={'float_kind': lambda x: "%.16f" % x}).replace('\n', '')

            data_dict['relaxed_cell'] = np.array2string(
                ase_atoms_obj_opt.get_cell(), separator=',',
                formatter={'float_kind': lambda x: "%.16f" % x}).replace('\n', '')

            data_dict['chem_formula'] = ase_atoms_obj_opt.get_chemical_formula()

            data_dict['relaxed_volume'] =ase_atoms_obj_opt.get_volume()

            cell_lengths_angles = ase_atoms_obj_opt.get_cell_lengths_and_angles()

            data_dict['relaxed_a_len'] = cell_lengths_angles[0]
            data_dict['relaxed_b_len'] = cell_lengths_angles[1]
            data_dict['relaxed_c_len'] = cell_lengths_angles[2]
            data_dict['relaxed_alpha_angle'] = cell_lengths_angles[3]
            data_dict['relaxed_beta_angle'] = cell_lengths_angles[4]
            data_dict['relaxed_gamma_angle'] = cell_lengths_angles[5]
            data_dict['status'] = 'calc_finished_after_parsing'
            # Otherwise everything has been parsed correclty, meaning
            # the calculation finished without issue.
            self.add_finished_correctly_path(submission_path)
            return data_dict

        else:  # The calc didn't expired but not ongoing or out of npl.
            print(f'path: {submission_path} added as misbehaving.')
            self.add_misbehaving_path(submission_path)
            data_dict['status'] = 'not_expired_misbehaving'
            return data_dict

    def check_if_exciting_info_out_does_not_exist(self, submission_dir):
        """Check if the simulation never ran."""
        if os.path.isfile(submission_dir + '/INFO.OUT'):
            return False
        else:
            return True

    def add_diag_fail_path(self, submission_path):
        """Add a path name for sim where SCF didnt converge.
        Args:
        path: (str) path name to submission script so that it
            can be resubmitted.
        """
        with open(
                self.paths_diag_fail, 'a') as fo:
            fo.writelines(submission_path + '\n')

    def add_mt_overlap_path(self, submission_path):
        """Add a path name for sim where SCF didnt converge.
        Args:
        path: (str) path name to submission script so that it
            can be resubmitted.
        """
        with open(
                self.paths_mt_overlap, 'a') as fo:
            fo.writelines(submission_path + '\n')

    def add_oom_path(self, submission_path):
        """Add a path name for an out of memory boolean path.

        Args:
        path: (str) path name to submission script so that it
            can be resubmitted.
        """
        with open(
                self.paths_out_of_memory, 'a') as fo:
            fo.writelines(submission_path + '\n')

    def add_ndirac_limit_path(self, submission_path):
        """Add a path name for where the ndirac limit was too large."""
        with open(
                self.paths_ndirac_limit, 'a') as fo:
            fo.writelines(submission_path + '\n')

    def add_expired_path(self, submission_path):
        """Add a path name where a simulation expired to due to being out of time.

        Args:
        path: (str) path name to submission script so that it
            can be resubmitted.
        """
        with open(
                self.paths_to_resubmit, 'a') as fo:
            fo.writelines(submission_path + '\n')

    def add_increase_optimizer_path(self, submission_path):
        """Add path where the geometry steps were too many.
        """
        with open(
                self.paths_increase_optimizer, 'a') as fo:
            fo.writelines(submission_path + '\n')


    def add_scalar_to_integer_path(self, submission_path):
        """Add a path name for a failed sim at start due to scalar2int issue."""
        with open(
                self.paths_scalar_to_integer, 'a') as fo:
            fo.writelines(submission_path + '\n')

    def add_resubmit_from_scratch_path(self, submission_path):
        """Add a path name for a failed sim at start due to scalar2int issue."""
        with open(
                self.paths_resubmit_from_scratch, 'a') as fo:
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

    def add_species_xml_missing_path(self, submission_path):
        """Add a path name for sim where SCF didnt converge.
        Args:
        path: (str) path name to submission script so that it
            can be resubmitted.
        """
        with open(
                self.paths_species_xml_missing, 'a') as fo:
            fo.writelines(submission_path + '\n')

    def add_out_of_npl_path(self, submission_path):
        """Add a path name for sim where sim didn't run due to out of npl.
        Args:
        path: (str) path name to submission script so that it
            can be resubmitted.
        """
        with open(
                self.paths_out_of_npl, 'a') as fo:
            fo.write(submission_path + '\n')

    def add_ongoing_path(self, submission_path):
        """Add a path name for sim where is still running.
        Args:
        path: (str) path name to submission script so that it
            can be resubmitted.
        """
        with open(
                self.paths_ongoing, 'a') as fo:
            fo.write(submission_path + '\n')

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

    def add_finished_correctly_path(self, submission_path):
        """Add a path name where the job finished correctly.

        Args:
        path: (str) path name to submission script so that it
            can be resubmitted.
        """
        with open(
                self.paths_finished_correclty, 'a') as fo:
            fo.writelines(submission_path + '\n')        

    def add_gs_not_started_path(self, submission_path):
        """Add a path name where ground state did not start.

        Args:
        path: (str) path name to submission script so that it
            can be resubmitted.
        """
        with open(
                self.paths_gs_not_started, 'a') as fo:
            fo.writelines(submission_path + '\n')

    def add_geo_opt_not_started_path(self, submission_path):
        """Add a path name where geometry optimization didn't start.

        Args:
        path: (str) path name to submission script so that it
            can be resubmitted.
        """
        with open(
                self.paths_geo_opt_not_started, 'a') as fo:
            fo.writelines(submission_path + '\n')

    def add_expired_time_path(self, submission_path):
        """Add a path name where a simulation didn't end nicely.

        Args:
        path: (str) path name to submission script so that it
            can be resubmitted.
        """
        with open(
                self.paths_expired, 'a') as fo:
            fo.writelines(submission_path + '\n')


    @staticmethod
    def get_opt_ase_object(geometry_opt_file):
        # First check if the exciting sim finished

        try:
            parsed_geometry = parse_structure(geometry_opt_file)

            new_structure = ExcitingStructure(
                atoms=parsed_geometry['atoms'],
                lattice=parsed_geometry['lattice'],
                species_path=parsed_geometry['species_path'],
                crystal_properties=parsed_geometry['crystal_properties'])

            ase_atoms_object = exciting_structure_to_ase(new_structure)
        except:
            ase_atoms_object = None

        return ase_atoms_object


    @staticmethod
    def check_calc_finished(parent_path):
        """Check if simulation exited nicely.

        We look line by line (from bottom of INFO.out)
        if 'EXCITING neon stopped' is there. If it is
        we return True. We continue looking for 50 lines
        and return False if we haven't found it by that
        point.

        Args:
            path: (str) path to folder containing INFO.OUT.

        Returns:
            calc_finished_bool: (bool) True if calc exited
            nicely.
            expired_time_bool: (bool) True if sim ran out of time.
            scf_bool: (bool) True if the sim's SCF cycle didn't
                converge.
        """
        exciting_output_file = parent_path + '/INFO.OUT'
        line_num = 0
        gs_started = False
        out_of_npl_bool = False
        expired_time_bool = False
        calc_finished_bool = False
        geo_opt_started_bool = False
        geo_opt_finished_bool = False
        diag_fail_bool = False
        mt_overlap_bool = False
        calc_ongoing_bool = False
        ndirac_limit_bool = False
        total_comp_time = None
        oom_bool = False
        # with open(aims_output_file, 'r') as fd:
        for line in reversed(list(open(exciting_output_file))):

            if "| EXCITING NEON stopped " in line:
                calc_finished_bool = True

            if "Total time spent (seconds)" in line:
                total_comp_time = float(line.strip(' ').split(':')[-1])

            if ("* Structure-optimization module stopped" in line) or (
                "Maximum force target reached already at the initial configuration" in line):
                geo_opt_finished_bool = True

            if "* Structure-optimization module started" in line:
                geo_opt_started_bool = True
                gs_started = True

            if "* Groundstate module started" in line:
                gs_started = True

            line_num += 1
            if line_num > 8000:  # Simply too long.
                break

        if calc_finished_bool is not True:
            # Then we need to take a look at the error jobs.
            # Call it with OutputParser namespace since
            # this method is static method and the method
            # we want to call is also a static method.
            expired_time_bool, out_of_npl_bool, oom_bool = OutputParserExciting.check_sim_time_lim(
                parent_path)
            if not expired_time_bool:
                diag_fail_bool, mt_overlap_bool, calc_ongoing_bool, ndirac_limit_bool = OutputParserExciting.check_out_file_fail(parent_path)

        # TODO(DTS): Change this into a dictionary.
        return calc_finished_bool, geo_opt_finished_bool, geo_opt_started_bool, gs_started, expired_time_bool, out_of_npl_bool, diag_fail_bool, mt_overlap_bool, calc_ongoing_bool, ndirac_limit_bool, oom_bool, total_comp_time

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
    def get_most_recent_djob_out(path):
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
        djob_out_list = [x for x in file_list if 'djob.out.' in x]
        # Check if the list is empty
        if not djob_out_list:
            return None
        # Now we need to choose the most recent djob err file. Let's
        # look at which has the largest integer after splitting the
        # file name by . and looking at last value. We first
        # list of integers of name of Jobs.
        djob_int_list = [x.split('.')[-1] for x in djob_out_list]
        # Now find the index with the largest integer.
        most_recent_job_num = djob_int_list.index(max(djob_int_list))
        # Use the index to get the djob err path name.
        most_recent_djob_out_path = djob_out_list[most_recent_job_num]

        return most_recent_djob_out_path

    @staticmethod
    def check_out_file_fail(path):
        """If time expired on sim, it returns True.

        If the sim didn't have a exciting neon stopped message in the INFO.out
        we call this method to check the djob.out file (output file) to see
        what type of messages are contained.

        First we check for any file names with /djob.out.* in our folder.
        We do so by first getting a list of all files in our folder.
        Then looking if there's a match with the type. Then we choose
        the djob.err with the largest #. We look in this djob.out for
        diagonalization errors or muffin tin overlap errors

        Args:
        path: (str) path to folder containing aims.out.

        Returns:
        expired_time_bool: (bool) True if the sim ran out of time.
            False otherwise.
        """

        # Now go through each line in the most recent djob.err.*
        # file in the folder and see if we can spot the marker.
        diag_marker = 'Error(seceqnfv): diagonalisation failed'
        mt_overlap_marker = 'Error(checkmt): muffin-tin spheres overlap between'
        calc_not_finished_marker = 'Elapsed: '
        ndirac_limit_marker = 'Error(rdirac): maximum iterations exceeded'
        # By default the bool we return is False since we
        # haven't seen markers.
        diag_fail_bool = False
        mt_overlap_bool = False
        calc_ongoing_bool = True
        ndirac_limit_bool = False
        # Get the path to most recent djob error.
        most_recent_djob_out = OutputParserExciting.get_most_recent_djob_out(path)
        if most_recent_djob_out is not None:
            # Print path of most recent djob error.
            for line in reversed(list(open(path + '/' + most_recent_djob_out))):
                if diag_marker in line:
                    diag_fail_bool = True
                if mt_overlap_marker in line:
                    mt_overlap_bool = True
                if calc_not_finished_marker in line:
                    calc_ongoing_bool = False
                if ndirac_limit_marker in line:
                    ndirac_limit_bool = True
                
        return diag_fail_bool, mt_overlap_bool, calc_ongoing_bool, ndirac_limit_bool

    @staticmethod
    def check_no_info_out_err_file(parent_path):
        """Check the error file when there is no info out file.

        If the sim didn't have a Have a nice day in the exciting INFO.out
        check the djob.err file if it says the xml species files are missing
        or if there is a scalar2integer issue.

        Args:
        path: (str) path to folder containing exciting INFO.out.

        Returns:
        species_xml_missing_bool: (bool) True if one or more species files are
            missing.
        scalar_to_integer_bool: (bool) True if there was a scalar to integer
            issue.
        """

        # Now go through each line in the most recent djob.err.*
        # file in the folder and see if we can spot the marker.
        species_marker = 'cp: cannot stat'
        scalar_to_integer_marker = 'Error in scalartointeger'

        # By default the bool we return is False since we
        # haven't seen markers.
        species_xml_missing_bool = False
        scalar_to_integer_bool = False
        # Get the path to most recent djob error.
        most_recent_djob_err = OutputParserExciting.get_most_recent_djob_err(parent_path)
        if most_recent_djob_err is not None:
            # Print path of most recent djob error.
            for line in reversed(list(open(os.path.join(parent_path, most_recent_djob_err)))):
                if species_marker in line:
                    species_xml_missing_bool = True
                elif scalar_to_integer_marker in line:
                    scalar_to_integer_bool = True
        return species_xml_missing_bool, scalar_to_integer_bool

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
        oom_marker = 'Out Of Memory'
        # By default the bool we return is False since we
        # haven't seen markers.
        expired_time_bool = False
        out_of_npl_bool = False
        oom_bool = False
        # Get the path to most recent djob error.
        most_recent_djob_err = OutputParserExciting.get_most_recent_djob_err(path)
        if most_recent_djob_err is not None:
            # Print path of most recent djob error.
            for line in reversed(list(open(path + '/' + most_recent_djob_err))):
                if first_marker in line and second_marker not in line:
                    out_of_npl_bool = True
                if first_marker in line and second_marker in line:
                    expired_time_bool = True
                    break
                if oom_marker in line:
                    oom_bool = True
        return expired_time_bool, out_of_npl_bool, oom_bool


    @staticmethod
    def check_state_out_read_issue(path):
        """Check if can't start because no STATE.out file exists and never ran.

        This occurs when trying to run a calc `fromfile` in the input.xml but
        there are no saved files since the calculation never ran. 
        """

        # Now go through each line in the most recent djob.err.*
        # file in the folder and see if we can spot the marker.
        state_out_marker = 'Error(readstate): error opening STATE.OUT'
        # By default the bool we return is False since we
        # haven't seen markers.
        state_out_bool = False
        # Get the path to most recent djob error.
        most_recent_djob_out = OutputParserExciting.get_most_recent_djob_out(path)
        if most_recent_djob_out is not None:
            # Print path of most recent djob error.
            for line in reversed(list(open(path + '/' + most_recent_djob_out))):
                if state_out_marker in line:
                    state_out_bool = True
        return state_out_bool



    def connect_to_csv(self):
        """Write simulation data to a CSV.

        We go through each path in the list of paths.
        """
        # Create header if csv file doesn't exist.
        if not os.path.isfile(self.csv_filename):
            self.create_header()

        with open(self.csv_filename, 'a') as csv_file:
            dict_writer = csv.DictWriter(
                csv_file, fieldnames=self.csv_columns)
            self.write_all_path_data(
                dict_writer)

    def write_all_path_data(self, dict_writer):
        """Write all data from path list to csv and db.

        Args:
        dict_writer: (csv.DictWriter() object) used as handle
            to write dictionary data to a row in an open CSV.
        db: (ASE db handle) db that has been opened and
            we can easily write to it.
        """
        for path in self.path_list:
            try:
                # Get data stored in files in the path folder.
                data_dict = self.gather_all_path_data(path)
                if data_dict is None:
                    raise ValueError(f'No data dict for path: {path}')
                # Commented out for speedup
                dict_writer.writerow(data_dict)

            except Exception as e:
                print("This path: %s has issues" % path)
                print("Error: %s" % e)
                self.logger.error(
                    "This path: %s gives this error: %s" % (path, e))


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


def main(argv):
    """Main fxn to allow us to call this method directly."""
    input_paths_txt_file = FLAGS.input_paths_txt_file
    if input_paths_txt_file == 'None':
        sys.exit('input_path_txt_files is None / not specified')
    print(f'paths txt file is {input_paths_txt_file}')
    save_directory = FLAGS.save_directory
    if save_directory == 'None':
        sys.exit('save_directoryname is None / not specified')
    print(f'csv filename is {save_directory}')

    parse_obj = OutputParserExciting(
        input_paths_txt_file=input_paths_txt_file,
        save_directory=save_directory)
    # Now submit all jobs
    parse_obj.connect_to_csv()


if __name__ == '__main__':
    app.run(main)
