"""Parsing class and methods to extract data for error prediction.

This file was created to create a single class to extract data for
the error estimation research project. The data was created using ASE.
This data was then either saved in an ASE db or in a JSON created from
ASE or another method. We choose to deal with the JSON file as it allows
for parsing of data that came from other methods than ASE as well. Each
DFT code (e.g. FHI-aims, exciting,...) has different key value pairs and
as such we are forced to create slightly different methods for each.
"""
import ast
import os
import sys
from absl import flags
from absl import app
from ase import Atoms
# from matid import Classifier
# from matid import SymmetryAnalyzer
# from matid.classifications import Class3D, Material2D, Surface
import json  # We use the default JSON module to read the JSON files.
import numpy as np
import logging
import csv
from collections import defaultdict  # We use this to create dicts of dicts.

aims_non_avail_monomers = [
    '57', '58', '59', '60', '61', '62', '63', '64', '65',
    '66', '67', '68', '69', '70', '88', '89', '90']

# base_folder = "/home/speckhard/Documents/theory/errorbar_project/error_modelling"
base_folder = "/home/dts/Documents/theory/errorbar_project/errorbar_modelling/"


FLAGS = flags.FLAGS
flags.DEFINE_string(
    'csv_filename_stub',
    base_folder +
    'parsing/data/json_parser',
    'file path stub that we will add monomers/binaries to save csv data.')
flags.DEFINE_string(
    'dft_code_name',
    'fhi_aims',
    'Name of DFT code data being processed.')
flags.DEFINE_string(
    'monomers_json',
    base_folder + "parsing/data/error_bar_monomers.json",
    'Where to find the monomers of JSON.')
flags.DEFINE_string(
    'binaries_json',
    base_folder + "parsing/data/error_bar_binaries.json",
    'Where to find the binaries of JSON.')
flags.DEFINE_string(
    'monomers_end_of_json',
    'ids',
    'How many entries are in the json.')
flags.DEFINE_string(
    'binaries_end_of_json',
    'ids',
    'How many entries are in the json.')


class JsonParser:
    """Parses JSON files containing DFT energy calculations."""

    def __init__(
            self, dft_code_name, json_filename,
            csv_filename, end_of_json, write_csv=True):
        """Construct a new JSON parser.

        Args:
        code_name: json contains results from what DFT code name?
        json_filename: (string) path name to json file to extract.

        Returns:
        monomer_df: (pandas dataframe) containing the atomic data.
        binary_df: (pandas dataframe) containing the binary data.

        Raises:
        ValueError: If the 'code_name' is not supported.
        """
        self.dft_code_name = dft_code_name
        self.json_filename = json_filename
        # self.numerical_setting = numerical_setting
        self.initalize_dicts()
        self.logger = logging.getLogger(__name__)
        # Keep track of how many atoms we've seen to report later
        # as statistics.
        self.monomer_counter = 0
        self.binary_counter = 0

        self.binary_expanded_counter = 0
        self.monomer_expanded_counter = 0
        self.end_of_json = end_of_json
        self.csv_filename = csv_filename
        if self.dft_code_name == 'fhi_aims':
            self.csv_columns = [
                    'chem_formula',
                    'category',
                    'crystal_system',
                    'bravais_lattice',
                    'precision_level',
                    'basis_set_size',
                    'numerical_setting',
                    'relativistic_setting',
                    'k_point_density',
                    'functional',
                    'total_energy',
                    'volume',
                    'bandstructure_gap',
                    'free_energy',
                    'gamma_gap',
                    'min_atom_num',
                    'max_atom_num',
                    'min_atom_occurence',
                    'max_atom_occurence',
                    'numbers',
                    'json_id_num',
                    'name',
                    'num_atoms']
        if self.dft_code_name == 'exciting':
            self.csv_columns = [
                    'chem_formula',
                    'category',
                    'crystal_system',
                    'bravais_lattice',
                    'precision_level',
                    'rgkmax',
                    'k_point_density',
                    'functional',
                    'total_energy',
                    'E_cut',
                    'precision_element1',
                    'precision_element2',
                    'precision_element3',
                    'min_atom_num',
                    'max_atom_num',
                    'min_atom_occurence',
                    'max_atom_occurence',
                    'numbers',
                    'json_id_num',
                    'name',
                    'num_atoms']
        # Create the CSV with the hemader
        if write_csv is True:
            self.create_csv_header()

    def create_csv_header(self):
        """Create csv with header.
        """
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

    def initalize_dicts(self):
        """Construct a defaultdict objects to store parsed data.

        We use dict of dicts to store data as we parse from a json file.
        Another option that we should look into is to use the JSON
        parser and try to flatted the JSON file when parsing so we can
        parse directly into a pandas dataframe.
        """
        # Initalize a dictionary that will hold energy/bandgap/
        # cell/position data.
        self.energy_binaries_dict = defaultdict(dict)
        self.energy_monomers_dict = defaultdict(dict)
        self.cell_binaries_dict = defaultdict(dict)
        self.cell_monomers_dict = defaultdict(dict)
        self.positions_binaries_dict = defaultdict(dict)
        self.positions_monomers_dict = defaultdict(dict)
        self.pbc_binaries_dict = defaultdict(dict)
        self.pbc_monomers_dict = defaultdict(dict)
        # Initialize a dicitonary that will hold energy/bandgap for a
        # 5% expanded cell.
        self.energy_expanded_binaries_dict = defaultdict(dict)
        self.energy_expanded_monomers_dict = defaultdict(dict)

        if self.dft_code_name in ['FHI-aims', 'GPAW', 'VASP']:
            self.bandgap_expanded_binaries_dict = defaultdict(dict)
            self.bandgap_expanded_monomers_dict = defaultdict(dict)
            self.bandgap_binaries_dict = defaultdict(dict)
            self.bandgap_monomers_dict = defaultdict(dict)
        elif self.dft_code_name == 'exciting':
            self.cutoff_monomers_dict = defaultdict(dict)
            self.cutoff_binaries_dict = defaultdict(dict)

    def convert_json_2_csv(
                self, category_list=None):
        """Get raw data from JSON and save it as rows to a csv.

        Since we can't save lists to a csv we remove the lists
        from the raw data otherwise everything stays. We go item
        by item through the json and take relevant info into a dict
        that is saved as a row value."""
        with open(self.json_filename, 'r') as f:  # open the json as read.
            datastore = json.load(f)  # load the json into a dictionary.
        # Here we expect that the file has a header written to it already. So
        # we append it.
        with open(self.csv_filename, 'a') as csv_file:
            dict_writer = csv.DictWriter(
                csv_file, fieldnames=self.csv_columns)
            # Go through each number (index), and atom object in the dictionary
            # (datastore) that we've parsed from the json.
            for number, atom_object in datastore.items():
                # Check if we've reached the end of useful data in json.
                # print(number)
                if number == self.end_of_json:
                    break  # if so stop extracting data.
                elif (int(number) % 1000) == 0:
                    print('Index processed from json: %s' % number)

                # Use dft code specific extracting methods.
                if self.dft_code_name == 'fhi_aims':
                    row_dict = self.extract_aims_directly_to_dict(
                        atom_object)
                    if row_dict is not None:
                        row_dict['json_id_num'] = number
                        dict_writer.writerow(row_dict)
                elif self.dft_code_name == 'exciting':
                    row_dict = self.extract_exciting_directly_to_dict(
                        atom_object)
                    if (row_dict is not None):
                        if ((category_list is not None) and (
                                row_dict['category'] in category_list)) or (
                                    category_list is None):
                            row_dict['json_id_num'] = number
                            dict_writer.writerow(row_dict)

    def get_data_from_json(self, end_of_json):
        """ Gets data from self.json_filename and stores in member vars.

        Args:
        json_filename: json filename to parse.
        end_of_json: the last index in the json that if we encounter we should
            stop parsing.

        """
        with open(self.json_filename, 'r') as f:  # open the json as read.
            datastore = json.load(f)  # load the json into a dictionary.

        # Go through each number (index), and atom object in the dictionary
        # (datastore) that we've parsed from the json.
        for number, atom_object in datastore.items():
            # Check if we've reached the end of useful data in json.
            # print(number)
            if number == end_of_json:
                # print('were breaking!')
                break  # if so stop extracting data.
            # print(number)
            # print(atom_object)
            numbers = atom_object["numbers"]  # Atom numbers
            min_atom_num = np.min(atom_object["numbers"])
            max_atom_num = np.max(atom_object["numbers"])
            category = atom_object["key_value_pairs"]["category"]
            # Use dft code specific extracting methods.
            if self.dft_code_name == 'FHI-aims':
                # print('exctract from aims')
                self.extract_aims_from_dict(
                    atom_object, numbers, min_atom_num, max_atom_num, category)
            elif self.dft_code_name == 'exciting':
                self.extract_exciting_from_dict(
                    atom_object, numbers, min_atom_num, max_atom_num, category)

        # Print summary statistics of data that was extracted.
        print('binary_counter is %d' % self.binary_counter)
        print('monomer_counter is %d' % self.monomer_counter)
        print('binary_expanded_counter is %d' % self.binary_expanded_counter)
        print('monomer_expanded_counter is %d' % self.monomer_expanded_counter)
        self.parser_sucess = True  # Confirm parsing was successful.

    def extract_exciting_from_dict(
            self, atom_object, numbers, min_atom_num, max_atom_num, category):
        """Method that looks through aims data json file and parses data into dicts.

        This method looks at a dictionary containing parsed json results.
        it iterates over the key value pairs in the dictionary and parses out the
        relevant quantities we are interested in like energy, basis set size,
        numerical settings and so forth. It saves these parsed values into a
        defaultdict of dictionaries where we save the key as as the atom number
        and precision level (which is the combination of basis set size and
        numerical setting). In the future it might be wise to parse the json
        file directly into a dataframe to avoid this copying of data from
        one dictionary to another dictionary.

        Args:
        atom_object: (dictionary) containing key values pairs of dft data.
        numbers: (list of integers) is a string containing
            the list of integers indicating which atoms are in the unit cell
            being simulated.
        min_atom_number: (integer) containing the atomic number of the atom in
            the unit cell with the lowest atomic number in a.m.u.
        max_atom_number (integer) containst he atomic number of the atom
            in the unit cell with the largest atomic number in a.m.u.
        category: (string) indicates what type of simulation this data belongs
            to, i.e. to binary expanded - meaning two types of atoms in a cell
            that's been expanded by 5% in all directions relative to
            experimental minimum.

        Raises: ValueError if any monomer data has two types of atoms.
        """
        k_point_density = atom_object["key_value_pairs"]["k_point_density"]
        energy = atom_object["key_value_pairs"]["total_energy"]
        numbers = atom_object["numbers"]
        # We want the energy per atom in the system.
        energy = np.divide(energy, len(numbers))
        total_precision = int(
            atom_object["key_value_pairs"]["total_precision"])
        cutoff_energy = atom_object["key_value_pairs"]["E_cut"]
        # print(total_precision)
        positions = atom_object["positions"]
        cell = atom_object["cell"]
        pbc = atom_object["pbc"]
        if k_point_density == 8.0:
            if category == "binaries":
                # Here we create a dictionary where keys are string version
                # of atom numbers since these identify which compounds are
                # simulated. The values are dictionarys of key, values:
                # (kpoints_density, energy) so we store them all for the same
                # compound.
                self.energy_binaries_dict[
                    str(numbers)][str(total_precision)] = energy
                self.cutoff_binaries_dict[
                    str(numbers)][str(total_precision)] = cutoff_energy
                self.binary_counter += 1
                self.positions_binaries_dict[
                    str(numbers)][str(total_precision)] = positions
                # Add the unit cell data as well.
                self.cell_binaries_dict[
                        str(numbers)][str(total_precision)] = cell
                self.pbc_binaries_dict[
                        str(numbers)][str(total_precision)] = pbc
            elif category == "binaries_expanded":
                self.binary_expanded_counter += 1
                self.energy_expanded_binaries_dict[
                    str(numbers)][str(total_precision)] = energy
            elif category == "monomers":
                # Since the monomers only have one type of atom_number
                # let's just save this number so we can use this to lookup values
                # of energy for this element.
                if not min_atom_num == max_atom_num:
                    self.logger.error(
                        "The monomer has two types of elements in the system?!")
                    raise ValueError(
                        "min_atom_num: %d is unequal to "
                        "max_atom_num: %d" % min_atom_num, max_atom_num)
                self.energy_monomers_dict[
                    str(min_atom_num)][str(total_precision)] = energy
                self.cutoff_monomers_dict[
                    str(min_atom_num)][str(total_precision)] = cutoff_energy
                self.monomer_counter += 1
                self.positions_monomers_dict[
                    str(min_atom_num)][str(total_precision)] = positions
            elif category == "monomers_expanded":
                self.monomer_expanded_counter += 1
                self.energy_expanded_monomers_dict[
                    str(min_atom_num)][str(total_precision)] = energy

    def extract_aims_from_dict(
            self, atom_object, numbers, min_atom_num, max_atom_num, category):
        """Method that looks through aims data json file and parses data into dicts.

        This method performs same proceedure as get_exciting_data_from_json.
        It is different in that it is looking for different key value pairs.

        atom_object: (dictionary) containing key values pairs of dft data.
        numbers: (string containing list of integers) is a string containing
            the list of integers indicating which atoms are in the unit cell
            being simulated.
        min_atom_number: (integer) containing the atomic number of the atom in
            the unit cell with the lowest atomic number in a.m.u.
        max_atom_number (integer) containst he atomic number of the atom
            in the unit cell with the largest atomic number in a.m.u.
        category: (string) indicates what type of simulation this data belongs
            to, i.e. to binary expanded - meaning two types of atoms in a cell
            that's been expanded by 5% in all directions relative to
            experimental minimum.

        Raises: ValueError if any monomer data has two types of atoms.
        """
        if "calculator" in atom_object and "energy" in atom_object:
            if atom_object["calculator"] != "aims":
                print("calculator not aims!")
                raise ValueError('Calculator for aims data is not aims!')
            # basis set size - minimal, standard, tier1, tier2
            # is named as tiers in the fhi json file.
            basis_set_size = atom_object["key_value_pairs"]["tiers"]
            # Grab the relativistic data. This can be either two values:
            # "atomic_zora" or a list ['zora scalar', 1e-12]
            relativistic_setting = atom_object[
                "key_value_pairs"]["relativistic_treatment"]
            k_point_density = atom_object[
                "key_value_pairs"]["k_point_density"]
            functional = atom_object["key_value_pairs"]["functional"]
            positions = atom_object["positions"]
            cell = atom_object["cell"]
            pbc = atom_object["pbc"]
            # numerical settings of light, tight, really tight.
            numerical_setting = atom_object[
                "key_value_pairs"]["basis_set"]

            if (category in [
                    "monomers_unrelaxed",
                    "binaries_unrelaxed",
                    "binaries_unrelaxed_expanded_5pc",
                    "monomers_unrelaxed_expanded_5pc"]) \
                    and (k_point_density == 8) \
                    and (functional == 'pbe') \
                    and (relativistic_setting == "atomic_zora"):
                    # and (numerical_setting == self.numerical_setting):

                energy = float(atom_object["energy"])
                # We want the energy per atom in the system.
                energy = np.divide(energy, len(numbers))
                name = atom_object["key_value_pairs"]["name"]
                # Now let's seperate into binary/monomer cateogries.
                basis_set_size = get_aims_precision_level(
                    basis_set_size=basis_set_size,
                    numerical_setting=numerical_setting)
                precision_level = basis_set_size
                bandstructure_gap = atom_object[
                    "key_value_pairs"]["gap_bandstructure"]
                if (category == "binaries_unrelaxed" and
                        str(max_atom_num) not in aims_non_avail_monomers):
                    # Here we create a dictionary where keys are string
                    # version of atom numbers since these identify which
                    # compounds are simulated. The values are dictionaries
                    # of key, values:(kpoints_density, energy) so we store
                    # them all for the same compound.
                    self.energy_binaries_dict[
                        str(numbers)][str(precision_level)] = energy
                    self.bandgap_binaries_dict[
                        str(numbers)][str(precision_level)] = bandstructure_gap
                    self.cell_binaries_dict[
                        str(numbers)][str(precision_level)] = cell
                    self.positions_binaries_dict[
                        str(numbers)][str(precision_level)] = positions
                    self.pbc_binaries_dict[
                        str(numbers)][str(precision_level)] = pbc
                    self.binary_counter += 1
                elif (category == "binaries_unrelaxed_expanded_5pc" and
                        str(max_atom_num) not in aims_non_avail_monomers):
                    # Then let's save the expanded data.
                    self.energy_expanded_binaries_dict[
                        str(numbers)][str(precision_level)] = energy
                    self.bandgap_expanded_binaries_dict[
                        str(numbers)][str(precision_level)] = bandstructure_gap
                    self.binary_expanded_counter += 1
                elif category == "monomers_unrelaxed":
                    # Since the monomers only have one type of atom_number
                    # let's just save number so we can use to lookup vals
                    # of energy for this element.
                    if not min_atom_num == max_atom_num:
                        print("The monomer has two diff types of atoms?!")
                    self.energy_monomers_dict[
                        str(min_atom_num)][str(precision_level)] = energy
                    self.bandgap_monomers_dict[
                        str(min_atom_num)][str(precision_level)] = bandstructure_gap
                    self.cell_monomers_dict[
                        str(min_atom_num)][str(precision_level)] = cell
                    self.positions_monomers_dict[
                        str(min_atom_num)][str(precision_level)] = positions
                    # self.atom_numbers_monomers_dict[
                    #     str(min_atom_num)][str(precision_level)] = numbers
                    self.monomer_counter += 1
                elif category == "monomers_unrelaxed_expanded_5pc":
                    if not min_atom_num == max_atom_num:
                        print(
                            "The monomer doesn't have 2 types of elements in sys?!")
                        raise ValueError(
                            "The monomer doesn't have 2 types of elements in sys?!")
                    self.energy_expanded_monomers_dict[
                        str(min_atom_num)][str(precision_level)] = energy
                    self.bandgap_expanded_monomers_dict[
                        str(min_atom_num)][str(precision_level)] = bandstructure_gap
                    self.monomer_expanded_counter += 1

    def extract_aims_directly_to_dict(self, atom_object):
        """Extract aims data and store it direclty into a dict.

        Parse the dictionaries into a single dict that will be saved
        as a row in a csv. The csv will eventually be loaded into
        a dataframe. This method avoids individual efforts to calc
        the error and simply saves raw data rows.

        Args:
        atom_object (dictionary): containing key values pairs of dft data.

        Raises: ValueError if any monomer data has two types of atoms.
        """
        # Wrap the code in this if statement since someimtes the
        # atom_object will be a dud and not have required data.
        row_dict = {}
        if "calculator" in atom_object and "energy" in atom_object:
            # Quick check to make sure we have aims data.
            if atom_object["calculator"] != "aims":
                print("calculator not aims!")
                raise ValueError('Calculator for aims data is not aims!')
            # Now parse relevant fields for this datapoint.
            # basis set size is one of {minimal, standard, tier1, tier2}
            # is named as tiers in the fhi json file.
            numbers = atom_object["numbers"]  # Atom numbers
            row_dict['min_atom_num'] = np.min(numbers)
            row_dict['max_atom_num'] = np.max(numbers)
            row_dict['min_atom_occurence'] = numbers.count(row_dict['min_atom_num'])
            row_dict['max_atom_occurence'] = numbers.count(row_dict['max_atom_num'])
            row_dict['num_atoms'] = len(numbers)
            row_dict['category'] = atom_object["key_value_pairs"]["category"]
            row_dict["basis_set_size"] = atom_object[
                "key_value_pairs"]["tiers"]
            # Grab the relativistic data. This can be either two values:
            # "pbe" or "lda".
            row_dict["relativistic_setting"] = atom_object[
                "key_value_pairs"]["relativistic_treatment"]
            row_dict["k_point_density"] = atom_object[
                "key_value_pairs"]["k_point_density"]
            row_dict["functional"] = atom_object[
                "key_value_pairs"]["functional"]
            # numerical settings of light, tight, really tight.
            row_dict["numerical_setting"] = atom_object[
                "key_value_pairs"]["basis_set"]
            # Grab the volume
            row_dict["volume"] = atom_object["key_value_pairs"]["Volume"]
            # Grab the energy
            total_energy = float(atom_object["energy"])
            # We want the energy per atom in the system.
            row_dict["total_energy"] = np.divide(total_energy, len(numbers))
            # Grab the free energy which is different in Bjorn's file.
            free_energy = atom_object[
                "key_value_pairs"]["free_energy"]
            row_dict["free_energy"] = np.divide(free_energy, len(numbers))
            # Grab the name of the binary.
            row_dict["name"] = atom_object["key_value_pairs"]["name"]
            # Grab the precision level.
            row_dict["precision_level"] = get_aims_precision_level(
                basis_set_size=row_dict['basis_set_size'],
                numerical_setting=row_dict['numerical_setting'])
            # Grab the bandgap aka gap_bandstructure.
            row_dict["bandstructure_gap"] = atom_object[
                "key_value_pairs"]["gap_bandstructure"]
            # Grab the gamma gap.
            row_dict["gamma_gap"] = atom_object[
                "key_value_pairs"]["gap_gamma"]

            crystal_system, bravais_lattice, chem_formula = self.get_xtal_structure(
                    numbers, atom_object["positions"],
                    atom_object["cell"],
                    atom_object["pbc"])
            row_dict['crystal_system'] = crystal_system
            row_dict['bravais_lattice'] = bravais_lattice
            row_dict['chem_formula'] = chem_formula
            return row_dict

    def extract_exciting_directly_to_dict(self, atom_object, category=None):
        """Extract aims data and store it direclty into a dict.

        Parse the dictionaries into a single dict that will be saved
        as a row in a csv. The csv will eventually be loaded into
        a dataframe. This method avoids individual efforts to calc
        the error and simply saves raw data rows.

        Args:
        atom_object (dictionary): containing key values pairs of dft data.

        Raises: ValueError if any monomer data has two types of atoms.
        """
        # Wrap the code in this if statement since someimtes the
        # atom_object will be a dud and not have required data.
        row_dict = {}
        if "ctime" in atom_object:
            # Now parse relevant fields for this datapoint.
            # basis set size is one of {minimal, standard, tier1, tier2}
            # is named as tiers in the fhi json file.
            numbers = atom_object["numbers"]
            row_dict['numbers'] = numbers  # Atom numbers
            row_dict['min_atom_num'] = np.min(numbers)
            row_dict['max_atom_num'] = np.max(numbers)
            row_dict['min_atom_occurence'] = numbers.count(row_dict['min_atom_num'])
            row_dict['max_atom_occurence'] = numbers.count(row_dict['max_atom_num'])
            row_dict['num_atoms'] = len(numbers)
            row_dict['category'] = atom_object["key_value_pairs"]["category"]

            # Grab the relativistic data. This can be either two values:
            # "pbe" or "lda".
            row_dict["k_point_density"] = atom_object[
                "key_value_pairs"]["k_point_density"]
            # Get the lower case since we want pbe instead of PBE.
            row_dict["functional"] = atom_object[
                "key_value_pairs"]["xc_functional"].lower()
            # Grab the total precision
            row_dict['precision_level'] = atom_object[
                "key_value_pairs"]["total_precision"]
            # Element 1/2  name: we don't need these.
            # row_dict['element1'] = atom_object["key_value_pairs"]["element1"]
            # row_dict['element2'] = atom_object["key_value_pairs"]["element2"]
            # Precision of element1
            row_dict['precision_element1'] = atom_object[
                "key_value_pairs"]["precision_of_element1"]
            # Find the # of different elements in the numbers list.
            num_elements = len(set(numbers))
            if num_elements > 1:
                row_dict['precision_element2'] = atom_object[
                    "key_value_pairs"]["precision_of_element2"]
            else:
                row_dict['precision_element2'] = np.nan
            if num_elements > 2:
                row_dict['precision_element3'] = atom_object[
                    "key_value_pairs"]["precision_of_element3"]
            else:
                row_dict['precision_element3'] = np.nan
            # E_cut
            row_dict['E_cut'] = atom_object["key_value_pairs"]["E_cut"]
            # Rgkmax
            row_dict['rgkmax'] = atom_object["key_value_pairs"]["rgkmax"]
            # Grab the energy
            total_energy = float(atom_object["key_value_pairs"]["total_energy"])
            # We want the energy per atom in the system.
            row_dict["total_energy"] = np.divide(total_energy, len(numbers))
            row_dict["name"] = atom_object["key_value_pairs"]["name"]

            crystal_system, bravais_lattice, chem_formula = self.get_xtal_structure(
                    numbers, atom_object["positions"],
                    atom_object["cell"],
                    atom_object["pbc"])
            row_dict['crystal_system'] = crystal_system
            row_dict['bravais_lattice'] = bravais_lattice
            row_dict['chem_formula'] = chem_formula
            return row_dict

    def get_xtal_structure(
                self, atom_numbers, positions, cell, pbc):
        """Get binary classification using Matid package.

        We create an ASE atoms object using the cell, positions
        and pbc of a binary. Then we use Matid to classify the
        atoms object as a string. We update the class member list
        binary_xtal_class_list.

        Args:
        atom_numbers: binary atom numbers in a.m.u.
        positions (matrix): positions of atoms in crystal unit cell.
        cell (matrix): lattice vector matrix for atoms.
        pbc (list): periodic boundary conditions (boolean for 
            each lattice vector).
        """
        # atom_numbers = ast.literal_eval(atom_numbers)
        binary_ase_obj = Atoms(
            cell=cell, positions=positions, pbc=pbc, symbols=atom_numbers)

        chem_formula = binary_ase_obj.get_chemical_formula()
        # Define the Matid classifier.
        # classifier = Classifier()
        classifier = None  # Temp fix to avoid matid.
        # Perform classification.
        classification = classifier.classify(binary_ase_obj)
        # In case crystal system calssification fails return None
        crystal_system = None
        bravais_lattice = None
        # if type(classification) in [Class3D, Surface]:
        #     symm_analyzer = SymmetryAnalyzer(binary_ase_obj)
        #     crystal_system = symm_analyzer.get_crystal_system()
        #     bravais_lattice = symm_analyzer.get_bravais_lattice()
        return crystal_system, bravais_lattice, chem_formula


def get_aims_precision_level(basis_set_size, numerical_setting):
    """Translate the tiers and basis set into a numeric number from 1-12.

    FHI-aims has an interesting way of specifying the numerical settings
    and basis set size. The tier level specifies the numerical setting
    precision level but also changes the basis set size. The basis set size
    controls how many basis set functions to use per atom. This number
    is a bit hand-picked and ranges from atom to atom and the relationship
    is not clear. This method uses a simple encoding that might be an issue.

    Args:
    numerical_setting: (string) aims numerical settings level.
    basis_set_size: (string) aims basis set size setting.

    Returns:
    precision_level: (int) encoding of precision based on basis set size
        and numerical setting.

    Raises:
    ValueError: if the pairing of numerical setting and basis set size
        are not recognized.

    """

    if numerical_setting == "light":
        if basis_set_size == "minimal":
            precision = 0
        if basis_set_size == "tier1":
            precision = 1
        if basis_set_size == "standard":
            precision = 2
        if basis_set_size == "tier2":
            precision = 3
    elif numerical_setting == "tight":
        if basis_set_size == "minimal":
            precision = 4
        if basis_set_size == "tier1":
            precision = 5
        if basis_set_size == "standard":
            precision = 6
        if basis_set_size == "tier2":
            precision = 7
    elif (numerical_setting == "really_tight" or
            numerical_setting == "really tight"):
        if basis_set_size == "minimal":
            precision = 8
        if basis_set_size == "tier1":
            precision = 9
        if basis_set_size == "standard":
            precision = 10
        if basis_set_size == "tier2":
            precision = 11
    else:
        print(
            "basis set: %s is unrecognized in dataset row." % basis_set_size)
        raise ValueError(
            "basis set: %s is unrecognized in dataset row." % basis_set_size)
    return precision


def main(argv):
    """Main function when json_parser.py is called."""
    # Parse the flags
    csv_filename_stub = FLAGS.csv_filename_stub
    dft_code_name = FLAGS.dft_code_name
    # Create name for binaries csv file
    binaries_csv = csv_filename_stub + '/' + dft_code_name + '_binaries.csv'
    # Create header for CSV
    binaries_obj = JsonParser(
        dft_code_name=FLAGS.dft_code_name,
        json_filename=FLAGS.binaries_json,
        csv_filename=binaries_csv,
        end_of_json=FLAGS.binaries_end_of_json)
    # Create JsonParser Object for binaries data.
    if dft_code_name == 'fhi_aims':
        binaries_obj.convert_json_2_csv()
    elif dft_code_name == 'exciting':
        binaries_obj.convert_json_2_csv(category_list=[
        'binaries', 'binaries_expanded'])
    # Now save monomers data, re-use object
    monomers_csv = csv_filename_stub + '/' + dft_code_name + '_monomers.csv'
    # write the header line for the csv file.
    monomers_obj = JsonParser(
        dft_code_name=FLAGS.dft_code_name,
        json_filename=FLAGS.monomers_json,
        csv_filename=monomers_csv,
        end_of_json=FLAGS.monomers_end_of_json)
    if dft_code_name == 'fhi_aims':
        monomers_obj.convert_json_2_csv()
    elif dft_code_name == 'exciting':
        monomers_obj.convert_json_2_csv(category_list=[
        'monomers', 'monomers_expanded'])


if __name__ == '__main__':
    app.run(main)
