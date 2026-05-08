"""Create new features in the dataset.

Class supports creating new features from old features from
a pandas dataframe.
"""

import pickle
import pandas as pd
import sys
import os
import numpy as np
import mendeleev
from absl import flags
from absl import app

BASE_FOLDER = "/home/speckhard/Documents/theory/errorbar_project/error_modelling/"
# BASE_FOLDER = "/Users/dts/Documents/playground/errorbar_modelling/"

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'csv_filename_stub',
    BASE_FOLDER + 
    'parsing/data/dataframe_prep',
    'file path stub that we will add monomers/binaries to save csv data.')
flags.DEFINE_string(
    'dft_code_name',
    'fhi_aims',
    'DFT-code-name')
flags.DEFINE_string(
    'csv_suffix',
    '_13_12_2021',
    'Identitfying suffix')


class FeatureEng():
    """Class to create new featues to enhance the problem."""
    def __init__(self, df_csv, max_precision=11):
        """Constructor for feature engineering class.
        dataframe: (pandas df) dataframe containing primary features.
        max_precision: (int) defines the converged precision setting.
        """
        self.df = pd.read_csv(df_csv, index_col=0)
        self.df = self.df.drop(columns=['numbers'])
        self.df = self.df.dropna()
        self.max_precision = max_precision

    def add_aims_basis_fxns(
                self, basis_dict_pkl_filename):
        """Add data columns to dataframe of basis fxns per valence electron.

        We load a dictionary from the basis_dikt_pkl_filename. The dictionary
        has format keys (numerical_setting, atomic_number, basis set size).
        and value - number of basis set functions. We first convert this
        basis set functions data to a per valence electron number using
        the mendeleev library. We then assign a new column based on min/max
        atom number columns in the dataframe containing the min/max
        basis set functions per valence electron."""

        basis_dict = pickle.load(
            open(basis_dict_pkl_filename, "rb"))

        self.df['A_basis_functions'] = [
            basis_dict[get_aims_basis_set_size(y)[0], x,
                get_aims_basis_set_size(y)[1]] for x, y in zip(
                    self.df['A_atom_num'], self.df['precision_level'])]

        self.df['B_basis_functions'] = [
            basis_dict[
                get_aims_basis_set_size(y)[0], x,
                get_aims_basis_set_size(y)[1]] for x, y in zip(
                    self.df['B_atom_num'], self.df['precision_level'])]
        
        # Let's also define a new column that is the difference between
        # basis set size converged setting and basis set size of unconverged.
        converged_max_basis_settings = get_aims_basis_set_size(11)
        self.df['B_diff_basis_functions'] = ([
            basis_dict[
                converged_max_basis_settings[0], x,
                converged_max_basis_settings[1]] for x in self.df[
                    'B_atom_num']] - self.df['B_basis_functions'])

        converged_max_basis_settings = get_aims_basis_set_size(11)
        self.df['A_diff_basis_functions'] = ([
            basis_dict[
                converged_max_basis_settings[0], x,
                converged_max_basis_settings[1]] for x in self.df[
                    'A_atom_num']] - self.df['A_basis_functions'])

    def basis_fxn_per_val(self):
        """Create a basis function per valence electron column."""

        self.df['binary_basis_functions'] = np.add(
            self.df['A_basis_functions'],
            self.df['B_basis_functions'])

        self.df['binary_basis_functions_pve'] = np.divide(
            self.df['binary_basis_functions'],
            np.add(
                self.df['A_valence_num'],
                self.df['B_valence_num']))

        self.df['A_basis_functions_pve'] = np.divide(
            self.df['A_basis_functions'],
            self.df['A_valence_num'])

        self.df['B_basis_functions_pve'] = np.divide(
            self.df['B_basis_functions'],
            self.df['B_valence_num'])

        self.df['A_diff_basis_functions_pve'] = np.divide(
            self.df['A_diff_basis_functions'],
            self.df['A_valence_num'])

        self.df['B_diff_basis_functions_pve'] = np.divide(
            self.df['B_diff_basis_functions'],
            self.df['B_valence_num'])

        self.df['binary_diff_basis_functions_pve'] = np.add(
            self.df['B_diff_basis_functions_pve'],
            self.df['A_diff_basis_functions_pve'])

        self.df['numerical_setting'] = [
            convert_num_setting_2_int(
                get_aims_basis_set_size(x)[0]) for x in self.df[
                    'precision_level']]

        self.df['basis_set_size'] = [
            get_aims_basis_set_size(x)[1] for x in self.df[
                'precision_level']]


def get_aims_basis_set_size(binary_precision):
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
        print('binary precision: %d - no good' % binary_precision)    
    return [numerical_setting, basis_set_size]


def convert_num_setting_2_int(numerical_setting):
    if numerical_setting == 'light':
        num_int = 1
    elif numerical_setting == 'tight':
        num_int = 2
    elif numerical_setting == 'really_tight':
        num_int = 3
    else:
        print('Wrong num setting % s' % numerical_setting)
        sys.exit(0)
    return num_int


def main(argv):
    """Main function feature eng v2 is called."""
    csv_filename_stub = FLAGS.csv_filename_stub
    dft_code_name = FLAGS.dft_code_name
    csv_suffix = FLAGS.csv_suffix

    pkl_filename = (
        BASE_FOLDER +
        "/modelling/data/aims_basis_function_dict.pickle")

    valence_csv_filename = (
        BASE_FOLDER +
        '/parsing/data/add_valence/' + dft_code_name +
        '_prepped_w_valence' + csv_suffix + '.csv')
    feat_eng_csv_filename = (
        BASE_FOLDER +
        '/parsing/data/feat_eng/' + dft_code_name +
        '_prepped_w_valence_w_feat_eng' + csv_suffix + '.csv')
    # Create object
    fe_obj = FeatureEng(df_csv=valence_csv_filename)
    fe_obj.add_aims_basis_fxns(
            basis_dict_pkl_filename=pkl_filename)
    fe_obj.basis_fxn_per_val()
    # Save dataframe to a csv
    fe_obj.df.to_csv(feat_eng_csv_filename, index='False')


if __name__ == '__main__':
    app.run(main)
