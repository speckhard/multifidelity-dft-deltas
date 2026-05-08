"""Create new features in the dataset.

Class supports creating new features from old features from
a pandas dataframe.
"""

import pickle
import sys
import os
import numpy as np
# Append path so that we can access packages that are above this directory.
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import parsing.json_parser as json_parser


class FeatureEng():
    """Class to create new featues to enhance the problem."""
    def __init__(self, dataframe, max_precision):
        """Constructor for feature engineering class.
        dataframe: (pandas df) dataframe containing primary features.
        max_precision: (int) defines the converged precision setting.
        """
        self.df = dataframe
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

        self.df['A Basis Functions'] = [
            basis_dict[get_aims_basis_set_size(y)[0], x,
                get_aims_basis_set_size(y)[1]] for x, y in zip(
                    self.df['A Atom Number'], self.df['Binary Precision'])]

        self.df['B Basis Functions'] = [
            basis_dict[
                get_aims_basis_set_size(y)[0], x,
                get_aims_basis_set_size(y)[1]] for x, y in zip(
                    self.df['B Atom Number'], self.df['Binary Precision'])]
        
        # Let's also define a new column that is the difference between
        # basis set size converged setting and basis set size of unconverged.
        converged_max_basis_settings = get_aims_basis_set_size(11)
        self.df['B Diff Basis Functions'] = ([
            basis_dict[
                converged_max_basis_settings[0], x,
                converged_max_basis_settings[1]] for x in self.df[
                    'B Atom Number']] - self.df['B Basis Functions'])

        converged_max_basis_settings = get_aims_basis_set_size(11)
        self.df['A Diff Basis Functions'] = ([
            basis_dict[
                converged_max_basis_settings[0], x,
                converged_max_basis_settings[1]] for x in self.df[
                    'A Atom Number']] - self.df['A Basis Functions'])

    def basis_fxn_per_val(self):
        """Create a basis function per valence electron column."""

        self.df['Binary Basis Functions'] = np.add(
            self.df['A Basis Functions'],
            self.df['B Basis Functions'])

        self.df['Binary Basis Functions P.V.E.'] = np.divide(
            self.df['Binary Basis Functions'],
            np.add(
                self.df['A Valence #'],
                self.df['B Valence #']))

        self.df['A Basis Functions P.V.E.'] = np.divide(
            self.df['A Basis Functions'],
            self.df['A Valence #'])

        self.df['B Basis Functions P.V.E.'] = np.divide(
            self.df['B Basis Functions'],
            self.df['B Valence #'])

        self.df['A Diff Basis Functions P.V.E.'] = np.divide(
            self.df['A Diff Basis Functions'],
            self.df['A Valence #'])

        self.df['B Diff Basis Functions P.V.E.'] = np.divide(
            self.df['B Diff Basis Functions'],
            self.df['B Valence #'])

        self.df['Binary Diff Basis Functions P.V.E.'] = np.add(
            self.df['B Diff Basis Functions P.V.E.'],
            self.df['A Diff Basis Functions P.V.E.'])

        self.df['Numerical Setting'] = [
            convert_num_setting_2_int(
                get_aims_basis_set_size(x)[0]) for x in self.df[
                    'Binary Precision']]

        # self.df['Basis Set Size'] = [
        #     get_aims_basis_set_size(x)[1] for x in self.df[
        #         'Binary Precision']]


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



