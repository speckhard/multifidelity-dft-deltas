import pickle
import pandas as pd
import sys
import os
import numpy as np
import mendeleev
from absl import flags
from absl import app

BASE_FOLDER = "/home/dts/Documents/theory/errorbar_modelling/"

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'csv_filename',
    BASE_FOLDER +
    'parsing/data/dataframe_prep',
    'file path stub that we will add monomers/binaries to save csv data.')
flags.DEFINE_string(
    'dft_code_name',
    'fhi_aims',
    'DFT-code-name')
flags.DEFINE_string(
    'csv_suffix',
    '_20_09_2023',
    'Identitfying suffix')


class AddValence():
    """Class to create new featues to enhance the problem."""
    def __init__(self, df_csv):
        """Constructor for feature engineering class.
        dataframe: (pandas df) dataframe containing primary features.
        max_precision: (int) defines the converged precision setting.
        """
        self.df = pd.read_csv(df_csv, index_col=0)
        # Right now numbers, giving a string inside of which is a list of the
        # the atom numbers presetn is a column. I thinnk this is causing
        # all rows to be dropped for exciting data so I've commented this line out.
        # self.df = self.df.dropna()
        self.val_dict = self.create_val_dict()

    def create_val_dict(self):
        """Create a valence electron dict."""
        val_dict = {}
        for i in range(91):
            val_dict[str(i+1)] = mendeleev.element(int(i+1)).nvalence()
        return val_dict

    def add_valence_num(self):
        """Add valence electron number to data."""
        self.df['a_valence_num'] = [self.val_dict[str(x)] for x in self.df[
            'a_atom_num']]
        self.df['b_valence_num'] = [self.val_dict[str(x)] for x in self.df[
            'b_atom_num']]


def main(argv):
    """Main function feature eng v2 is called."""
    # data_folder = "/home/speckhard/Documents/theory/errorbar_project/error_modelling"
    # data_folder = '/u/dansp/gen_error_data/errorbar_modelling'
    # aims_csv_file = '/parsing/data/dataframe_prep/aims_prepped_eof.csv'
    prepped_csv = FLAGS.csv_filename
    dft_code_name = FLAGS.dft_code_name
    csv_suffix = FLAGS.dft_code_name
    # aims_csv_file = '/parsing/data/dataframe_prep/aims_prepped.csv'
    target_folder = (
        BASE_FOLDER + 'parsing/data/add_valence')
    target_csv_filename = (
        target_folder + '/' + dft_code_name + '_prepped_w_valence_geo_opt' + csv_suffix + '.csv')
    # Create object
    av_obj = AddValence(df_csv=prepped_csv)
    av_obj.add_valence_num()
    # Save dataframe to a csv
    av_obj.df.to_csv(target_csv_filename, index='False')


if __name__ == '__main__':
    app.run(main)
