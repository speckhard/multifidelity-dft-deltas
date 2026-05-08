"""Convert categorical features."""
from absl import flags
from absl import app
import pandas as pd
from absl import flags
from absl import app

# BASE_FOLDER = "/home/speckhard/Documents/theory/errorbar_project/error_modelling/"
# BASE_FOLDER = "/Users/dts/Documents/playground/errorbar_modelling/"
BASE_FOLDER = "/home/dts/Documents/theory/errorbar_project/errorbar_modelling/"

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

class ConvertCat():
    def __init__(self, df_csv, dft_code_name):
        self.df = pd.read_csv(df_csv)
        self.dft_code_name = dft_code_name
        if self.dft_code_name == 'exciting':
            self.df = self.df.drop(columns=['precision_element3'])
        self.get_cat_cols()
        # Remove na rows
        print('Unique binaries is:')
        print(len(set(self.df['chem_formula'])))
        print('len of rows is %d' % len(self.df['chem_formula']))

        print('Number of NA rows')
        print(self.df.isna().sum())
        self.df = self.df.dropna()
        print('Unique binaries is:')
        print(len(set(self.df['chem_formula'])))
        print('len of rows after dropping nans is %d' % len(self.df['chem_formula']))
        # Convert cat cols to ordinal #s
        self.convert_cols()

    def get_cat_cols(self):
        """Get the categorical columns"""
        numeric_cols = self.df._get_numeric_data().columns
        columns = self.df.columns
        self.cat_cols = list(set(columns) - set(numeric_cols))
        print(self.cat_cols)

    def convert_cols(self):
        for col in self.cat_cols:
            print(col)
            self.df[col] = self.df[col].astype('category')
            # print(self.df[col].iloc[[0,1,2,3]])
            # print(self.df[col].dtypes)
            # print(self.df[col].cat.categories)
            d = dict(enumerate(self.df[col].cat.categories))
            # print(d)
            self.df[col] = self.df[col].cat.codes
            # print(self.df[col].iloc[[0,1,2,3]])
            # print(self.df[col].cat.categories)
        # print(self.df[self.cat_cols].dtypes)
        # cat_columns = self.df.select_dtypes(['category']).columns
        # self.df[cat_columns] = self.df[cat_columns].apply(
        #     lambda x: x.cat.codes)

    # def print_cat_encodings(self):
    #     for col in self.cat_cols:
    #         print(self.df[col].cat.categories)


def main(argv):
    """Main function when json_parser.py is called."""
    csv_filename_stub = FLAGS.csv_filename_stub
    dft_code_name = FLAGS.dft_code_name
    csv_suffix = FLAGS.csv_suffix

    if dft_code_name == 'exciting':
        source_csv = (
            BASE_FOLDER + 'parsing/data/add_valence/' +
            dft_code_name + '_prepped_w_valence'
            + csv_suffix + '.csv') 
        encoded_csv_filename = (
            BASE_FOLDER + 'parsing/data/encoded_data/' +
            dft_code_name + '_prepped_w_valence_encoded'
            + csv_suffix + '.csv')
    else:
        source_csv = (
            BASE_FOLDER + 'parsing/data/feat_eng/' +
            dft_code_name + '_prepped_w_valence_w_feat_eng'
            + csv_suffix + '.csv')
        encoded_csv_filename = (
            BASE_FOLDER + 'parsing/data/encoded_data/' +
            dft_code_name + '_prepped_w_valence_w_feat_eng_encoded'
            + csv_suffix + '.csv')


    # Create object
    cc_obj = ConvertCat(df_csv=source_csv, dft_code_name=dft_code_name)
    # Save dataframe to a csv.
    cc_obj.df.to_csv(encoded_csv_filename, index=False)


if __name__ == '__main__':
    app.run(main)