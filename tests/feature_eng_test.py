"""Tests for feature engineering methods.

We test whether methods in feature_eng.py file actually create
basis functions per valence electron for min and max atoms.
"""

import os
import sys
import pandas as pd
import pickle
# Append path so that we can access packages that are above this directory.
print(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import feat_eng.feature_eng as feat_eng
print(os.getcwd())


mock_pkl_filename = "tests/data/mock_basis_fxn_dict.pickle"


def get_mock_df():
    """Gets mock dataframe to use in testing."""
    data = {
        'Min Atom Number': [1, 1, 1, 2],
        'Max Atom Number': [6, 6, 6, 7],
        'Binary Precision': [8, 9, 10, 9],
        'A Valence #': [1, 1, 1, 0],
        'B Valence #': [12, 12, 12, 13]
    }
    df = pd.DataFrame(data)
    return df


def test_get_mock_df():
    df = get_mock_df()
    print(df[df['Min Atom Number'] == 2]['Binary Precision'])
    assert df[df['Min Atom Number'] == 2]['Binary Precision'][3] == 9


def save_mock_pkl_dict():
    """Save a dictionary as a pkl file to tests/data/ folder."""
    basis_dict = {}
    basis_dict['really tight', 1, 'minimal'] = 10

    basis_dict['really tight', 6, 'minimal'] = 30
    basis_dict['really tight', 1, 'standard'] = 13
    basis_dict['really tight', 6, 'standard'] = 40
    basis_dict['really tight', 1, 'tier1'] = 20
    basis_dict['really tight', 6, 'tier1'] = 24
    basis_dict['really tight', 2, 'standard'] = 60
    basis_dict['really tight', 7, 'standard'] = 70
    with open(mock_pkl_filename, 'wb') as handle:
        pickle.dump(
            basis_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return basis_dict


def test_save_mock_pkl_dict():
    basis_dict = save_mock_pkl_dict()
    print(basis_dict)
    assert basis_dict['really tight', 7, 'standard'] == 70
    basis_dict_saved = pickle.load(
        open(mock_pkl_filename, "rb"))
    assert basis_dict_saved['really tight', 7, 'standard'] == 70


def create_mock_fe_obj():
    """Create a mock feature engineering object."""
    fe_obj = feat_eng.FeatureEng(dataframe=get_mock_df())
    fe_obj.add_aims_basis_fxns(
        basis_dict_pkl_filename=mock_pkl_filename,
        numerical_setting='really tight')
    return fe_obj


def test_add_aims_basis_fxns():
    """Test ability to add basis functions per min/max atom."""
    fe_obj = create_mock_fe_obj()
    print(fe_obj.df)
    print(list(fe_obj.df[
        (fe_obj.df['Binary Precision'] == 9) & (fe_obj.df[
            'Max Atom Number'] == 7)]['Min Basis Functions']))
    assert list(fe_obj.df[
        (fe_obj.df['Binary Precision'] == 9) & (fe_obj.df[
            'Max Atom Number'] == 7)]['Max Basis Functions'])[0] == 70
    assert list(fe_obj.df[
        (fe_obj.df['Binary Precision'] == 9) & (fe_obj.df[
            'Min Atom Number'] == 2)]['Min Basis Functions'])[0] == 60
    assert list(fe_obj.df[
        (fe_obj.df['Binary Precision'] == 8) & (fe_obj.df[
            'Min Atom Number'] == 1)]['Min Basis Functions'])[0] == 10


def test_basis_fxn_per_val():
    """Test method to get basis functions per valence electron."""
    fe_obj = create_mock_fe_obj()
    fe_obj.basis_fxn_per_val()
    assert list(fe_obj.df[
        (fe_obj.df['Binary Precision'] == 9) & (fe_obj.df[
            'Min Atom Number'] == 2)]['Binary Basis Functions'])[0] == 130
    assert list(fe_obj.df[
        (fe_obj.df['Binary Precision'] == 9) & (fe_obj.df[
            'Min Atom Number'] == 2)]['Binary Basis Functions P.V.E.'])[0] == 130/13