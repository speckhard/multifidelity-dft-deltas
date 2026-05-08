"""Test adding valence data for feature engineering."""

import pytest
import os, sys
import unittest
# Append path so that we can access packages that are above this directory.
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import pandas as pd
import numpy as np
import feat_eng.add_valence as av

test_csv = 'parsing/data/dataframe_prep/exciting_prepped.csv'


def create_mock_av_obj():
    """Create mock add valence (av) object"""
    av_obj = av.AddValence(df_csv=test_csv)
    return av_obj


def test_create_val_dict():
    """Test creation of a dictionary where k,v is (atomic number, # of valence electrons)"""
    av_obj = create_mock_av_obj()
    valence_dict = av_obj.create_val_dict()
    assert valence_dict['1'] == 1
    assert valence_dict['4'] == 2


def test_add_valence_num():
    """Test that the A_valence_num and B_valence_num cols are correctly added."""
    av_obj = create_mock_av_obj()
    # Make sure we are testing the right row, where A refers to Hydrogen in H4Li4.
    assert 'A_atom_num' in av_obj.df.columns
    assert len(av_obj.df['A_atom_num']) > 0
    assert av_obj.df['A_atom_num'].iloc[[0]].values[0] == 1
    assert av_obj.df['B_atom_num'].iloc[[0]].values[0] == 3
    av_obj.add_valence_num()
    assert av_obj.df['A_valence_num'].iloc[[0]].values[0] == 1
    assert av_obj.df['B_valence_num'].iloc[[0]].values[0] == 1
