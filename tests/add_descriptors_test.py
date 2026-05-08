"""Test file for add descriptors."""

import pytest
import os, sys
# Append path so that we can access packages that are above this directory.
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import pandas as pd 
import numpy as np
import feat_eng.add_descriptors as ad

csv_filename = 'tests/data/ad_test.csv'


def setup_mock_object():
    """Method used to create a mock dataframe that we will use for tests."""
    # initialize list of lists
    # def __init__(self, dataframe, functional, csv_filename=None): 
    mock_add_descrip = ad.AddDescriptors(setup_mock_dataframe(),
            functional='pbe', csv_filename=csv_filename)
    return mock_add_descrip

def setup_mock_dataframe():
    data = {'Min Atom Number': [1,2, 3, 4],
            'Max Atom Number': [4, 3, 2, 1],
            'Energy': [1.1, 2.2, 3.3, 4.4]
    }
    return pd.DataFrame(data)


def test_add_descriptors():
    print("HEres the directory in test")
    print(os. getcwd())
    mock_add_descrip = setup_mock_object()
    # Add descriptors
    mock_add_descrip.add_descriptors()
    assert mock_add_descrip.dataframe['Min EA Half'][0] == -0.69804885
    assert mock_add_descrip.dataframe['Min EA Half'][1] == 3.05683208
    assert mock_add_descrip.dataframe['Min EA Half'][2] == -0.43620871
    assert mock_add_descrip.dataframe['Min EA Half'][3] == 0.74699855
    
    assert mock_add_descrip.dataframe['Max EA Half'][0] == 0.74699855
    assert mock_add_descrip.dataframe['Max EA Half'][1] == -0.43620871
    assert mock_add_descrip.dataframe['Max EA Half'][2] == 3.05683208
    assert mock_add_descrip.dataframe['Max EA Half'][3] == -0.69804885

    assert mock_add_descrip.dataframe['Max f index'][0] == 0
    assert mock_add_descrip.dataframe['Max f index'][1] == 7
    assert mock_add_descrip.dataframe['Max f index'][2] == 0
    assert mock_add_descrip.dataframe['Max f index'][3] == 0