import argparse

import matplotlib.pyplot as plt
import torch

from src.data.make_dataset import CorruptMnist
from src.models.model import MyAwesomeModel

import numpy as np
import os

import os.path

from tests import _PATH_DATA
import pytest

train_set = CorruptMnist(train=True, in_folder="data/raw", out_folder="data/processed")
test_set = CorruptMnist(train=False, in_folder="data/raw", out_folder="data/processed")

N_train = 40000
N_test = 5000

class TestClass:
    @pytest.mark.skipif(not os.path.exists(_PATH_DATA+'/this_folder_doesnt_exist'), reason="Data files not found")
    def test_len(self): #Test length of dataset
        assert len(train_set) == N_train, "Training dataset did not have the correct number of samples"
        assert len(test_set) == N_test, "Test dataset did not have the correct number of samples"

    def test_torch_shape(self):
        assert train_set[0][0].shape == torch.Size([1, 28, 28]) #The training data tensor did not have the right format
        assert test_set[0][0].shape == torch.Size([1, 28, 28]) #The training data tensor did not have the right format

    def test_labels(self):
        assert len(train_set[:][1].unique()) == 10 #You don't have 10 different labels in your training data set as you should have.
        assert len(test_set[:][1].unique()) == 10 #You don't have 10 different labels in your training data set as you should have.

    def test_error_on_wrong_shape(self):
        with pytest.raises(ValueError, match='Expected input to a 4D tensor'):
            model = MyAwesomeModel()
            model(torch.randn(1,2,3))

    def test_error_on_wrong_shape2(self):
        with pytest.raises(ValueError, match=r'Expected each sample to have shape \[1, 28, 28\]'):
            model = MyAwesomeModel()
            model(torch.randn(64,1,28,27))

    @pytest.mark.parametrize("test_input,expected", [("3+5", 8), ("2+4", 6), ("9*9", 81)])
    def test_eval(self, test_input, expected):
        assert eval(test_input) == expected