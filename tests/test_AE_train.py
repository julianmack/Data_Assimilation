import torch
from pipeline.utils import ML_utils as ML
from pipeline import config
from pipeline import TrainAE
import pytest
import os

class TestAE_train():
    def test_AE_train_linear(self, tmpdir):
        epochs = 1
        settings = config.ToyAEConfig()
        expdir = tmpdir.mkdir("experiments/")

        trainer = TrainAE(settings, str(expdir))
        model = trainer.train(epochs)

    def test_AE_train_linear_DA(self, tmpdir):
        epochs = 1
        settings = config.ToyAEConfig()
        expdir = tmpdir.mkdir("experiments/")
        calc_DA_MAE = True
        
        trainer = TrainAE(settings, str(expdir), calc_DA_MAE)
        model = trainer.train(epochs)