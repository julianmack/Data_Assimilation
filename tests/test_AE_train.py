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
        print(expdir)
        print(os.getcwd())
        trainer = TrainAE(settings, str(expdir))
        model = trainer.train(epochs)