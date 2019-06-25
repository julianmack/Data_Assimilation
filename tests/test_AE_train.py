import torch
from pipeline.utils import ML_utils as ML
from pipeline import config
from pipeline import TrainAE
import pytest
import os
import numpy as np

class TestAE_TrainLinear():
    def __settings(self, tmpdir, force_init=False):
        if hasattr(self, "settings") and not force_init:
            return self.settings
        else:
            X = np.random.rand(10, 15)
            X[0, 3] = 1
            X = X.T

            INTERMEDIATE_FP = "inter"
            p = tmpdir.mkdir(INTERMEDIATE_FP).join("X_fp.npy")
            p.dump(X)
            p.allow_pickel = True

            settings = config.ToyAEConfig()
            settings.X_FP = str(p)
            settings.FORCE_GEN_X = False
            settings.set_n(10)
            self.settings = settings
            return settings

    def test_AE_train_linear(self, tmpdir):
        """Test no exception thrown"""
        epochs = 1
        settings = self.__settings(tmpdir)
        expdir = tmpdir.mkdir("experiments/")

        trainer = TrainAE(settings, str(expdir))
        model = trainer.train(epochs)

    def test_AE_train_linear_DA(self, tmpdir):
        """Test no exception thrown"""

        epochs = 1
        settings = self.__settings(tmpdir)
        expdir = tmpdir.mkdir("experiments/")
        calc_DA_MAE = True

        trainer = TrainAE(settings, str(expdir), calc_DA_MAE)
        model = trainer.train(epochs)

class TestAE_Train3D():
    def __settings(self, tmpdir, force_init=False):
        if hasattr(self, "settings") and not force_init:
            return self.settings
        else:
            X = np.random.rand(6, 5, 7, 4)


            INTERMEDIATE_FP = "inter"
            p = tmpdir.mkdir(INTERMEDIATE_FP).join("X_fp.npy")
            p.dump(X)
            p.allow_pickel = True

            settings = config.CAEConfig()
            settings.CHANGEOVERS = (7, 7, 7)
            settings.X_FP = str(p)
            settings.FORCE_GEN_X = False
            settings.n3d = tuple(X.shape[1:])
            self.settings = settings
            return settings

    def test_AE_train_3d(self, tmpdir):
        """Test no exception thrown"""
        epochs = 1
        settings = self.__settings(tmpdir)
        expdir = tmpdir.mkdir("experiments/")
        trainer = TrainAE(settings, str(expdir))
        model = trainer.train(epochs)

    def test_AE_train_3D_DA(self, tmpdir):
        """Test no exception thrown"""
        return
        epochs = 1
        settings = self.__settings(tmpdir, force_init= True)
        expdir = tmpdir.mkdir("experiments/")
        calc_DA_MAE = True

        trainer = TrainAE(settings, str(expdir), calc_DA_MAE)
        model = trainer.train(epochs)