import torch
from VarDACAE import ML_utils as ML
from VarDACAE.settings import base_CAE as config
from VarDACAE import TrainAE
import pytest
import os
import numpy as np

class TestAE_TrainLinear():
    def __settings(self, tmpdir, force_init=False):
        if hasattr(self, "settings") and not force_init:
            return self.settings
        else:
            n = 3
            M = 6
            X = np.random.rand(n, M)
            X[0, 3] = 1
            X = X.T

            INTERMEDIATE_FP = "inter"
            p = tmpdir.mkdir(INTERMEDIATE_FP).join("X_fp.npy")
            p.dump(X)
            p.allow_pickel = True

            settings = config.ToyAEConfig()
            settings.set_X_fp(str(p))
            settings.REDUCED_SPACE = True
            settings.FORCE_GEN_X = False
            settings.calc_DA_MAE = False
            settings.OBS_FRAC = 0.5
            settings.set_n(n)
            self.settings = settings
            return settings

    def test_AE_train_linear(self, tmpdir):
        """Test no exception thrown"""
        epochs = 1
        settings = self.__settings(tmpdir)
        expdir = tmpdir.mkdir("experiments/")

        trainer = TrainAE(settings, str(expdir))
        model = trainer.train(epochs, num_workers=0)

    def test_AE_train_linear_DA(self, tmpdir):
        """Test no exception thrown"""

        epochs = 1
        settings = self.__settings(tmpdir)
        expdir = tmpdir.mkdir("experiments/")
        calc_DA_MAE = True

        trainer = TrainAE(settings, str(expdir), calc_DA_MAE)
        model = trainer.train(epochs, num_workers=0)

class TestAE_Train3D():
    def __settings(self, tmpdir, force_init=False):
        if hasattr(self, "settings") and not force_init:
            return self.settings
        else:
            X = np.random.rand(6, 2, 2, 2)


            INTERMEDIATE_FP = "inter"
            p = tmpdir.mkdir(INTERMEDIATE_FP).join("X_fp.npy")
            p.dump(X)
            p.allow_pickel = True

            settings = config.CAEConfig()
            settings.REDUCED_SPACE = True
            settings.CHANGEOVERS = (7, 7, 7)
            settings.set_X_fp(str(p))
            settings.FORCE_GEN_X = False
            settings.n3d = tuple(X.shape[1:])
            settings.OBS_FRAC = 1 / 6.0


            self.settings = settings

            return settings

    def test_AE_train_3d(self, tmpdir):
        """Test no exception thrown"""
        epochs = 1
        settings = self.__settings(tmpdir)
        expdir = tmpdir.mkdir("experiments/")
        trainer = TrainAE(settings, str(expdir))
        model = trainer.train(epochs, num_workers=0)

    def test_AE_train_3D_DA(self, tmpdir):
        """Test no exception thrown"""
        epochs = 1
        settings = self.__settings(tmpdir, force_init= True)
        expdir = tmpdir.mkdir("experiments/")
        calc_DA_MAE = True

        trainer = TrainAE(settings, str(expdir), calc_DA_MAE)
        model = trainer.train(epochs, num_workers=0)