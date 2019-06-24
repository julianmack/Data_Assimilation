"""Run training for AE"""
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pickle

import pipeline.config as config

from pipeline import utils
import os

BATCH = 256

class TrainAE():
    def __init__(self, AE_settings, expdir):
        self.settings = AE_settings

        err_msg = """AE_settings must be an AE configuration class"""
        assert self.settings.COMPRESSION_METHOD == "AE", err_msg

        self.expdir = self.__init_expdir(expdir)

        self.test_fp = self.expdir + "test.csv"
        self.train_fp = self.expdir + "train.csv"
        self.settings_fp = self.expdir + "settings.txt"

    def train(self, num_epoch = 100, learning_rate = 0.001):
        settings = self.settings
        #data
        loader = utils.DataLoader()
        X = loader.get_X(settings)

        train_X, test_X, _, X_norm,  mean, std = loader.test_train_DA_split_maybe_normalize(X, settings)

        #Add Channel
        train_X = np.expand_dims(train_X, 1)
        test_X = np.expand_dims(test_X, 1)

        #Dataloaders
        train_dataset = TensorDataset(torch.Tensor(train_X))
        train_loader = DataLoader(train_dataset, BATCH, shuffle=True)
        test_dataset = TensorDataset(torch.Tensor(test_X))
        test_loader = DataLoader(test_dataset, test_X.shape[0])

        print("train_size = ", len(train_loader.dataset))
        print("test_size = ", len(test_loader.dataset))


        device = utils.ML_utils.get_device()

        loss_fn = torch.nn.L1Loss(reduction='sum')
        model = settings.AE_MODEL_TYPE(**settings.get_kwargs())

        self.model = model

        optimizer = optim.Adam(model.parameters(), learning_rate)


        print("Number of parameters:", sum(p.numel() for p in model.parameters()))


        train_losses, test_losses = utils.ML_utils.training_loop_AE(model, optimizer,
                                loss_fn, train_loader, test_loader,
                                num_epoch, device, print_every=1, test_every=1, model_dir = self.expdir)


        #Save results and settings file (so that it can be exactly reproduced)
        self.__write_csv(train_losses, self.train_fp)
        self.__write_csv(test_losses, self.test_losses)
        pickle.dump(settings, self.settings_fp)

        return model

    def __write_csv(np_array, fp):
        header = "epoch,loss,DA_MAE"
        np.savetxt(fp, np_array, delimiter=",", header=header)


    def __init_expdir(self, expdir):
        wd = utils.get_home_dir()
        try:
            dir_ls = expdir.split("/")
            assert dir_ls[0] == "experiments"
        except (AssertionError, KeyError, AttributeError):
            raise ValueError("expdir must be in the experiments/ directory")

        if os.path.isdir(expdir):
            if len(os.listdir(expdir)) > 0:
                raise "Cannot overwrite files in expdir. Exit-ing."
        else:
            if expdir[0] == "/":
                expdir = expdir[1:]
            if not expdir[-1] == "/":
                expdir += "/"
            expdir = wd + expdir
            os.makedirs(expdir)
        return expdir


if __name__ == "__main__":
    settings = config.ToyAEConfig
    main(settings)
