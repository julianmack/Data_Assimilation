"""Run training for AE"""
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pickle

import pipeline.config as config

from pipeline import utils


BATCH = 256

class TrainAE():
    def __init__(self, settings):
        self.settings = settings

        err_msg = """settings must be an AE configuration class"""
        assert self.settings.COMPRESSION_METHOD == "AE", err_msg

    def train(self, num_epoch = 100, learning_rate = 0.001):
        settings = self.settings
        #data
        X = np.load(settings.X_FP)
        n, M = X.shape
        hist_idx = int(M * settings.HIST_FRAC)
        hist_X = X[:, : hist_idx]

        #Normalize:
        #use only the training set to calculate mean and std
        mean = np.mean(hist_X, axis=1)
        std = np.std(hist_X, axis=1)

        X_centered = (X.T - mean).T
        X_norm = (X_centered.T / std).T

        train_X = X_norm[:, : hist_idx]
        test_X = X_norm[:, hist_idx : -(settings.TDA_IDX_FROM_END+1)] #leave final elements for DA

        #Dataloaders
        train_dataset = TensorDataset(torch.Tensor(train_X.T))
        train_loader = DataLoader(train_dataset, BATCH, shuffle=True)
        test_dataset = TensorDataset(torch.Tensor(test_X.T))
        test_loader = DataLoader(test_dataset, test_X.shape[1])

        print("train_size = ", len(train_loader.dataset))
        print("test_size = ", len(test_loader.dataset))


        device = utils.ML_utils.get_device()

        model_fp = settings.AE_MODEL_FP
        results_fp_train = settings.RESULTS_FP + "toy_train_mode{}_hid{}.txt".format(settings.NUMBER_MODES, settings.HIDDEN)
        results_fp_test = settings.RESULTS_FP + "toy_test_mode{}_hid{}.txt".format(settings.NUMBER_MODES, settings.HIDDEN)

        loss_fn = torch.nn.L1Loss(reduction='sum')
        model = settings.AE_MODEL_TYPE(**settings.get_kwargs())
        self.model = model

        optimizer = optim.Adam(model.parameters(), learning_rate)

        print(model)
        print("Number of parameters:", sum(p.numel() for p in model.parameters()))

        train_losses, test_losses = utils.ML_utils.training_loop_AE(model, optimizer,
                                loss_fn, train_loader, test_loader,
                                num_epoch, device, print_every=1, test_every=5)

        torch.save(model.state_dict(), model_fp)

        with open(results_fp_train, 'wb') as fp:
            pickle.dump(train_losses, fp)
        with open(results_fp_test, 'wb') as fp:
            pickle.dump(test_losses, fp)


        return model



if __name__ == "__main__":
    settings = config.ToyAEConfig
    main(settings)
