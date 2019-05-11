"""Run training for AE"""
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pickle

import config
from AutoEncoders import VanillaAE, ToyNet
import utils


settings = config.ToyAEConfig

RESULTS_FP = "results/"
BATCH = 256

def main():
    #data
    X = np.load(settings.X_FP)
    n, M = X.shape
    hist_idx = int(M * settings.HIST_FRAC)
    hist_X = X[:, : hist_idx]
    test_X = X[:, hist_idx : -3] #leave final three elements for DA

    #Dataloaders
    train_dataset = TensorDataset(torch.Tensor(hist_X.T))
    train_loader = DataLoader(train_dataset, BATCH, shuffle=True)
    test_dataset = TensorDataset(torch.Tensor(test_X.T))
    test_loader = DataLoader(test_dataset, test_X.shape[1])

    print("train_size = ", len(train_loader.dataset))
    print("test_size = ", len(test_loader.dataset))

    #training hyperparams
    num_epoch = 1000
    device = utils.ML_utils.get_device()
    #AE hyperparams
    input_size = n

    learning_rate = 0.0001
    # layers = [1000, 100]
    # latent_size = 10

    model_fp = settings.AE_MODEL_FP
    results_fp_train = "{}toy_train.txt".format(RESULTS_FP)
    results_fp_test = "{}toy_test.txt".format(RESULTS_FP)

    loss_fn = torch.nn.L1Loss(reduction='sum')
    model = settings.AE_MODEL_TYPE(**settings.kwargs)
    optimizer = optim.Adam(model.parameters(), learning_rate)


    print(model)
    print("Number of parameters:", sum(p.numel() for p in model.parameters()))

    train_losses, test_losses = utils.ML_utils.training_loop_AE(model, optimizer, loss_fn, train_loader, test_loader,
        num_epoch, device, print_every=1, test_every=5)
    with open(results_fp_train, 'wb') as fp:
        pickle.dump(train_losses, fp)
    with open(results_fp_test, 'wb') as fp:
        pickle.dump(test_losses, fp)
    torch.save(model.state_dict(), model_fp)

if __name__ == "__main__":
    main()
