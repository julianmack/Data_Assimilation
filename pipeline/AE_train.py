"""Run training for AE"""
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch
import numpy as np
import settings
from AutoEncoders import VanillaAE
import sys
sys.path.append('/home/jfm1118')
import utils

import pickle


MODEL_FP = "models/AE"
RESULTS_FP = "results/"
BATCH = 128

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
    num_epoch = 120
    device = utils.ML_utils.get_device()
    #AE hyperparams
    input_size = n

    layers = [60, 50, 40]
    latent_size = 40
    learning_rate = 0.0001
    # layers = [1000, 100]
    # latent_size = 10

    model_fp = "{}_dim{}_epoch{}.pth".format(MODEL_FP, latent_size, num_epoch)
    results_fp_train = "{}train_dim{}_epoch{}.txt".format(RESULTS_FP, latent_size, num_epoch)
    results_fp_test = "{}test_dim{}_epoch{}.txt".format(RESULTS_FP, latent_size, num_epoch)

    loss_fn = torch.nn.L1Loss(reduction='sum')
    model = VanillaAE(input_size, latent_size, layers)
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
