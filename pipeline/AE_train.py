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


MODEL_FP = "models/AE_1.pth"
BATCH = 64

def main():
    #data
    X = np.load(settings.X_FP)
    n, M = X.shape
    hist_idx = int(M * settings.HIST_FRAC)
    hist_X = X[:, : hist_idx]
    test_X = X[:, hist_idx : -3] #leave final three elements for DA

    print("train shape:", hist_X.shape)
    print("train tensor shape:", torch.Tensor(hist_X.T).shape)
    #Dataloaders
    train_dataset = TensorDataset(torch.Tensor(hist_X.T))
    train_loader = DataLoader(train_dataset, BATCH, shuffle=True)
    test_dataset = TensorDataset(torch.Tensor(test_X.T))
    test_loader = DataLoader(test_dataset, test_X.shape[1])

    #AE hyperparams
    input_size = n
    latent_size = 1
    layers = [1000, 100]

    #training hyperparams
    num_epoch = 20
    device = utils.ML_utils.get_device()
    print("Device:", device)


    loss_fn = torch.nn.L1Loss(reduction='sum')
    model = VanillaAE(input_size, latent_size, layers)
    optimizer = optim.Adam(model.parameters())

    print(model)
    utils.ML_utils.training_loop_AE(model, optimizer, loss_fn, train_loader, test_loader,
            num_epoch, device, print_every=1, test_every=1)
    torch.save(model.state_dict(), MODEL_FP)

if __name__ == "__main__":
    main()
