import numpy as np
import os, sys
#import pipeline
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(os.getcwd()) #to import pipeline
import pipeline

from pipeline.VarDA import SVD
from pipeline.settings.CAE7 import CAE7
from pipeline import SplitData, GetData, DAPipeline, ML_utils

import torch
from torch.utils.data import TensorDataset, DataLoader


def main():
    dir = "/data/home/jfm1118/DA/experiments/train_DA_Pressure/2-l4NBN/" # 299.pth"
    model, settings = ML_utils.load_model_and_settings_from_dir(dir)

    settings.TOL = 1e-3
    settings.SAVE = False
    settings.export_env_vars()

    #Load data
    loader, splitter = GetData(), SplitData()
    X = loader.get_X(settings)
    train_X, test_X, u_c, X, mean, std = splitter.train_test_DA_split_maybe_normalize(X, settings)

    L1 = torch.nn.L1Loss(reduction='sum')
    L2 = torch.nn.MSELoss(reduction="sum")


    datasets = {"train": train_X,
                "test": test_X,
                "u_c": u_c,
                "mean": mean,
                "u_0": np.zeros_like(u_c)}
    # datasets = {
    #             "train1": train_X[:1],
    #             "train2": train_X[:2],
    #             "train3": train_X[:3],
    #             "train4": train_X[:4],
    #             "train5": train_X[:5],
    #             "train6": train_X[:6],
    #             }

    for name, data in datasets.items():

        if len(data.shape) in [1, 3]:
            num_states = 1
        else:
            num_states = data.shape[0]

        device = ML_utils.get_device()
        data_tensor = torch.Tensor(data)
        data_tensor = data_tensor.to(device)


        DA_pipeline = DAPipeline(settings, model)


        encoder = DA_pipeline.data.get("encoder")
        decoder = DA_pipeline.data.get("decoder")

        data_hat = decoder(encoder(data))
        data_hat = torch.Tensor(data_hat)
        data_hat = data_hat.to(device)

        
        l1 = L1(data_hat, data_tensor)
        l2 = L2(data_hat, data_tensor)

        print("name: {}, \nL1: {:.2f}\nL2: {:.2f}\n".format(name, l1 / num_states, l2 / num_states,))
        print()

        # data_2 = model.decode(model.encode(data_tensor.unsqueeze(1))).squeeze(1)
        # model3 = DA_pipeline.data.get("model").to(device)
        # data_3 = model3(data_tensor.unsqueeze(1)).squeeze(1)
        # assert torch.allclose(data_hat, data_2)
        # assert torch.allclose(data_hat, data_3)
        # for x in [data_hat, data_2, data_3]:
        #
        #     l1 = L1(x, data_tensor)
        #     l2 = L2(x, data_tensor)
        #
        #     print("name: {}, \nL1: {:.2f}\nL2: {:.2f}\n".format(name, l1 / num_states, l2 / num_states,))
        # print()



def see_how_close(x1, x2, rtol = 0.1):
    assert x1.shape == x2.shape
    shape = x1.shape
    npoints = prod(shape)

    mean = (x1 + x2) / 2
    mean = np.where(mean <= 0., 1, mean)
    diff = x1 - x2

    relative = diff / mean
    num_above = relative > rtol
    num_above = num_above.sum()
    mean_rel = relative.mean()

    print("num_above", num_above, "/", npoints)
    print("mean_rel", mean_rel)

if __name__ == "__main__":
    main()