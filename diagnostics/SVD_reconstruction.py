import numpy as np
import os, sys
#import pipeline
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(os.getcwd()) #to import pipeline
import pipeline

from pipeline.VarDA import SVD
from pipeline.settings import config
from pipeline import SplitData, GetData

import torch

import operator
from functools import reduce
def prod(iterable):
    return reduce(operator.mul, iterable, 1)


def main():
    settings = config.Config()
    settings.THREE_DIM = True

    settings.set_X_fp(settings.INTERMEDIATE_FP + "X_3D_{}.npy".format(settings.FIELD_NAME))
    settings.set_n( (91, 85, 32))
    settings.OBS_FRAC = 0.3

    #LOAD U, s, W
    fp_base = settings.get_X_fp().split("/")[-1][1:]

    U = np.load(settings.INTERMEDIATE_FP  + "U" + fp_base)
    s = np.load(settings.INTERMEDIATE_FP  + "s" + fp_base)
    W = np.load(settings.INTERMEDIATE_FP  + "W" + fp_base)

    #Load data
    loader, splitter = GetData(), SplitData()
    X = loader.get_X(settings)
    train_X, test_X, u_c, X, mean, std = splitter.train_test_DA_split_maybe_normalize(X, settings)

    modes = [-1, 500, 100, 50, 30, 20, 10, 8, 6, 4, 3, 2, 1]

    L1 = torch.nn.L1Loss(reduction='sum')
    L2 = torch.nn.MSELoss(reduction="sum")


    datasets = {"train": train_X,
                "test": test_X,
                "u_c": u_c,
                "mean": mean,
                "u_0": np.zeros_like(u_c)}
    # datasets = {"train": train_X[0],
    #             "train1": train_X[:1],
    #             "train2": train_X[:2],
    #             "train3": train_X[:3],
    #             "train4": train_X[:4],
    #             "train5": train_X[:5],
    #             "train6": train_X[:6],
    #             "train7": train_X[:7],
    #             }

    for name, data in datasets.items():

        if len(data.shape) in [1, 3]:
            num_states = 1
        else:
            num_states = data.shape[0]

        data_tensor = torch.Tensor(data)

        for mode in modes:

            data_hat = SVD.SVD_reconstruction_trunc(data, U, s, W, mode)

            data_hat = torch.Tensor(data_hat)

            l1 = L1(data_hat, data_tensor)
            l2 = L2(data_hat, data_tensor)

            print("name: {}, modes = {}\nL1: {:.2f}\nL2: {:.2f}\n".format(name, mode, l1 / num_states, l2 / num_states,))

        print()




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