import numpy as np
import os, sys
#import pipeline
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(os.getcwd()) #to import pipeline
import pipeline

from pipeline.VarDA import SVD
from pipeline.settings import config
from pipeline import SplitData, GetData
from pipeline import DAPipeline
DA_results = self.DA_pipeline.DA_AE()
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
    settings.DEBUG = False

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

    obs_fracs = [0.001, 0.01, 0.03]

    modes = [-1]
    obs_fracs = [0.001]


    L1 = torch.nn.L1Loss(reduction='sum')
    L2 = torch.nn.MSELoss(reduction="sum")


    datasets = {"train": train_X,
                "test": test_X,
                "u_c": u_c,
                "mean": mean,
                "u_0": np.zeros_like(u_c)}
    datasets = {"train1": train_X[:1],
                "train2": train_X[:2] }

    DA_pipeline = DAPipeline(settings)

    for name, data in datasets.items():
        for obs_frac in obs_fracs:
            settings.OBS_FRAC = obs_frac

            if len(data.shape) in [1, 3]:
                num_states = 1
            else:
                num_states = data.shape[0]

            for mode in modes:
                totals = {"ref_MAE": np.zeros_like(u_c),
                        "da_MAE": np.zeros_like(u_c),
                        "ref_MAE_mean": 0,
                        "da_MAE_mean": 0,
                        "counts": 0}
                for idx in range(num_states):

                    data = DA_pipeline.data
                    V_trunc = SVD.SVD_V_trunc(U, s, W, modes=mode)
                    data["V_trunc"] = V_trunc
                    data["V"] = None
                    data["V_grad"] = None

                    DA_results = DA_pipeline.perform_VarDA(data, settings)

                    # ref_MAE = DA_results["ref_MAE"]
                    # da_MAE = DA_results["da_MAE"]
                    ref_MAE_mean = DA_results["ref_MAE_mean"]
                    da_MAE_mean = DA_results["da_MAE_mean"]
                    counts = (DA_results["da_MAE"] < DA_results["ref_MAE"]).sum()

                    #add to dict results
                    # totals["ref_MAE"] += ref_MAE
                    # totals["da_MAE"] += da_MAE
                    totals["ref_MAE_mean"] += ref_MAE_mean
                    totals["da_MAE_mean"] += da_MAE_mean
                    totals["counts"] += counts

                print(name.upper(), ": obs_frac:", obs_frac, "number_modes:", mode)
                for k, v in totals:
                    print(k, v / num_states)

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