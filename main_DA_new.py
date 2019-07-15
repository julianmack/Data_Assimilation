"""File to run elements of pipeline module from"""

from pipeline import TrainAE
from pipeline.settings import CAE_configs
import os
import sys
from matplotlib import pyplot as plt


def main():
    #get models fp

    expdir = "experiments/DA/cae6"
    settings = CAE_configs.CAE6()
    settings.BATCH_NORM = False
    settings.CHANGEOVER_DEFAULT = 0

    print(os.listdir(expdir))
    model_fp = settings.HOME_DIR + expdir + "/29.pth"



    settings.OBS_FRAC = 0.01  #1 % of observations
    settings.OBS_VARIANCE = 0.01
    settings.AE_MODEL_FP = model_fp
    da = DAPipeline(settings)
    w_opt = da.Var_DA_routine(settings)



if __name__ == "__main__":
    main()
