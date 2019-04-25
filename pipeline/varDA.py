#!/usr/bin/python3
"""3D VarDA pipeline to assimilate a single observation"""

import numpy as np
from helpers import VarDataAssimilationPipeline as VarDA
import settings
import sys
import random

sys.path.append('/home/jfm1118')
import utils

#hyperparameters
ALPHA = 1.0
OBS_VARIANCE = 0.01 #TODO - CHECK this is specific to the sensors (in this case - the error in model predictions)

OBS_FRAC = 0.001 #fraction of state used as "observations"
HIST_FRAC = 1 / 3.0 #fraction of data used as "history"

def main():
    #initialize helper function class
    vda = VarDA()

    #The X array should already be saved in settings.X_FP
    #but can be created from .vtu fps if required. see trunc_SVD.py for an example
    X = np.load(settings.X_FP)
    n, M = X.shape

    # Split X into historical and present data. We will
    # assimilate "observations" at a single timestep t_DA
    # which corresponds to the control state u_c
    # We will take initial condition u_0, as mean of historical data
    hist_idx = int(M * HIST_FRAC)
    hist_X = X[:, : hist_idx]
    t_DA = M - 2 #i.e. second to last
    u_c = X[:, t_DA]
    V, u_0 = vda.create_V_from_X(hist_X, return_mean = True)

    # Define observations as a random subset of the control state.
    nobs = int(OBS_FRAC * n) #number of observations
    utils.set_seeds(seed = settings.SEED) #set seeds so that the selected subset is the same every time
    obs_idx = random.sample(range(n), nobs) #select nobs integers w/o replacement
    observations = np.take(u_c, obs_idx)

    #Now define quantities required for 3D-VarDA
    H = vda.create_H(obs_idx, n, nobs)
    R_inv = vda.create_R_inv(OBS_VARIANCE, nobs)
    



if __name__ == "__main__":
    main()
