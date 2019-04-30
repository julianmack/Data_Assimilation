#!/usr/bin/python3
"""3D VarDA pipeline to assimilate a single observation"""

import numpy as np
from helpers import VarDataAssimilationPipeline as VarDA
import settings
from scipy.optimize import minimize

#hyperparameters
ALPHA = 0
OBS_VARIANCE = 0.01 #TODO - CHECK this is specific to the sensors (in this case - the error in model predictions)
NUMBER_MODES = 4  #Set this to None if you want to use the Rossella et al. selection of truncation parameter

OBS_FRAC = 0.01 #fraction of state used as "observations"
MODE = "single_max" #"single_max" or "rand"

HIST_FRAC = 1 / 3.0 #fraction of data used as "history"
TOL = 1e-3

def main():
    print("alpha =", ALPHA)
    print("obs_var =", OBS_VARIANCE)
    print("obs_frac =", OBS_FRAC)
    print("hist_frac =", HIST_FRAC)
    print("Tol =", TOL)

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

    observations, obs_idx, nobs = vda.select_obs(MODE, u_c, {"fraction": OBS_FRAC}) #options are specific for rand


    #Now define quantities required for 3D-VarDA - see Algorithm 1 in Rossella et al (2019)
    H_0 = vda.create_H(obs_idx, n, nobs)
    d = observations - H_0 @ u_0 #'d' in literature
    #R_inv = vda.create_R_inv(OBS_VARIANCE, nobs)
    V_trunc, U, s, W = vda.trunc_SVD(V, NUMBER_MODES)



    num_modes = s.shape[0]
    #Define intial w_0
    w_0 = np.zeros((W.shape[-1],)) #TODO - I'm not sure about this - can we assume is it 0?
        # in algorithm 2 we use:
        # V_plus_trunc = W.T * (1 / s) @  U.T

    #Define costJ and grad_J
    args =  (d, H_0, V_trunc, ALPHA, OBS_VARIANCE) # list of all args required for cost_function_J and grad_J
    #args =  (d, H_0, V_trunc, ALPHA, None, R_inv) # list of all args required for cost_function_J and grad_J
    res = minimize(vda.cost_function_J, w_0, args = args, method='L-BFGS-B', jac=vda.grad_J, tol=TOL)

    w_opt = res.x
    delta_u_DA = V_trunc @ w_opt
    u_DA = u_0 + delta_u_DA

    ref_MAE = np.abs(u_0 - u_c)
    da_MAE = np.abs(u_DA - u_c)

    ref_MAE_mean = np.mean(ref_MAE)
    da_MAE_mean = np.mean(da_MAE)

    print("RESULTS")
    print("Reference MAE: ", ref_MAE_mean)
    print("DA MAE: ", da_MAE_mean)
    print("If DA has worked, DA MAE > Ref_MAE")
    #Compare abs(u_0 - u_c).sum() with abs(u_DA - u_c).sum() in paraview

    #Save .vtu files so that I can look @ in paraview
    sample_fp = vda.get_sorted_fps_U(settings.DATA_FP)[0]
    out_fp_ref = settings.INTERMEDIATE_FP + "ref_MAE.vtu"
    out_fp_DA =  settings.INTERMEDIATE_FP + "DA_MAE.vtu"

    vda.save_vtu_file(ref_MAE, "ref_MAE", out_fp_ref, sample_fp)
    vda.save_vtu_file(da_MAE, "DA_MAE", out_fp_DA, sample_fp)


if __name__ == "__main__":
    main()
