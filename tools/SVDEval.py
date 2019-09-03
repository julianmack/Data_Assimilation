"""File to run elements of VarDACAE module from"""

from VarDACAE.settings.base_3D import Config3D
from VarDACAE import BatchDA, SplitData

import numpy as np

DEBUG = False
SINGLE_STATE = False
SAVE_VTU = False
TEST = False

def VarDASVD(num_modes, csv_fp=None, debug = DEBUG,
            single_state = False, all_obs=False, save_vtu=False, nobs=None):
    """Calculates VarDA percentage improvement averaged across the
    whole test set. """

    settings = Config3D()

    settings.NORMALIZE = True
    settings.UNDO_NORMALIZE = True
    settings.SHUFFLE_DATA = True
    settings.OBS_VARIANCE = 0.005
    settings.NUMBER_MODES = num_modes
    settings.DEBUG = debug
    settings.TOL = 1e-3
    if nobs is not None and all_obs is True:
        raise ValueError("Only set one of: nobs or all_obs=True")

    if nobs:
        settings.NOBS = nobs
        settings.OBS_MODE = "rand"

    if all_obs == True:
        settings.OBS_FRAC = 1.0
        settings.OBS_MODE = "all"

    settings.SAVE = True

    loader, splitter = settings.get_loader(), SplitData()
    X = loader.get_X(settings)

    _, test_X, u_c_std, _, _, _ = splitter.train_test_DA_split_maybe_normalize(X, settings)
    if single_state:
        u_c = np.expand_dims(u_c_std, 0)
    else:
        u_c = test_X

    batcher = BatchDA(settings, u_c, csv_fp=csv_fp, AEModel=None,
                        reconstruction=True, save_vtu=save_vtu)

    res = batcher.run(print_every=5, print_small=True)


if __name__ == "__main__":
    num_obs = [248, 25, 2]
    mode = 32
    exp_base = "experiments/TSVD3/extra/"
    for nobs in num_obs:
        csv_fp = "{}modes{}_{}_obs.csv".format(exp_base, str(mode), str(nobs))
        VarDASVD(mode, csv_fp, False, SINGLE_STATE, False, SAVE_VTU, nobs)
    exit()
    modes = [1, 2, 4, 8, 12, 16, 32, 60, 100, 150, 250, 350, 450, 550, 700, 791]
    if TEST:
        modes = [1, 791]
        SINGLE_STATE = True
        DEBUG = True

    num_obs = [24750]
    exp_base = "experiments/TSVD3/modes/"
    for nobs in num_obs:
        for mode in modes:

            csv_fp = "{}modes{}_{}_obs.csv".format(exp_base, str(mode), str(nobs))
            print("nobs", nobs, "mode", mode)

            VarDASVD(mode, csv_fp, False, SINGLE_STATE, False, SAVE_VTU, nobs)

    total = 91 * 85 * 32
    num_obs = [2 ** x for x in range(20)]
    if TEST:
        num_obs = [2 ** x for x in range(16, 19)]
    final = False
    exp_base = "experiments/TSVD3/nobs/"
    modes = [32, 791]
    for mode in modes:
        for nobs in num_obs:

            if final:
                final = False
                break
            if nobs > total:
                nobs = total
                final = True

            print("nobs", nobs, "mode", mode)

            csv_fp = "{}modes{}_{}_obs.csv".format(exp_base, str(mode), str(nobs))
            VarDASVD(mode, csv_fp, DEBUG, SINGLE_STATE, False, SAVE_VTU, nobs)
