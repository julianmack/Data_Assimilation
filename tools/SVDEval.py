"""File to run elements of VarDACAE module from"""

from VarDACAE.settings.base_3D import Config3D
from VarDACAE import BatchDA, SplitData

import numpy as np

DEBUG = False
SINGLE_STATE = False
ALL_OBSERVATIONS = True
SAVE_VTU = True

def VarDASVD(num_modes, csv_fp=None, debug = DEBUG,
            single_state = False, all_obs=False, save_vtu=False):
    """Calculates VarDA percentage improvement averaged across the
    whole test set. """

    settings = Config3D()

    settings.NORMALIZE = True
    settings.UNDO_NORMALIZE = True
    settings.SHUFFLE_DATA = True
    settings.OBS_VARIANCE = 0.5
    settings.NUMBER_MODES = num_modes
    settings.DEBUG = debug
    settings.TOL = 1e-3
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
    modes = [1, 2, 4, 8, 12, 16, None, 60, 100, 150, 250, 350, 450, 550, 700, 791]

    ALL_OBSERVATIONS = True

    exp_base = "experiments/TSVD/modes/"
    for mode in modes:
        print(mode)
        csv_fp = "{}modes{}ALL_obs.csv".format(exp_base, str(mode))
        VarDASVD(mode, csv_fp, DEBUG, SINGLE_STATE, ALL_OBSERVATIONS, SAVE_VTU)
