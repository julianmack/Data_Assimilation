"""File to run elements of pipeline module from"""

from pipeline.settings import config
from pipeline.VarDA.batch_DA import BatchDA
from pipeline import GetData, SplitData
import numpy as np

DEBUG = False
SINGLE_STATE = False

def VarDASVD(num_modes, csv_fp=None, debug = DEBUG, single_state = False):
    """Calculates VarDA percentage improvement averaged across the
    whole test set. """

    settings = config.Config3D()

    settings.NORMALIZE = True
    settings.UNDO_NORMALIZE = True
    settings.SHUFFLE_DATA = True
    settings.OBS_VARIANCE = 0.5
    settings.NUMBER_MODES = num_modes
    settings.DEBUG = debug
    settings.TOL = 1e-2

    loader, splitter = GetData(), SplitData()
    X = loader.get_X(settings)

    _, test_X, u_c_std, _, _, _ = splitter.train_test_DA_split_maybe_normalize(X, settings)
    if single_state:
        u_c = np.expand_dims(u_c_std, 0)
    else:
        u_c = test_X
    batcher = BatchDA(settings, u_c, csv_fp=csv_fp, AEModel=None,
                        reconstruction=True)

    res = batcher.run(print_every=5, print_small=True)


if __name__ == "__main__":
    modes = [1, 2, 4, 8, 16, 33, 150,  750]
    exp_base = "experiments/TSVD/"
    for mode in modes:
        print(mode)
        csv_fp = "{}modes{}.csv".format(exp_base, str(mode))
        VarDASVD(mode, csv_fp, DEBUG, SINGLE_STATE)
