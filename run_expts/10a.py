"""
Timing experiments
"""

from VarDACAE.settings.models.resNeXt import ResStack3
from VarDACAE.settings.models.CLIC import CLIC, GRDNBaseline

from VarDACAE import TrainAE, ML_utils, BatchDA
from tools.calc_DA_dir_best import calc_DA_dir
from run_expts.expt_config import ExptConfigTest


TEST = False
EXPDIR = "experiments/time/10a/gpu/"
ALL_DATA = True
PARAMS = {"var": 0.005, "tol":1e-3}
GPU_DEVICE = "CPU" #CPU
DIR = "experiments/models/"
PRINT = True

def main():

    mse_DA, model_data = calc_DA_dir(DIR, PARAMS, EXPDIR,
                            prnt=PRINT, all_data=ALL_DATA,
                            save_vtu=False, gpu_device=GPU_DEVICE)

if __name__ == "__main__":
    main()


