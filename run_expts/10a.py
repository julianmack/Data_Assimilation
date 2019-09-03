"""
Timing experiments
"""

from VarDACAE.settings.models.resNeXt import ResStack3
from VarDACAE.settings.models.CLIC import CLIC, GRDNBaseline

from VarDACAE import TrainAE, ML_utils, BatchDA
from tools.calc_DA_dir_best import calc_DA_dir
from run_expts.expt_config import ExptConfigTest, ExptConfig
from collections import OrderedDict
import pickle
import pandas as pd


TEST = False
PRINT = True
PARAMS = {"var": 0.005, "tol":1e-3}
GPU_DEVICE = "CPU"
NUM_GPU = 1
DIR = "experiments/models/"

NEW_EXPDIR_BASE = "experiments/time/best/"
LOCATION_BASE = "models_/best/"

########################
TRAIN1 = [("L1", 50)]
TRAIN2 = [("L2", 150), ("L1", 50)]

#NOTE: these locs are NOT used
models = OrderedDict([
    ("Backbone", {})
    #("tucodec_relu_vanilla", {"loc": 'experiments/06a5/12', "sched": TRAIN1}),
    #("tucodec_prelu_next", {"loc": 'experiments/DA3/06a/1/', "sched": TRAIN1}),
    # ("RDB3_27_4",  {"loc": 'experiments/09a/09a/0', "sched": TRAIN2}),
    # ("ResNeXt_27_1", {"loc": 'experiments/09a/09a/2', "sched": TRAIN2}),
    # ("RAB_4_next",   {"loc": 'experiments/03c/10/', "sched": TRAIN2}),
    # ("GDRN_CBAM",  {"loc": 'experiments/09c/0', "sched": TRAIN2})
    ])

def main():
    if TEST:
        expt = ExptConfigTest()
        expt.calc_DA_MAE = False
    else:
        expt = ExptConfig()

    idx = 0
    times = []
    for name in models.keys():
        idx_ = idx
        idx += 1
        if GPU_DEVICE != "CPU" and idx_ % NUM_GPU != GPU_DEVICE:
            continue

        expdir_new = NEW_EXPDIR_BASE + name + "/"
        location = LOCATION_BASE + name + "/"


        mse_DA, model_data, df = calc_DA_dir(location, PARAMS, expdir_new,
                            prnt=PRINT, all_data=not expt.SMALL_DEBUG_DOM,
                            save_vtu=False, gpu_device=GPU_DEVICE, return_df=True)
        time_online = df["time_online"].mean()
        df = df.drop(0)
        time = df["time"].mean()

        num_params = model_data["num_params"]
        results = {"model": name, "num_params": num_params, "time": time,
                    "time_online": time_online}
        times.append(results)

    final = pd.DataFrame(times)
    final.to_csv(NEW_EXPDIR_BASE + "time.csv")

if __name__ == "__main__":
    main()


