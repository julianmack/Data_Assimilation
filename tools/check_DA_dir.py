"""File to run elements of pipeline module from"""
from pipeline import ML_utils, GetData, SplitData
from pipeline.VarDA.batch_DA import BatchDA
from pipeline.settings import config
from tools.check_train_load_DA import run_DA_batch
import shutil


DIR = "experiments/train/01_resNeXt_3/0/2/"
DIR = "experiments/train/00c_baseResNext/" #DA4/DA3


CONFIGS = [DIR]
KWARGS = [0,]
##################


#global variables for DA:
ALL_DATA = False
EXPDIR = "experiments/DA/load/"
SAVE = False

def main(params, prnt=True):

    if isinstance(CONFIGS, list):
        configs = CONFIGS
    else:
        configs = (CONFIGS, ) * len(KWARGS)
    assert len(configs) == len(KWARGS)

    for idx, conf in enumerate(configs):
        check_DA_dir(configs[idx], KWARGS[idx], ALL_DATA, EXPDIR, params, prnt)
        print()


def check_DA_dir(dir, kwargs, all_data, expdir, params, prnt):
    try:
        model, settings = ML_utils.load_model_and_settings_from_dir(dir)
        df = run_DA_batch(settings, model, all_data, expdir, params)
        if prnt:
            print(df.tail(10))
    except Exception as e:
        try:
            shutil.rmtree(expdir, ignore_errors=False, onerror=None)
            raise e
        except Exception as z:
            raise e



if __name__ == "__main__":
    main(params = {"var": 0.005, "tol":1e-3})
    exit()
    prnt = False
    for tol in [1e-5, 1e-4, 1e-3, 1e-2]:
        for var in [5, 0.5, 0.005, 0.0005, 0.00005]:
            params = {"var": var, "tol":tol}
            print("var: {:.5f}, tol: {:.5f}".format(var, tol))
            main(params, prnt)



