"""File to run elements of pipeline module from"""
from pipeline import ML_utils, GetData, SplitData
from pipeline.VarDA.batch_DA import BatchDA
from pipeline.settings import config
from tools.check_train_load_DA import run_DA_batch
import shutil


DIR = "experiments/train/01_resNeXt_3/0/2/"
DIR = "experiments/train/00c_baseResNext/"


CONFIGS = [DIR]
KWARGS = [0,]
##################


#global variables for DA:
ALL_DATA = True
EXPDIR = "experiments/DA/load/"
SAVE = False

def main():

    if isinstance(CONFIGS, list):
        configs = CONFIGS
    else:
        configs = (CONFIGS, ) * len(KWARGS)
    assert len(configs) == len(KWARGS)

    for idx, conf in enumerate(configs):
        check_DA_dir(configs[idx], KWARGS[idx], ALL_DATA, EXPDIR)
        print()


def check_DA_dir(dir, kwargs, all_data, expdir):
    try:
        model, settings = ML_utils.load_model_and_settings_from_dir(dir)
        df = run_DA_batch(settings, model, all_data, expdir)
        print(df.tail(10))
    except Exception as e:
        try:
            shutil.rmtree(expdir, ignore_errors=False, onerror=None)
            raise e
        except Exception as z:
            raise e



if __name__ == "__main__":
    main()

