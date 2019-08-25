"""This is used to check that a new model sucessfully initializes
Trains, runs and can be used for DA"""

from VarDACAE import ML_utils, GetData, SplitData
from VarDACAE import TrainAE
from VarDACAE.VarDA.batch_DA import BatchDA
from VarDACAE.settings.base import Config

import shutil

################# Import models
from VarDACAE.settings.explore.block_models import BaselineRes
from VarDACAE.settings.explore.block_models import Res34AE, Res34AE_Stacked, Cho2019
from VarDACAE.settings.models.resNeXt import Baseline1Block, ResNeXt, ResStack3
from VarDACAE.settings.explore.baseline_explore import Baseline1
from VarDACAE.settings.models.CLIC import CLIC, GRDNBaseline
import os

VAR = 0.05
TOL = 1e-2
ACTIVATION = "GDN"
EXPDIR = "experiments/CTL/"

resNext3_k = {"layers": 3, "cardinality": 1, "block_type": "RNAB",
                "module_type": "Bespoke", "sigmoid": True}
clic_K = {"model_name": "Tucodec", "block_type": "NeXt", "Cstd": 64}
grdn_k = {"block_type": "NeXt", "Cstd": 2}

CONFIGS = [ResStack3, CLIC, GRDNBaseline]
KWARGS = ( resNext3_k, clic_K, grdn_k)
#################
CONFIGS = CONFIGS[1]
KWARGS = (KWARGS[1],)


#global variables for DA and training:
EPOCHS = 1
SMALL_DEBUG_DOM = True #For training
ALL_DATA = False
PRINT_MODEL = False

def main():

    if isinstance(CONFIGS, list):
        configs = CONFIGS
    else:
        configs = (CONFIGS, ) * len(KWARGS)
    assert len(configs) == len(KWARGS)

    for idx, conf in enumerate(configs):
        check_train_load_DA(configs[idx], KWARGS[idx], SMALL_DEBUG_DOM, ALL_DATA, ACTIVATION)
        print()

def run_DA_batch(settings, model, all_data, expdir, params={"var": VAR, "tol": TOL}):
    """By default it evaluates over the whole test set"""
    settings.DEBUG = False
    settings.NORMALIZE = True
    settings.UNDO_NORMALIZE = True
    settings.SHUFFLE_DATA = True
    settings.OBS_VARIANCE = params.get("var") if params.get("var") else 0.005
    settings.TOL = params.get("tol") if params.get("tol") else 1e-2
    #set control_states
    #Load data
    loader, splitter = GetData(), SplitData()
    X = loader.get_X(settings)

    train_X, test_X, u_c_std, X, mean, std = splitter.train_test_DA_split_maybe_normalize(X, settings)

    if all_data:
        control_states = test_X
        if settings.COMPRESSION_METHOD == "AE":
            print_every = 50
        else:
            print_every = 500 #i.e. don't print
    else:
        NUM_STATES = 5
        START = 100
        control_states = test_X[START:NUM_STATES + START]
        print_every = 10
    if settings.COMPRESSION_METHOD == "AE":
        out_fp = os.path.join(expdir, "AE.csv")#this will be written and then deleted
    else:
        out_fp = os.path.join(expdir, "SVD.csv")

    batch_DA_AE = BatchDA(settings, control_states, csv_fp= out_fp, AEModel=model,
                        reconstruction=True, plot=False)

    return batch_DA_AE.run(print_every=print_every, print_small=True)

def check_train_load_DA(config, config_kwargs,  small_debug=True, all_data=False,
                        activation=None, params={"var": VAR, "tol": TOL}):
    expdir = EXPDIR
    try:
        if not config_kwargs:
            config_kwargs = {}
        assert isinstance(config_kwargs, dict)

        settings = config(**config_kwargs)
        settings.DEBUG = False
        if activation:
            settings.ACTIVATION = activation

        calc_DA_MAE = True
        num_epochs_cv = 0
        print_every = 1
        test_every = 1
        lr = 0.0003

        print(settings.__class__.__name__)
        if config_kwargs:
            print(list([(k, v) for (k, v) in config_kwargs.items()]))
        trainer = TrainAE(settings, expdir, calc_DA_MAE)
        expdir = trainer.expdir #get full path

        model = trainer.train(EPOCHS, learning_rate=lr, test_every=test_every, num_epochs_cv=num_epochs_cv,
                                print_every=print_every, small_debug=small_debug)

        if PRINT_MODEL:
            print(model.layers_encode)
        #test loading
        model, settings = ML_utils.load_model_and_settings_from_dir(expdir)


        model.to(ML_utils.get_device()) #TODO

        x_fp = settings.get_X_fp(True) #force init X_FP

        res_AE = run_DA_batch(settings, model, all_data, expdir, params)

        print(res_AE.head(10))
        shutil.rmtree(expdir, ignore_errors=False, onerror=None)
    except Exception as e:
        try:
            shutil.rmtree(expdir, ignore_errors=False, onerror=None)
            raise e
        except Exception as z:
            raise e



if __name__ == "__main__":
    main()

