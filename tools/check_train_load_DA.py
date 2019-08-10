"""This is used to check that a new model sucessfully initializes
Trains, runs and can be used for DA"""

from pipeline import ML_utils, GetData, SplitData
from pipeline import TrainAE
from pipeline.VarDA.batch_DA import BatchDA
from pipeline.settings.config import Config
import shutil

################# Import models
from pipeline.settings.block_models import BaselineRes
from pipeline.settings.block_models import Res34AE, Res34AE_Stacked, Cho2019
from pipeline.settings.models_.resNeXt import Baseline1Block, ResNeXt
from pipeline.settings.baseline_explore import Baseline1



#################### Models to init
resNext_1 = {"layers": 2, "cardinality": 2}
resNext_2 = {"layers": 4, "cardinality": 4}
resNext_3 = {"layers": 2, "cardinality": 2}

CONFIGS = [ResNeXt, ResNeXt, ResNeXt]
KWARGS = (resNext_1,  resNext_2, resNext_3)
##################
CONFIGS = CONFIGS[2]
KWARGS = (KWARGS[2],)

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
        check_train_load_DA(configs[idx], KWARGS[idx], SMALL_DEBUG_DOM, ALL_DATA)
        print()

def run_DA_batch(settings, model, all_data, expdir):
    settings.DEBUG = False
    #set control_states
    #Load data
    loader, splitter = GetData(), SplitData()
    X = loader.get_X(settings)

    train_X, test_X, u_c_std, X, mean, std = splitter.train_test_DA_split_maybe_normalize(X, settings)

    if all_data:
        control_states = X
        print_every = 50
    else:
        NUM_STATES = 5
        START = 100
        control_states = train_X[START:NUM_STATES + START]
        print_every = 10
    if settings.COMPRESSION_METHOD == "AE":
        out_fp = expdir + "AE.csv" #this will be written and then deleted
    else:
        out_fp = expdir + "SVD.csv"

    batch_DA_AE = BatchDA(settings, control_states, csv_fp= out_fp, AEModel=model,
                        reconstruction=True, plot=False)

    return batch_DA_AE.run(print_every=print_every, print_small=True)

def check_train_load_DA(config, config_kwargs, small_debug=True, all_data=False):
    expdir = "experiments/CTL/"
    try:
        if not config_kwargs:
            config_kwargs = {}
        assert isinstance(config_kwargs, dict)

        settings = config(**config_kwargs)
        settings.DEBUG = False

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

        res_AE = run_DA_batch(settings, model, all_data, expdir)

        print(res_AE.tail())
        shutil.rmtree(expdir, ignore_errors=False, onerror=None)
    except Exception as e:
        try:
            shutil.rmtree(expdir, ignore_errors=False, onerror=None)
            raise e
        except Exception as z:
            raise e



if __name__ == "__main__":
    main()

