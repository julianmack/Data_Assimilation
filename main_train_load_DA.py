"""File to run elements of pipeline module from"""
from pipeline.settings.baseline import Baseline
from pipeline.settings.block_models import Baseline1_replicate, BaselineRes

from pipeline import TrainAE, ML_utils, GetData, SplitData
from pipeline.VarDA.batch_DA import BatchDA
from pipeline.settings.block_explore import Block
from pipeline.settings.block_models import Res34AE, Res34AE_Stacked, Cho2019


import shutil

#global variables for DA and training:
EPOCHS = 1
SMALL_DEBUG_DOM = True #For training
ALL_DATA = False #for DA
SAVE = False

def main():

    settings = BaselineRes()
    #model = ML_utils.load_model_from_settings(settings)


    settings.BATCH_NORM = False
    settings.REDUCED_SPACE = True
    settings.DEBUG = False
    settings.SHUFFLE_DATA = True #Set this =False for harder test and train set
    settings.FIELD_NAME = "Pressure"

    expdir = "experiments/train/block/test/"


    calc_DA_MAE = True
    num_epochs_cv = 0
    print_every = 5
    test_every = 2
    lr = 0.001
    trainer = TrainAE(settings, expdir, calc_DA_MAE)
    expdir = trainer.expdir #get full path


    model = trainer.train(EPOCHS, learning_rate=lr, test_every=test_every, num_epochs_cv=num_epochs_cv,
                            print_every=print_every, small_debug=SMALL_DEBUG_DOM)


    #test loading
    model, settings = ML_utils.load_model_and_settings_from_dir(expdir)


    model.to(ML_utils.get_device()) #TODO

    settings.DEBUG = False
    x_fp = settings.get_X_fp(True) #force init X_FP

    #set control_states
    #Load data
    loader, splitter = GetData(), SplitData()
    X = loader.get_X(settings)

    train_X, test_X, u_c_std, X, mean, std = splitter.train_test_DA_split_maybe_normalize(X, settings)

    if ALL_DATA:
        control_states = X
        print_every = 50
    else:
        NUM_STATES = 5
        START = 100
        control_states = train_X[START:NUM_STATES + START]
        print_every = 10

    if SAVE:
        out_fp = expdir + "AE.csv"
    else:
        out_fp = None


    batch_DA_AE = BatchDA(settings, control_states, csv_fp= out_fp, AEModel=model,
                        reconstruction=True, plot=False)

    res_AE = batch_DA_AE.run(print_every=print_every)

    print(res_AE.tail())
    #Uncomment line below if you want to automatically delete expdir (useful during testing)
    if not SAVE:
        shutil.rmtree(expdir, ignore_errors=False, onerror=None)


if __name__ == "__main__":
    main()

