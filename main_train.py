"""File to run elements of pipeline module from"""
from pipeline.settings import config
from pipeline import TrainAE
from pipeline.settings.CAE7 import CAE7

import shutil

def main():

    settings = CAE7(1)
    settings.BATCH_NORM = False
    settings.CHANGEOVER_DEFAULT = 0
    settings.REDUCED_SPACE = True
    settings.DEBUG = False
    settings.SHUFFLE_DATA = True #Set this =False for harder test and train set
    settings.FIELD_NAME = "Pressure"

    expdir = "experiments/new_try"
    epochs = 1
    calc_DA_MAE = True
    num_epochs_cv = 0
    small_debug = True
    print_every = 1
    lr = 0.001
    trainer = TrainAE(settings, expdir, calc_DA_MAE)
    num_encode = len(trainer.model.layers_encode)

    model = trainer.train(epochs, learning_rate=lr, test_every=1, num_epochs_cv=num_epochs_cv,
                            print_every=print_every, small_debug=small_debug)

    #Uncomment line below if you want to automatically delete expdir (useful during testing)
    shutil.rmtree(expdir, ignore_errors=False, onerror=None)

if __name__ == "__main__":
    main()
