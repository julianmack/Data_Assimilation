"""File to run elements of pipeline module from"""
from pipeline.settings import config
from pipeline import TrainAE
from pipeline.settings.CAE7 import CAE7
import shutil

def main():
    epochs = 40
    settings = CAE7()
    settings.BATCH_NORM = False
    settings.CHANGEOVER_DEFAULT = 0
    settings.REDUCED_SPACE = True
    settings.DEBUG = False
    settings.SHUFFLE_DATA = True
    settings.FIELD_NAME = "Tracer"

    expdir = "experiments/train_DA_{}".format(settings.FIELD_NAME)
    calc_DA_MAE = True
    num_epochs_cv = 0
    small_debug = True
    trainer = TrainAE(settings, expdir, calc_DA_MAE)
    num_encode = len(trainer.model.layers_encode)

    model = trainer.train(epochs, test_every=1, num_epochs_cv=num_epochs_cv,
                            small_debug=small_debug)

    #Uncomment line below if you want to automatically delete expdir (useful during testing)
    shutil.rmtree(expdir, ignore_errors=False, onerror=None)

if __name__ == "__main__":
    main()
