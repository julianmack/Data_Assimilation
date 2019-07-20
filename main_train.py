"""File to run elements of pipeline module from"""
from pipeline.settings import config
from pipeline import TrainAE
from pipeline.settings.CAE7 import CAE7
import shutil

def main():
    epochs = 3
    settings = CAE7()
    settings.BATCH_NORM = False
    settings.CHANGEOVER_DEFAULT = 0
    expdir = "experiments/train_DA"
    calc_DA_MAE = True
    num_epochs_cv = 0
    trainer = TrainAE(settings, expdir, calc_DA_MAE)
    num_encode = len(trainer.model.layers_encode)

    model = trainer.train(epochs, test_every=1, num_epochs_cv=num_epochs_cv)

    #shutil.rmtree(expdir, ignore_errors=False, onerror=None)

if __name__ == "__main__":
    main()
