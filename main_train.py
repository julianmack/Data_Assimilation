"""File to run elements of pipeline module from"""
from pipeline import config
from pipeline import TrainAE
from pipeline.AEs import CAE_configs
import shutil

def main():
    epochs = 1
    settings = CAE_configs.CAE1()
    settings.BATCH_NORM = False
    settings.CHANGEOVER_DEFAULT = 0
    expdir = "experiments/rand/x"
    trainer = TrainAE(settings, expdir)
    num_encode = len(trainer.model.layers_encode)

    model = trainer.train(epochs)

    shutil.rmtree(expdir, ignore_errors=False, onerror=None)

if __name__ == "__main__":
    main()
