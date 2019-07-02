"""File to run elements of pipeline module from"""
from pipeline.settings import config
from pipeline import TrainAE
from pipeline.settings import CAE_configs
import shutil

def main():
    epochs = 3
    settings = CAE_configs.CAE1()
    settings.BATCH_NORM = False
    settings.CHANGEOVER_DEFAULT = 0
    expdir = "experiments/rand/cae1"
    trainer = TrainAE(settings, expdir)
    num_encode = len(trainer.model.layers_encode)

    model = trainer.train(epochs)

    shutil.rmtree(expdir, ignore_errors=False, onerror=None)

if __name__ == "__main__":
    main()
