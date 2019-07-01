"""File to run elements of pipeline module from"""
from pipeline import config
from pipeline import TrainAE
from pipeline.AEs import CAE_configs


def main():
    epochs = 4
    settings = CAE_configs.CAE3()
    settings.BATCH_NORM = False
    expdir = "experiments/rand/CAE3"
    trainer = TrainAE(settings, expdir)
    num_encode = len(trainer.model.layers_encode)

    model = trainer.train(epochs)



if __name__ == "__main__":
    main()
