"""File to run elements of pipeline module from"""
from pipeline.settings import config
from pipeline import TrainAE
from pipeline.settings import CAE_configs


def main():
    lrs = [0.001, 0.003, 0.009, 0.02]
    expdir_base =  "experiments/BN-prelim/"

    for lr in lrs[::-1]:
        epochs = 7
        settings = CAE_configs.CAE1()
        settings.BATCH_NORM = True
        settings.CHANGEOVER_DEFAULT = 2
        print(settings.get_channels(), lr, "BN")
        expdir = expdir_base + "BN-" + str(lr)
        trainer = TrainAE(settings, expdir)
        num_encode = len(trainer.model.layers_encode)

        model = trainer.train(epochs, learning_rate= lr)

    #non BATCH_NORM
    for lr in lrs:

        epochs = 8
        settings = CAE_configs.CAE1()
        settings.BATCH_NORM = False
        settings.CHANGEOVER_DEFAULT = 2
        print(settings.get_channels(), lr, "NBN")

        expdir = expdir_base + "NBN-" + str(lr)
        trainer = TrainAE(settings, expdir)
        num_encode = len(trainer.model.layers_encode)

        model = trainer.train(epochs, learning_rate= lr)


if __name__ == "__main__":
    main()
