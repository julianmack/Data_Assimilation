"""File to run elements of pipeline module from"""
from pipeline.settings import config
from pipeline import TrainAE
from pipeline.settings.CAE7 import CAE7


def main():
    mults = [1, 2, 4, 8]
    latent_mults = [2, 4]
    BNs = [ False]
    for latent in latent_mults:
        for mult in mults:
            for BN in BNs:
                settings = CAE7(mult, latent)
                settings.BATCH_NORM = BN
                settings.CHANGEOVER_DEFAULT = 0
                settings.REDUCED_SPACE = True
                settings.DEBUG = False
                settings.SHUFFLE_DATA = True
                settings.FIELD_NAME = "Pressure"


                if BN:
                    batch = "BN"
                else:
                    batch = "NBN"
                expdir = "experiments/train_DA_{}/{}-l{}{}".format(settings.FIELD_NAME, mult, latent, batch)
                epochs = 300
                calc_DA_MAE = True
                num_epochs_cv = 20
                small_debug = False
                print_every = 5
                test_every = 5
                lr = 0.001
                try:
                    trainer = TrainAE(settings, expdir, calc_DA_MAE)

                    model = trainer.train(epochs, learning_rate=lr, test_every=test_every, num_epochs_cv=num_epochs_cv,
                                            print_every=print_every, small_debug=small_debug)
                except:
                    try:
                        trainer = TrainAE(settings, expdir, calc_DA_MAE, batch_sz=8)

                        model = trainer.train(epochs, learning_rate=lr, test_every=test_every, num_epochs_cv=num_epochs_cv,
                                                print_every=print_every, small_debug=small_debug)
                    except:
                        pass


if __name__ == "__main__":
    main()
