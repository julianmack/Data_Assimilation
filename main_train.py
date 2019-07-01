"""File to run elements of pipeline module from"""
from pipeline import config
from pipeline import TrainAE
from pipeline.AEs import CAE_configs


def main():
    epochs = 1
    settings = CAE_configs.CAE6()
    expdir = "experiments/CAE6_test"
    trainer = TrainAE(settings, expdir)
    model = trainer.train(epochs)



if __name__ == "__main__":
    main()
