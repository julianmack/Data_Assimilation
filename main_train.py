"""File to run elements of pipeline module from"""
from pipeline import config
from pipeline import TrainAE

def main():

    settings = config.CAEConfig()
    expdir = "experiments/CAE_zoo"
    trainer = TrainAE(settings, expdir)
    model = trainer.train(2)


if __name__ == "__main__":
    main()
