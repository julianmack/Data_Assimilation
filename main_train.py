"""File to run elements of pipeline module from"""
from pipeline import config
from pipeline import TrainAE

def main():

    settings = config.CAEConfig()
    expdir = "experiments/CAE_first"
    trainer = TrainAE(settings, expdir)
    model = trainer.train(1)


if __name__ == "__main__":
    main()
