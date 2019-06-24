"""File to run elements of pipeline module from"""
from pipeline import config
from pipeline import TrainAE

def main():
    epochs = 1
    settings = config.CAEConfig()
    expdir = "experiments/CAE_first"
    trainer = TrainAE(settings, expdir)
    model = trainer.train(epochs)


if __name__ == "__main__":
    main()
