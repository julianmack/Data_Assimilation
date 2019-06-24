"""File to run elements of pipeline module from"""
from pipeline import config
from pipeline import TrainAE

def main():

    settings = config.CAEConfig()
    number_modes = settings.NUMBER_MODES
    trainer = TrainAE(settings)
    model = trainer.train()


if __name__ == "__main__":
    main()
