"""File to run elements of pipeline module from"""
from pipeline import DAPipeline
from pipeline.settings.CAE_configs import CAE5
from pipeline.settings import config

def main():

    settings = config.Config()

    settings.NORMALIZE = True
    settings.UNDO_NORMALIZE = True
    settings.SHUFFLE_DATA = True
    settings.OBS_VARIANCE = 0.05
    settings.SAVE = True
    settings.FORCE_GEN_X = False
    settings.AZURE_DOWNLOAD = True

    settings.THREE_DIM = True
    settings.set_n((91, 85, 32))

    DA = DAPipeline(settings)

    print(settings.COMPRESSION_METHOD)
    DA.run()


if __name__ == "__main__":
    main()
