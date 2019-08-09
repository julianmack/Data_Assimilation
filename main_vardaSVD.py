"""File to run elements of pipeline module from"""
from pipeline import DAPipeline
from pipeline.settings.CAE_configs import CAE5
from pipeline.settings import config

def main():

    settings = config.Config()

    settings.NORMALIZE = False
    settings.UNDO_NORMALIZE = False
    settings.SHUFFLE_DATA = False
    settings.OBS_VARIANCE = 10
    settings.SAVE = True
    settings.FORCE_GEN_X = False
    settings.AZURE_DOWNLOAD = True

    settings.THREE_DIM = True
    
    DA = DAPipeline(settings)

    print(settings.COMPRESSION_METHOD)
    DA.run()


if __name__ == "__main__":
    main()
