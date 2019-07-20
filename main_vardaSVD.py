"""File to run elements of pipeline module from"""
from pipeline import DAPipeline
from pipeline.settings.CAE_configs import CAE5
from pipeline.settings import config

def main():

    settings = config.Config()

    settings.NORMALIZE = False
    settings.UNDO_NORMALIZE = False
    settings.SHUFFLE_DATA = False
    settings.OBS_VARIANCE = 10000000000000000000
    settings.SAVE = False
    settings.THREE_DIM = True
    settings.set_X_fp(settings.INTERMEDIATE_FP + "X_3D_{}.npy".format(settings.FIELD_NAME))
    settings.set_n( (91, 85, 32))

    print(settings.get_n())
    DA = DAPipeline(settings)

    print(settings.COMPRESSION_METHOD)
    DA.run()


if __name__ == "__main__":
    main()
