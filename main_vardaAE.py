"""File to run elements of pipeline module from"""
from pipeline import DAPipeline
from pipeline.settings.CAE7 import CAE7
from pipeline.settings import config

def main():




    model_pth = "/data/home/jfm1118/DA/experiments/train_DA_Pressure/2-l4NBN/299.pth"
    settings = CAE7(2, 4) #see path above
    settings.REDUCED_SPACE = True
    settings.AE_MODEL_FP = model_pth
    settings.TOL = 1e-3

    settings.ACTIVATION = "lrelu"
    settings.CHANGEOVER_DEFAULT = 0
    settings.BATCH_NORM = False
    settings.SAVE = False

    settings.NORMALIZE = False
    settings.UNDO_NORMALIZE = False
    settings.SHUFFLE_DATA = False
    settings.OBS_VARIANCE = 10


    #settings = config.Config()
    DA = DAPipeline(settings)

    #settings = config.Config()



    print(settings.COMPRESSION_METHOD)
    DA.run()


if __name__ == "__main__":
    main()
