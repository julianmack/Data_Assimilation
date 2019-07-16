"""File to run elements of pipeline module from"""
from pipeline import DAPipeline
from pipeline.settings.CAE_configs import CAE5
from pipeline.settings import config

def main():

    settings = CAE5()
    model_pth = "models/29.pth"
    settings.AE_MODEL_FP = model_pth

    #CAE5-7-lrelu-0-0.5-NBN
    settings.ACTIVATION = "lrelu"
    settings.CHANGEOVER_DEFAULT = 0
    settings.BATCH_NORM = False
    channels = settings.get_channels()
    final_channel = channels[-1]
    channels[-1] = int(channels[-1] * 0.5)

    #settings = config.Config()
    DA = DAPipeline(settings)

    #settings = config.Config()



    print(settings.COMPRESSION_METHOD)
    DA.run()


if __name__ == "__main__":
    main()
