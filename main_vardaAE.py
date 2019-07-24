"""File to run elements of pipeline module from"""
from pipeline import DAPipeline, ML_utils
from pipeline.settings.CAE7 import CAE7
from pipeline.settings import config
from pipeline import DAPipeline
from pipeline.settings.CAE_configs import CAE5

def main():
    azure = False #set to true when working in Azure (different model available)

    if azure:

        dir = "/data/home/jfm1118/DA/experiments/train_DA_Pressure/2-l4NBN/" # 299.pth"
        model, settings = ML_utils.load_model_and_settings_from_dir(dir)

        settings.TOL = 1e-3
        settings.FORCE_GEN_X = False
        settings.SAVE = False
        settings.export_env_vars()
    else:
        settings = CAE5()
        settings.REDUCED_SPACE = True

        model_pth = "models/29.pth"
        settings.AE_MODEL_FP = model_pth

        #CAE57lrelu00.5NBN
        settings.ACTIVATION = "lrelu"
        settings.CHANGEOVER_DEFAULT = 0
        settings.BATCH_NORM = False
        settings.SAVE = False

        settings.NORMALIZE = False
        settings.UNDO_NORMALIZE = False
        settings.SHUFFLE_DATA = False
        settings.OBS_VARIANCE = 10
        channels = settings.get_channels()
        final_channel = channels[1]
        channels[-1] = int(channels[-1] * 0.5)
        model = ML_utils.load_model_from_settings(settings)

    #settings = config.Config()
    DA = DAPipeline(settings, model)

    print(settings.COMPRESSION_METHOD)
    DA.run()


if __name__ == "__main__":
    main()
