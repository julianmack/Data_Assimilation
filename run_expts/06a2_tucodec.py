"""
After success of first experiment (particularly with vanilla blocks)
 I want to investigate the effect of
 1) Increasing the number of channels
 2) Increasing the amount of augmentation (as overfitting was a problem)
        i.e. I have removed the FieldJitter(0, 0) augmentation

"""

from pipeline.settings.models_.resNeXt import ResStack3
from pipeline.settings.models_.CLIC import CLIC


from pipeline import TrainAE, ML_utils, GetData, SplitData
from pipeline.VarDA.batch_DA import BatchDA
from run_expts.expt_config import ExptConfigTest


TEST = True
GPU_DEVICE = 0
exp_base = "experiments/train/06a2/"

#global variables for DA and training:
class ExptConfig():
    EPOCHS = 300
    SMALL_DEBUG_DOM = False #For training
    calc_DA_MAE = True
    num_epochs_cv = 0
    LR = 0.0002
    print_every = 10
    test_every = 10

def main():
    blocks = ["vanilla"]
    channels = [128, 196]
    augmentation = []


    if TEST:
        expt = ExptConfigTest()
    else:
        expt = ExptConfig()

    idx = 0

    for block in blocks:
        for Cstd in channels:
            kwargs = {"model_name": "Tucodec", "block_type": block, "Cstd": Cstd}
            idx += 1

            for k, v in kwargs.items():
                print("{}={}, ".format(k, v), end="")
            print()

            settings = CLIC(**kwargs)
            settings.GPU_DEVICE = GPU_DEVICE
            settings.export_env_vars()

            expdir = exp_base + str(idx - 1) + "/"


            trainer = TrainAE(settings, expdir, expt.calc_DA_MAE)
            expdir = trainer.expdir #get full path


            model = trainer.train(expt.EPOCHS, test_every=expt.test_every,
                                    num_epochs_cv=expt.num_epochs_cv,
                                    learning_rate = expt.LR, print_every=expt.print_every,
                                    small_debug=expt.SMALL_DEBUG_DOM)



if __name__ == "__main__":
    main()


