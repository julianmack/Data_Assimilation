"""
Augmentation comparison. Use best Tucodec model.

"""

from varda_cae.settings.models.resNeXt import ResStack3
from varda_cae.settings.models.CLIC import CLIC


from varda_cae import TrainAE, ML_utils, GetData, SplitData
from varda_cae.VarDA.batch_DA import BatchDA
from run_expts.expt_config import ExptConfigTest


TEST = False
GPU_DEVICE = 1
NUM_GPU = 2
exp_base = "experiments/train2/07a/"

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

    kwargs = {"model_name": "Tucodec", "block_type": "NeXt",
            "Cstd": 64, "sigmoid": False,
            "activation": "prelu"}
    aug_schemes = list(range(-1, 5))

    if TEST:
        expt = ExptConfigTest()
    else:
        expt = ExptConfig()

    idx = 0
    for aug_scheme in aug_schemes:
        kwargs["aug_scheme"] = aug_scheme

        idx_ = idx
        idx += 1
        if idx_ % NUM_GPU != GPU_DEVICE:
            continue


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


