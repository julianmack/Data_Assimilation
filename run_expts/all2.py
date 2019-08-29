"""
Expt 07c and 09b

"""

from VarDACAE.settings.models.resNeXt import ResStack3
from VarDACAE.settings.models.CLIC import CLIC

from VarDACAE import TrainAE, ML_utils, BatchDA

from run_expts.expt_config import ExptConfigTest


TEST = False
GPU_DEVICE = 0
NUM_GPU = 1


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
    exp_base = "experiments/train2/07c/"
    kwargs = {"model_name": "Tucodec", "block_type": "NeXt",
            "Cstd": 64, "sigmoid": False,
            "activation": "prelu"}
    aug_schemes = list(range(5, 7))

    if TEST:
        expt = ExptConfigTest()
    else:
        expt = ExptConfig()
        expt.EPOCHS = 300

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

    exp_base = "experiments/train2/09b/"

    activations = ["relu", "GDN"]
    if TEST:
        expt = ExptConfigTest()
    else:
        expt = ExptConfig()
        expt.EPOCHS = 150

    for act in activations:

        kwargs = {"layers": 0, "cardinality": 2,
                    "aug_scheme": 0, "activation": act}


        idx_ = idx
        idx += 1
        if idx_ % NUM_GPU != GPU_DEVICE:
            continue


        for k, v in kwargs.items():
            print("{}={}, ".format(k, v), end="")
        print()

        settings = ResStack3(**kwargs)
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


