"""
Train varying numbers of RNABs within basic structure.

Experiment with removing the attenuation

"""

from varda_cae.settings.models.resNeXt import ResStack3


from varda_cae import TrainAE, ML_utils, GetData, SplitData
from varda_cae.VarDA.batch_DA import BatchDA
from run_expts.expt_config import ExptConfigTest


TEST = False
GPU_DEVICE = 1
exp_base = "experiments/train/03b/"

#global variables for DA and training:
class ExptConfig():
    EPOCHS = 150
    SMALL_DEBUG_DOM = False #For training
    calc_DA_MAE = True
    num_epochs_cv = 0
    LR = 0.0002
    print_every = 10
    test_every = 10

def main():
    kwargs = {"cardinality": 1, "block_type": "RNAB",
                    "module_type": "Bespoke", "attenuation": False}
    layers = [1, 2, 4, 8]

    if TEST:
        expt = ExptConfigTest()
    else:
        expt = ExptConfig()

    idx = 0

    for layer in layers:
        kwargs["subBlock"] = "NeXt" #this performed slightly better on first case
        kwargs["layers"] = layer
        idx += 1

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


