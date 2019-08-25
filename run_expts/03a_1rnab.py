"""
Experiment to see if RNAB works within the baseline.
Just train two models - if either works then expt 03b will expand on this
If not - move on.
Train for two block types:
    blocks = ["NeXt", "vanilla"]

"""

from varda_cae.settings.models.resNeXt import ResStack3


from varda_cae import TrainAE, ML_utils, GetData, SplitData
from varda_cae.VarDA.batch_DA import BatchDA
from run_expts.expt_config import ExptConfigTest


TEST = False
GPU_DEVICE = 0
exp_base = "experiments/train/03a/"

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
    blocks = ["NeXt", "vanilla"]
    kwargs = {"layers": 1, "cardinality": 1, "block_type": "RNAB",
                    "module_type": "Bespoke"}
    if TEST:
        expt = ExptConfigTest()
    else:
        expt = ExptConfig()

    idx = 0

    for block in blocks:
        kwargs["subBlock"] = block
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


