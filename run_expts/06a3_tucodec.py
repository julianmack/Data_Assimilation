"""
I latterly realised that the RNAB/RAB module had no sigmoid in the attention calc.
The 6a [and 6a2] results are good though so I want to test here if addition of this
sigmoid makes the results better/worse.

6a2 is was *very* slow with 96 and the results were not better than w. 64.
Therefore I will stick with Cstd = 64

Run on a subset of previous results:
Cstd = 64, block = [vanilla, NeXt], sigmoid=True

Only 2 but split between
"""

from VarDACAE.settings.models.resNeXt import ResStack3
from VarDACAE.settings.models.CLIC import CLIC


from VarDACAE import TrainAE, ML_utils, GetData, SplitData
from VarDACAE.VarDA.batch_DA import BatchDA
from run_expts.expt_config import ExptConfigTest


TEST = False
GPU_DEVICE = 1
NUM_GPU = 2
exp_base = "experiments/train/06a3/"

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
    blocks = ["NeXt", "vanilla",]
    channels = [64]
    sigmoid = True

    if TEST:
        expt = ExptConfigTest()
    else:
        expt = ExptConfig()

    idx = 0

    for block in blocks:
        for Cstd in channels:
            kwargs = {"model_name": "Tucodec", "block_type": block,
                    "Cstd": Cstd, "sigmoid": sigmoid}
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


