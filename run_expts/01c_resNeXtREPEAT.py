"""
Duplicate experiment 01c with a larger set of values.

Split over 4 nodes. and use training scheme from 02a_rbtype.py

"""
from pipeline import TrainAE, ML_utils, GetData, SplitData
from pipeline.VarDA.batch_DA import BatchDA

from run_expts.expt_config import ExptConfigTest
from pipeline.settings.models.resNeXt import ResStack3


TEST = False
GPU_DEVICE = 3
exp_base = "experiments/train/01cREPEAT/"

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

    layers = [3, 6, 9, 18, 27]
    cardinalities = [1, 4, 8, 16, 32]
    substructure = "ResNeXt3"
    block = "NeXt"

    if TEST:
        expt = ExptConfigTest()
    else:
        expt = ExptConfig()

    idx = 0
    for layer in layers:
        for cardinality in cardinalities:
            idx_ = idx
            idx += 1
            #split work across 4 gpus:
            if idx_ % 4 != GPU_DEVICE:
                continue

            kwargs = {"layers": layer, "cardinality": cardinality,
                    "block_type": block,
                    "module_type": substructure}
            for k, v in kwargs.items():
                print("{}={}, ".format(k, v), end="")
            print()

            settings = ResStack3(**kwargs)
            settings.GPU_DEVICE = GPU_DEVICE
            settings.export_env_vars()

            expdir = exp_base + str(idx_) + "/"

            trainer = TrainAE(settings, expdir, expt.calc_DA_MAE)
            expdir = trainer.expdir #get full path


            model = trainer.train(expt.EPOCHS, test_every=expt.test_every,
                                    num_epochs_cv=expt.num_epochs_cv,
                                    learning_rate = expt.LR, print_every=expt.print_every,
                                    small_debug=expt.SMALL_DEBUG_DOM)




if __name__ == "__main__":
    main()

