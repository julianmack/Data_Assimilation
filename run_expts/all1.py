"""
Run a large number of experiments back to back to avoid
wasting computation. Running experiments:
02c, 03c, 06a5


"""

from VarDACAE.settings.models.resNeXt import ResStack3
from VarDACAE.settings.models.CLIC import CLIC

from VarDACAE import TrainAE, ML_utils, BatchDA

from run_expts.expt_config import ExptConfigTest


TEST = True
GPU_DEVICE = 0
NUM_GPU = 1
exp_base = "experiments/train2/08b/"

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
    idx = 0
    ##################### 02c

    structure = (4, 27) #(cardinality, layers)
    substructures = ["ResNeXt3", "RDB3"]
    blocks = ["CBAM_vanilla", "vanilla", "NeXt", "CBAM_NeXt"]

    if TEST:
        expt = ExptConfigTest()
    else:
        expt = ExptConfig()
        expt.EPOCHS = 150

    for substruct in substructures:

        for block in blocks:
            (cardinality, layers) = structure
            idx_ = idx
            idx += 1
            if idx_ % NUM_GPU != GPU_DEVICE:
                continue

            kwargs = {"layers": layers, "cardinality": cardinality,
                    "block_type": block,
                    "module_type": substruct,
                    "aug_scheme": 0}

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

    ################# 03c
    blocks = ["CBAM_vanilla", "vanilla", "CBAM_NeXt"]
    kwargs = {"cardinality": 1, "block_type": "RAB",
                    "sigmoid": True, "module_type": "Bespoke",
                    "attenuation": False, "layers": 4,
                    "aug_scheme": 0}

    if TEST:
        expt = ExptConfigTest()
    else:
        expt = ExptConfig()
        expt.EPOCHS = 150

    for block in blocks:
        kwargs["subBlock"] = block

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

        expdir = exp_base + str(idx_) + "/"

        trainer = TrainAE(settings, expdir, expt.calc_DA_MAE)
        expdir = trainer.expdir #get full path


        model = trainer.train(expt.EPOCHS, test_every=expt.test_every,
                                num_epochs_cv=expt.num_epochs_cv,
                                learning_rate = expt.LR, print_every=expt.print_every,
                                small_debug=expt.SMALL_DEBUG_DOM)

    ################################06a5
    blocks = ["NeXt", "vanilla", "CBAM_vanilla", "CBAM_NeXt", ]
    Cstd = 64
    sigmoid = False
    activations = ["relu"]
    if TEST:
        expt = ExptConfigTest()
    else:
        expt = ExptConfig()
        expt.EPOCHS = 300

    for activation in activations:
        for block in blocks:
            kwargs = {"model_name": "Tucodec", "block_type": block,
                    "Cstd": Cstd, "sigmoid": sigmoid,
                    "activation": activation, "aug_scheme": 0}
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


