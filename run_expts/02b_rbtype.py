"""
Read expt 02a first
This extends expt 02a by adding CBAM blocks
"""
from VarDACAE.settings.models.resNeXt import ResStack3

from VarDACAE import TrainAE, ML_utils, BatchDA

from run_expts.expt_config import ExptConfigTest


TEST = False
GPU_DEVICE = 1
exp_base = "experiments/train/02b/"

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

    structures = [(8, 3), (1, 27)] #(cardinality, layers)
    substructures = ["ResNeXt3", "RDB3"]
    blocks = ["CBAM_NeXt", "CBAM_vanilla"]


    if TEST:
        expt = ExptConfigTest()
    else:
        expt = ExptConfig()

    idx = 8
    for block in blocks:
        for substruct in substructures:
            for struct in structures:
                idx += 1
                #split work across 2 gpus:
                if idx - 1 < 12 and GPU_DEVICE == 1:
                    continue
                elif idx - 1 >= 12 and  GPU_DEVICE == 0:
                    continue

                (cardinality, layers) = struct

                kwargs = {"layers": layers, "cardinality": cardinality,
                        "block_type": block,
                        "module_type": substruct}

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

