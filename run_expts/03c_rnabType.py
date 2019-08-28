"""
RAB choice of block
"""
from VarDACAE import TrainAE, ML_utils, BatchDA

from run_expts.expt_config import ExptConfigTest
from VarDACAE.settings.models.resNeXt import ResStack3


TEST = False
GPU_DEVICE = 0
NUM_GPU = 2
exp_base = "experiments/train2/03c/"

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


    blocks = ["CBAM_vanilla", "vanilla", "CBAM_NeXt"]
    kwargs = {"cardinality": 1, "block_type": "RAB",
                    "sigmoid": True, "module_type": "Bespoke",
                    "attenuation": False, "layers": 4, 
                    "aug_scheme": 0}

    if TEST:
        expt = ExptConfigTest()
    else:
        expt = ExptConfig()

    idx = 0

    for block in blocks:
        kwargs["subBlock"] = block

        idx_ = idx
        idx += 1
        if idx_ % NUM_GPU != GPU_DEVICE:
            continue


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

