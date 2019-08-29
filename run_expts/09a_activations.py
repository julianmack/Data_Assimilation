"""
Run outstanding activation experiments for GDN and relu
    RNAB
    GRDN
    ResNeXt x3
"""

from VarDACAE.settings.models.resNeXt import ResStack3
from VarDACAE.settings.models.CLIC import CLIC, GRDNBaseline

from VarDACAE import TrainAE, ML_utils, BatchDA

from run_expts.expt_config import ExptConfigTest


TEST = True
GPU_DEVICE = 0
NUM_GPU = 4
exp_base = "experiments/train2/09a/"

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

    activations = ["relu", "GDN"]


    resNextk1 = {"layers": layers, "cardinality": cardinality,
            "block_type": block,
            "module_type": substruct,
            "aug_scheme": 0}
    resNextk2 = {"layers": layers, "cardinality": cardinality,
            "block_type": block,
            "module_type": substruct,
            "aug_scheme": 0}
    resNextk3 = {"layers": layers, "cardinality": cardinality,
            "block_type": block,
            "module_type": substruct,
            "aug_scheme": 0}

    rabkwargs = {"cardinality": 1, "block_type": "RAB",
                    "sigmoid": True, "module_type": "Bespoke",
                    "attenuation": False, "layers": 4,
                    "aug_scheme": 0, "subBlock": INSERT,}

    kwarg_lst = (resNextk1, resNextk2, resNextk3, rabkwargs,)
    models = (ResStack3, ResStack3, ResStack3, ResStack3,)


    assert len(models) == len(kwarg_lst)

    if TEST:
        expt = ExptConfigTest()
    else:
        expt = ExptConfig()

    idx = 0

    for idx, kwargs in enumerate(kwarg_lst):
        for act in activations:
            Model = models[idx]

            kwargs["activation"] = act

            idx_ = idx
            idx += 1
            if idx_ % NUM_GPU != GPU_DEVICE:
                continue


            for k, v in kwargs.items():
                print("{}={}, ".format(k, v), end="")
            print()

            settings = Model(**kwargs)
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


