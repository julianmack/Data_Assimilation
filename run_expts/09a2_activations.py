"""
Run outstanding activation experiments for GDN and relu
    RNAB
    ResNeXt x3
"""

from VarDACAE.settings.models.resNeXt import ResStack3
from VarDACAE.settings.models.CLIC import CLIC, GRDNBaseline

from VarDACAE import TrainAE, ML_utils, BatchDA

from run_expts.expt_config import ExptConfigTest


TEST = False
GPU_DEVICE = 1
NUM_GPU = 2
exp_base = "experiments/train2/09a2/"

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

    activations = ["GDN", "relu"]
    lr_factors = [1, 0.2]


    resNextk3 = {"layers": 3, "cardinality": 8,
            "block_type": "vanilla",
            "module_type": "RDB3",
            "aug_scheme": 0}

    rabkwargs = {"cardinality": 1, "block_type": "RAB",
                    "sigmoid": True, "module_type": "Bespoke",
                    "attenuation": False, "layers": 4,
                    "aug_scheme": 0, "subBlock": "NeXt",}

    kwarg_lst = ( resNextk3, rabkwargs,)
    models = ( ResStack3, ResStack3,)


    assert len(models) == len(kwarg_lst)



    idx = 0

    for index, kwargs in enumerate(kwarg_lst):
        for idx2, act in enumerate(activations):
            if TEST:
                expt = ExptConfigTest()
            else:
                expt = ExptConfig()
                expt.LR = expt.LR * lr_factors[idx2]

            Model = models[index]

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


