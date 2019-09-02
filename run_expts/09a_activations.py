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
GPU_DEVICE = 3
NUM_GPU = 4
exp_base = "experiments/train2/09a/"
GPU_OFFSET = 0

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
    lr_factors = [1, 0.10]

    resNextk1 = {"layers": 27, "cardinality": 4,
            "block_type": "CBAM_vanilla",
            "module_type": "RDB3",
            "aug_scheme": 0}


    resNextk2 = {"layers": 27, "cardinality": 1,
            "block_type": "CBAM_vanilla",
            "module_type": "ResNeXt3",
            "aug_scheme": 0}


    kwarg_lst = (resNextk1, resNextk2, )
    models = (ResStack3, ResStack3, )


    assert len(models) == len(kwarg_lst)



    idx = 0

    for index, kwargs in enumerate(kwarg_lst):
        for idx2, act in enumerate(activations):
            if TEST:
                expt = ExptConfigTest()
            else:
                expt = ExptConfig()
                expt.LR = expt.LR * lr_factors[idx2]

            batch_sz = 16
            # if act == "GDN" and "module_type" == "RDB3":
            #     batch_sz = 8

            if act == "relu":
                kwargs["aug_scheme"] = 4

            Model = models[index]

            kwargs["activation"] = act

            idx_ = idx
            idx += 1

            if idx_ % NUM_GPU != GPU_DEVICE - GPU_OFFSET:
                continue

            for k, v in kwargs.items():
                print("{}={}, ".format(k, v), end="")
            print()

            settings = Model(**kwargs)
            settings.GPU_DEVICE = GPU_DEVICE
            settings.export_env_vars()

            expdir = exp_base + str(idx - 1) + "/"

            print(expdir)
            trainer = TrainAE(settings, expdir, expt.calc_DA_MAE, batch_sz=batch_sz)
            expdir = trainer.expdir #get full path


            model = trainer.train(expt.EPOCHS, test_every=expt.test_every,
                                    num_epochs_cv=expt.num_epochs_cv,
                                    learning_rate = expt.LR, print_every=expt.print_every,
                                    small_debug=expt.SMALL_DEBUG_DOM)



if __name__ == "__main__":
    main()


