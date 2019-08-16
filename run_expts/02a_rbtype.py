"""
Experiment to find the best residual block in our case.
Using result from 01c_resNext, the best results were with:
a) cardinality 8, layers 3
b) cardinality 1, layers 27 - this was actually quite a bit worse but
                                it is diverse to a) so carry it forward

I also want to investigate the following substructures:
i) x3 residual blocks stacked with an extra residual connection
        across all three blocks: `ResNeXt3`
ii) x3 residual blocks in the RDB format: https://arxiv.org/pdf/1608.06993.pdf.
        This also has an extra residual connection over all three blocks: `RDB3`

In this experiment I will investigate the following:
    structures a) and b)
    with ["NeXt", "vanilla"] and blocks
    with substructures i) and ii)


x8 experiments in total: split across 2 GPUS
"""
from pipeline.settings.models_.resNeXt import ResStack3


from pipeline import TrainAE, ML_utils, GetData, SplitData
from pipeline.VarDA.batch_DA import BatchDA
from run_expts.expt_config import ExptConfigTest


TEST = False
GPU_DEVICE = 1
exp_base = "experiments/train/02a/"

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
    blocks = ["NeXt", "vanilla"]



    if TEST:
        expt = ExptConfigTest()
    else:
        expt = ExptConfig()

    idx = 0
    for struct in structures:
        for substruct in substructures:
            for block in blocks:
                idx += 1
                #split work across 2 gpus:
                if idx - 1 < 4 and GPU_DEVICE == 1:
                    continue
                elif idx - 1 >= 4 and  GPU_DEVICE == 0:
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

