"""
Train Etridgu model: CLIC joint-winner 2019: http://openaccess.thecvf.com/content_CVPRW_2019/html/CLIC_2019/Cho_Low_Bit-rate_Image_Compression_based_on_Post-processing_with_Grouped_Residual_CVPRW_2019_paper.html
Train for two block types and two #channels:
    blocks = ["NeXt", "vanilla"]

Only train for a single channel because it is **VERY** slow

"""

from VarDACAE.settings.models.resNeXt import ResStack3
from VarDACAE.settings.models.CLIC import CLIC, GRDNBaseline


from VarDACAE import TrainAE, ML_utils, GetData, SplitData
from VarDACAE.VarDA.batch_DA import BatchDA
from run_expts.expt_config import ExptConfigTest


TEST = False
GPU_DEVICE = 0
exp_base = "experiments/train/06b/"

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
    blocks = ["NeXt", "vanilla"]
    channels = [32,]



    if TEST:
        expt = ExptConfigTest()
    else:
        expt = ExptConfig()

    idx = 0

    for block in blocks:
        for Cstd in channels:
            kwargs = { "block_type": block, "Cstd": Cstd}
            idx += 1

            for k, v in kwargs.items():
                print("{}={}, ".format(k, v), end="")
            print()

            settings = GRDNBaseline(**kwargs)
            settings.GPU_DEVICE = GPU_DEVICE
            settings.export_env_vars()

            expdir = exp_base + str(idx - 1) + "/"


            trainer = TrainAE(settings, expdir, expt.calc_DA_MAE, batch_sz=8)
            expdir = trainer.expdir #get full path


            model = trainer.train(expt.EPOCHS, test_every=expt.test_every,
                                    num_epochs_cv=expt.num_epochs_cv,
                                    learning_rate = expt.LR, print_every=expt.print_every,
                                    small_debug=expt.SMALL_DEBUG_DOM)



if __name__ == "__main__":
    main()


