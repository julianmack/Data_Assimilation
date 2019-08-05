"""
After looking at results of 00_train_baseline:
    1) BatchNorm and dropout are both v poor in this use-case
    2) Two training runs diverged after ~ 75 epochs.

Hence - run again w/o BN and dropout at a lower lr.
Don't use the cross-validation routine so that it is
possible to compare experiments.

In addition, Add a final candidate baseline model:
    BaselineBlock - this has been created since I ran expt 0
"""
from pipeline.settings.baseline_explore import Baseline1, Baseline2
from pipeline.settings.block_models import BaselineBlock

from pipeline import TrainAE, ML_utils, GetData, SplitData
from pipeline.VarDA.batch_DA import BatchDA

import shutil

#global variables for DA and training:
EPOCHS = 150
SMALL_DEBUG_DOM = False #For training
calc_DA_MAE = True
num_epochs_cv = 0 #unlike in expt 00
LR = 0.0005
print_every = 10
test_every = 10
exp_base = "experiments/train/baseline_0b/"

def main():
    model_settings = [BaselineBlock, Baseline2, Baseline1,]
    augments = [True, False]

    idx = 0

    for aug in augments:
        for config in model_settings:
            print("Augmentation", aug)
            print("model", config)

            settings = config()
            settings.AUGMENTATION = aug
            settings.DEBUG = False
            expdir = exp_base + str(idx) + "/"


            trainer = TrainAE(settings, expdir, calc_DA_MAE)
            expdir = trainer.expdir #get full path


            model = trainer.train(EPOCHS, test_every=test_every, num_epochs_cv=num_epochs_cv,
                                    learning_rate = LR, print_every=print_every, small_debug=SMALL_DEBUG_DOM)

            idx += 1



if __name__ == "__main__":
    main()

