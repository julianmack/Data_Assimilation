"""File to run elements of pipeline module from"""
from pipeline.settings.baseline_explore import Baseline1, Baseline2
from pipeline import TrainAE, ML_utils, GetData, SplitData
from pipeline.VarDA.batch_DA import BatchDA

import shutil

#global variables for DA and training:
EPOCHS = 150
SMALL_DEBUG_DOM = False #False #For training
calc_DA_MAE = True
num_epochs_cv = 25
print_every = 10
test_every = 10


def main():
    model_settings = [Baseline2, Baseline1]
    BNs = [True, False]
    dropouts = [True, False]
    augments = [True, False]

    idx = 0
    for BN in BNs[::-1]:
        for aug in augments:
            for dropout in dropouts:
                for config in model_settings:
                    print("BN", BN)
                    print("Augmentation", aug)
                    print("dropout", dropout)
                    print("config", config)

                    settings = config()
                    settings.BATCH_NORM = BN
                    settings.DROPOUT = dropout
                    settings.AUGMENTATION = aug

                    expdir = "experiments/train/baseline/" + str(idx) + "/"


                    trainer = TrainAE(settings, expdir, calc_DA_MAE)
                    expdir = trainer.expdir #get full path


                    model = trainer.train(EPOCHS, test_every=test_every, num_epochs_cv=num_epochs_cv,
                                            print_every=print_every, small_debug=SMALL_DEBUG_DOM)

                    idx += 1



if __name__ == "__main__":
    main()

