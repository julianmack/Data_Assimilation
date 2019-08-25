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
from VarDACAE.settings.models.resNeXt import ResNeXt


from VarDACAE import TrainAE, ML_utils, GetData, SplitData
from VarDACAE.VarDA.batch_DA import BatchDA

import shutil

#global variables for DA and training:
EPOCHS = 150
SMALL_DEBUG_DOM = False #For training
calc_DA_MAE = True
num_epochs_cv = 25
LR = 0.0003
print_every = 10
test_every = 10
exp_base = "experiments/train/01_resNeXt_3/"

def main():
    res_layers = [3, 9, 27]
    cardinalities = [1, 8, 32]


    idx = 0

    for layer in res_layers:
        for cardinality in cardinalities:
            print("Layers", layer)
            print("Cardinality", cardinality)

            kwargs = {"layers": layer, "cardinality": cardinality}

            settings = ResNeXt(**kwargs)
            settings.AUGMENTATION = True
            settings.DEBUG = False
            expdir = exp_base + str(idx) + "/"


            trainer = TrainAE(settings, expdir, calc_DA_MAE)
            expdir = trainer.expdir #get full path


            model = trainer.train(EPOCHS, test_every=test_every, num_epochs_cv=num_epochs_cv,
                                    learning_rate = LR, print_every=print_every, small_debug=SMALL_DEBUG_DOM)

            idx += 1



if __name__ == "__main__":
    main()

