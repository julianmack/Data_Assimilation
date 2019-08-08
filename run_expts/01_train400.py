"""
After looking at results of 00_train_baseline:
    1) BatchNorm and dropout are both v poor in this use-case
    2) Two training runs diverged after ~ 75 epochs.

"""
from pipeline.settings.models_.resNeXt import ResNeXt


from pipeline import TrainAE, ML_utils, GetData, SplitData
from pipeline.VarDA.batch_DA import BatchDA

import shutil

#global variables for DA and training:
EPOCHS = 250 #50 #train for 150 more epochs
SMALL_DEBUG_DOM = False #For training
calc_DA_MAE = True
num_epochs_cv = 25
LR = 0.0003
print_every = 10
test_every = 10
exp_base = "experiments/train/01_resNeXt_2/cont/"
exp_load = "experiments/train/01_resNeXt_2/0/"
def main():
    res_layers = [3, 9, 27]
    cardinalities = [1, 8, 32]


    idx = 0
    layer = 3
    cardinality = 1
    expdir = exp_base + str(0) + "/"

    print("Layers", layer)
    print("Cardinality", cardinality)

    kwargs = {"layers": layer, "cardinality": cardinality}
    _, settings = ML_utils.load_model_and_settings_from_dir(exp_load)

    expdir = exp_base + str(idx) + "/"


    trainer = TrainAE(settings, expdir, calc_DA_MAE)
    expdir = trainer.expdir #get full path


    model = trainer.train(EPOCHS, test_every=test_every, num_epochs_cv=num_epochs_cv,
                            learning_rate = LR, print_every=print_every, small_debug=SMALL_DEBUG_DOM)




if __name__ == "__main__":
    main()

