"""
Train all resNext - split over two nodes.
lower learning rate to 0.0002 (struggling to learn at low values)
"""
from varda_cae.settings.models.resNeXt import ResNeXt


from varda_cae import TrainAE, ML_utils, GetData, SplitData
from varda_cae.VarDA.batch_DA import BatchDA

import shutil

#global variables for DA and training:
EPOCHS = 150
SMALL_DEBUG_DOM = False #For training
calc_DA_MAE = True
num_epochs_cv = 0
LR = 0.0002
print_every = 10
test_every = 10
PARAM_IDX = 0

exp_base = "experiments/train/01_resNeXt_3/{}/".format(PARAM_IDX)
def main():
    #of form (layers, cardinality)
    param_vals = [(3, 1), (3, 8), (3, 32), (27, 32)]
    param_vals_2 = [(9, 1), (9, 8), (9, 32), (27, 1), (27, 8),]
    param_options = [param_vals, param_vals_2]

    idx = 0
    params = param_options[PARAM_IDX]
    for param in params:
        layer, cardinality = param
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

