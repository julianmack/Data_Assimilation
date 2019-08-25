"""
Check that you can get a resNext system to train

"""

from pipeline.settings.models.resNeXt import ResNeXt


from pipeline import TrainAE, ML_utils, GetData, SplitData
from pipeline.VarDA.batch_DA import BatchDA


#global variables for DA and training:
EPOCHS = 100
SMALL_DEBUG_DOM = False #False #False #For training
calc_DA_MAE = True
num_epochs_cv = 0
LR = 0.0003
print_every = 5
test_every = 5
exp_base = "experiments/train/01b/"
GPU_DEVICE = 0
def main():
    layer = 6
    cardinality = 4
    print("Layers", layer)
    print("Cardinality", cardinality)

    kwargs = {"layers": layer, "cardinality": cardinality}

    settings = ResNeXt(**kwargs)
    settings.AUGMENTATION = True
    settings.DEBUG = False
    settings.GPU_DEVICE = GPU_DEVICE
    settings.SEED = 19
    settings.export_env_vars()

    expdir = exp_base

    trainer = TrainAE(settings, expdir, calc_DA_MAE)
    expdir = trainer.expdir #get full path


    model = trainer.train(EPOCHS, test_every=test_every, num_epochs_cv=num_epochs_cv,
                            learning_rate = LR, print_every=print_every, small_debug=SMALL_DEBUG_DOM)




if __name__ == "__main__":
    main()

