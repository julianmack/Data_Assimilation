"""
Running to create a baseline using ResNext param names so that:
1) expt 1 and onwards results have a point of comparison
2) retraining this model with extra resnet layers added will be feasible

Train for 400 epochs (to see if this makes large difference) but we will use
the model after 150 epochs for comparison.
"""

from pipeline.settings.models_.resNeXt import ResNeXt


from pipeline import TrainAE, ML_utils, GetData, SplitData
from pipeline.VarDA.batch_DA import BatchDA


#global variables for DA and training:
EPOCHS = 400
SMALL_DEBUG_DOM = False #False #False #For training
calc_DA_MAE = True
num_epochs_cv = 25
LR = 0.0003
print_every = 10
test_every = 10
exp_base = "experiments/train/00c_baseResNext/"

def main():
    layer = 0
    print("Layers", 0)
    print("Cardinality", 2)

    kwargs = {"layers": layer, "cardinality": 2}

    settings = ResNeXt(**kwargs)
    settings.AUGMENTATION = True
    settings.DEBUG = False
    expdir = exp_base

    trainer = TrainAE(settings, expdir, calc_DA_MAE)
    expdir = trainer.expdir #get full path


    model = trainer.train(EPOCHS, test_every=test_every, num_epochs_cv=num_epochs_cv,
                            learning_rate = LR, print_every=print_every, small_debug=SMALL_DEBUG_DOM)




if __name__ == "__main__":
    main()

