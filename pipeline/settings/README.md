### Configuration Files

Here we have used hierarchical configuration files that inherit from the base class Config() to avoid code repetition.

An initialised config instance is required for all elements of the DA pipeline and/or AE training/inference as all key methods accept a `settings` parameter - i.e. an initialized class that inherits from the base Config() class.

All parameters for Data Assimilation are specified in the config classes.
In addition, when an AE is used for DA, its architecture is specified by the config class.
The model definitions used in the final paper are given in the `models_` subdirectory.
Other model definitions that were not used in the final paper but that were relevant in the exploration phase are in the sub directory `models_explore`.

### Example Usage: TSVD
```from pipeline.settings.base import Config
from pipeline import DA_pipeline

settings = Config()
#the default settings.COMPRESSION_METHOD = "SVD" so no need to change anything
DA = DAPipeline(settings)
DA.run()
```


  
### Example Usage: Training ResNeXt model
```from pipeline.settings.models_.resNeXt import ResNeXt
from pipeline import TrainAE

init_params = {"layers": 3, "cardinality": 2}
settings = ResNeXt(**init_params)
expdir = "experiments/TEST"


trainer = TrainAE(settings, expdir)

EPOCHS = 150
SMALL_DEBUG_DOM = False #For training
num_epochs_cv = 20 #Number of epochs to cross validate for
LR = 0.0003        #This is overridden by num_epochs_cv
print_every = 10
test_every = 10

model = trainer.train(EPOCHS, test_every=test_every, num_epochs_cv=num_epochs_cv,
                learning_rate = LR, print_every=print_every, small_debug=SMALL_DEBUG_DOM)
```
