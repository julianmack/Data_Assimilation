# VarDACAE
This module is used to create Convolutional AutoEncoders for Variational Data Assimilation. A  user can define, create and train an AE for Data Assimilation with just a few lines of code.

## Introduction

Data Assimilation (DA) is an uncertainty quantification technique used to reduce the error in  predictions by combining forecasting data with observation of the state. The most common techniques for DA are Variational approaches and Kalman Filters.

In this work, we propose a method of using Autoencoders to model the Background error covariance matrix, to greatly reduce the computational cost of solving 3D Variational DA **while increasing the quality of the Data Assimilation**.

## Installation
To install do the following:
```
git clone https://github.com/julianmack/Data_Assimilation.git
cd Data_Assimilation
pip install -e .    #or use `pip install .`
                    #if you don't intend to update the module
                    #for your own models/data
```

## Tests
From the project home directory run `pytest`.

## Getting Started
To train and evaluate a [Tucodec](http://openaccess.thecvf.com/content_CVPRW_2019/papers/CLIC%202019/Zhou_End-to-end_Optimized_Image_Compression_with_Attention_Mechanism_CVPRW_2019_paper.pdf "Tucodec CLIC-2019 paper") model on Fluidity data:
```
from VarDACAE import TrainAE, BatchDA
from VarDACAE.settings.models.CLIC import CLIC

model_kwargs = {"model_name": "Tucodec", "block_type": "NeXt", "Cstd": 64}

settings = CLIC(**model_kwargs)    #settings describing experimental setup
expdir = "experiments/expt1/"      #dir to save results data and models

trainer = TrainAE(settings, expdir, batch_sz=16)
model = trainer.train(num_epochs=150)   #this will take approximately 8 hrs on a K80

#evaluate DA on the test set:
results_df = BatchDA(settings, AEModel=model).run()

```
## Settings Instance
The API is based around a monolithic ```settings``` object that is used to define all configuration parameters, from the model definition to the seed. This single point of truth is used so that, an experiment can be repeated _exactly_ by simply loading a pickled  ```settings``` object. All key classes like ```TrainAE``` and ```BatchDA``` require a ```settings``` object at initialisation.

## Train a model on your *own* data

To train a model on your own 3D data you must do the following:
* Override the default ```get_X(...)``` method in the ```GetData``` loader class:
```
from VarDACAE import GetData

class NewLoaderClass(GetData):
    def get_X(self, settings):
        "Arguments:
               settings: (A settings.Config class)
        returns:
            np.array of dimensions B x nx x ny x nz "

        # ... calculate / load or download X
        # For an example see VarDACAE.data.load.GetData.get_X"""
        return X
```

* Create a new settings class that inherits from your desired model's settings class (e.g. `VarDACAE.settings.models.CLIC.CLIC`) and update the data dimensions:
```
from VarDACAE.settings.models.CLIC import CLIC

class NewConfig(CLIC):
    def __init__(self, CLIC_kwargs, opt_kwargs):
        super(CLIC, self).__init__(**CLIC_kwargs)
        self.n3d = (100, 200, 300)  # Define input domain size
                                    # This is used by ConvScheduler
        self.X_FP = "SET_IF_REQ_BY_get_X"
        # ... use opt_kwargs as desired

CLIC_kwargs =  {"model_name": "Tucodec", "block_type": "NeXt",
                "Cstd": 64, "loader": NewLoaderClass}
                # NOTE: do not initialize NewLoaderClass

settings = NewConfig(CLIC_kwargs, opt_kwargs)

```
This ```settings``` object can now be used to train a model with the `TrainAE` method as shown above.

