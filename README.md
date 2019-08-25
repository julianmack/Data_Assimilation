# Data Assimilation
This module is used to create AutoEncoders that are useful for Data Assimilation. A  user can define, create and train an AE for Data Assimilation with just a few lines of code.

## Introduction

Data Assimilation (DA) is an uncertainty quantification technique used to reduce the error in  predictions by combining forecasting data with observation of the state. The most common techniques for DA are Variational approaches and Kalman Filters.

In this work, I propose a method of using Autoencoders to model the Background error covariance matrix, to greatly reduce the computational cost of solving 3D Variational DA **while increasing the quality of the Data Assimilation**.

## Installation
```git clone https://github.com/julianmack/Data_Assimilation.git
cd Data_Assimilation
pip install -e .    #or use `pip install .`
                    #if you don't intend to update the module
                    #for your own models/data
```

## Getting Started
To train and evaluate a Tucodec model on Fluidity data:
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


### Data Assimilation

### AE Define and Train

## Repo Structure
To run tests ...:
### Config classes

## Using this repo to training your own models
If you would like to use this repo to create an AE for an arbitrary dataset you must update the files in `VarDACAE/data` for dataloading and in `VarDACAE/settings/base.py` as the data input size is hard-coded.

