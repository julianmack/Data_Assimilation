# Data Assimilation
This module is used to create AutoEncoders that are useful for Data Assimilation. A  user can define, create and train an AE for Data Assimilation with just a few lines of code.

## Introduction

Data Assimilation (DA) is an uncertainty quantification technique used to reduce the error in  predictions by combining forecasting data with observation of the state. The most common techniques for DA are Variational approaches and Kalman Filters.

In this work, I propose a method of using Autoencoders to model the Background error covariance matrix, to greatly reduce the computational cost of solving 3D Variational DA **while increasing the quality of the Data Assimilation**.

## Installation

## Getting Started
Defined for Fluidity data

### Data Assimilation

### AE Define and Train

## Repo Structure
To run tests ...: 
### Config classes

## Using this repo to training your own models
If you would like to use this repo to create an AE for an arbitrary dataset you must update the files in `pipeline/data_` for dataloading and in `pipeline/settings/config.py` as the data input size is hard-coded.

