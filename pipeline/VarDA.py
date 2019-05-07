#!/usr/bin/python3
"""3D VarDA pipeline. See settings.py for options"""

import numpy as np
import torch
from helpers import VarDataAssimilationPipeline as VarDA
import AutoEncoders as AE
import sys

sys.path.append('/home/jfm1118')

import utils

import config
from scipy.optimize import minimize

TOL = 1e-3


if __name__ == "__main__":
    settings = config.Config

    vda = VarDA()
    vda.Var_DA_routine(settings)
