

import numpy as np
import pandas as pd

import sys
import os
fluidity_fp  = '/mnt/c/Users/julia/fluidity/fluidity-master/python'
DA_project_fp = '/mnt/c/Users/julia/Documents/Imperial/DA_project'
sys.path.append(fluidity_fp)
sys.path.append(DA_project_fp)

import vtktools
ug1=vtktools.vtu(DA_project_fp + '/data/LSBU_c_0.vtu')
ug2=vtktools.vtu(DA_project_fp + '/data/LSBU_0.vtu')
print(ug1.GetFieldNames())
print(ug2.GetFieldNames())

#read the values of the tracers and copy in a vector named p
p=ug2.GetVectorField('Velocity')

n=len(p)
print(n)


Background = np.matmul(p, p.T)
print(Background.shape)
