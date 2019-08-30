"""File to run elements of VarDACAE module from"""

import VarDACAE
from VarDACAE.settings.base_3D import Config3D
from VarDACAE import BatchDA, SplitData
from VarDACAE.data.load import GetData
import numpy as np
from VarDACAE import fluidity


def main():
    settings = VarDACAE.settings.base_3D.Config3D()
    settings.FORCE_GEN_X = True

    #create sg
    fp = settings.DATA_FP + "LSBU_8.vtu"
    ug = fluidity.vtktools.vtu(fp)
    GetData.get_3D_np_from_ug(ug, settings, save_newgrid_fp=None)

    #load sg
    s_grid = fluidity.vtktools.vtu(settings.VTU_FP)
    print(s_grid)

if __name__ == "__main__":
    main()

