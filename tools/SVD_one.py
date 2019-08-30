"""File to run elements of VarDACAE module from"""

import VarDACAE
from VarDACAE.settings.base_3D import Config3D
from VarDACAE import BatchDA, SplitData

import numpy as np



def main():
    settings = VarDACAE.settings.base_3D.Config3D()
    #settings.NUMBER_MODES = None
    #settings.OBS_FRAC = 1.0
    settings.FORCE_GEN_X = False
    settings.AZURE_DOWNLOAD = False
    settings.OBS_MODE = "all"
    da = VarDACAE.VarDA.DataAssimilation.DAPipeline(settings)

    da.run()
if __name__ == "__main__":
    main()