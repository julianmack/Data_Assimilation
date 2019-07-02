from pipeline import  utils
from pipeline.settings import config

import numpy as np
import vtk.util.numpy_support as ns

def main():
    fluidity = utils.FluidityUtils()

    settings = config.Config()

    fps = utils.DataLoader.get_sorted_fps_U(settings.DATA_FP)

    #fp_alt = settings.HOME_DIR + "data/" + "DA_MAE.vtu"

    in_3d = fluidity.get_3D_grid(fps[2], settings.FIELD_NAME,
                        save_newgrid_fp = "data/interpolate2",
                        ret_torch = False, factor_inc = 2.43)

    #vtk_file = ns.numpy_to_vtk(in_3d, 1)

    print(in_3d.shape)
    prod = 1
    for x in in_3d.shape:
        prod = prod * x
    print(prod)

if __name__ == "__main__":
    main()