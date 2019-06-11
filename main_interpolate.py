from pipeline import DataAssimilation, config, utils
import numpy as np
import vtk.util.numpy_support as ns

def main():
    fluidity = utils.FluidityUtils()

    settings = config.Config()
    DA = DataAssimilation.DAPipeline(settings)

    fps = DA.get_sorted_fps_U(settings.DATA_FP)

    #fp_alt = settings.HOME_DIR + "data/" + "DA_MAE.vtu"

    in_3d = fluidity.get_3D_grid(fps[1], settings.FIELD_NAME, save_newgrid_fp = "data/interpolate2", ret_torch = False)

    #vtk_file = ns.numpy_to_vtk(in_3d, 1)

    print(in_3d.shape)

if __name__ == "__main__":
    main()