from pipeline import DataAssimilation, config, utils
import numpy as np
import vtk.util.numpy_support as ns

def main():
    fluidity = utils.FluidityUtils()

    settings = config.Config()
    DA = DataAssimilation.DAPipeline(settings)

    fps = DA.get_sorted_fps_U(settings.DATA_FP)

    fp2 = settings.HOME_DIR + "data/" + "DA_MAE.vtu"
    in_3d = fluidity.get_3D_grid(fps[0], settings.FIELD_NAME, ret_torch = True)

    print(in_3d.shape)

if __name__ == "__main__":
    main()