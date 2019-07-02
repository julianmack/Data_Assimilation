from pipeline import utils, AutoEncoders
from pipeline.settings import config
import numpy as np
import vtk.util.numpy_support as ns


def main():
    fluidity = utils.FluidityUtils()

    settings = config.Config()

    fps = utils.DataLoader.get_sorted_fps_U(settings.DATA_FP)

    #fp_alt = settings.HOME_DIR + "data/" + "DA_MAE.vtu"

    # in_3d = fluidity.get_3D_grid(fps[2], settings.FIELD_NAME,
    #                     save_newgrid_fp = "data/interpolate2",
    #                     ret_torch = False, factor_inc = 2.43)

    #vtk_file = ns.numpy_to_vtk(in_3d, 1)
    n = (91, 85, 32)
    #inx, iny, inz = (91, 85, 32)
    conv_data = utils.ML_utils.conv_scheduler3D(n, None, 1, True)
    init_data = utils.ML_utils.get_init_data_from_schedule(conv_data)
    print(init_data)
    channels = list(range(1, len(init_data) + 2))[::-1]
    cae = AutoEncoders.CAE_3D(init_data, channels)


if __name__ == "__main__":
    main()