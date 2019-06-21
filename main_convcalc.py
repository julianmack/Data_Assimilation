from pipeline import config, utils
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

    exit()
    res = utils.ML_utils().conv_scheduler1D_stride2(86, 1)
    n = (91, 85, 32)
    #inx, iny, inz = (91, 85, 32)
    results = []
    for n_i in n:
        res_i = utils.ML_utils.conv_scheduler1D_stride2(n_i, 1)
        results.append(res_i)
    min_len = min([len(i) for i in results])
    for layer in range(min_len):
        for dim in range(len(n)):
            print(results[dim][layer]["in"], end=", ")
        print()

    #final out
    for dim in range(len(n)):
        print(results[dim][min_len - 1]["out"], end=", ")
    exit()
    inx, iny, inz = (91, 85, 32)
    stride = 1
    pad = 0
    kernel = (2, 2, 2)
    while inx*iny*inz != 1 and inx > 0:
        inx = utils.ML_utils.conv_formula(inx, stride, pad, kernel[0])
        iny = utils.ML_utils.conv_formula(iny, stride, pad, kernel[1])
        inz = utils.ML_utils.conv_formula(inz, stride, pad, kernel[2])
        print(inx, iny, inz)

if __name__ == "__main__":
    main()