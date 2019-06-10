from pipeline import DataAssimilation, config, utils

def main():
    fluidity = utils.FluidityUtils()

    settings = config.Config()
    DA = DataAssimilation.DAPipeline(settings)

    fps = DA.get_sorted_fps_U(settings.DATA_FP)
    print(fps)
    fp2 = settings.HOME_DIR + "data/" + "DA_MAE.vtu"
    in_3d = fluidity.get_3D_grid(fps[1], torch = True)

    print(in_3d.shape)

if __name__ == "__main__":
    main()