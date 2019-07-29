"""File to run elements of pipeline module from"""
from pipeline import ML_utils, GetData, SplitData
from pipeline.VarDA.batch_DA import BatchDA
from pipeline.settings import config

ALL_DATA = False
SAVE = False
def main():
    save_fp = "/experiments/batch_DA/2/"
    init_settings =  config.Config3D()

    #Load data
    loader, splitter = GetData(), SplitData()
    X = loader.get_X(init_settings)

    train_X, test_X, u_c_std, X, mean, std = splitter.train_test_DA_split_maybe_normalize(X, init_settings)

    #set control_states
    if ALL_DATA:
        control_states = X
    else:
        NUM_STATES = 100
        START = 100
        control_states = train_X[START:NUM_STATES + START]

    #AE
    dir = "/data/home/jfm1118/DA/experiments/train_DA_Pressure/2-l4NBN/2-l4NBN/" # 299.pth"
    #dir = "/Users/julia/Documents/Imperial/DA_project/experiments/azure/train_DA_Pressure/2-l4NBN/" # 299.pth"

    model, settings = ML_utils.load_model_and_settings_from_dir(dir)
    model.to(ML_utils.get_device()) #TODO
    settings.HOME_DIR = init_settings.HOME_DIR
    settings.INTERMEDIATE_FP = init_settings.INTERMEDIATE_FP
    #settings.UNDO_NORMALIZE = True
    settings.DEBUG = False
    settings.TOL = 1e-6
    settings.OBS_VARIANCE = 0.5
    x_fp = settings.get_X_fp(True) #force init X_FP

    out_fp = save_fp + "AE.csv"
    if SAVE:
        out_fp = save_fp + "AE.csv"
    else:
        out_fp = None

    batch_DA_AE = BatchDA(settings, control_states, csv_fp= out_fp, AEModel=model,
                        reconstruction=True, plot=False)

    res_AE = batch_DA_AE.run(print_every=10)





if __name__ == "__main__":
    main()
