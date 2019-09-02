"""File to run elements of VarDACAE module from"""
from VarDACAE import ML_utils,  SplitData
from VarDACAE.VarDA.batch_DA import BatchDA
from tools.check_train_load_DA import run_DA_batch
from notebooks.utils import get_model_specific_data
import os
import pickle


################ DA3 06a/06b

fp = "experiments/06a4/"
fp2 = "experiments/06a5/"
fp3 = "experiments/DA3/06a/1/"
fp4 = "experiments/DA3/06a/3/"

DIRS = [fp, fp2, fp3, fp4]

EXPDIR = "experiments/DA/06b/"
EPOCH = None #choose latest epoch if this is None

fp1 = "experiments/06b2"
fp2 = "experiments/09c/"

DIRS = [fp1, fp2]

#Expt 02
# B = "/home/jfm1118/DA/experiments/train2/"
# DIRS = ["experiments/06b2"]
# B = "/home/jfm1118/DA/experiments/train2/08b/"
# DIRS = [B + str(x) + "/" for x in range(8, 11)]

#DIRS = ["experiments/06a5/12/"]
#global variables for DA:
PRINT = True
ALL_DATA = True
SAVE_VTU = False
def calc_DA_best(dirs, params, expdir, prnt=True, all_data=True, epoch=None,
                save_vtu=False):
    if isinstance(dirs, list):
        pass
    elif isinstance(dirs, str):
        dirs = [dirs]

    results = []
    index = 0

    for idx, dir in enumerate(dirs):
        for path, subdirs, files in os.walk(dir):
            for file in files:

                if file == "settings.txt":
                    #this is a model directory
                    dir = path
                    expt_name = path.split("/")
                    expt_name = "{}_{}".format(expt_name[-1], str(index))
                    expdirnew = os.path.join(expdir, expt_name)

                    mse_DA, model_data = calc_DA_dir(dir, params, expdirnew,
                                            prnt=True, all_data=all_data,
                                            epoch=epoch, save_vtu=save_vtu)
                    results.append((mse_DA, model_data, path, expdirnew))
                    index += 1
    print(results)
    print()
    res = sorted(results)
    out_fp = os.path.join(expdir, "final.txt")

    with open(out_fp, "wb") as f:
        results = pickle.dump(res, f)


    print(res)
    print("BEST")
    print(res[0])
    print()
    print("SECOND")
    print(res[1])
    print("THIRD")
    print(res[2])
    print("FOURTH")
    print(res[3])
    print("FIFTH")
    print(res[4])

def calc_DA_dir(dir, params, expdir, prnt=True, all_data=True, epoch=None,
                save_vtu=False, gpu_device=0):
    gpu = False
    if gpu_device is not "CPU":
        gpu = True

    model, settings = ML_utils.load_model_and_settings_from_dir(dir,
                        device_idx= gpu_device, choose_epoch=epoch, gpu=gpu)

    df = run_DA_batch(settings, model, all_data, expdir, params, save_vtu,
                gpu_device=gpu_device)
    mse_DA = df["mse_DA"].mean()
    model_data = get_model_specific_data(settings, dir, model=model)
    if prnt:
        print(mse_DA, model_data, expdir)
        print(df.tail(5))
    return mse_DA, model_data


if __name__ == "__main__":
    params = {"var": 0.005, "tol":1e-3}
    calc_DA_best(DIRS, params, EXPDIR, PRINT, ALL_DATA, EPOCH, SAVE_VTU)



