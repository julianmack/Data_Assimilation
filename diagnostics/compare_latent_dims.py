#!/usr/bin/python3
import numpy as np
from pipeline import ML_utils
from pipeline.VarDA import VDAInit
from pipeline.data_.load import GetData
from pipeline.data_.split import SplitData

DIR = "experiments/train/00c_baseResNext/"
#DIR = "experiments/train/01_resNeXt_3/0/2/"


def main():

    model, settings = ML_utils.load_model_and_settings_from_dir(DIR)
    initializer = VDAInit(settings, model)
    vdadata = initializer.run()
    encoder = vdadata.get("encoder")


    loader, splitter = GetData(), SplitData()
    X = loader.get_X(settings)

    train_X, test_X, u_c, X, mean, std = splitter.train_test_DA_split_maybe_normalize(X, settings)
    settings.AUGMENTATION = False


    train_loader, test_loader = loader.get_train_test_loaders(settings, 32, num_workers = 6)

    Z = []
    for Bidx, x in enumerate(test_loader):
        x, = x
        z = encoder(x)
        Z.append(z)
    Z_test = np.concatenate(Z, axis = 0)
    print(Z_test.shape)
    #Compare all test data with all other test data
    tot, tot_abs = 0, 0
    number = 0
    for idx1 in range(Z_test.shape[0]):
        for idx2 in range(Z_test.shape[0]):
            if idx2 <= idx1:
                continue
            res = compare_overlap(Z_test[idx1], Z_test[idx2])
            tot_abs += np.abs(res)
            tot += res
            number += 1
            if number % 10000 == 0 and number !=0:
                print(number, tot_abs/number, tot/number)
    print("Final", tot_abs/number, tot/number)

def compare_overlap(z_1, z_2):

    z1_norm, z2_norm = z_1 / np.linalg.norm(z_1), z_2 / np.linalg.norm(z_2)

    return np.dot(z1_norm, z2_norm)

def compare_overlap_lantent(u_1, u_2, encoder):
    z_1, z_2 = encoder(u_1), encoder(u_2)

    return compare_overlap(z_1, z_2)

if __name__ == "__main__":
    main()

