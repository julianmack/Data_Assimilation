#!/usr/bin/python3.6

import numpy as np
import sys
import os

sys.path.append('fluidity-master')
import vtktools


DATA_FP = "data/small3DLSBU/"
U_FP = "data/U_small3D_Tracer"

def main():
    field_name = "Tracer"
    fps_sorted = get_sorted_fps_U(DATA_FP, field_name)
    U = create_U_from_fps(fps_sorted, field_name)
    print(U.shape)
    np.save(U_FP, U)
    V = create_V_from_U(U_FP)

def get_sorted_fps_U(data_dir, field_name):
    """Creates and returns list of .vtu filepaths sorted according to timestamp in name.
    Input files in data_dir must be of the form XXXLSBU_<TIMESTAMP_IDX>.vtu"""

    fps = os.listdir(data_dir)
ls
    #extract index of timestep from file name
    idx_fps = []
    for fp in fps:
        _, file_number = fp.split("LSBU_")
        #file_number is of form 'XXX.vtu'
        idx = int(file_number.replace(".vtu", ""))
        idx_fps.append(idx)

    #sort by timestep
    assert len(idx_fps) == len(fps)
    zipped_pairs = zip(idx_fps, fps)
    fps_sorted = [x for _, x in sorted(zipped_pairs)]

    #add absolute path
    fps_sorted = [data_dir + x for x in fps_sorted]

    return fps_sorted

def create_U_from_fps(fps, field_name, field_type  = "scalar"):
    """Creates a numpy array of values of scalar field_name
    Input list must be sorted"""
    M = len(fps) #number timesteps
    n = 100040 #hardcode for now

    output = np.zeros((M, n))
    for idx, fp in enumerate(fps):
        # create array of tracer
        ug = vtktools.vtu(fp)
        if field_type == "scalar":
            vector = ug.GetScalarField(field_name)
        elif field_type == "vector":
            vector = ug.GetVectorField(field_name)
        else:
            raise ValueError("field_name must be in {\'scalar\', \'vector\'}")
        #print("Length of vector:", vector.shape)
        output[idx] = vector

    return output.T #return (n x M)

def create_V_from_U(input_FP = U_FP):
    pass


if __name__ == "__main__":
    main()
