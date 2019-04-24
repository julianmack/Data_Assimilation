#!/usr/bin/python3
"""Ingests data and creates """
import numpy as np
import sys
import os

sys.path.append('fluidity-master')
sys.path.append('fluidity-master/python')
sys.path.append('simulation')
import vtktools


DATA_FP = "data/small3DLSBU/"
X_FP = "data/small3D_intermediate/X_small3D_Tracer.npy"
INTERMEDIATE_FP = "data/small3D_intermediate/"


def main():
    field_name = "Tracer"
    fps_sorted = get_sorted_fps_U(DATA_FP, field_name)
    print("Got sorted list")
    X = create_X_from_fps(fps_sorted, field_name)
    print(X.shape)
    np.save(X_FP, X)

    V = create_V_from_X(X)
    V_T = trunc_SVD(V)

class VarDataAssimilationPipeline():
    """Class to hold @static_method pipeline functions for
    Variational DA
    """

    def __init__(self):
        pass

    @staticmethod
    def get_sorted_fps_U(data_dir, field_name):
        """Creates and returns list of .vtu filepaths sorted according
        to timestamp in name.
        Input files in data_dir must be of the
        form <XXXX>LSBU_<TIMESTEP INDEX>.vtu"""

        fps = os.listdir(data_dir)

        #extract index of timestep from file name
        idx_fps = []
        for fp in fps:
            _, file_number = fp.split("LSBU_")
            #file_number is of form '<IDX>.vtu'
            idx = int(file_number.replace(".vtu", ""))
            idx_fps.append(idx)

        #sort by timestep
        assert len(idx_fps) == len(fps)
        zipped_pairs = zip(idx_fps, fps)
        fps_sorted = [x for _, x in sorted(zipped_pairs)]

        #add absolute path
        fps_sorted = [data_dir + x for x in fps_sorted]

        return fps_sorted

    @staticmethod
    def create_X_from_fps(fps, field_name, field_type  = "scalar"):
        """Creates a numpy array of values of scalar field_name
        Input list must be sorted"""
        M = len(fps) #number timesteps

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

            vec_len, = vector.shape
            if idx == 0:
                #fix length of vectors and initialize the output array:
                n = vec_len
                output = np.zeros((M, n))
            else:
                #enforce all vectors are of the same length
                assert vec_len == n, "All input .vtu files must be of the same length."


            output[idx] = vector

        return output.T #return (n x M)
    @staticmethod
    def create_V_from_X(X_fp):
        """Creates a mean centred matrix V from input matrix X.
        X_FP can be a numpy matrix or a fp to X"""
        if type(X_fp) == str:
            X = np.load(X_fp)
        elif type(X_fp) == np.ndarray:
            X = X_fp
        else:
            raise TypeError("X_fp must be a filpath or a numpy.ndarray")

        n, M = X.shape
        V = X - np.mean(X, axis=1) @ np.ones(n)
        V = (M - 1) ** (- 0.5) * V
        return V

    @staticmethod
    def trunc_SVD(V):
        """Performs Truncated SVD where Truncation parameter is calculated
        according to Rossella et al. 2018 (Optimal Reduced space ...)"""

        print("Starting SVD")
        U, s, VH = np.linalg.svd(V, False)
        print("U:", U.shape)
        print("s:", s.shape)
        print("VH:",VH.shape)
        print(s[0], s[1], s[2])

        np.save(INTERMEDIATE_FP + "U.npy", U)
        np.save(INTERMEDIATE_FP + "s.npy", s)
        np.save(INTERMEDIATE_FP + "VH.npy", VH)
        #first singular value
        sing_1 = s[0]
        threshold = np.sqrt(sing_1)
        trunc_idx = 0 #number of modes to retain
        for sing in s:
            if sing > threshold:
                trunc_idx += 1

        print("# modes kept: ", trunc_idx)
        singular = np.zeros_like(s)
        singular[: trunc_idx] = s[: trunc_idx]
        V_trunc = np.matmul(U, np.matmul(np.diag(singular), VH))

        return V_trunc

if __name__ == "__main__":
    main()
