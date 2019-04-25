"""All VarDA ingesting and evaluation helpers"""

import numpy as np
import vtktools
import settings
import os

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
    def create_V_from_X(X_fp, return_mean = False):
        """Creates a mean centred matrix V from input matrix X.
        X_FP can be a numpy matrix or a fp to X"""
        if type(X_fp) == str:
            X = np.load(X_fp)
        elif type(X_fp) == np.ndarray:
            X = X_fp
        else:
            raise TypeError("X_fp must be a filpath or a numpy.ndarray")

        n, M = X.shape
        mean = np.mean(X, axis=1)
        V = X - mean @ np.ones(n)
        V = (M - 1) ** (- 0.5) * V

        if return_mean:
            return V, mean
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

        np.save(settings.INTERMEDIATE_FP + "U.npy", U)
        np.save(settings.INTERMEDIATE_FP + "s.npy", s)
        np.save(settings.INTERMEDIATE_FP + "VH.npy", VH)
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

    @staticmethod
    def create_H(obs_idxs, n, nobs):
        """Creates the mapping matrix from the statespace to the observations.
            :obs_idxs - an iterable of indexes @ which the observations are made
            :n - size of state space
            :nobs - number of observations
        returns
            :H - numpy array of size (nobs x n)
        """

        H = np.zeros((nobs, n))
        H[range(nobs), obs_idxs] = 1

        assert H[0, obs_idxs[0]] == 1, "1s were not correctly assigned"
        assert H[0, (obs_idxs[0] + 1) % n ] == 0, "0s were not correctly assigned"
        assert H[nobs - 1, obs_idxs[-1]] == 1, "1s were not correctly assigned"
        assert H.shape == (nobs, n)

        return H

    @staticmethod
    def create_R_inv(sigma, nobs):
        """Creates inverse of R: the observation error matrix.
        Assume all observations are independent s.t. R = sigma**2 * identity.
        args
            :sigma - observation error variance
            :nobs - number of observations
        returns
            :R_inv - (nobs x nobs) array"""

        R_inv = 1.0 / sigma ** 2 * np.eye(nobs)

        assert R_inv.shape == (nobs, nobs)
        
        return R_inv
