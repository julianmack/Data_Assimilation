"""All VarDA ingesting and evaluation helpers"""

import numpy as np
import vtktools
import settings
import os
import sys
import random
sys.path.append('/home/jfm1118')
import utils

class VarDataAssimilationPipeline():
    """Class to hold @static_method pipeline functions for
    Variational DA
    """

    def __init__(self):
        pass

    @staticmethod
    def get_sorted_fps_U(data_dir):
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
        V = X.T - mean
        V = V.T
        V = (M - 1) ** (- 0.5) * V
        if return_mean:
            return V, mean
        return V
    @staticmethod
    def AE_forward():
        pass
    @staticmethod
    def trunc_SVD(V, trunc_idx=None, test=False):
        """Performs Truncated SVD where Truncation parameter is calculated
        via one of two methods:
            1) according to Rossella et al. 2018 (Optimal Reduced space ...).
            2) Alternatively, if trunc_ixd=n (where n is int), choose n modes with
                largest variance
        arguments
            :V - numpy array (n x M)
            :trunc_idx (opt) - index at which to truncate V.
        returns
            :V_trunc - truncated V (n x trunc_idx)
            :U, :s, :W - i.e. V can be factorized as:
                        V = U @ np.diag(s) @ W = U * s @ W
        """

        U, s, W = np.linalg.svd(V, False)

        np.save(settings.INTERMEDIATE_FP + "U.npy", U)
        np.save(settings.INTERMEDIATE_FP + "s.npy", s)
        np.save(settings.INTERMEDIATE_FP + "W.npy", W)
        #first singular value
        sing_1 = s[0]
        threshold = np.sqrt(sing_1)

        if not trunc_idx:
            trunc_idx = 0 #number of modes to retain
            for sing in s:
                if sing > threshold:
                    trunc_idx += 1
            if trunc_idx == 0: #when all singular values are < 1
                trunc_idx = 1
        else:
            assert type(trunc_idx) == int, "trunc_idx must be an integer"

        print("# modes kept: ", trunc_idx)
        U_trunc = U[:, :trunc_idx]
        W_trunc = W[:trunc_idx, :]
        s_trunc = s[:trunc_idx]
        V_trunc = U_trunc * s_trunc @ W_trunc

        if test:
            #1) Check generalized inverses
            V_plus = W.T * (1 / s) @  U.T #Equivalent to W.T @ np.diag(1 / s) @  U.T
            V_plus_trunc =  W_trunc.T * (1 / s_trunc) @  U_trunc.T

            assert np.allclose(V @ V_plus @ V, V), "V_plus should be generalized inverse of V"
            assert np.allclose(V_trunc @ V_plus_trunc @ V_trunc, V_trunc), "V_plus_trunc should be generalized inverse of V_trunc"

            #2) Check both methods to find V_trunc are equivalent
            # Another way to calculate V_trunc is as follows:
            singular = np.zeros_like(s)
            singular[: trunc_idx] = s[: trunc_idx]
            V_trunc2 = U * singular @ W
            assert np.allclose(V_trunc, V_trunc2)

        return V_trunc, U_trunc, s_trunc, W_trunc
    @staticmethod
    def select_obs(mode, vec, options):
        """Selects and return a subset of observations and their indexes
        from vec according to a user selected mode"""
        n = vec.shape[0]

        if mode == "rand":
            # Define observations as a random subset of the control state.
            frac = options["fraction"]
            nobs = int(frac * n) #number of observations
            utils.set_seeds(seed = settings.SEED) #set seeds so that the selected subset is the same every time
            obs_idx = random.sample(range(n), nobs) #select nobs integers w/o replacement
            observations = np.take(vec, obs_idx)
        elif mode == "single_max":
            nobs = 1
            obs_idx = np.argmax(vec)
            obs_idx = [obs_idx]
            observations = np.take(vec, obs_idx)

        return observations, obs_idx, nobs

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

    @staticmethod
    def cost_function_J(w, d, G, V, alpha, sigma = None, R_inv = None, test=True,
            mode=settings.TRUNCATION_METHOD):
        """Computes VarDA cost function.
        NOTE: eventually - implement this by hand as grad_J and J share quantity Q"""
        # trunc, = w.shape
        # n, = u_0.shape
        # nobs, = R_inv.shape[0]

        #print("G {}, V {}, w {}, d {}".format(G.shape, V.shape, w.shape, d.shape))
        if mode == "SVM":
            Q = (G @ V @ w - d)
        elif mode == "AE":
            Q = (G @ V(w) - d)

        if not R_inv and sigma:
            #When R is proportional to identity
            J_o = 0.5 / sigma ** 2 * np.dot(Q, Q)
        elif R_inv:
            J_o = 0.5 * Q.T @ R_inv @ Q
        else:
            raise ValueError("Either R_inv or sigma must be non-zero")

        J_b = 0.5 * alpha * np.dot(w, w)
        J = J_b + J_o
        print("J_b = {:.2f}, J_o = {:.2f}, w[3] = {}".format(J_b, J_o, w[3]))
        return J


        # if test:
        #     #check dimensions
        #     assert G.shape[1] == V.shape[0]
        #     assert np.allclose(V @ V_plus @ V, V), "V_plus must be the generalized inverse of V"
        #     assert  R_inv.shape[0] == nobs
        #     assert d.shape == (nobs,)
    @staticmethod
    def grad_J(w, d, G, V, alpha, V_grad = None, sigma = None, R_inv = None):

        if mode == "SVM":
            Q = (G @ V @ w - d)
            P = V.T @ G.T
        elif mode == "AE":
            assert type(V_grad) = function, "V_grad must be a function if mode=AE is used"
            Q = (G @ V(w) - d)
            P = V.T @ G.T
        if not R_inv and sigma:
            #When R is proportional to identity
            grad_o = 0.5 / sigma ** 2 * np.dot(P, Q)
        elif R_inv:
            J_o = 0.5 * P @ R_inv @ Q
        else:
            raise ValueError("Either R_inv or sigma must be non-zero")

        grad_J = alpha * w + grad_o

        return grad_J

    @staticmethod
    def save_vtu_file(arr, name, filename, sample_fp=None):
        """Saves a VTU file - NOTE TODO - should be using deep copy method in vtktools.py -> VtuDiff()"""
        if sample_fp == None:
            sample_fp = vda.get_sorted_fps_U(settings.DATA_FP)[0]

        ug = vtktools.vtu(sample_fp) #use sample fp to initialize positions on grid

        ug.AddScalarField('name', arr)
        ug.Write(filename)
