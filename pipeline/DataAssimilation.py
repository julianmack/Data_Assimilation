"""All VarDA ingesting and evaluation helpers"""

import numpy as np
import os
import random
import torch
from scipy.optimize import minimize
import vtktools
import config
import utils

SETTINGS = config.Config

class DAPipeline():
    """Class to hold @static_method pipeline functions for
    Variational DA and Kalman DA
    """

    def __init__(self):
        pass

    def Var_DA_routine(self, settings = config.Config):
        """Runs the variational DA routine using settings from the passed config class
        (see config.py for example)"""
        #initialize helper function class
        X, n, M, hist_idx, hist_X, t_DA, u_c, V, u_0, \
                        observations, obs_idx, nobs, H_0, d, std, mean = self.vda_setup(settings)

        if settings.COMPRESSION_METHOD == "SVD":
            V_trunc, U, s, W = self.trunc_SVD(V, settings.NUMBER_MODES)
            #Define intial w_0
            w_0 = np.zeros((W.shape[-1],)) #TODO - I'm not sure about this - can we assume is it 0?

            #OR - Alternatively, use the following:
            # V_plus_trunc = W.T * (1 / s) @  U.T
            # w_0_v2 = V_plus_trunc @ u_0 #i.e. this is the value given in Rossella et al (2019).
            #     #I'm not clear if there is any difference - we are minimizing so expect them to
            #     #be equivalent
            # w_0 = w_0_v2

        elif settings.COMPRESSION_METHOD == "AE":

            latent_size = 2
            kwargs = {"input_size": n, "latent_size": latent_size,"hid_layers":[1000, 200]}
            encoder, decoder = utils.ML_utils.load_AE(settings.AE_MODEL_TYPE, settings.AE_MODEL_FP, **kwargs)
            w_0 = torch.zeros((1, latent_size), requires_grad = True)
            u_0 = decoder(w_0)

            raise NotImplementedError("AE not implemented. Need to calculate NN gradient")

        else:
            raise ValueError("COMPRESSION_METHOD must be in {SVD, AE}")

        #Define costJ and grad_J
        args =  (d, H_0, V_trunc, settings.ALPHA, settings.OBS_VARIANCE) # list of all args required for cost_function_J and grad_J
        #args =  (d, H_0, V_trunc, ALPHA, None, R_inv) # list of all args required for cost_function_J and grad_J
        res = minimize(self.cost_function_J, w_0, args = args, method='L-BFGS-B',
                jac=self.grad_J, tol=settings.TOL)

        w_opt = res.x
        delta_u_DA = V_trunc @ w_opt
        u_DA = u_0 + delta_u_DA

        #Undo normalization
        if settings.NORMALIZE:
            u_DA = (u_DA.T * std + mean).T
            u_c = (u_c.T * std + mean).T
            u_0 = (u_0.T * std + mean).T

        ref_MAE = np.abs(u_0 - u_c)
        da_MAE = np.abs(u_DA - u_c)

        ref_MAE_mean = np.mean(ref_MAE)
        da_MAE_mean = np.mean(da_MAE)

        print("RESULTS")

        print("Reference MAE: ", ref_MAE_mean)
        print("DA MAE: ", da_MAE_mean)
        print("If DA has worked, DA MAE > Ref_MAE")
        #Compare abs(u_0 - u_c).sum() with abs(u_DA - u_c).sum() in paraview

        #Save .vtu files so that I can look @ in paraview
        sample_fp = self.get_sorted_fps_U(settings.DATA_FP)[0]
        out_fp_ref = settings.INTERMEDIATE_FP + "ref_MAE.vtu"
        out_fp_DA =  settings.INTERMEDIATE_FP + "DA_MAE.vtu"

        self.save_vtu_file(ref_MAE, "ref_MAE", out_fp_ref, sample_fp)
        self.save_vtu_file(da_MAE, "DA_MAE", out_fp_DA, sample_fp)


    def vda_setup(self, settings):
        #The X array should already be saved in settings.X_FP
        #but can be created from .vtu fps if required. see trunc_SVD.py for an example
        X = np.load(settings.X_FP)
        n, M = X.shape

        # Split X into historical and present data. We will
        # assimilate "observations" at a single timestep t_DA
        # which corresponds to the control state u_c
        # We will take initial condition u_0, as mean of historical data
        hist_idx = int(M * settings.HIST_FRAC)
        t_DA = M - settings.TDA_IDX_FROM_END
        assert t_DA > hist_idx, "Cannot select observation from historical data. \
                    Reduce HIST_FRAC or reduce TDA_IDX_FROM_END to prevent overlap"

        hist_X = X[:, : hist_idx] #select training set data

        if settings.NORMALIZE:
            #use only the training set to calculate mean and std
            mean = np.mean(X, axis=1)
            std = np.std(X, axis=1)
            #NOTE: when hist_X -> X in 2 lines above, the MAE reduces massively
            #In preliminary experiments, this is not true with hist_X

            X = (X.T - mean).T
            X = (X.T / std).T

            hist_X = X[:, : hist_idx]
            V, u_0, _ = self.create_V_from_X(hist_X, return_mean = True)
        else:
            V, mean, std = self.create_V_from_X(hist_X, return_mean = True)
            u_0 = mean

        u_c = X[:, t_DA]
        observations, obs_idx, nobs = self.select_obs(settings.OBS_MODE, u_c, {"fraction": settings.OBS_FRAC}) #options are specific for rand

        #Now define quantities required for 3D-VarDA - see Algorithm 1 in Rossella et al (2019)
        H_0 = self.create_H(obs_idx, n, nobs)
        d = observations - H_0 @ u_0 #'d' in literature
        #R_inv = self.create_R_inv(OBS_VARIANCE, nobs)

        return X, n, M, hist_idx, hist_X, t_DA, u_c, V, u_0, \
                        observations, obs_idx, nobs, H_0, d, std, mean

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
        std = np.std(X, axis=1)

        V = (X.T - mean).T

        V = (M - 1) ** (- 0.5) * V
        if return_mean:
            return V, mean, std
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

        np.save(SETTINGS.INTERMEDIATE_FP + "U.npy", U)
        np.save(SETTINGS.INTERMEDIATE_FP + "s.npy", s)
        np.save(SETTINGS.INTERMEDIATE_FP + "W.npy", W)
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
            utils.set_seeds(seed = SETTINGS.SEED) #set seeds so that the selected subset is the same every time
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
            mode=SETTINGS.COMPRESSION_METHOD):
        """Computes VarDA cost function.
        NOTE: eventually - implement this by hand as grad_J and J share quantity Q"""
        # trunc, = w.shape
        # n, = u_0.shape
        # nobs, = R_inv.shape[0]

        #print("G {}, V {}, w {}, d {}".format(G.shape, V.shape, w.shape, d.shape))
        if mode == "SVD":
            Q = (G @ V @ w - d)
        elif mode == "AE":
            Q = (G @ V(w) - d)
        else:
            raise ValueError("Invalid mode")

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
    def grad_J(w, d, G, V, alpha, sigma = None, V_grad = None, R_inv = None, mode=SETTINGS.COMPRESSION_METHOD):

        if mode == "SVD":
            Q = (G @ V @ w - d)
            P = V.T @ G.T
        elif mode == "AE":
            assert type(V_grad) == "function", "V_grad must be a function if mode=AE is used"
            x = V(w)
            Q = (G @ x - d)
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
            sample_fp = vda.get_sorted_fps_U(SETTINGS.DATA_FP)[0]

        ug = vtktools.vtu(sample_fp) #use sample fp to initialize positions on grid

        ug.AddScalarField('name', arr)
        ug.Write(filename)

if __name__ == "__main__":
    SETTINGS = config.Config

    DA = DAPipeline()
    DA.Var_DA_routine(SETTINGS)
