"""All VarDA ingesting and evaluation helpers"""

import numpy as np
import os
import random
import torch
from scipy.optimize import minimize
import vtktools


from pipeline import config, utils


class DAPipeline():
    """Class to hold pipeline functions for Variational DA
    """

    def __init__(self, settings = None):
        if settings:
            self.settings = settings

    def Var_DA_routine(self, settings, return_stats=False):
        """Runs the variational DA routine using settings from the passed config class
        (see config.py for example)"""

        self.settings = settings

        self.data, std, mean = self.vda_setup(settings)

        u_0 = self.data.get("u_0")
        u_c = self.data.get("u_c")

        self.init_VarDA()

        w_0 = self.data.get("w_0")

        DA_results = self.perform_VarDA(self.data, self.settings)

        ref_MAE = DA_results["ref_MAE"]
        da_MAE = DA_results["da_MAE"]
        u_DA = DA_results["u_DA"]
        ref_MAE_mean = DA_results["ref_MAE_mean"]
        da_MAE_mean = DA_results["da_MAE_mean"]
        w_opt = DA_results["w_opt"]


        if settings.DEBUG:
            size = len(std)
            if size > 4:
                size = 4
            print("std:    ", std[-size:])
            print("mean:   ", mean[-size:])
            print("u_0:    ", u_0[-size:])
            print("u_c:    ", u_c[-size:])
            print("u_DA:   ", u_DA[-size:])
            print("ref_MAE:", ref_MAE[-size:])
            print("da_MAE: ", da_MAE[-size:])

        counts = (ref_MAE > da_MAE).sum()

        print("RESULTS")

        print("Reference MAE: ", ref_MAE_mean)
        print("DA MAE: ", da_MAE_mean)
        print("ref_MAE_mean > da_MAE_mean for {}/{}".format(counts, da_MAE.shape[0]))
        print("If DA has worked, DA MAE > Ref_MAE")
        print("Percentage improvement: {:.2f}%".format(100*(ref_MAE_mean - da_MAE_mean)/ref_MAE_mean))
        #Compare abs(u_0 - u_c).sum() with abs(u_DA - u_c).sum() in paraview

        if settings.SAVE:
            #Save .vtu files so that I can look @ in paraview
            sample_fp = utils.DataLoader.get_sorted_fps_U(settings.DATA_FP)[0]
            out_fp_ref = settings.INTERMEDIATE_FP + "ref_MAE.vtu"
            out_fp_DA =  settings.INTERMEDIATE_FP + "DA_MAE.vtu"

            utils.FluidityUtils.save_vtu_file(ref_MAE, "ref_MAE", out_fp_ref, sample_fp)
            utils.FluidityUtils.save_vtu_file(da_MAE, "DA_MAE", out_fp_DA, sample_fp)
        if return_stats:
            stats = {}
            stats["Percent_improvement"] = 100*(ref_MAE_mean - da_MAE_mean)/ref_MAE_mean
            stats["ref_MAE_mean"] = ref_MAE_mean
            stats["da_MAE_mean"] = da_MAE_mean
            return w_opt, stats
        return w_opt


    def vda_setup(self, settings):
        """Generates matrices for VarDA. All returned matrices are in the
        (n X M) format (as typical in VarDA) although when
        settings.THREE_DIM = True, some are 4-dimensional"""
        data = {}
        loader = utils.DataLoader()
        X = loader.get_X(settings)

        train_X, test_X, u_c, X, mean, std = loader.test_train_DA_split_maybe_normalize(X, settings)

        V = self.create_V_from_X(train_X, settings)

        if settings.THREE_DIM:
            #MUST return in ( nx x ny x nz x M) form
            raise NotImplementedError("Must deal with 3d case")
        else:
            #Deal with dimensions:
            #currently dim are: (M x nx x ny x nz ) or (M x n )
            X = X.T
            train_X = train_X.T
            test_X = test_X.T
            V = V.T


        # We will take initial condition u_0, as mean of historical data
        if settings.NORMALIZE:
            u_0 = np.zeros(settings.n) #since the data is mean centred
        else:
            u_0 = mean

        observations, obs_idx, nobs = self.select_obs(settings.OBS_MODE, u_c, settings.OBS_FRAC) #options are specific for rand

        #Now define quantities required for 3D-VarDA - see Algorithm 1 in Rossella et al (2019)
        H_0 = self.create_H(obs_idx, settings.n, nobs)
        d = observations - H_0 @ u_0 #'d' in literature
        #R_inv = self.create_R_inv(OBS_VARIANCE, nobs)
        data = {"d": d, "G": H_0, "V": V,
                "observations": observations,
                "u_c": u_c, "u_0": u_0, "X": X,
                "train_X": train_X, "test_X":test_X,
                "std": std, "mean": mean}


        return data, std, mean

    def init_VarDA(self):
        settings = self.settings
        data = self.data
        V = self.data["V"]

        if settings.COMPRESSION_METHOD == "SVD":
            V_trunc, U, s, W = self.trunc_SVD(V, settings.NUMBER_MODES)
            data["V_trunc"] = V_trunc
            #Define intial w_0
            self.data["w_0"] = np.zeros((W.shape[-1],)) #TODO - I'm not sure about this - can we assume is it 0?

            self.data["V_grad"] = None
            # OR - Alternatively, use the following:
            # V_plus_trunc = W.T * (1 / s) @  U.T
            # w_0_v2 = V_plus_trunc @ u_0 #i.e. this is the value given in Rossella et al (2019).
            #     #I'm not clear if there is any difference - we are minimizing so expect them to
            #     #be equivalent
            # w_0 = w_0_v2

        elif settings.COMPRESSION_METHOD == "AE":
            kwargs = settings.get_kwargs()

            encoder, decoder = utils.ML_utils.load_AE(settings.AE_MODEL_TYPE, settings.AE_MODEL_FP, **kwargs)

            V_trunc = decoder
            self.data["V_trunc"] = V_trunc

            self.data["w_0"] = torch.zeros((settings.NUMBER_MODES))
            #u_0 = decoder(w_0).detach().numpy()

            # Now access explicit gradient function
            try:
                self.data["V_grad"] = settings.AE_MODEL_TYPE(**kwargs).jac_explicit
            except:
                raise NotImpelemtedError("This model type does not have a gradient available")
        else:
            raise ValueError("COMPRESSION_METHOD must be in {SVD, AE}")

    @staticmethod
    def perform_VarDA(data, settings):
        """This is a static method so that it can be performed in AE_train with user specified data"""
        args = (data, settings)
        res = minimize(DAPipeline.cost_function_J, data.get("w_0"), args = args, method='L-BFGS-B',
                jac=DAPipeline.grad_J, tol=settings.TOL)

        w_opt = res.x
        if settings.COMPRESSION_METHOD == "SVD":
            delta_u_DA = data.get("V_trunc") @ w_opt
        elif settings.COMPRESSION_METHOD == "AE":
            delta_u_DA = data.get("V_trunc")(torch.Tensor(w_opt)).detach().numpy()

        u_0 = data.get("u_0")
        u_c = data.get("u_c")

        u_DA = u_0 + delta_u_DA

        #Undo normalization
        if settings.UNDO_NORMALIZE:
            std = data.get("std")
            mean = data.get("mean")
            u_DA = (u_DA.T * std + mean).T
            u_c = (u_c.T * std + mean).T
            u_0 = (u_0.T * std + mean).T
        elif settings.NORMALIZE:
            print("Normalization not undone")

        ref_MAE = np.abs(u_0 - u_c)
        da_MAE = np.abs(u_DA - u_c)
        ref_MAE_mean = np.mean(ref_MAE)
        da_MAE_mean = np.mean(da_MAE)

        results_data = {"ref_MAE": ref_MAE,
                    "da_MAE": da_MAE,
                    "u_DA": u_DA,
                    "ref_MAE_mean": ref_MAE_mean,
                    "da_MAE_mean": da_MAE_mean,
                    "w_opt": w_opt}
        return results_data


    @staticmethod
    def create_V_from_X(X_fp, settings):
        """Creates a mean centred matrix V from input matrix X.
        X_FP can be a numpy matrix or a fp to X.
        returns V in the  M x n format"""

        if type(X_fp) == str:
            X = np.load(X_fp)
        elif type(X_fp) == np.ndarray:
            X = X_fp
        else:
            raise TypeError("X_fp must be a filpath or a numpy.ndarray")

        M, n = utils.DataLoader.get_dim_X(X, settings)

        mean = np.mean(X, axis=0)

        V = (X - mean)

        # V = (M - 1) ** (- 0.5) * V

        return V

    def get_npoints_from_shape(self, n):
        if type(n) == tuple:
            npoints = 1
            for val in n:
                npoints *= val
        elif type(n) == int:
            npoints = n
        else:
            raise TypeError("Size n must be of type int or tuple")
        return npoints

    def select_obs(self, mode, vec, frac=None):
        """Selects and return a subset of observations and their indexes
        from vec according to a user selected mode"""
        npoints = self.get_npoints_from_shape(vec.shape)

        if mode == "rand":
            # Define observations as a random subset of the control state.
            nobs = int(frac * npoints) #number of observations

            utils.set_seeds(seed = self.settings.SEED) #set seeds so that the selected subset is the same every time
            obs_idx = random.sample(range(npoints), nobs) #select nobs integers w/o replacement
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
        #raise NotImplementedError("Haven't worked out what to do with 3D observation operator")

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


    def trunc_SVD(self, V, trunc_idx=None, test=False):
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
        settings = self.settings
        U, s, W = np.linalg.svd(V, False)

        if settings.SAVE:
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
    def cost_function_J(w, data, settings):
        """Computes VarDA cost function.
        NOTE: eventually - implement this by hand as grad_J and J share quantity Q"""


        d = data.get("d")
        G = data.get("G")
        V_trunc = data.get("V_trunc")
        V =  V_trunc if V_trunc is not None else data.get("V")
        V_grad = data.get("V_grad")
        R_inv = data.get("R_inv")

        sigma_2 = settings.OBS_VARIANCE
        mode = settings.COMPRESSION_METHOD
        alpha = settings.ALPHA

        if mode == "SVD":
            Q = (G @ V @ w - d)

        elif mode == "AE":
            assert callable(V), "V must be a function if mode=AE is used"
            w_tensor = torch.Tensor(w)
            V_w = V(w_tensor).detach().numpy()
            print(V)
            print(w_tensor.shape, "w_tensor.shape")
            print(G.shape, "G.shape")
            print(V_w.shape, "(V_w.shape)")
            print(d.shape, "d.shape")
            Q = (G @ V_w - d)

        else:
            raise ValueError("Invalid mode")

        if sigma_2 and not R_inv:
            #When R is proportional to identity
            J_o = 0.5 / sigma_2 * np.dot(Q, Q)
        elif R_inv:
            J_o = 0.5 * Q.T @ R_inv @ Q
        else:
            raise ValueError("Either R_inv or sigma must be provided")

        J_b = 0.5 * alpha * np.dot(w, w)
        J = J_b + J_o

        if settings.DEBUG:
            print("J_b = {:.2f}, J_o = {:.2f}".format(J_b, J_o))
        return J


    @staticmethod
    def grad_J(w, data, settings):
        d = data.get("d")
        G = data.get("G")
        V_trunc = data.get("V_trunc")
        V =  V_trunc if V_trunc is not None else data.get("V")
        V_grad = data.get("V_grad")
        R_inv = data.get("R_inv")

        sigma_2 = settings.OBS_VARIANCE
        mode = settings.COMPRESSION_METHOD
        alpha = settings.ALPHA

        if mode == "SVD":
            Q = (G @ V @ w - d)
            P = V.T @ G.T
        elif mode == "AE":
            assert callable(V_grad), "V_grad must be a function if mode=AE is used"
            w_tensor = torch.Tensor(w)

            V_w = V(w_tensor).detach().numpy()

            V_grad_w = V_grad(w_tensor).detach().numpy()

            Q = (G @ V_w - d)
            P = V_grad_w.T @ G.T
        if not R_inv and sigma_2:
            #When R is proportional to identity
            grad_o = (1.0 / sigma_2 ) * np.dot(P, Q)
        elif R_inv:
            J_o = 1.0 * P @ R_inv @ Q
        else:
            raise ValueError("Either R_inv or sigma must be non-zero")

        grad_J = alpha * w + grad_o

        return grad_J


if __name__ == "__main__":


    DA = DAPipeline()
    settings = config.Config()
    DA.Var_DA_routine(settings)
    exit()

    #create X:
    loader = utils.DataLoader()
    X = loader.get_X(settings)
    np.save(settings.X_FP, X)
    exit()
