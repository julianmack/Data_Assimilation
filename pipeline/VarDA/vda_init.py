import numpy as np
import random
import torch

from pipeline import ML_utils
from pipeline import GetData, SplitData

class VDAInit:
    def __init__(self, settings, AEmodel=None, u_c=None):
        self.AEmodel = AEmodel
        self.settings = settings
        self.u_c = u_c #user can pass their own u_c

    def run(self):
        """Generates matrices for VarDA. All returned matrices are in the
        (M X n) or (M x nx x ny x nz) format """

        data = {}
        loader = GetData()
        splitter = SplitData()
        settings = self.settings

        X = loader.get_X(settings)

        train_X, test_X, u_c_std, X, mean, std = splitter.train_test_DA_split_maybe_normalize(X, settings)

        if self.u_c is None:
            self.u_c = u_c_std

        #self.u_c = train_X[62] #good
        #self.u_c = train_X[-1] #bad

        # We will take initial condition u_0, as mean of historical data
        if settings.NORMALIZE:
            u_0 = np.zeros_like(mean) #since the data is mean centred
        else:
            u_0 = mean

        encoder = None
        decoder = None
        model = self.AEmodel

        if self.settings.COMPRESSION_METHOD == "AE":
            #get encoder
            device = ML_utils.get_device()
            if model is None:
                model = ML_utils.load_model_from_settings(settings)


            def __create_encoderOrDecoder(fn):
                """This returns a function that deals with encoder/decoder
                input dimensions (e.g. adds channel dim for 3D case)"""
                def ret_fn(vec):
                    vec = torch.Tensor(vec).to(device)

                    #for 3D case, unsqueeze for channel
                    if self.settings.THREE_DIM:
                        dims = len(vec.shape)
                        if dims == 3:

                            vec = vec.unsqueeze(0)
                        elif dims == 4:
                            #batched input
                            vec = vec.unsqueeze(1)
                    res = fn(vec).detach().cpu()
                    #for 3D case, squeeze for channel
                    dims = len(res.shape)
                    if self.settings.THREE_DIM and dims > 2:
                        if dims == 4:
                            res = res.squeeze(0)
                        elif dims == 5:   #batched input
                            res = res.squeeze(1)
                    return res.numpy()

                return ret_fn

            encoder = __create_encoderOrDecoder(model.encode)
            decoder = __create_encoderOrDecoder(model.decode)


        if self.settings.REDUCED_SPACE == True:
            if self.settings.COMPRESSION_METHOD == "SVD":
                raise NotImplementedError("SVD in reduced space not implemented")

            self.settings.OBS_MODE = "all"

            observations, H_0, w_0, d = self.__get_obs_and_d_reduced_space(self.settings, self.u_c, u_0, encoder)

        else:
            observations, H_0, w_0, d = self.__get_obs_and_d_not_reduced(self.settings, self.u_c, u_0, encoder)

        device = ML_utils.get_device()

        #TODO - **maybe** get rid of this monstrosity...:
        #i.e. you could return a class that has these attributes:

        data = {"d": d, "G": H_0,
                "observations": observations,
                "model": model,
                "encoder": encoder, "decoder": decoder,
                "u_c": self.u_c, "u_0": u_0, "X": X,
                "train_X": train_X, "test_X":test_X,
                "std": std, "mean": mean, "device": device}

        if w_0 is not None:
            data["w_0"] = w_0

        return data

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

        M, n = SplitData.get_dim_X(X, settings)

        mean = np.mean(X, axis=0)

        V = (X - mean)

        # V = (M - 1) ** (- 0.5) * V

        return V

    @staticmethod
    def select_obs(settings, vec):
        """Selects and return a subset of observations and their indexes
        from vec according to a user selected mode"""
        npoints = VDAInit.__get_npoints_from_shape(vec.shape)
        if settings.OBS_MODE == "rand":
            # Define observations as a random subset of the control state.
            nobs = int(settings.OBS_FRAC * npoints) #number of observations

            if nobs == npoints: #then we are selecting all points
                settings.OBS_MODE = "all"
                return VDAInit.__select_all_obs(vec)

            ML_utils.set_seeds(seed = settings.SEED) #set seeds so that the selected subset is the same every time
            obs_idx = random.sample(range(npoints), nobs) #select nobs integers w/o replacement
            observations = np.take(vec, obs_idx)
        elif settings.OBS_MODE == "single_max":
            nobs = 1
            obs_idx = np.argmax(vec)
            obs_idx = [obs_idx]
            observations = np.take(vec, obs_idx)
        elif settings.OBS_MODE == "all":
            observations, obs_idx, nobs = VDAInit.__select_all_obs(vec)
        else:
            raise ValueError("OBS_MODE = {} is not allowed.".format(settings.OBS_MODE))
        return observations, obs_idx, nobs

    @staticmethod
    def create_H(obs_idxs, n, nobs, three_dim=False):
        """Creates the mapping matrix from the statespace to the observations.
            :obs_idxs - an iterable of indexes @ which the observations are made
            :n - size of state space
            :nobs - number of observations
        returns
            :H - numpy array of size (nobs x n)
        """
        if three_dim:
             nx, ny, nz = n
             n = nx * ny * nz
        else:
            assert type(n) == int

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
    def __get_obs_and_d_not_reduced(settings, u_c, u_0, encoder=None):
        if settings.COMPRESSION_METHOD == "AE":
            if encoder is None:
                raise ValueError("Encoder must be provided if settings.COMPRESSION_METHOD == `AE` ")
            w_0 = encoder(u_0)
            #w_0_v1 = torch.zeros((settings.get_number_modes())).to(device)
        else:
            w_0 = None #this will be initialized in SVD_DA()
        observations, obs_idx, nobs = VDAInit.select_obs(settings, u_c) #options are specific for rand
        H_0 = VDAInit.create_H(obs_idx, settings.get_n(), nobs, settings.THREE_DIM)
        d = observations - H_0 @ u_0.flatten() #'d' in literature
        #R_inv = self.create_R_inv(OBS_VARIANCE, nobs)
        return observations, H_0, w_0, d

    @staticmethod
    def __get_obs_and_d_reduced_space(settings, u_c, u_0, encoder):
        """helper function to get observations and H_0 and d in reduced space"""
        if len(u_c.shape) not in [1, 3]:
            raise ValueError("This function does not accept batched input with {} dimensions".format(len(u_c.shape)))
        w_c = encoder(u_c)
        observations, _, nobs = VDAInit.select_obs(settings, w_c)
        H_0 = np.eye(nobs) #i.e. using all observations
        w_0 = encoder(u_0)

        d = observations - H_0 @ w_0.flatten()
        return observations, H_0, w_0, d

    @staticmethod
    def provide_u_c_update_data_full_space(data, settings, u_c):
        assert settings.REDUCED_SPACE == False, "This function only works in reduced space"
        encoder = data.get("encoder")
        u_0 = data.get("u_0")
        if encoder is None and settings.COMPRESSION_METHOD == "AE":
            raise ValueError("Encoder must be initialized in `data` dict")
        if u_0 is None:
            raise ValueError("u_0 must be initialized in `data` dict")

        observations, H_0, w_0, d = VDAInit.__get_obs_and_d_not_reduced(settings, u_c, u_0, encoder)
        data["observations"] = observations
        data["G"] = H_0

        if w_0 is not None: #i.e. don't update if no result was returned
            data["w_0"] = w_0

        data["d"] = d
        data["u_c"] = u_c
        return data


    @staticmethod
    def provide_u_c_update_data_reduced_AE(data, settings, u_c):
        assert settings.REDUCED_SPACE == True, "This function only works in reduced space"
        assert settings.COMPRESSION_METHOD == "AE", "This function only works for AE"
        encoder = data.get("encoder")
        u_0 = data.get("u_0")
        if encoder is None:
            raise ValueError("Encoder must be initialized in `data` dict")
        if u_0 is None:
            raise ValueError("u_0 must be initialized in `data` dict")

        observations, H_0, w_0, d = VDAInit.__get_obs_and_d_reduced_space(settings, u_c, u_0, encoder)
        data["observations"] = observations
        data["G"] = H_0
        data["w_0"] = w_0
        data["d"] = d
        data["u_c"] = u_c
        return data

    @staticmethod
    def create_V_red(X, encoder, num_modes, settings):
        V = VDAInit.create_V_from_X(X, settings)
        assert V.shape[0] >= num_modes

        #take random selection of V:
        #idxs = random.sample(range(V.shape[0]), num_modes)

        #take evenly spaced states
        step = V.shape[0] / num_modes
        idxs = [int(x * step) for x in range(num_modes)]

        assert len(idxs) == num_modes

        V_selected = V[idxs]

        V_red = encoder(V_selected)

        return V_red

    @staticmethod
    def __select_all_obs(vec):
        nobs = VDAInit.__get_npoints_from_shape(vec.shape)
        return vec, list(range(nobs)), nobs

    @staticmethod
    def __get_npoints_from_shape(n):
        if type(n) == tuple:
            npoints = 1
            for val in n:
                npoints *= val
        elif type(n) == int:
            npoints = n
        else:
            raise TypeError("Size n must be of type int or tuple")
        return npoints