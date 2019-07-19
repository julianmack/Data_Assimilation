import numpy as np
import random

from pipeline import ML_utils
from pipeline import GetData, SplitData

class VDAInit:
    def __init__(self, settings):
        self.settings = settings

    def run(self):
        """Generates matrices for VarDA. All returned matrices are in the
        (n X M) format (as typical in VarDA) although when
        settings.THREE_DIM = True, some are 4-dimensional"""

        data = {}
        loader = GetData()
        splitter = SplitData()
        settings = self.settings

        X = loader.get_X(settings)

        train_X, test_X, u_c, X, mean, std = splitter.train_test_DA_split_maybe_normalize(X, settings)

        #Deal with dimensions:
        #currently dim are: (M x n ) or (M x nx x ny x nz)
        #change to (n x M) or (nx x ny x nz x M)
        if not settings.THREE_DIM:
            pass
            # X = X.T
            # train_X = train_X.T
            # test_X = test_X.T

        else:
            pass
            # X = np.moveaxis(X, 0, 3)
            # train_X = np.moveaxis(train_X, 0, 3)
            # test_X = np.moveaxis(test_X, 0, 3)


        # We will take initial condition u_0, as mean of historical data
        if settings.NORMALIZE:
            u_0 = np.zeros(settings.get_n()) #since the data is mean centred
        else:
            u_0 = mean

        #TODO - possible don't flatten these:
        # i.e. deal with in the SVD fn

        #flatten 3D vectors:
        u_c = u_c.flatten()
        std = std.flatten()
        mean = mean.flatten()
        u_0_not_flat = u_0 #TODO: get rid of this when nothing is flat
        u_0 = u_0.flatten()

        #TODO - the reduced space idea should (maybe) be able to work for SVD too??

        observations, obs_idx, nobs = self.select_obs(u_c) #options are specific for rand

        #Now define quantities required for 3D-VarDA - see Algorithm 1 in Rossella et al (2019)
        H_0 = self.create_H(obs_idx, settings.get_n(), nobs, settings.THREE_DIM)
        d = observations - H_0 @ u_0 #'d' in literature
        #R_inv = self.create_R_inv(OBS_VARIANCE, nobs)

        device = ML_utils.get_device()

        #TODO - **maybe** get rid of this monstrosity...:
        #i.e. you could return a class that has these attributes:

        data = {"d": d, "G": H_0,
                "observations": observations,
                "u_c": u_c, "u_0": u_0, "u_0_not_flat": u_0_not_flat, "X": X,
                "train_X": train_X, "test_X":test_X,
                "std": std, "mean": mean, "device": device}

        #TODO - if you keep data, get rid of std and mean
        return data, std, mean

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

    def select_obs(self, vec):
        """Selects and return a subset of observations and their indexes
        from vec according to a user selected mode"""
        npoints = self.__get_npoints_from_shape(vec.shape)

        if self.settings.REDUCED_SPACE == True:
            raise NotImplementedError("Reduced Space method not implemented")
            #Then go one of three ways (rand, single_max, all)
        if self.settings.OBS_MODE == "rand":

            # Define observations as a random subset of the control state.
            nobs = int(self.settings.OBS_FRAC * npoints) #number of observations

            ML_utils.set_seeds(seed = self.settings.SEED) #set seeds so that the selected subset is the same every time
            obs_idx = random.sample(range(npoints), nobs) #select nobs integers w/o replacement
            observations = np.take(vec, obs_idx)
        elif self.settings.OBS_MODE == "single_max":
            nobs = 1
            obs_idx = np.argmax(vec)
            obs_idx = [obs_idx]
            observations = np.take(vec, obs_idx)
        elif self.settings.OBS_MODE == "all":
            raise NotImplementedError("select all obs not impelemented")
        else:
            raise ValueError("OBS_MODE = {} is not allowed.".format(self.settings.OBS_MODE))
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


    def __get_npoints_from_shape(self, n):
        if type(n) == tuple:
            npoints = 1
            for val in n:
                npoints *= val
        elif type(n) == int:
            npoints = n
        else:
            raise TypeError("Size n must be of type int or tuple")
        return npoints