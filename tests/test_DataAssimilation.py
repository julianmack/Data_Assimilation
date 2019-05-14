from pipeline import DAPipeline
from pipeline import config
import pytest
import numpy as np
DA = DAPipeline()
import os

class Test_Setup():

    def test_select_obs1(self):
        import numpy.random as random
        mode = "rand"
        u_c = random.rand(10,)
        frac =  1.0/3.0
        observations, obs_idx, nobs = DA.select_obs(mode, u_c, frac)

        for idx, obs_idx in enumerate(obs_idx):
            assert u_c[obs_idx] == observations[idx]

        assert nobs == 3, "nobs should be 3"

    def test_select_obs2(self):
        import numpy.random as random
        mode = "single_max"
        u_c = random.rand(10,) - 1
        u_c[3] = 1 #this will be max value

        observations, obs_idx, nobs = DA.select_obs(mode, u_c)
        assert nobs == 1
        assert obs_idx == [3]
        assert observations == [1]

    def test_create_H(self):
        import numpy.random as random
        mode = "single_max"
        n = 3
        u_c = random.rand(n,) - 1
        u_c[2] = 1 #this will be max value

        observations, obs_idx, nobs = DA.select_obs(mode, u_c)
        H = DA.create_H(obs_idx, n, nobs)
        assert H @ u_c == [1]
        assert H.shape == (1, 3)
        assert np.array_equal(H, np.array([[0, 0, 1]]))

    def test_vda_setup_not_normalized(self, tmpdir):
        """End-to-end seteup test"""
        X = np.zeros((3, 4))
        X[:,:2] = np.arange(6).reshape((3, 2))
        X[0, 3] = 1

        INTERMEDIATE_FP = "inter"
        p = tmpdir.mkdir(INTERMEDIATE_FP).join("X_fp.npy")
        p.dump(X)
        p.allow_pickel = True

        settings = config.Config()
        settings.X_FP = str(p)
        settings.n = 3
        settings.OBS_MODE = "single_max"
        settings.OBS_VARIANCE = 0.1
        settings.TDA_IDX_FROM_END = 0
        settings.HIST_FRAC = 0.5

        settings.NORMALIZE = False

        X_ret, n, M, hist_idx, hist_X, t_DA, u_c, V, u_0, \
                        observations, obs_idx, nobs, H_0, d,\
                        std, mean = DA.vda_setup(settings)

        mean_exp = np.array([0.5, 2.5, 4.5])
        std_exp = np.array([0.5, 0.5, 0.5])
        V_exp = X[0] - mean_exp

        assert np.array_equal(X, X_ret)
        assert (3, 4) == (n, M)
        assert [2] == hist_idx
        assert np.array_equal(np.array(X[:1]), hist_X)
        assert np.array_equal(np.array(X[-1]), u_c)
        assert t_DA == 3
        assert np.array_equal(V_exp, V)
        assert np.array_equal(mean_exp, u_0)
        assert observation == [1]
        assert obs_idx == [0]
        assert nobs == 1
        assert np.array_equal(np.array([1, 0, 0]), H_0)
        assert [0.5] == d
        assert np.array_equal(std_exp, std)
        assert np.array_equal(mean_exp, mean)

def test_vda_setup(self):
    """End-to-end test for vda_setup()"""
    settings = config.Config()
    settings.n = 5


class TestDA():
    """End-to-end tests"""
    def __settings(self):
        return config.Config()

    def test_check_import(self):
        DA = DAPipeline()
        method = DA.vda_setup
        assert callable(method), "Should be able to import DA method"

    def test_vda_setup(self):
        DA = DAPipeline()
        settings = self.__settings()
        settings.NORMALIZE = True
        #DA.vda_setup(settings)

    def test_cost_fn_J(self):
        # w, d, G,
        # V, alpha, sigma = None
        # V_grad = None, R_inv = None
        # mode=SETTINGS.COMPRESSION_METHOD
        pass


if __name__ == "__main__":
    pytest.main()
    pass
