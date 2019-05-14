from pipeline import DAPipeline
from pipeline import config
import pytest
DA = DAPipeline()

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
        import numpy as np
        mode = "single_max"
        n = 3
        u_c = random.rand(n,) - 1
        u_c[2] = 1 #this will be max value

        observations, obs_idx, nobs = DA.select_obs(mode, u_c)
        H = DA.create_H(obs_idx, n, nobs)

        assert H @ u_c == [1]
        assert H.shape == (1, 3)
        assert np.array_equal(H, np.array([[0, 0, 1]]))

    def test_vda_setup(self):
        """End-to-end test for vda_setup()"""
        


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
        DA.vda_setup(settings)

    def test_cost_fn_J(self):
        # w, d, G,
        # V, alpha, sigma = None
        # V_grad = None, R_inv = None
        # mode=SETTINGS.COMPRESSION_METHOD
        pass


if __name__ == "__main__":
    pytest.main()
    pass
