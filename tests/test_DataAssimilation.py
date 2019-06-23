from pipeline import DAPipeline
from pipeline import config
import pytest
import numpy as np

DA = DAPipeline(config.Config())
import os

class TestSetup():
    def test_check_import(self):
        DA = DAPipeline()
        method = DA.vda_setup
        assert callable(method), "Should be able to import DA method"


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


    def test_vda_setup_notnormalized(self, tmpdir):
        """End-to-end setup test"""
        X = np.zeros((3, 4))
        X[:,:2] = np.arange(6).reshape((3, 2))
        X[0, 3] = 1
        X = X.T


        INTERMEDIATE_FP = "inter"
        p = tmpdir.mkdir(INTERMEDIATE_FP).join("X_fp.npy")
        p.dump(X)
        p.allow_pickel = True

        settings = config.Config()
        settings.X_FP = str(p)
        settings.n = 3
        settings.FORCE_GEN_X = False
        settings.OBS_MODE = "single_max"
        settings.OBS_VARIANCE = 0.1
        settings.TDA_IDX_FROM_END = 0
        settings.HIST_FRAC = 0.5

        settings.NORMALIZE = False

        data,  std, mean = DA.vda_setup(settings)
        X_ret = data.get("X")
        V = data.get("V")
        n, M = X_ret.shape
        u_c = data.get("u_c")
        u_0 = data.get("u_0")
        nobs = len(data["observations"])

        X_expt = X.T
        mean_exp = np.array([0.5, 2.5, 4.5])
        std_exp = np.array([0.5, 0.5, 0.5])


        V_exp = X_expt[:,:2] - mean_exp.reshape((-1, 1))

        assert np.array_equal(np.array(X_expt[:,:2]), data.get("train_X"))
        assert np.array_equal(X_expt, X_ret)
        assert (3, 4) == (n, M)
        assert np.array_equal(np.array(X[-1]), u_c)
        assert np.array_equal(V_exp, V)
        assert np.array_equal(mean_exp, u_0)
        assert data.get("observations") == [1]
        assert nobs == 1
        assert np.allclose(np.array([1, 0, 0]), data.get("G"))
        assert [0.5] == data.get("d")
        assert np.array_equal(std_exp, std)
        assert np.array_equal(mean_exp, mean)


    def test_vda_setup_normalized(self, tmpdir):
        """End-to-end setup test"""
        X = np.zeros((3, 4))
        X[:,:2] = np.arange(6).reshape((3, 2))
        X[0, 3] = 1
        X = X.T

        INTERMEDIATE_FP = "inter"
        p = tmpdir.mkdir(INTERMEDIATE_FP).join("X_fp.npy")
        p.dump(X)
        p.allow_pickel = True

        settings = config.Config()
        settings.X_FP = str(p)
        settings.n = 3
        settings.FORCE_GEN_X = False
        settings.OBS_MODE = "single_max"
        settings.OBS_VARIANCE = 0.1
        settings.TDA_IDX_FROM_END = 0
        settings.HIST_FRAC = 0.5

        settings.NORMALIZE = True

        data,  std, mean = DA.vda_setup(settings)
        X_ret = data.get("X")
        X_train = data.get("train_X")
        V = data.get("V")
        n, M = X_ret.shape
        u_c = data.get("u_c")
        u_0 = data.get("u_0")
        nobs = len(data["observations"])

        mean_exp = np.array([0.5, 2.5, 4.5])
        std_exp = np.array([0.5, 0.5, 0.5])
        X_exp = (( X - mean_exp) * 2).T
        X_00_exp = -1

        assert X_00_exp == X_train[0, 0] and  X_00_exp == X_ret[0, 0]
        assert np.array_equal(X_exp, X_ret)
        assert (3, 4) == (n, M)
        assert np.array_equal(np.array(X_exp[:, :2]), data.get("train_X"))
        assert np.array_equal(np.array(X_exp[:,-1]), u_c)
        assert np.array_equal(X_exp[:, :2], V)
        assert np.allclose(np.zeros((3)), u_0)
        assert data.get("observations") == [1.]
        assert nobs == 1
        assert np.allclose(np.array([1, 0, 0]), data.get("G"))
        assert [1.] == data.get("d")
        assert np.array_equal(std_exp, std)
        assert np.array_equal(mean_exp, mean)

class TestTruncSVD():


    def test_trunc_SVD(self):
        "Test truncated SVD Rossella et al. mode"
        self.settings = config.Config()
        self.settings.COMPRESSION_METHOD = "SVD"
        DA = DAPipeline(self.settings)
        V = np.random.rand(10, 8)

        #There are tests embedded in DA.trunc_SVD:
        V_trunc, U_trunc, s_trunc, W_trunc = DA.trunc_SVD(V, trunc_idx = 5, test=True)
        V_trunc, U_trunc, s_trunc, W_trunc = DA.trunc_SVD(V, trunc_idx = None, test=True)


class TestMinimizeJ():
    """End-to-end tests"""
    def __settings(self, tmpdir, normalize, force_init=False):
        if hasattr(self, "settings") and not force_init:
            return self.settings
        else:
            X = np.zeros((3, 4))
            X[:,:2] = np.arange(6).reshape((3, 2))
            X[0, 3] = 1
            X = X.T

            INTERMEDIATE_FP = "inter"
            p = tmpdir.mkdir(INTERMEDIATE_FP).join("X_fp.npy")
            p.dump(X)
            p.allow_pickel = True

            settings = config.Config()
            settings.X_FP = str(p)
            settings.n = 3
            settings.FORCE_GEN_X = False
            settings.OBS_MODE = "single_max"
            settings.OBS_VARIANCE = 0.5
            settings.TDA_IDX_FROM_END = 0
            settings.HIST_FRAC = 0.5
            settings.ALPHA = 1.0
            settings.COMPRESSION_METHOD = "SVD" #we aren't actually using SVD truncated matrix here
            settings.SAVE = False
            settings.TOL = 1e-8
            settings.NORMALIZE = normalize

            data,  std, mean = DA.vda_setup(settings)


            self.u_0 = data.get("u_0")
            self.V = data.get("V")
            self.H_0 = data.get("G")
            self.d = data.get("d")
            nobs = len(data["observations"])
            self.R_inv = (1 / settings.OBS_VARIANCE) * np.eye(nobs)
            data["R_inv"] = self.R_inv

            self.nobs = nobs
            self.settings = settings
            self.data = data

            return settings

    def test_cost_fn_J(self, tmpdir):
        #Now check for normalized system
        normalize = False
        settings = self.__settings(tmpdir, normalize, force_init=True)

        w_1 = np.array([1, 2])
        data = self.data

        #init with sigma_2
        sigma_2 = settings.OBS_VARIANCE
        R_inv = data.get("R_inv")
        data["R_inv"] = None
        J_1_sigma = DA.cost_function_J(w_1, data, settings)
        data["R_inv"] = R_inv
        settings.OBS_VARIANCE = None
        J_1_R_inv = DA.cost_function_J(w_1, data, settings)

        assert np.isclose(J_1_sigma, J_1_sigma)
        assert np.isclose(J_1_sigma, 5/2)

        settings.OBS_VARIANCE = sigma_2

        w_2  = np.array([1, 0])
        J_2 = DA.cost_function_J(w_2, data, settings)
        assert np.isclose(J_2, 1.5)


    def test_minimize_J_normalized(self, tmpdir):
        """Check that system is finding correct answer found by
        rearranging gradient eqn"""
        normalize = True
        settings = self.__settings(tmpdir, normalize, force_init=True)

        alpha = settings.ALPHA

        w_opt_ret = DA.Var_DA_routine(settings)

        prefix = self.V.T @ self.H_0.T @ self.R_inv

        LHS = prefix @ self.d
        RHS_ = prefix @ self.H_0 @ self.V + alpha * np.eye(w_opt_ret.shape[0])
        RHS = RHS_ @ w_opt_ret

        assert np.allclose(LHS, RHS)


    def test_minimize_J_unnormalized(self, tmpdir):
        #Now check for normalized system
        normalize = False
        settings = self.__settings(tmpdir, normalize, force_init=True)

        alpha = settings.ALPHA
        w_opt_ret = DA.Var_DA_routine(settings)
        #w_opt_ret = 0.25 * np.array([-1, 1]) #This is value we want

        prefix = self.V.T @ self.H_0.T @ self.R_inv

        LHS = prefix @ self.d
        RHS_ = prefix @ self.H_0 @ self.V + alpha * np.eye(w_opt_ret.shape[0])
        RHS = RHS_ @ w_opt_ret

        # print("prefix @ self.H_0 @ self.V\n", prefix @ self.H_0 @ self.V)
        # print("RHS_", RHS_)
        # print("LHS", LHS)
        # print("----------")
        # print("V", self.V)
        # print("RHS_", RHS_)
        # print("w_opt", w_opt_ret)
        # print("RHS_ @ w_opt_ret", RHS_ @ w_opt_ret)
        # print("prefix", prefix)
        # print("d",  self.d)
        # print("LHS", LHS)
        # print("LHS.shape", LHS.shape)
        # print("RHS", RHS)
        # print("RHS.shape", RHS.shape

        assert np.allclose(LHS, RHS)



if __name__ == "__main__":
    pytest.main()
