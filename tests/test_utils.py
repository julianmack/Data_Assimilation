from pipeline import utils
from pipeline import config
import os
import pytest
import numpy as np
import torch

class TestSeed():
    def test_set_seeds_normal(self):
        seed = 42
        utils.set_seeds(seed)
        a = np.random.randn(45)
        utils.set_seeds(seed)
        b = np.random.randn(45)
        c = np.random.randn(45)
        assert np.allclose(a, b)
        assert not np.allclose(b, c)

    def test_set_seeds_raiseNameError(self):
        env = os.environ
        if env.get("SEED"):
            del env["SEED"]
        with pytest.raises(NameError):
            utils.set_seeds()

class TestJacSlow():
    def test_jac_slow_no_batch(self):
        input_size = 3
        hidden = 5
        latent_dim = 2

        x = torch.rand((latent_dim,), requires_grad=True)
        W = torch.rand((hidden, latent_dim))
        b = torch.rand((hidden, ))
        y = W @ x + b

        grad = utils.ML_utils.jacobian_slow_torch(x, y)

        assert np.allclose(W, grad)

    def test_jac_slow_batch(self):
        input_size = 3
        hidden = 5
        latent_dim = 2
        batch_sz = 4

        X = torch.rand((batch_sz, latent_dim,), requires_grad=True)
        W = torch.rand((hidden, latent_dim))
        b = torch.rand((hidden, ))
        y = X @ W.t() + b
        W_stacked = W.expand((batch_sz, -1, -1))
        grad = utils.ML_utils.jacobian_slow_torch(X, y)

        assert np.allclose(W_stacked, grad)