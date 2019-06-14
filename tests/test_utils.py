from pipeline import utils
from pipeline import config
import os
import pytest
import numpy as np

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
        if env["SEED"]:
            del env["SEED"]
        with pytest.raises(NameError):
            utils.set_seeds()