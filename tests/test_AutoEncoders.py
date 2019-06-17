import torch
from pipeline.utils import ML_utils as ML
from pipeline import config
from pipeline.AutoEncoders import ToyAE, VanillaAE
import pytest

class TestAEInit():
    def test_ToyAE_init_base_config(self):
        settings = config.ToyAEConfig()
        try:
            model = settings.AE_MODEL_TYPE(**settings.get_kwargs())
        except:
            pytest.fail("Unable to init model")
    def test_VanillaAE_init_base_config(self):
        settings = config.ConfigAE()
        try:
            model = settings.AE_MODEL_TYPE(**settings.get_kwargs())
        except:
            pytest.fail("Unable to init model")

class TestAEForward():
    def test_ToyAE_forward_single_hid(self):
        settings = config.ToyAEConfig()
        settings.n = 128
        settings.HIDDEN = 64
        settings.NUMBER_MODES = 4
        Batch_sz = 16
        x = torch.rand((Batch_sz, settings.n), requires_grad=True)

        model = ToyAE(**settings.get_kwargs())
        try:
            y = model(x)
        except:
            pytest.fail("Unable to do forward pass")

    def test_ToyAE_forward_mult_hid(self):
        settings = config.ToyAEConfig()
        settings.n = 128
        settings.HIDDEN = [128, 128, 64]
        settings.NUMBER_MODES = 4
        Batch_sz = 16
        x = torch.rand((Batch_sz, settings.n), requires_grad=True)

        model = ToyAE(**settings.get_kwargs())
        try:
            y = model(x)
        except:
            pytest.fail("Unable to do forward pass")

    def test_VanillaAE_forward(self):
        settings = config.ConfigAE()
        settings.n = 128
        settings.HIDDEN = [128, 128, 64]
        settings.NUMBER_MODES = 4
        Batch_sz = 16
        x = torch.rand((Batch_sz, settings.n), requires_grad=True)

        model = VanillaAE(**settings.get_kwargs())
        try:
            y = model(x)
        except:
            pytest.fail("Unable to do forward pass")
