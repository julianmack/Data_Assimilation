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
        except Exception as e:
            print(e)
            pytest.fail("Unable to init model")

    def test_VanillaAE_init_base_config(self):
        settings = config.ConfigAE()
        try:
            model = settings.AE_MODEL_TYPE(**settings.get_kwargs())
        except:
            pytest.fail("Unable to init model")

    def test_VanillaAE_init_hidden(self):
        settings = config.ConfigAE()
        settings.HIDDEN = 4
        try:
            model = settings.AE_MODEL_TYPE(**settings.get_kwargs())
        except:
            pytest.fail("Unable to init model")

        settings.HIDDEN = None
        try:
            model = settings.AE_MODEL_TYPE(**settings.get_kwargs())
        except:
            pytest.fail("Unable to init model")

class TestAEForward():
    def test_ToyAE_forward_nobatch(self):
        settings = config.ToyAEConfig()
        settings.n = 3
        settings.HIDDEN = 4
        settings.NUMBER_MODES = 2
        x = torch.rand((settings.n), requires_grad=True)

        model = ToyAE(**settings.get_kwargs())
        try:
            y = model(x)
        except:
            pytest.fail("Unable to do forward pass")

    def test_ToyAE_forward_single_hid(self):
        settings = config.ToyAEConfig()
        settings.n = 3
        settings.HIDDEN = 4
        settings.NUMBER_MODES = 2
        Batch_sz = 3
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

class TestJacExplicit():
    def test_jac_one_hid(self):
        input_size = 2
        hidden = 2
        latent_dim = 2
        Batch_sz = 3
        activation = "relu"

        decoder_input = torch.rand((Batch_sz, latent_dim), requires_grad=True)
        model = ToyAE(input_size, latent_dim, activation, hidden)
        decoder_output = model.decode(decoder_input)

        jac_true = ML.jacobian_slow_torch(decoder_input, decoder_output)
        jac_expl = model.jac_explicit(decoder_input)

        assert torch.allclose(jac_true, jac_expl, rtol=1e-02), "Two jacobians are not equal"

    def test_jac_mult_hid(self):
        input_size = 128
        hidden = [128, 128, 64]
        latent_dim = 4
        Batch_sz = 64
        activation = "relu"

        decoder_input = torch.rand((Batch_sz, latent_dim), requires_grad=True)

        model = ToyAE(input_size, latent_dim, activation, hidden)

        output = model.decode(decoder_input)

        jac_true = ML.jacobian_slow_torch(decoder_input, output)
        jac_expl = model.jac_explicit(decoder_input)

        assert torch.allclose(jac_true, jac_expl, rtol=1e-02), "Two jacobians are not equal"
