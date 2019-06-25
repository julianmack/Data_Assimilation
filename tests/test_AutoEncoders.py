import torch
from pipeline.utils import ML_utils as ML
from pipeline import config
from pipeline.AutoEncoders import ToyAE, VanillaAE, CAE_3D
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
        settings.set_n(3)
        settings.HIDDEN = 4
        settings.__NUMBER_MODES = 2
        x = torch.rand((settings.get_n()), requires_grad=True)

        model = ToyAE(**settings.get_kwargs())
        try:
            y = model(x)
        except:
            pytest.fail("Unable to do forward pass")

    def test_ToyAE_forward_single_hid(self):
        settings = config.ToyAEConfig()
        settings.set_n(3)
        settings.HIDDEN = 4
        settings.__NUMBER_MODES = 2
        Batch_sz = 3
        x = torch.rand((Batch_sz, settings.get_n()), requires_grad=True)

        model = ToyAE(**settings.get_kwargs())
        try:
            y = model(x)
        except:
            pytest.fail("Unable to do forward pass")

    def test_ToyAE_forward_mult_hid(self):
        settings = config.ToyAEConfig()
        settings.set_n(128)
        settings.HIDDEN = [128, 128, 64]
        settings.__NUMBER_MODES = 4
        Batch_sz = 16
        x = torch.rand((Batch_sz, settings.get_n()), requires_grad=True)

        model = ToyAE(**settings.get_kwargs())
        try:
            y = model(x)
        except:
            pytest.fail("Unable to do forward pass")

    def test_VanillaAE_forward(self):
        settings = config.ConfigAE()
        settings.set_n(3)
        settings.HIDDEN = [5, 6, 7]
        settings.__NUMBER_MODES = 2
        Batch_sz = 4
        x = torch.rand((Batch_sz, settings.get_n()), requires_grad=True)

        model = VanillaAE(**settings.get_kwargs())
        try:
            y = model(x)
        except:
            pytest.fail("Unable to do forward pass")

class TestJacExplicit():
    def test_jac_one_hid(self):
        input_size = 3
        hidden = 5
        latent_dim = 2
        Batch_sz = 4
        activation = "relu"

        decoder_input = torch.rand((Batch_sz, latent_dim), requires_grad=True)
        model = ToyAE(input_size, latent_dim, activation, hidden)
        decoder_output = model.decode(decoder_input)

        jac_true = ML.jacobian_slow_torch(decoder_input, decoder_output)
        jac_expl = model.jac_explicit(decoder_input)


        assert torch.allclose(jac_true, jac_expl, rtol=1e-02), "Two jacobians are not equal"

    def test_jac_mult_hid(self):
        input_size = 3
        hidden = [5, 6, 7]
        latent_dim = 2
        Batch_sz = 4
        activation = "relu"

        decoder_input = torch.rand((Batch_sz, latent_dim), requires_grad=True)

        model = ToyAE(input_size, latent_dim, activation, hidden)

        output = model.decode(decoder_input)

        jac_true = ML.jacobian_slow_torch(decoder_input, output)
        jac_expl = model.jac_explicit(decoder_input)

        assert torch.allclose(jac_true, jac_expl, rtol=1e-02), "Two jacobians are not equal"

    def test_jac_no_batch_one_hid(self):
        input_size = 3
        hidden = 5
        latent_dim = 2
        activation = "relu"

        decoder_input = torch.rand((latent_dim,), requires_grad=True)
        model = ToyAE(input_size, latent_dim, activation, hidden)
        decoder_output = model.decode(decoder_input)

        jac_expl = model.jac_explicit(decoder_input)

        jac_true = ML.jacobian_slow_torch(decoder_input, decoder_output)

        assert torch.allclose(jac_true, jac_expl, rtol=1e-02), "Two jacobians are not equal"

    def test_jac_no_batch_mult_hid(self):
        input_size = 3
        hidden = [5, 6, 7]
        latent_dim = 2
        activation = "relu"

        decoder_input = torch.rand((latent_dim,), requires_grad=True)
        model = ToyAE(input_size, latent_dim, activation, hidden)
        decoder_output = model.decode(decoder_input)

        jac_expl = model.jac_explicit(decoder_input)
        jac_true = ML.jacobian_slow_torch(decoder_input, decoder_output)


        assert torch.allclose(jac_true, jac_expl, rtol=1e-02), "Two jacobians are not equal"

class TestCAE_3D():
    """These tests are the ToyAE equivalents of the above but are all placed here
    (rather than in respective classes such as TestAEInit and TestAEForward
    so that they can be run w/o the standard AE tests)"""

    def test_CAE_init_base_config(self):
        settings = config.CAEConfig()
        try:
            model = settings.AE_MODEL_TYPE(**settings.get_kwargs())
        except Exception as e:
            print(e)
            pytest.fail("Unable to init model")

    def test_CAE_forward_batched(self):
        settings = config.CAEConfig()
        batch_sz = 2
        Cin = settings.get_channels()[0]
        size = (batch_sz, Cin) + settings.get_n()
        device = ML.get_device()
        x = torch.rand(size, requires_grad=True, device = device)

        model = CAE_3D(**settings.get_kwargs())

        model.to(device)
        try:

            y = model(x)
        except:
            pytest.fail("Unable to do forward pass")

    def test_CAE_forward_nobatch(self):
        settings = config.CAEConfig()
        Cin = settings.get_channels()[0]
        size = (Cin,) + settings.get_n()
        device = ML.get_device()
        x = torch.rand(size, requires_grad=True, device = device)

        model = CAE_3D(**settings.get_kwargs())

        model.to(device)
        try:

            y = model(x)
        except:
            pytest.fail("Unable to do forward pass")

    def test_CAE_linear_latent_batched(self):
        settings = config.CAEConfig()
        batch_sz = 2
        Cin = settings.get_channels()[0]
        size = (batch_sz, Cin) + settings.get_n()
        device = ML.get_device()
        x = torch.rand(size, requires_grad=True, device = device)

        model = CAE_3D(**settings.get_kwargs())



        model.to(device)
        encode = model.encode
        try:

            w = encode(x)
        except:
            pytest.fail("Unable to do forward pass")

        assert w.shape[0] == batch_sz
        assert len(w.shape) == 2, "There should only be one (non batched) dimension"
        assert w.shape[1] == settings.get_number_modes()


    def test_CAE_linear_latent_nonbatched(self):
        settings = config.CAEConfig()
        Cin = settings.get_channels()[0]
        size = (Cin, ) + settings.get_n()
        device = ML.get_device()
        x = torch.rand(size, requires_grad=True, device = device)

        model = CAE_3D(**settings.get_kwargs())



        model.to(device)
        encode = model.encode
        try:

            w = encode(x)
        except:
            pytest.fail("Unable to do forward pass")

        assert len(w.shape) == 1, "There should only be one dimension"
        assert w.shape[0] == settings.get_number_modes()