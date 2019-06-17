import torch
from pipeline.utils import ML_utils as ML
from pipeline import config
from pipeline.AutoEncoders import ToyAE

class TestToyAEJacExplicit():
    def test_jac_explicit_correct_single_hid_layer(self):
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

    def test_jac_explicit_correct_mult_hid_layer(self):
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
