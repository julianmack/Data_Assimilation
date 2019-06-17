import torch
from pipeline.utils import ML_utils as ML
from pipeline import config
from pipeline.AutoEncoders import ToyAE


def test_jac_explicit_correct_single_hid_layer():
    input_size = 128
    hidden = 8
    latent_dim = 4
    Batch_sz = 64

    decoder_input = torch.rand((Batch_sz, latent_dim), requires_grad=True)
    model = ToyAE(input_size, hidden, latent_dim)

    decoder_output = model.decode(decoder_input)

    jac_true = ML.jacobian_slow_torch(decoder_input, decoder_output)
    jac_expl = model.jac_explicit(decoder_input)

    assert torch.allclose(jac_true, jac_expl, rtol=1e-02), "Two jacobians are not equal"

def test_jac_explicit_correct_mult_hid_layer():
    input_size = 128
    hidden = [128, 128, 64]
    latent_dim = 4
    Batch_sz = 64
    decoder_input = torch.rand((Batch_sz, latent_dim), requires_grad=True)
    model = ToyAE(input_size, hidden, latent_dim)

    output = model.decode(decoder_input)

    jac_true = ML.jacobian_slow_torch(decoder_input, output)
    jac_expl = model.jac_explicit(decoder_input)

    assert torch.allclose(jac_true, jac_expl, rtol=1e-02), "Two jacobians are not equal"
