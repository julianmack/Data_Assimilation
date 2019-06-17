import torch
from pipeline.utils import ML_utils as ML

from pipeline.AutoEncoders import ToyAE

def test_jac_explicit_correct_single_hid_layer():
    INPUT = 32
    OUT = 4
    HIDDEN = 128
    Batch_sz = 64
    input = torch.rand((Batch_sz, INPUT), requires_grad=True)
    model = ToyAE(INPUT, HIDDEN, OUT)

    output = model.decode(input)

    jac_true = ML.jacobian_slow_torch(input, output)
    jac_expl = model.jac_explicit(input)

    assert torch.allclose(jac_true, jac_expl, rtol=1e-02), "Two jacobians are not equal"

def test_jac_explicit_correct_mult_hid_layer():
    INPUT = 32
    OUT = 4
    HIDDEN = [128, 128, 64]
    Batch_sz = 64
    input = torch.rand((Batch_sz, INPUT), requires_grad=True)
    model = ToyAE(INPUT, HIDDEN, OUT)

    output = model.decode(input)

    jac_true = ML.jacobian_slow_torch(input, output)
    jac_expl = model.jac_explicit(input)

    assert torch.allclose(jac_true, jac_expl, rtol=1e-02), "Two jacobians are not equal"
