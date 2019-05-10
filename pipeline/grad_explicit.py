import torch.nn as nn
import torch.nn.functional as F

import torch
import numpy as np
from utils import ML_utils as ML
import utils

import time
class ToyNet(nn.Module):
    """Creates simple toy network with one fc hidden layer"""
    def __init__(self, inn, hid, out):
        super(ToyNet, self).__init__()
        self.fc1 = nn.Linear(inn, hid, bias = True)
        self.fc2 = nn.Linear(hid, out, bias = True)
        self.sizes = [inn, hid, out]

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = self.fc2(h)
        return h

    def gen_rand_weights(self):
        """Generates random weights for simple two layer fc network.
        """
        [inn, hid, out] = self.sizes
        #weights
        W_a = torch.rand((hid, inn), requires_grad=True) - 0.5
        W_b = torch.rand((out, hid), requires_grad=True) - 0.5

        #biases
        b_a = torch.rand((hid,), requires_grad=True)
        b_b = torch.rand((out,), requires_grad=True)

        #assign
        self.fc1.weight = nn.Parameter(W_a)
        self.fc2.weight = nn.Parameter(W_b)
        self.fc1.bias = nn.Parameter(b_a)
        self.fc2.bias = nn.Parameter(b_b)

    def jac_explicit(self, x):
        """Generate explicit gradient
        (from hand calculated expression)"""

        W_a = self.fc1.weight
        W_b = self.fc2.weight
        b_a = self.fc1.bias
        b_b = self.fc2.bias

        z_1 = (x @ W_a.t()) + b_a
        #A = torch.sign(z_1).unsqueeze(2)
        A = (z_1 > 0).unsqueeze(2).type(torch.FloatTensor)

        B = W_b.t().expand((z_1.shape[0], -1, -1))


        first = torch.mul(A, B)
        first = torch.transpose(first, 1, 2)

        jac = first @ W_a

        jac = torch.transpose(jac, 1, 2)

        return jac
        # print(A)
        # print(B)
        # print("A.shape:", A.shape, )
        # print("B.shape:", B.shape, )
        # print(first.shape)
        # print(jac.shape)

if __name__ == "__main__":
    INPUT = 40
    HIDDEN = 50
    OUT = 60
    Batch_sz = 4
    utils.set_seeds()

    model = ToyNet(INPUT, HIDDEN, OUT)
    model.gen_rand_weights()

    input = torch.rand((Batch_sz, INPUT), requires_grad=True)
    output = model(input)

    jac_true = ML.jacobian_slow_torch(input, output)

    jac_expl = model.jac_explicit(input)

    # print("True jac shape", jac_true.shape)
    # print("jac explicit shape", jac_expl.shape)
    #
    # print(jac_true)
    # print(jac_expl)

    assert torch.allclose(jac_true, jac_expl), "Two jacobians are not equal"
    print("SUCCESS")
