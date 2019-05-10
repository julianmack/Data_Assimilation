import torch.nn as nn
import torch
import numpy as np
from utils import ML_utils as ML
import utils

class ToyNet(nn.Module):
    """Creates simple toy network with one fc hidden layer"""
    def __init__(self, inn, hid, out):
        super(ToyNet, self).__init__()
        self.fc1 = nn.Linear(inn, hid, bias = True)
        self.fc2 = nn.Linear(hid, out, bias = True)
        self.sizes = [inn, hid, out]

    def forward(self, x):
        h = torch.F.ReLU(self.fc1(x))
        h = self.fc2(h)
        return h

    def gen_rand_weights(self):
        """Generates random weights for simple two layer fc network.
        """
        [inn, hid, out] = self.sizes
        #weights
        W_a = torch.rand((hid, inn), requires_grad=True)
        W_b = torch.rand((out, hid), requires_grad=True)

        #biases
        b_a = torch.rand((hid,), requires_grad=True)
        b_b = torch.rand((out,), requires_grad=True)

        #assign
        self.fc1.weight = nn.Parameter(W_a)
        self.fc2.weight = nn.Parameter(W_b)
        self.fc1.bias = nn.Parameter(b_a)
        self.fc2.bias = nn.Parameter(b_b)




if __name__ == "__main__":
    INPUT = 3
    OUT = 16
    HIDDEN = 8
    Batch_sz = 1
    utils.set_seeds()

    model = ToyNet(INPUT, HIDDEN, OUT)
    model.gen_rand_weights()

    input = torch.rand((Batch_sz, INPUT))
    output = model(input)

    jac_true = ML.jacobian_slow_torch(input, output)
    print(jac_true)
