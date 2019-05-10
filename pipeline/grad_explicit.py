import torch.nn as nn
import torch.nn.functional as F

import torch
import numpy as np
from utils import ML_utils as ML
import utils

import time
import matplotlib.pyplot as plt

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
def plot_time_w_output(outputs, inn, hidden, batch_sz):
    T_2s = []
    T_1s = []
    factors = []
    for out_sz in outputs:
        utils.set_seeds()
        model = ToyNet(inn, hidden, out_sz)
        model.gen_rand_weights()

        input = torch.rand((Batch_sz, INPUT), requires_grad=True)
        output = model(input)

        t0 = time.time()
        jac_true = ML.jacobian_slow_torch(input, output)
        t1 = time.time()
        jac_expl = model.jac_explicit(input)
        t2 = time.time()
        T_1 = t1-t0
        T_2 = t2-t1

        try:
            factor = T_1 / T_2
        except ZeroDivisionError:
            factor = 0

        T_1s.append(T_1)
        T_2s.append(T_2)
        factors.append(factor)

        print("out = {}. Explicit x{:.1f} faster than loop method".format(out_sz, factor))


    plt.plot(outputs, T_1s)
    plt.show()
    plt.plot(outputs, T_2s)
    plt.show()
    plt.plot(outputs, factors)
    plt.show()

if __name__ == "__main__":
    INPUT = 32
    HIDDEN = 128
    Batch_sz = 64
    outputs = [2**x for x in range(17)]
    plot_time_w_output(outputs, INPUT, HIDDEN, Batch_sz)
    exit()
    utils.set_seeds()

    model = ToyNet(INPUT, HIDDEN, OUT)
    model.gen_rand_weights()

    input = torch.rand((Batch_sz, INPUT), requires_grad=True)
    output = model(input)

    t0 = time.time()
    jac_true = ML.jacobian_slow_torch(input, output)
    t1 = time.time()
    jac_expl = model.jac_explicit(input)
    t2 = time.time()

    # print("True jac shape", jac_true.shape)
    # print("jac explicit shape", jac_expl.shape)
    #
    # print(jac_true)
    # print(jac_expl)


    T_1 = t1-t0
    T_2 = t2-t1
    print("Time for loop method: {:.4f}s".format(T_1))
    print("Time for explicit method: {:.4f}s".format(T_2))

    print("Explicit x{:.1f} faster than loop method".format(T_1 / T_2))
    assert torch.allclose(jac_true, jac_expl, rtol=1e-02), "Two jacobians are not equal"
    print("SUCCESS")
