import torch
from pipeline.utils import ML_utils as ML
from pipeline.AutoEncoders import ToyAE
from pipeline import utils

import time
import matplotlib.pyplot as plt

def plot_time_w_output(outputs, inn, hidden, batch_sz, loop=True, no_batch=False):

    T_2s = []
    T_1s = []
    factors = []

    utils.set_seeds()
    input = torch.rand((Batch_sz, inn), requires_grad=True)
    if no_batch:
        input = input[0]

    for out_sz in outputs:
        model = ToyAE(inn, hidden, out_sz)
        model.gen_rand_weights()
        output = model.decode(input)
        t0 = time.time()
        if loop:
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
        if loop:
            print("out = {}. Explicit x{:.1f} faster than loop method".format(out_sz, factor))
    if loop:
        plt.plot(outputs, T_1s)
        plt.show()
        plt.plot(outputs, factors)
        plt.show()
    plt.plot(outputs, T_2s)
    plt.show()


if __name__ == "__main__":
    INPUT = 32
    HIDDEN = 128
    Batch_sz = 64
    outputs = [2**x for x in range(8)]
    plot_time_w_output(outputs, INPUT, HIDDEN, Batch_sz, loop=True, no_batch=False)

    exit()
    utils.set_seeds()

    model = ToyAE(INPUT, HIDDEN, OUT)
    model.gen_rand_weights()

    input = torch.rand((Batch_sz, INPUT), requires_grad=True)
    output = model.decode(input)

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
