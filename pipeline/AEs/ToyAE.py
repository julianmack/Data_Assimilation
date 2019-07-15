
import torch.nn as nn
import torch
import torch.nn.functional as F
from collections import OrderedDict
from pipeline.AEs import VanillaAE


class ToyAE(VanillaAE):
    """Creates simple toy network with one fc hidden layer.
    I have worked out the explicit differential for this newtork.

    The .forward, .encode and .decode methods are inherited from VanillaAE
    """
    def __init__(self, input_size, latent_dim, activation = "relu", hidden = 128, batch_norm=False):

        if activation == "lrelu":
            raise NotImplementedError("Leaky ReLU not implemented for ToyAE")

        super(ToyAE, self).__init__(input_size, latent_dim, activation, hidden)

    def jac_explicit(self, x):
        """Generate explicit gradient for decoder
        (from hand calculated expression)"""

        self.eval() #Edit this if you add dropout

        z_i = x
        jac_running = None
        for idx, layer in enumerate(self.layers_decode[:-1]):
            jac_par, z_i = self.__jac_single_fc_layer(z_i, layer)
            if idx == 0:
                jac_running = jac_par
            else:
                if self.batch:
                    jac_par = jac_par.unsqueeze(1)
                    jac_running = jac_running.unsqueeze(2)
                    jac_running = (jac_running @ jac_par).squeeze(2)
                else:
                    jac_running = jac_par @ jac_running
        W_i = self.layers_decode[-1].weight

        if self.batch:
            W_i = W_i.t().expand((x.shape[0], -1, -1))

        if type(jac_running) != torch.Tensor:
            jac = W_i
        else:
            if self.batch:
                W_i = W_i.unsqueeze(1)
                jac_running = jac_running.unsqueeze(2)
                jac = (jac_running @ W_i).squeeze(2).transpose(2, 1)
            else:
                jac = W_i @ jac_running
        return jac

    def __jac_single_fc_layer(self, x, layer):
        if self.activation != "relu":
            raise NotImpelemtedError()

        W_i = layer.weight
        b_i = layer.bias
        a_i = (x @ W_i.t()) + b_i

        # In order to handle both batched and non-batched input:
        if len(a_i.shape) > len(b_i.shape): #batched
            A = (a_i > 0).unsqueeze(2).type(torch.FloatTensor)
            A = torch.transpose(A, 1, 2)
            B = W_i.t().expand((a_i.shape[0], -1, -1))
            self.batch = True
        else: #non-batched
            A = (a_i > 0).unsqueeze(1).type(torch.FloatTensor)
            B = W_i
            self.batch = False

        jac_partial = torch.mul(A, B)

        z_i = self.act_fn(a_i)
        return jac_partial, z_i