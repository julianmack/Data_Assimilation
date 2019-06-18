"""Autoencoders and other ML methods.
import with:
    from ML_AEs import AEs"""

import torch.nn as nn
import torch
import torch.nn.functional as F


class BaseAE(nn.Module):
    """Base AE class which all should inherit from
    The following instance variables must be initialized:
        self.layers_encode - an nn.ModuleList of all encoding layers in the network
        self.layers_decode - an nn.ModuleList of all decoding layers in the network"""
    def forward(self, x):
        self.__check_instance_vars()
        x = self.encode(x)
        x = self.decode(x)
        return x

    def encode(self, x):
        layers = self.layers_encode
        for layer in layers[:-1]:
            x = self.act_fn(layer(x))

        x = layers[-1](x) #no activation function for latent space

        return x

    def decode(self, x):
        layers = self.layers_decode
        for layer in layers[:-1]:
            x = self.act_fn(layer(x))

        x = layers[-1](x) #no activation function for output
        return x
    def __check_instance_vars(self):
        try:
            x = self.layers_decode
            y = self.layers_encode
        except:
            raise ValueError("Must init model with instance variables layers_decode and layers_encode")

        assert isinstance(x, nn.ModuleList), "model.layers_decode must be of type nn.ModuleList"
        assert isinstance(y, nn.ModuleList), "model.layers_encode must be of type nn.ModuleList"

class VanillaAE(BaseAE):
    """Variable size AE - using only fully connected layers.
    Arguments (for initialization):
        :input_size - int. size of input (and output)
        :latent_dim - int. size of latent representation
        :hidden - int list. size of hidden layers"""

    def __init__(self, input_size, latent_dim, activation = "relu", hidden=None):
        super(VanillaAE, self).__init__()
        assert hidden == None or type(hidden) == list or type(hidden) == int, "hidden must be a list an int or None"
        assert activation in ["relu", "lrelu"]
        self.input_size = input_size
        self.hidden = hidden
        self.latent_dim = latent_dim
        self.activation = activation
        self.__init_multilayer_AE()


    def __init_multilayer_AE(self):
        input_size = self.input_size
        hidden = self.hidden
        latent_dim = self.latent_dim
        activation = self.activation

        if type(hidden) == int:
            hidden = [hidden]
        elif not hidden:
            hidden = []

        #create a list of all dimension sizes (including input/output)
        layers = [input_size]
        #encoder:
        for size in hidden:
            layers.append(size)
        #latent representation:
        layers.append(latent_dim)
        #decoder:
        for size in hidden[::-1]: #reversed list
            layers.append(size)
        layers.append(input_size)

        #now create the fc layers and store in nn.module list
        self.layers = nn.ModuleList([])
        for idx, n_hidden in enumerate(layers[:-1]):
            fc = nn.Linear(n_hidden, layers[idx + 1])
            nn.init.xavier_uniform_(fc.weight)
            self.layers.append(fc)

        if activation == "lrelu":
            self.act_fn = nn.LeakyReLU(negative_slope = 0.05, inplace=False)
        elif activation == "relu":
            self.act_fn = F.relu
        num_encode = len(hidden) + 1 #=num_decode

        self.layers_encode = self.layers[:num_encode]
        self.layers_decode = self.layers[num_encode:]


class ToyAE(VanillaAE):
    """Creates simple toy network with one fc hidden layer.
    I have worked out the explicit differential for this newtork.

    The .forward, .encode and .decode methods are inherited from VanillaAE
    """
    def __init__(self, input_size, latent_dim, activation = "relu", hidden = 128):

        if activation == "lrelu":
            raise NotImpelemtedError("Leaky ReLU not implemented for ToyAE")

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


class ToyCAE(BaseAE):
    """Creates a simple CAE for which
    I have worked out the explicit differential
    """
    def __init__(self, input_size, latent_dim, activation, hidden):
        if activation == "lrelu":
            raise NotImpelemtedError("Leaky ReLU not implemented for ToyAE")
        super(ToyCAE, self).__init__()

        #Now initialize


    def decode(self, x):

        raise NotImpelemtedError()
    def encode(self, x):

        raise NotImpelemtedError()

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x


    def jac_explicit(self, x):
        """Generate explicit gradient for decoder
        (from hand calculated expression)"""


        raise NotImpelemtedError()

