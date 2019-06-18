"""Autoencoders and other ML methods.
import with:
    from ML_AEs import AEs"""

import torch.nn as nn
import torch
import torch.nn.functional as F

class VanillaAE(nn.Module):
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

        decode_fc = self.fclayers[self.num_decode:]


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
        self.fclayers = nn.ModuleList([])
        for idx, n_hidden in enumerate(layers[:-1]):
            fc = nn.Linear(n_hidden, layers[idx + 1])
            nn.init.xavier_uniform_(fc.weight)
            self.fclayers.append(fc)

        if activation == "lrelu":
            self.act_fn = nn.LeakyReLU(negative_slope = 0.05, inplace=False)
        elif activation == "relu":
            self.act_fn = F.relu

        self.num_encode = len(hidden) + 1
        self.num_decode = self.num_encode


    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def encode(self, x):
        encode_fc = self.fclayers[:self.num_encode]
        assert len(encode_fc) == self.num_encode
        for fc in encode_fc[:-1]:
            x = self.act_fn(fc(x))

        x = encode_fc[-1](x) #no activation function for latent space

        return x

    def decode(self, x):
        decode_fc = self.fclayers[self.num_decode:]

        assert len(decode_fc) == self.num_decode

        for fc in decode_fc[:-1]:
            x = self.act_fn(fc(x))

        x = decode_fc[-1](x) #no activation function for output
        return x

class ToyAE(VanillaAE):
    """Creates simple toy network with one fc hidden layer.
    I have worked out the explicit differential for this newtork.

    The .forward, .encode and .decode methods are inherited from VanillaAE
    """
    def __init__(self, input_size, latent_dim, activation = "relu", hidden = 128):

        if activation == "lrelu":
            raise NotImpelemtedError("Leaky ReLU not implemented for ToyAE")

        super(ToyAE, self).__init__(input_size, latent_dim, activation, hidden)

    #     # #encoder
    #     self.fc00 = nn.Linear(input_size, hidden, bias = True)
    #     self.fc01 = nn.Linear(hidden, latent_dim, bias = True)
    #
    #     #decoder
    #     self.fc1 = nn.Linear(latent_dim, hidden, bias = True)
    #     self.fc2 = nn.Linear(hidden, input_size, bias = True)
    #     self.sizes = [latent_dim, hidden, input_size]
    #
    # def decode(self, x):
    #     h = F.relu(self.fc1(x))
    #     h = self.fc2(h)
    #     return h
    #
    # def encode(self, x):
    #     h = F.relu(self.fc00(x))
    #     h = self.fc01(h)
    #     return h
    #
    # def forward(self, x):
    #     x = self.encode(x)
    #     x = self.decode(x)
    #     return x

    def gen_rand_weights(self):
        """Generates random weights for simple two layer fc decoder network.
        """
        [latent_dim, hidden, input_size] = self.sizes
        #weights
        W_a = torch.rand((hidden, latent_dim), requires_grad=True) - 0.5
        W_b = torch.rand((input_size, hidden), requires_grad=True) - 0.5

        #biases
        b_a = torch.rand((hidden,), requires_grad=True)
        b_b = torch.rand((input_size,), requires_grad=True)

        #assign
        self.fc1.weight = nn.Parameter(W_a)
        self.fc2.weight = nn.Parameter(W_b)
        self.fc1.bias = nn.Parameter(b_a)
        self.fc2.bias = nn.Parameter(b_b)

    def jac_explicit(self, x):
        """Generate explicit gradient for decoder
        (from hand calculated expression)"""

        self.eval() #Edit this if you add dropout

        decode_layers = self.fclayers[self.num_decode:]

        z_i = x
        jac_running = None
        for idx, layer in enumerate(decode_layers[:-1]):
            jac_par, z_i = self.__jac_single_fc_layer(z_i, layer)
            if idx == 0:
                jac_running = jac_par
            else:
                if self.batch:
                    jac_par = jac_par.unsqueeze(1)
                    jac_running = jac_running.unsqueeze(2)
                    jac_running = (jac_running @ jac_par).squeeze(2)
                else:
                    raise NotImpelemtedError("Non-batch not implemented")
        W_i = decode_layers[-1].weight
        
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

    def jac_explicit_old(self, x):
        """Generate explicit gradient for decoder
        (from hand calculated expression)"""

        W_a = self.fc1.weight
        W_b = self.fc2.weight
        b_a = self.fc1.bias
        b_b = self.fc2.bias

        z_1 = (x @ W_a.t()) + b_a

        #A = torch.sign(z_1).unsqueeze(2)
        # In order to handle both batched and non-batched input:
        batch = False
        if len(z_1.shape) > len(b_a.shape):
            batch = True
            A = (z_1 > 0).unsqueeze(2).type(torch.FloatTensor)
            B = W_b.t().expand((z_1.shape[0], -1, -1))
        else:
            A = (z_1 > 0).unsqueeze(1).type(torch.FloatTensor)
            B = W_b.t().expand((z_1.shape[0], -1))


        first = torch.mul(A, B)
        if batch:
            first = torch.transpose(first, 1, 2)
        else:
            first = torch.transpose(first, 0, 1)
        jac = first @ W_a


        return jac


class ToyCAE(nn.Module):
    """Creates a simple CAE for which
    I have worked out the explicit differential
    """
    def __init__(self, latent_dim, hidden, input_size):
        super(ToyCAE, self).__init__()


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

