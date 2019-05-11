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
        :latent_size - int. size of latent representation
        :layers - int list. size of hidden layers"""

    def __init__(self, input_size, latent_size, hid_layers=None):
        super(VanillaAE, self).__init__()
        assert hid_layers == None or type(hid_layers) == list

        #create a list of all dimension sizes (including input/output)

        if not hid_layers:
            layers = [input_size, latent_size, input_size]
        else:
            layers = [input_size]
            #encoder:
            for size in hid_layers:
                layers.append(size)
            #latent representation:
            layers.append(latent_size)
            #decoder:
            for size in hid_layers[::-1]: #reversed list
                layers.append(size)
            layers.append(input_size)

        #now create the fc layers and store in nn.module list
        self.fclayers = nn.ModuleList([])
        for idx, n_hidden in enumerate(layers[:-1]):
            fc = nn.Linear(n_hidden, layers[idx + 1])
            nn.init.xavier_uniform_(fc.weight)
            self.fclayers.append(fc)

        self.lrelu = nn.LeakyReLU(negative_slope = 0.05, inplace=False)

        self.num_encode = len(hid_layers) + 1
        self.num_decode = self.num_encode

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def encode(self, x):
        encode_fc = self.fclayers[:self.num_encode]
        assert len(encode_fc) == self.num_encode
        for fc in encode_fc[:-1]:
            x = self.lrelu(fc(x))

        x = encode_fc[-1](x) #no activation function for latent space

        return x

    def decode(self, x):
        decode_fc = self.fclayers[self.num_decode:]
        assert len(decode_fc) == self.num_decode

        for fc in decode_fc[:-1]:
            x = self.lrelu(fc(x))

        x = decode_fc[-1](x) #no activation function for output
        return x

class ToyNet(nn.Module):
    """Creates simple toy network with one fc hidden layer.
    I have worked out the explicit differential for this newtork.
    """
    def __init__(self, inn, hid, out):
        super(ToyNet, self).__init__()
        #encoder
        self.fc00 = nn.Linear(out, hid, bias = True)
        self.fc01 = nn.Linear(hid, inn, bias = True)

        #decoder
        self.fc1 = nn.Linear(inn, hid, bias = True)
        self.fc2 = nn.Linear(hid, out, bias = True)
        self.sizes = [inn, hid, out]

    def decode(self, x):
        h = F.relu(self.fc1(x))
        h = self.fc2(h)
        return h
    def encode(self, x):
        h = F.relu(self.fc00(x))
        h = self.fc01(h)
        return h

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def gen_rand_weights(self):
        """Generates random weights for simple two layer fc decoder network.
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
        """Generate explicit gradient for decoder
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

import torch
