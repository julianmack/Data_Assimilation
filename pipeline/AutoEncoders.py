"""Autoencoders and other ML methods.
import with:
    from ML_AEs import AEs"""

import torch.nn as nn
import numpy
import config

settings = config.Config



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
