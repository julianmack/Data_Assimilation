import torch.nn as nn
from pipeline.AEs import BaseAE 
import torch.nn.functional as F


class VanillaAE(BaseAE):
    """Variable size AE - using only fully connected layers.
    Arguments (for initialization):
        :input_size - int. size of input (and output)
        :latent_dim - int. size of latent representation
        :hidden - int list. size of hidden layers"""

    def __init__(self, input_size, latent_dim, activation = "relu", hidden=None, batch_norm=False):
        super(VanillaAE, self).__init__()
        assert hidden == None or type(hidden) == list or type(hidden) == int, "hidden must be a list an int or None"
        assert activation in ["relu", "lrelu"]

        if batch_norm == True:
            raise NotImplementedError("Batch Norm not implemented for ToyAE")

        self.input_size = input_size
        self.hidden = hidden
        self.latent_dim = latent_dim
        self.latent_sz = (latent_dim, )
        self.activation = activation
        self.batch_norm = batch_norm
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


        layers = self.get_list_AE_layers(input_size, latent_dim, hidden)

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
