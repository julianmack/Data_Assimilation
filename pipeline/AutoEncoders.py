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
        self.layers_decode - an nn.ModuleList of all decoding layers in the network
        self.act_fn - the activation function to use in between layers"""
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

    def get_list_AE_layers(self, input_size, latent_dim, hidden):
        """Helper function to get a list of the number of fc nodes or conv
        channels in an autoencoder"""
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

        return layers


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

class CAE_3D(BaseAE):
    def __init__(self, layer_data, channels, activation = "relu"):
        super(CAE_3D, self).__init__()
        assert len(layer_data) + 1 == len(channels)

        self.layers = nn.ModuleList([])

        channels = self.get_list_AE_layers(channels[0], channels[-1], channels[1:-1])

        layer_data_list = layer_data + layer_data[::-1]
        num_encode = len(layer_data)

        assert len(channels) == len(layer_data_list) + 1

        for idx, data in enumerate(layer_data_list):
            data = layer_data_list[idx]

            if idx  < num_encode:
                conv = nn.Conv3d(channels[idx], channels[idx + 1], **data)
            else:
                conv = nn.ConvTranspose3d(channels[idx], channels[idx + 1], **data)
            self.layers.append(conv)

        self.layers_encode = self.layers[:num_encode]
        self.layers_decode = self.layers[num_encode:]

        if activation == "lrelu":
            self.act_fn = nn.LeakyReLU(negative_slope = 0.05, inplace=False)
        elif activation == "relu":
            self.act_fn = F.relu
        else:
            raise NotImplemtedError("Activation function must be in {'lrelu', 'relu'}")


class BaselineCAE(nn.Module):
    def __init__(self, channels):
        super(BaselineCAE, self).__init__()
        """
        TODO: Define here the layers (convolutions, relu etc.) that will be
        used in the encoder and decoder pipelines.
        """
        C_in = 1
        C1, C2, C3, C4 = channels

        self.C4 = C4 #save this for .decode()

        # encoder layers
        self.conv1 = nn.Conv3d(C_in, C1, 3, stride=2, padding=1)
        self.norm1 = nn.BatchNorm3d(C1)
        self.conv2 = nn.Conv3d(C1, C2, 3, stride=2, padding=1)
        self.norm2 = nn.BatchNorm3d(C2)
        self.conv3 = nn.Conv3d(C2, C3, 3, stride=2, padding=1)
        self.norm3 = nn.BatchNorm3d(C3)
        self.conv4 = nn.Conv3d(C3, C4, 3, stride=2, padding=1)
        self.norm4 = nn.BatchNorm3d(C4)
        self.fc = nn.Linear(C4 * 2 * 2, hidden_size)
        self.normfc = nn.BatchNorm1d(hidden_size)

        # decoder layers                                      #B, 8, 2, 2
        self.deconv1 = nn.ConvTranspose3d(C4, C3, 3, stride=2) #B, 16, 5
        self.normd1 = nn.BatchNorm3d(C3)
        self.deconv2 = nn.ConvTranspose3d(C3, C2, 3, stride=2, padding=1) #B, 32, 9
        self.normd2 = nn.BatchNorm3d(C2)
        self.deconv3 = nn.ConvTranspose3d(C2, C1, 3, stride=2, padding=1) #B, 16, 17
        self.normd3 =nn.BatchNorm3d(C1)
        self.deconv4 = nn.ConvTranspose3d(C1, C_in, 2, stride=2, padding=1) #B, 3, 32, 32
        self.normd4 = nn.BatchNorm3d(C_in)

        self.sigmoid = nn.Sigmoid()
        # relu
        self.relu = nn.ReLU(True)


    def encode(self, x):
        """
        TODO: Construct the encoder pipeline here. The encoder's
        output will be the laten space representation of x.
        """

        x = self.norm1(self.relu(self.conv1(x)))
        x = self.norm2(self.relu(self.conv2(x)))
        x = self.norm3(self.relu(self.conv3(x)))
        x = self.norm4(self.relu(self.conv4(x)))
        x = x.view(x.size(0), -1) # b, 8*2*2
        x = self.normfc(self.fc(x)) # b, h_dim
        return x

    def decode(self, z):
        """
        TODO: Construct the decoder pipeline here. The decoder should
        generate an output tensor with equal dimenssions to the
        encoder's input tensor.
        """
        z = z.view(z.size(0), self.C4, 2, 2)
        z = self.normd1(self.relu(self.deconv1(z)))
        z = self.normd2(self.relu(self.deconv2(z)) )
        z = self.normd3(self.relu(self.deconv3(z)) )
        z = self.normd4(self.sigmoid(self.deconv4(z)))

        return z

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon

class ToyCAE(BaseAE):
    """Creates a simple CAE for which
    I have worked out the explicit differential
    """
    def __init__(self, input_size, latent_dim, activation, hidden):
        if activation == "lrelu":
            raise NotImpelemtedError("Leaky ReLU not implemented for ToyAE")
        super(ToyCAE, self).__init__()


    def jac_explicit(self, x):
        """Generate explicit gradient for decoder
        (from hand calculated expression)"""

        # use ML_utils.jacobian_slow_torch() for now
        raise NotImpelemtedError()

