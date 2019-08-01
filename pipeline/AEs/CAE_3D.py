import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from pipeline.AEs import BaseAE


class CAE_3D(BaseAE):
    def __init__(self, layer_data, channels, activation = "relu", latent_sz=None,
                jac_explicit=None, batch_norm=False):
        super(CAE_3D, self).__init__()
        assert len(layer_data) + 1 == len(channels)
        self.batch_norm = batch_norm

        layers = nn.ModuleList([])

        channels = self.get_list_AE_layers(channels[0], channels[-1], channels[1:-1])

        layer_data_list = layer_data + layer_data[::-1]
        num_encode = len(layer_data)

        assert len(channels) == len(layer_data_list) + 1

        for idx, data in enumerate(layer_data_list):
            data = layer_data_list[idx]

            if idx  == 0: #no batch_norm on input
                self.batch_norm = False
                conv = self.__conv_maybe_batch_norm(channels[idx], channels[idx + 1], data, False)
                self.batch_norm = batch_norm
            elif idx  < num_encode:
                conv = self.__conv_maybe_batch_norm(channels[idx], channels[idx + 1], data, False)
            else:
                conv = self.__conv_maybe_batch_norm(channels[idx], channels[idx + 1], data, True)

            layers.append(conv)

        #init instance variables
        self.latent_sz = latent_sz
        self.layers_encode = layers[:num_encode]
        self.layers_decode = layers[num_encode:]


        if activation == "lrelu":
            self.act_fn = nn.LeakyReLU(negative_slope = 0.05, inplace=False)
        elif activation == "relu":
            self.act_fn = F.relu
        else:
            raise NotImplemtedError("Activation function must be in {'lrelu', 'relu'}")

    def __conv_maybe_batch_norm(self, Cin, Cout, data, transpose):
        layer = OrderedDict()

        if self.batch_norm:
            layer.update({"0": nn.BatchNorm3d(Cin)})
        if not transpose:
            layer.update({"1": nn.Conv3d(Cin, Cout, **data)})
        else:
            layer.update({"1": nn.ConvTranspose3d(Cin, Cout, **data)})
        conv = nn.Sequential(layer)
        return conv
