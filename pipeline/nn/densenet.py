
"""
Implementation of  <https://arxiv.org/pdf/1608.06993.pdf> for 3D case.

The following code is heavily based on the DenseNet implementation
in torchvision.models.DenseNet - it was not possible to use the original
implementation directly because:
    1) It is for 2d input rather than 3D
    2) The order of convolution, BN, Activation of (BN -> ReLU -> conv -> BN -> ...) is not
    typically used anymore (Now BN -> conv -> ReLU -> BN) is more common

"""

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
from pipeline.nn.res_complex import ResNextBlock


class _DenseBlock(nn.Module):
    def __init__(self, activation_constructor, Cin, growth_rate, Csmall,
                    dense_layers, Block=ResNextBlock):
        super(_DenseBlock, self).__init__()
        for i in range(dense_layers):

            layer = Block(activation_constructor,
                Cin = Cin + i * growth_rate,
                channel_small = Csmall,
                Cout = growth_rate,
                residual=False,
            )

            self.add_module('denselayer%d' % (i + 1), layer)

        squeeze =  nn.Conv3d(Cin + (i + 1) * growth_rate, Cin, kernel_size=(1, 1, 1), stride=(1,1,1))

        self.add_module('denseSqueeze', squeeze)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            if name == "denseSqueeze":
                continue
            new_features = layer(torch.cat(features, 1))
            features.append(new_features)
        h = torch.cat(features, 1)
        return self.denseSqueeze(h)


