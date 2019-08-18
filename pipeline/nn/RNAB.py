
"""
Implementation of  RESIDUAL NON-LOCAL ATTENTION NETWORKS FOR
IMAGE RESTORATION for 3D case.

Influneced by authors implementation at: https://github.com/yulunzhang/RNAN
But more closely based on design in:
http://openaccess.thecvf.com/content_CVPRW_2019/papers/CLIC%202019/Zhou_End-to-end_Optimized_Image_Compression_with_Attention_Mechanism_CVPRW_2019_paper.pdf

It was not possible to use the original
implementation directly because:
    1) It is for 2d input rather than 3D
"""

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
from pipeline.nn.res import ResNextBlock, ResVanilla
from pipeline.nn.explore.empty import Empty
from pipeline.nn.helpers import get_activation

class RNAB(nn.Module):

    def __init__(self, encode, activation_constructor, Cin, Block = ResVanilla, channel_small=None,
                    down_sf=4, residual=True, downsample=None, upsample=None):
        super(RNAB, self).__init__()
        if downsample is not None:
            assert upsample is not None
        self.residual = residual
        if get_activation(activation_constructor) == "GDN":
            raise NotImplementedError("Must deal with GDN w. RNAB crossover")

        #init trunk: 3 res blocks
        self.trunk = nn.Sequential()
        for i in range(3):
            res = Block(encode, activation_constructor, Cin, channel_small)
            self.trunk.add_module('res%d' % (i), res)

        #init mask
        self.mask = nn.Sequential()
        for i in range(2):
            res = Block(encode, activation_constructor, Cin, channel_small)
            self.mask.add_module('res%d' % (i), res)
        if not downsample:
            downsample = self.__build_downsample(encode, activation_constructor, Cin, channel_small)
        self.mask.add_module('downsample', downsample)
        for i in range(2, 4):
            res = Block(encode, activation_constructor, Cin, channel_small)
            self.mask.add_module('res%d' % (i), res)
        if not upsample:
            upsample = self.__build_upsample(encode, activation_constructor, Cin, channel_small)
        self.mask.add_module('upsample', upsample)

        for i in range(4, 6):
            res = Block(encode, activation_constructor, Cin, channel_small)
            self.mask.add_module('res%d' % (i), res)

        self.mask.add_module('conv1x1', nn.Conv3d(Cin, Cin, kernel_size=(1, 1, 1)))

    def __build_downsample(self, encode, activation_constructor, Cin, channel_small):
        """This downsample is specific to out input size in this case of
        C, x, y, z = 32, 11, 11, 3"""
        conv1 = nn.Conv3d(Cin, Cin, kernel_size=(3, 3, 2), stride=(2,2,1))
        conv2 = nn.Conv3d(Cin, Cin, kernel_size=(3, 3, 2), stride=(2,2,1), padding=(1, 1, 0))
        conv3 = nn.Conv3d(Cin, Cin, kernel_size=(3, 3, 1), stride=(1,1,1),)
        return nn.Sequential(conv1, activation_constructor(Cin, False),
                            conv2, activation_constructor(Cin, False),
                            conv3, )

    def __build_upsample(self, encode, activation_constructor, Cin, channel_small):
        """This downsample is specific to out input size in this case of
        C, x, y, z = 32, 11, 11, 3"""
        conv1 = nn.ConvTranspose3d(Cin, Cin, kernel_size=(3, 3, 1), stride=(1,1,1),)
        conv2 = nn.ConvTranspose3d(Cin, Cin, kernel_size=(3, 3, 2), stride=(2,2,1), padding=(1, 1, 0))
        conv3 = nn.ConvTranspose3d(Cin, Cin, kernel_size=(3, 3, 2), stride=(2,2,1))

        return nn.Sequential(conv1, activation_constructor(Cin, True),
                            conv2, activation_constructor(Cin, True),
                            conv3)

    def forward(self, x):
        mask = self.mask(x)
        trunk = self.trunk(x)
        
        h = trunk * mask

        if self.residual:
            res = h + x
        return res
