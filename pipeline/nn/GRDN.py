"""
Implementation of  GRDN from: https://arxiv.org/pdf/1905.11172.pdf
for 3D case.

Using implementation discussed in: http://openaccess.thecvf.com/content_CVPRW_2019/html/CLIC_2019/Cho_Low_Bit-rate_Image_Compression_based_on_Post-processing_with_Grouped_Residual_CVPRW_2019_paper.html
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.nn.res import ResNextBlock, ResVanilla
from pipeline.nn.explore.empty import Empty
from pipeline.nn.helpers import get_activation
from pipeline.nn.densenet import _DenseBlock
from pipeline.nn.helpers import get_activation
from pipeline.nn.CBAM import CBAM


class GRDB(nn.Module):
    def __init__(self, encode, activation_constructor, Cin,  Block, RDB_kwargs,
                    num_rdb=4):

        super(GRDB, self).__init__()

        blocks = []
        for i in range(num_rdb):
            rdb = _DenseBlock(**RDB_kwargs)
            blocks.append(rdb)
        self.rdbs = nn.ModuleList(blocks)
        self.conv1x1 = nn.Conv3d(num_rdb * Cin, Cin, kernel_size=(1, 1, 1), stride=(1,1,1))

    def forward(self, x):
        res = []
        h = x
        for rdb in self.rdbs:
            h = rdb(h)
            res.append(h)
        h = torch.cat(res, dim=1) #concat on channel
        h = self.conv1x1(h)
        return x + h

class GRDN(nn.Module):

    def __init__(self, encode, activation_constructor, Cstd, Block, RDB_kwargs, num_rdb):
        super(GRDN, self).__init__()
        Cin = Cstd #`Cin` is an alias for Cstd since the true Cin=1 for the GRDN
        activation = get_activation(activation_constructor)

        self.conv1  = nn.Conv3d(1, 2, kernel_size=3, stride=1, padding=1)
        self.downsample, self.upsample = self.get_updown(Cin, activation)
        # Note that downsample and upsample already contain the `conv`
        # shown in Figure 3 of: http://openaccess.thecvf.com/content_CVPRW_2019/html/CLIC_2019/Cho_Low_Bit-rate_Image_Compression_based_on_Post-processing_with_Grouped_Residual_CVPRW_2019_paper.html
        self.conv2 = nn.Conv3d(4, Cin, kernel_size=3, stride=1, padding=1)

        self.grdb1 = GRDB(encode, activation_constructor, Cin, Block, RDB_kwargs, num_rdb=4)
        self.grdb2 = GRDB(encode, activation_constructor, Cin, Block, RDB_kwargs, num_rdb=4)
        self.grdb3 = GRDB(encode, activation_constructor, Cin, Block, RDB_kwargs, num_rdb=4)
        self.grdb4 = GRDB(encode, activation_constructor, Cin, Block, RDB_kwargs, num_rdb=4)
        self.conv2a = nn.Conv3d(Cin, 4, kernel_size=3, stride=1, padding=1)

        self.conv3 = nn.Conv3d(2, Cin, kernel_size=3, stride=1, padding=1)
        self.cbam  = CBAM(encode, activation_constructor, Cin, reduction_ratio=8,
                        pool_types=['avg', 'max'], no_spatial=False)
        self.conv4  = nn.Conv3d(Cin, 2, kernel_size=3, stride=1, padding=1)
        self.conv5  = nn.Conv3d(2, 1, kernel_size=(1, 1, 1), stride=(1,1,1))


    def forward(self, x):

        h = self.conv1(x)
        h = self.downsample(h)
        h = self.conv2(h)
        h = self.grdb1(h)
        h = self.grdb2(h)
        h = self.grdb3(h)
        h = self.grdb4(h)
        h = self.conv2a(h)
        h = self.upsample(h)
        h = self.conv3(h)
        h = self.cbam(h)
        h = self.conv4(h)
        h = self.conv5(h)
        
        return h + x

    def get_updown(self, Cin, activation):
        """Helper function to get up down by using the settings BLOCK api.
        This is appropriate in this case as the system is still of
        original size"""
        #import here to avoid circular imports
        from pipeline.settings.block_base import Block
        from pipeline.AEs.AE_general import GenCAE
        from pipeline.AEs.AE_general import MODES as M

        class ConfigTemp(Block):
            def __init__(self):
                super(ConfigTemp, self).__init__()
                self.ACTIVATION = activation
                layers = 5
                self.BLOCKS = [M.S, (layers, "conv")]
                down = [0, 0, 1, 1, 1, ]
                self.DOWNSAMPLE = (down, down, down)
                channels = self.get_channels()
                self.CHANNELS = channels[:layers + 1]
                self.CHANNELS[-1] = 4
                self.CHANNELS[-2] = 8
                self.CHANNELS[0] =  2
                self.CHANNELS[1] =  4



        settings_tmp = ConfigTemp()
        tmpCAE = GenCAE(**settings_tmp.get_kwargs())

        downsample = tmpCAE.layers_encode
        upsample = tmpCAE.layers_decode
        return downsample, upsample