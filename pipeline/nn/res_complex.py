import torch
from torch import nn
from pipeline.nn.conv import FactorizedConv

class ResNextBlock(nn.Module):
    """Single res-block from arXiv:1611.05431v2

    It is really just a standard res_block with sqeezed 1x1 convs
    at input and output
    """
    def __init__(self, activation_fn, Cin, channel_small=None, down_sf=4):
        super(ResNextBlock, self).__init__()
        self.act_fn = activation_fn

        if not channel_small:
            #Minimum of 4 channels
            channel_small = (Cin // down_sf) if (Cin // down_sf > 0) else 4

        self.conv1x1_1 = nn.Conv3d(Cin, channel_small, kernel_size=(1, 1, 1), stride=(1,1,1))
        self.conv3x3 = nn.Conv3d(channel_small, channel_small, kernel_size=(3, 3, 3), stride=(1,1,1), padding=(1,1,1))
        self.conv1x1_2 = nn.Conv3d(channel_small, Cin, kernel_size=(1, 1, 1), stride=(1,1,1))

    def forward(self, x):
        h = self.act_fn(self.conv1x1_1(x))
        h = self.act_fn(self.conv3x3(h))
        h = self.conv1x1_2(h)
        return h + x

class ResNeXt(nn.Module):
    """Full ResNext module from : arXiv:1611.05431v2
    """

    def __init__(self, activation_fn, Cin, cardinality):
        super(ResNeXt, self).__init__()
        self.act_fn = activation_fn

        channel_small = 4 #fixed for ResNeXt system
        blocks = nn.ModuleList([])
        for i in range(cardinality):
            resblock = ResNextBlock(activation_fn, Cin, channel_small=4)
            blocks.append(resblock)
        self.resblocks = blocks


    def forward(self, x):
        for idx, block in enumerate(self.resblocks):
            if idx == 0:
                h = self.act_fn(block(x))
            else:
                h += self.act_fn(block(x))

        return h + x


class ResBlockSlim(nn.Module):

    def __init__(self, activation_fn, in_channels, out_channels):
        super(ResBlockSlim, self).__init__()
        self.act_fn = activation_fn

        self.conv1x1 = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1), stride=(1,1,1))
        conv_kwargs1 = {"in_channels": out_channels,
                        "out_channels": out_channels,
                        "kernel_size": (3, 3, 3),
                        "stride": (1,1,1),
                        "padding": (1,1,1)}
        conv_kwargs2 = conv_kwargs1.copy()
        conv_kwargs2["out_channels"] = in_channels
        self.conv1 = FactorizedConv(self.act_fn, **conv_kwargs1)
        self.conv2 = FactorizedConv(self.act_fn, **conv_kwargs2)


    def forward(self, x):
        h = self.act_fn(self.conv1x1(x))
        h = self.act_fn(self.conv1(h))
        h = self.conv2(h)
        return h + x

class DRU(nn.Module):

    """Dense Residual Unit as described in: CNN-Optimized Image
    Compression with Uncertainty based Resource Allocation.

    These blocks can be used as a method of upsampling
    - they double the input size (in all directions) by s.f. 2 """


    def __init__(self, activation_fn, channel, downsample_div=4):
        super(DRU, self).__init__()
        self.act_fn = activation_fn
        channel_down = (channel // downsample_div) if (channel // downsample_div > 0) else 1
        self.conv1x1 = nn.Conv3d(channel, channel_down, kernel_size=(1, 1, 1), stride=(1,1,1))
        self.conv2 = nn.Conv3d(channel_down, channel_down, kernel_size=(3, 3, 3), stride=(1,1,1), padding=(1,1,1))
        self.conv3 = nn.Conv3d(channel_down, channel, kernel_size=(3, 3, 3), stride=(1,1,1), padding=(1,1,1))

    def forward(self, x):
        h = self.act_fn(self.conv1x1(x))
        h = self.act_fn(self.conv2(h))
        h = self.conv3(h)
        h = h + x
        return torch.cat([h, x])