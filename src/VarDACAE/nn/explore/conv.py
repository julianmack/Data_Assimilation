import torch
from torch import nn

from collections import OrderedDict


class FactorizedConv(nn.Module):
    """Replaces a 3d convolution with its factorized `equivalent`.
    self.conv2 = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
    """
    def __init__(self, activation, in_channels, out_channels, kernel_size, stride, padding):
        super(FactorizedConv, self).__init__()
        layer = OrderedDict()

        if in_channels < out_channels:
            Cs = (in_channels, in_channels, in_channels, out_channels)
        else:
            Cs = (in_channels, out_channels, out_channels, out_channels)

        layer.update({"0": nn.Conv3d(Cs[0], Cs[1], kernel_size=(kernel_size[0], 1, 1),
                                    stride=1, padding=(padding[0], 0, 0))})
        layer.update({"0a": activation})
        layer.update({"1": nn.Conv3d(Cs[1], Cs[2], kernel_size=(1, kernel_size[1], 1),
                                    stride=1, padding=(0, padding[1], 0))})
        layer.update({"1a": activation})
        layer.update({"2": nn.Conv3d(Cs[2], Cs[3], kernel_size=(1, 1, kernel_size[2],),
                                    stride=1, padding=(0, 0, padding[2]))})
        self.conv = nn.Sequential(layer)

    def forward(self, x):
        return self.conv(x)