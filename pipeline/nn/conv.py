import torch
from collections import OrderedDict


class Conv(torch.nn):
    """Convolutional block that works in both encoder and decoder.
    Args:
        """

    def __init__(self, **kwargs):
        layer = OrderedDict()
        if dropout:
            layer.update({"00": nn.Dropout3d(0.33)})
        if batch_norm:
            layer.update({"0": nn.BatchNorm3d(Cin)})

        if encode:
            layer.update({"1": nn.Conv3d(Cin, Cout, **data)})
        else:
            layer.update({"1": nn.ConvTranspose3d(Cin, Cout, **data)})
        self.conv = nn.Sequential(layer)

    def forward(self, x):
        return self.conv(x)
