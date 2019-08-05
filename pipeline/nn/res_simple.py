from torch import nn
import torch

class ResBlock(nn.Module):

    def __init__(self, activation_fn, channel):
        super(ResBlock, self).__init__()
        self.act_fn = activation_fn

        self.conv1 = nn.Conv3d(channel, channel, kernel_size=(3, 3, 3), stride=(1,1,1), padding=(1,1,1))
        self.conv2 = nn.Conv3d(channel, channel, kernel_size=(3, 3, 3), stride=(1,1,1), padding=(1,1,1))

    def forward(self, x):
        h = self.act_fn(self.conv1(x))
        h = self.conv2(h)
        return h + x

class ResBlockStack3(nn.Module):

    def __init__(self, activation_fn, channel):
        super(ResBlockStack3, self).__init__()
        self.act_fn = activation_fn

        self.res1 = ResBlock(activation_fn, channel)
        self.res2 = ResBlock(activation_fn, channel)
        self.res3 = ResBlock(activation_fn, channel)

    def forward(self, x):
        h = self.act_fn(self.res1(x))
        h = self.act_fn(self.res2(h))
        h = self.res3(h)
        return h + x


