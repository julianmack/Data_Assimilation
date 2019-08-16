
"""
Implementation of CBAM: Convolutional Block Attention Module
(https://arxiv.org/pdf/1807.06521.pdf) for 3D case.

This is based on the following opensource implementation:
https://github.com/Jongchan/attention-module/blob/master/MODELS/cbam.py
"""

import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class BasicConv(nn.Module):
    """This is edited so that it works for 3D case"""

    def __init__(self, activation_constructor, in_planes, out_planes, kernel_size,
                stride=1, padding=0, dilation=1, groups=1, activation=True,
                bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size,
                            stride=stride, padding=padding, dilation=dilation,
                            groups=groups, bias=bias)
        self.bn = nn.BatchNorm3d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.act = activation_constructor(out_planes) if activation else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, activation_constructor, gate_channels,
                reduction_ratio=8, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            activation_constructor(gate_channels // reduction_ratio),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None

        k_size = (x.size(2), x.size(3), x.size(4))
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool3d( x, k_size, stride=k_size)
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool3d( x, k_size, stride=k_size)
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool3d( x, 2, k_size, stride=k_size)
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw


        scale = torch.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(x)

        return x * scale

def logsumexp_2d(tensor):

    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self, activation_constructor):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(activation_constructor, 2, 1, kernel_size,
                                    stride=1, padding=(kernel_size-1) // 2,
                                    activation=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, activation_constructor, gate_channels, reduction_ratio=8,
                    pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.channelgate = ChannelGate(activation_constructor, gate_channels,
                            reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.spatialgate = SpatialGate(activation_constructor)
    def forward(self, x):
        x_out = self.channelgate(x)
        if not self.no_spatial:
            x_out = self.spatialgate(x_out)
        return x_out




if __name__ == "__main__":
    model = CBAM(activation_constructor, 32, reduction_ratio=16, pool_types=['avg', 'max'])
    print(model)
    num_params = sum(p.numel() for p in model.parameters())
    print("number params:", num_params)