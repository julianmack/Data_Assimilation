import torch
from torch import nn
from pipeline.nn import init
from torch.nn.parameter import Parameter
from pipeline.nn.CBAM import CBAM


class ResVanilla(nn.Module):
    """Standard residual block (slightly adapted to our use case)
    """
    def __init__(self, encode, activation_constructor, Cin, channel_small=None,
                    down_sf=4, Cout=None, residual=True):
        super(ResVanilla, self).__init__()
        self.residual = residual
        if Cout is None:
            Cout = Cin
        if channel_small is None:
            channel_small = Cin
        conv1 = nn.Conv3d(Cin, channel_small, kernel_size=(3, 3, 3), stride=(1,1,1), padding=(1,1,1))
        conv2 = nn.Conv3d(channel_small, Cout, kernel_size=(3, 3, 3), stride=(1,1,1), padding=(1,1,1))


        #Initializations
        init.conv(conv1.weight, activation_constructor)
        init.conv(conv2.weight, activation_constructor)

        #ADD batch norms automatically
        self.ResLayers = nn.Sequential(conv1,
            activation_constructor(channel_small, not encode), nn.BatchNorm3d(channel_small), conv2,
            activation_constructor(Cout, not encode))

    def forward(self, x):
        h = self.ResLayers(x)
        if self.residual:
            h = h + x
        return h

class ResNextBlock(nn.Module):
    """Single res-block from arXiv:1611.05431v2

    It is really just a standard res_block with sqeezed 1x1 convs
    at input and output
    """
    def __init__(self, encode, activation_constructor, Cin, channel_small=None,
                    down_sf=4, Cout=None, residual=True):
        super(ResNextBlock, self).__init__()
        self.residual = residual
        if Cout is None:
            Cout = Cin

        if not channel_small:
            #Minimum of 4 channels
            channel_small = (Cin // down_sf) if (Cin // down_sf > 0) else 4


        conv1x1_1 = nn.Conv3d(Cin, channel_small, kernel_size=(1, 1, 1), stride=(1,1,1))
        conv3x3 = nn.Conv3d(channel_small, channel_small, kernel_size=(3, 3, 3), stride=(1,1,1), padding=(1,1,1))
        conv1x1_2 = nn.Conv3d(channel_small, Cout, kernel_size=(1, 1, 1), stride=(1,1,1))


        #Initializations
        init.conv(conv1x1_1.weight, activation_constructor)
        init.conv(conv3x3.weight, activation_constructor)
        init.conv(conv1x1_2.weight, activation_constructor)

        #ADD batch norms automatically
        self.ResLayers = nn.Sequential(conv1x1_1,
            activation_constructor(channel_small, not encode), nn.BatchNorm3d(channel_small), conv3x3,
            activation_constructor(channel_small, not encode), nn.BatchNorm3d(channel_small),
            conv1x1_2, activation_constructor(Cout, not encode))
            #nn.BatchNorm3d(Cin))

    def forward(self, x):
        h = self.ResLayers(x)
        if self.residual:
            h = h + x
        return h

class CBAMBlock(nn.Module):
    def __init__(self, ):
        super(CBAMBlock, self).__init__()

    def forward(self, x):
        h = self.block(x)
        x = self.cbam(h)
        return x + h

class CBAM_vanilla(CBAMBlock):
    def __init__(self, encode, activation_constructor, Cin, channel_small=None,
                    down_sf=4, Cout=None, residual=True):
        super(CBAM_vanilla, self).__init__()
        if Cout is None:
            Cout = Cin
        self.block = ResVanilla(encode, activation_constructor, Cin, channel_small,
                        down_sf, Cout, False)
        self.cbam = CBAM(encode, activation_constructor, Cout, reduction_ratio=down_sf,
                    pool_types=['avg', 'max'])
        print(Cout, Cin)
        print("Number of CBAM parameters:", sum(p.numel() for p in self.cbam.parameters()))

class CBAM_NeXt(CBAMBlock):
    def __init__(self, encode, activation_constructor, Cin, channel_small=None,
                    down_sf=4, Cout=None, residual=True):
        super(CBAM_NeXt, self).__init__()
        if Cout is None:
            Cout = Cin
        self.block = ResNextBlock(encode, activation_constructor, Cin, channel_small,
                        down_sf, Cout, False)
        self.cbam = CBAM(encode, activation_constructor, Cout, reduction_ratio=down_sf,
                    pool_types=['avg', 'max'])
        print("Number of CBAM parameters:", sum(p.numel() for p in self.cbam.parameters()))





class ResNeXt(nn.Module):
    """Full ResNext module from : arXiv:1611.05431v2
    """

    def __init__(self, encode, activation_constructor, Cin, cardinality, k, Cs,
                    Block=ResNextBlock, Cout=None, block_kwargs=None):
        super(ResNeXt, self).__init__()

        init_block = self.__init_block_helper(Block, encode, activation_constructor, Cin, Cs,
                                Cout, block_kwargs)
        if isinstance(init_block, ResNextBlock):
            if Cs:
                assert Cs == 4, "Cs must be 4 when block is ResNextBlock"
            Cs = 4 #fixed for ResNextBlock system
            assert k is None, "k should not be initialized for ResNextBlock"
        elif isinstance(init_block, (ResVanilla, CBAM_NeXt, CBAM_vanilla)):
            pass
        elif init_block.__class__.__name__ ==  "_DenseBlock":
            assert Cs is not None or block_kwargs.get("Csmall") is not None, \
                                    "Cs must be initlized by user if Block is _DenseBlock"
        else:
            raise NotImplementedError("Block must be in [ResNextBlock, ResVanilla, CBAM_NeXt, CBAM_vanilla]")

        blocks = nn.ModuleList([])
        for i in range(cardinality):
            resblock = self.__init_block_helper(Block, encode, activation_constructor,
                                    Cin, Cs, Cout, block_kwargs)
            blocks.append(resblock)
        self.resblocks = blocks
        self.ResNeXtBN = nn.BatchNorm3d(Cin)

    def forward(self, x):
        for idx, Block in enumerate(self.resblocks):
            if idx == 0:
                h = Block(x)
            else:
                h += Block(x)
        h = h
        h = self.ResNeXtBN(h)

        return h + x

    def __init_block_helper(self, Block, encode, activation_constructor, Cin, Cs,
                            Cout, block_kwargs):
        if block_kwargs is None:
            resblock = Block(encode, activation_constructor, Cin,
                            channel_small=Cs, Cout=Cout)
        else:
            resblock = Block(**block_kwargs)
        return resblock


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


class ResBlock1x1(nn.Module):

    def __init__(self, activation_fn, in_channels, out_channels):
        super(ResBlock1x1, self).__init__()
        self.act_fn = activation_fn

        self.conv1x1 = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1), stride=(1,1,1))
        self.conv1 = nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), stride=(1,1,1), padding=(1,1,1))
        self.conv2 = nn.Conv3d(out_channels, in_channels, kernel_size=(3, 3, 3), stride=(1,1,1), padding=(1,1,1))

    def forward(self, x):
        h = self.act_fn(self.conv1x1(x))
        h = self.act_fn(self.conv1(h))
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


