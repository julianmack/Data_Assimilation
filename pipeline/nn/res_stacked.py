import torch
from torch import nn
from pipeline.nn import res
from pipeline.nn.densenet import _DenseBlock
from torch.nn.parameter import Parameter
from pipeline import ML_utils

class Res3(nn.Module):
    """Module that implements a skip connection
    over every third residual block"""
    def __init__(self, **kwargs):
        super(Res3, self).__init__()

    def forward(self, x):
        h = self.l1of3(x)
        h = self.l2of3(h)
        h = self.l3of3(h)
        return x + h

#res modules with an extra skip connection over every third:

class ResNeXt3(Res3):
    def __init__(self, encode, activation_constructor, Cin, cardinality, layers, Block, k, Cs, subBlock):
        super(ResNeXt3, self).__init__()

        self.l1of3 = res.ResNeXt(encode, activation_constructor, Cin, cardinality, k, Cs, Block)
        self.l2of3 = res.ResNeXt(encode, activation_constructor, Cin, cardinality, k, Cs, Block)
        self.l3of3 = res.ResNeXt(encode, activation_constructor, Cin, cardinality, k, Cs, Block)

class RBD3(nn.Module):
    """Creates an RDB3 module within the ResNeXt system.
    This is from the paper: https://arxiv.org/pdf/1608.06993.pdf
    with number_layers = 3 fixed
    """
    def __init__(self, encode, activation_constructor, Cin, cardinality, layers, Block,
                    k, Cs, subBlock):
        super(RBD3, self).__init__()
        if k is None:
            k = 16
        if Cs is None:
            Cs = 64
        assert subBlock is None

        dense_block_kwargs = { "encode": encode,
                                "activation_constructor": activation_constructor,
                                "Cin": Cin, "growth_rate": k,
                                "Csmall": Cs, "Block": Block,
                                "dense_layers": 3}
        Block_in_ResNext = _DenseBlock
        self.rdb3 = res.ResNeXt(encode, activation_constructor, Cin,
                                        cardinality, k, Cs, Block_in_ResNext,
                                        block_kwargs=dense_block_kwargs)
    def forward(self, x):
        h = self.rdb3(x)
        return x + h

class ResBespoke(nn.Module):
    """Adds a single bespoke module"""
    def __init__(self, encode, activation_constructor, Cin, cardinality, layers, Block, k, Cs, subBlock=None):
        super(ResBespoke, self).__init__()
        assert subBlock is not None

        self.res = Block(encode, activation_constructor, Cin, channel_small=Cs, Block=subBlock)
    def forward(self, x):
        h = self.res(x)
        return x + h

class resOver(nn.Module):
    """Adds a skip connection and attentuation over the whole module.
    Also add a final Batch Norm to prevent gradients/values exploding
    in resNext layers.

    This is an extension of res.resResNeXt - the original
    class is kept so that it is possible to load models trained under
    the old scheme
    """

    def __init__(self, encode, activation_constructor, Cin, cardinality, layers, block,
                    k, Csmall=None, module=ResNeXt3, subBlock=None, attentuation=True):
        super(resOver, self).__init__()
        blocks = []
        for i in range(layers):
            res = module(encode, activation_constructor, Cin, cardinality, layers,
                            block, k=k, Cs=Csmall, subBlock=subBlock)
            blocks.append(res)
        self.resRes = nn.Sequential(*blocks, activation_constructor(Cin, not encode),
                                    nn.BatchNorm3d(Cin))
        if attentuation:
            self.attenuate_res = Parameter(torch.tensor([0.05], requires_grad=True))
        else:
            self.attenuate_res = 1.


    def forward(self, x):
        h = self.resRes(x)
        h = h * self.attenuate_res #To give less importance to residual network (at least initially)
        return h + x



class resResNeXt(nn.Module):
    """Adds a skip connection over the whole ResNeXt module.
    Also add a final Batch Norm to prevent gradients/values exploding
    in resNext layers.
    """

    def __init__(self, encode, activation_constructor, Cin, cardinality, layers, Cout=None,
                    k=None, Cs=None):
        super(resResNeXt, self).__init__()
        blocks = []
        for i in range(layers):
            reslayer = res.ResNeXt(encode, activation_constructor, Cin, cardinality, Cout=Cout, k=k, Cs=Cs)
            blocks.append(reslayer)
        self.resRes = nn.Sequential(*blocks, activation_constructor(Cin, not encode),
                                    nn.BatchNorm3d(Cin))
        self.attenuate_res = Parameter(torch.tensor([0.05], requires_grad=True))


    def forward(self, x):
        h = self.resRes(x)
        h = h * self.attenuate_res #To give less importance to residual network (at least initially)
        return h + x

