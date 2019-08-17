from torch import nn
from pipeline.nn import res, res_stacked
from collections import OrderedDict
from pipeline.nn import init
from pipeline.nn.explore.empty import Empty
from pipeline.nn.RNAB import RNAB
from pipeline.nn.pytorch_gdn.gdn import GDN
from pipeline import ML_utils

class NNBuilder():
    """Class to build nn blocks"""

    def __init__(self):
        pass

    @staticmethod
    def conv(encode, activation, conv_kwargs, dropout, batch_norm, final=False):
        if not encode:
            # First must update conv_kwargs for decoder.
            # i.e. switch in_channels <--> out_channels
            Cin = conv_kwargs["in_channels"]
            Cout = conv_kwargs["out_channels"]
            conv_kwargs["in_channels"] = Cout
            conv_kwargs["out_channels"] = Cin

        act_fn_constructor = NNBuilder.act_constr(activation)

        if not dropout and not batch_norm and final:
            conv = nn.Conv3d(**conv_kwargs) if encode else nn.ConvTranspose3d(**conv_kwargs)
            init.conv(conv.weight, act_fn_constructor)
            return conv


        #else
        layer = OrderedDict()
        #layer.update({"00": Empty()})
        if dropout:
            #TODO - make dropout rate variable
            layer.update({"0": nn.Dropout3d(0.33)})
        if batch_norm:
            layer.update({"1": nn.BatchNorm3d(conv_kwargs["in_channels"])})

        if encode:
            conv = nn.Conv3d(**conv_kwargs)
        else:
            conv = nn.ConvTranspose3d(**conv_kwargs)
        init.conv(conv.weight, act_fn_constructor)
        layer.update({"2": conv})

        #layer.update({"2a": Empty()})
        if not final:
            layer.update({"3": act_fn_constructor(conv_kwargs["out_channels"], not encode)})

        conv = nn.Sequential(layer)

        return conv

    @staticmethod
    def conv1x1(encode, D, I, final=False):
        channel_down = (channel // D) if (channel // D > 0) else 1
        module = nn.Conv3d(I, channel_down, kernel_size=(1, 1, 1), stride=(1,1,1))
        return NNBuilder.maybe_add_activation(encode, module, act_fn_constructor, final, I)

    @staticmethod
    def ResNeXt(encode, activation_fn, C, N, final=False):
        act_fn_constructor = NNBuilder.act_constr(activation_fn)
        module = res.ResNeXt(encode, act_fn_constructor, C, N)
        return NNBuilder.maybe_add_activation(encode, module, act_fn_constructor, final, C)
    @staticmethod
    def resResNeXt(encode, activation_fn, C, N, L, final=False):
        if L < 1 or C < 1:
            return nn.Sequential()
        act_fn_constructor = NNBuilder.act_constr(activation_fn)
        module = res_stacked.resResNeXt(encode, act_fn_constructor, C, N, L)
        return module #No activation - this is already in the resNext

    @staticmethod
    def ResNeXt3(encode, activation_fn, C, N, L, B, CS, k, SB, final=False):
        if L < 1 or C < 1:
            return nn.Sequential()
        assert L % 3 == 0
        assert SB is None

        resOver_layers =  int(L / 3)
        block = NNBuilder.get_block(B)
        act_fn_constructor = NNBuilder.act_constr(activation_fn)
        module = res_stacked.resOver(encode, act_fn_constructor, C, N,
                            resOver_layers, block, k, CS,
                            res_stacked.ResNeXt3)
        return module

    def ResBespoke(encode, activation_fn, C, N, L, B, CS, k, SB, final=False):
        if L < 1:
            return nn.Sequential()
        block = NNBuilder.get_block(B)
        subBlock = NNBuilder.get_block(SB)

        act_fn_constructor = NNBuilder.act_constr(activation_fn)
        module = res_stacked.resOver(encode, act_fn_constructor, C, N,
                            L, block, k, CS, module=res_stacked.ResBespoke,
                            subBlock=subBlock)
        return module

    @staticmethod
    def ResNeXtRDB3(encode, activation_fn, C, N, L, B, CS, k, SB, final=False):
        if L < 1 or C < 1:
            return nn.Sequential()
        assert SB is None
        resOver_layers =  int(L / 3)
        block = NNBuilder.get_block(B)
        act_fn_constructor = NNBuilder.act_constr(activation_fn)
        module = res_stacked.resOver(encode, act_fn_constructor, C, N,
                            resOver_layers, block, k, CS,
                            res_stacked.RBD3)
        return module

    @staticmethod
    def resB(encode, activation_fn, C, final=False):
        """Returns Residual block of structure:
        conv -> activation -> conv -> sum both conv.

        These enforce that Cin == Cout == C"""
        act_fn_constructor = NNBuilder.act_constr(activation_fn)
        module = res.ResBlock(encode, activation_fn, C)
        return NNBuilder.maybe_add_activation(encode, module, act_fn_constructor, final, C)

    @staticmethod
    def resB_3(encode, activation_fn, C, final=False):
        """Returns 3 stacked residual blocks each of structure:
            conv -> activation -> conv -> sum both conv.
        There is then a skip connection from first to output of stacked
        residual block as in 10.1109/CVPR.2018.00462

        Note: enforce that Cin == Cout == C"""
        act_fn_constructor = NNBuilder.act_constr(activation_fn)
        module =  res.ResBlockStack3(encode, activation_fn, C)
        return NNBuilder.maybe_add_activation(encode, module, act_fn_constructor, final, C)

    @staticmethod
    def resB1x1(encode, activation_fn, I, O, final=False):
        act_fn_constructor = NNBuilder.act_constr(activation_fn)
        module =  res.ResBlock1x1(encode, activation_fn, I, O)
        return NNBuilder.maybe_add_activation(encode, module, act_fn_constructor, final, O)

    @staticmethod
    def resBslim(encode, activation_fn, I, O, final=False):
        act_fn_constructor = NNBuilder.act_constr(activation_fn)
        module =  res.ResBlockSlim(encode, activation_fn, I, O)
        return NNBuilder.maybe_add_activation(encode, module, act_fn_constructor, final, O)

    @staticmethod
    def DRU(encode, activation_fn, C, final=False):
        """Returns A Dense Residual Unit

        Note: enforce that Cin == Cout == C"""
        act_fn_constructor = NNBuilder.act_constr(activation_fn)

        module =  res.DRU(encode, activation_fn, C)
        return NNBuilder.maybe_add_activation(encode, module, act_fn_constructor, final, C)

    @staticmethod
    def get_block(block):
        assert isinstance(block, str)
        if block == "vanilla":
            return res.ResVanilla
        elif block == "NeXt":
            return res.ResNextBlock
        elif block == "CBAM_NeXt":
            return res.CBAM_NeXt
        elif block == "CBAM_vanilla":
            return res.CBAM_vanilla
        elif block == "RNAB":
            return RNAB
        else:
            raise ValueError("`block`={} is not in [vanilla, NeXt]".format(block))

    @staticmethod
    def act_constr(activation_fn):
        if  activation_fn == "relu":
            activation_constructor = lambda x, y: nn.ReLU()()
        elif activation_fn == "lrelu":
            activation_constructor = lambda x, y: nn.LeakyReLU(0.05)
        elif activation_fn == "GDN":
            activation_constructor = lambda x, y: GDN(x,  ML_utils.get_device(), y)
        elif callable(activation_fn):
            activation_constructor = lambda x, y: activation_fn
        elif activation_fn == "prelu": # must be initilalized in situ
            activation_constructor = lambda x, y: nn.PReLU(x)
        else:
            raise NotImplementedError("Activation function not implemented")
        return activation_constructor

    @staticmethod
    def maybe_add_activation(encode, module, act_fn_constructor, final, C):
        if final:
            return module
        else:
            return nn.Sequential(module, act_fn_constructor(C, not encode))
            # BN = nn.BatchNorm3d(C)
            #return nn.Sequential(BN, module, act_fn_constructor(C))


