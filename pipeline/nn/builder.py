from torch import nn
from pipeline.nn import res_complex, res_simple
from collections import OrderedDict

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

        if not dropout and not batch_norm and final:
            return nn.Conv3d(**conv_kwargs) if encode else nn.ConvTranspose3d(**conv_kwargs)

        #else
        layer = OrderedDict()
        act_fn_constructor = NNBuilder.act_constr(activation)
        if dropout:
            #TODO - make dropout rate variable
            layer.update({"0": nn.Dropout3d(0.33)})
        if batch_norm:
            layer.update({"1": nn.BatchNorm3d(conv_kwargs["in_channels"])})
        if encode:
            layer.update({"2": nn.Conv3d(**conv_kwargs)})
        else:
            layer.update({"2": nn.ConvTranspose3d(**conv_kwargs)})
        if not final:
            layer.update({"3": act_fn_constructor(conv_kwargs["out_channels"])})
        conv = nn.Sequential(layer)

        return conv

    @staticmethod
    def conv1x1(D, I, final=False):
        channel_down = (channel // D) if (channel // D > 0) else 1
        module = nn.Conv3d(I, channel_down, kernel_size=(1, 1, 1), stride=(1,1,1))
        return NNBuilder.maybe_add_activation(module, act_fn_constructor, final, I)

    @staticmethod
    def ResNeXt(activation_fn, C, N, final=False):
        act_fn_constructor = NNBuilder.act_constr(activation_fn)
        module = res_complex.ResNeXt(act_fn_constructor, C, N)
        return NNBuilder.maybe_add_activation(module, act_fn_constructor, final, C)
    @staticmethod
    def resB(activation_fn, C, final=False):
        """Returns Residual block of structure:
        conv -> activation -> conv -> sum both conv.

        These enforce that Cin == Cout == C"""
        act_fn_constructor = NNBuilder.act_constr(activation_fn)
        module = res_simple.ResBlock(activation_fn, C)
        return NNBuilder.maybe_add_activation(module, act_fn_constructor, final, C)

    @staticmethod
    def resB_3(activation_fn, C, final=False):
        """Returns 3 stacked residual blocks each of structure:
            conv -> activation -> conv -> sum both conv.
        There is then a skip connection from first to output of stacked
        residual block as in 10.1109/CVPR.2018.00462

        Note: enforce that Cin == Cout == C"""
        act_fn_constructor = NNBuilder.act_constr(activation_fn)
        module =  res_simple.ResBlockStack3(activation_fn, C)
        return NNBuilder.maybe_add_activation(module, act_fn_constructor, final, C)

    @staticmethod
    def resB1x1(activation_fn, I, O, final=False):
        act_fn_constructor = NNBuilder.act_constr(activation_fn)
        module =  res_simple.ResBlock1x1(activation_fn, I, O)
        return NNBuilder.maybe_add_activation(module, act_fn_constructor, final, O)

    @staticmethod
    def resBslim(activation_fn, I, O, final=False):
        act_fn_constructor = NNBuilder.act_constr(activation_fn)
        module =  res_complex.ResBlockSlim(activation_fn, I, O)
        return NNBuilder.maybe_add_activation(module, act_fn_constructor, final, O)

    @staticmethod
    def DRU(activation_fn, C, final=False):
        """Returns A Dense Residual Unit

        Note: enforce that Cin == Cout == C"""
        act_fn_constructor = NNBuilder.act_constr(activation_fn)

        module =  res_complex.DRU(activation_fn, C)
        return NNBuilder.maybe_add_activation(module, act_fn_constructor, final, C)

    @staticmethod
    def act_constr(activation_fn):
        if  activation_fn == "relu":
            activation_constructor = lambda x: nn.ReLU()
        elif activation_fn == "lrelu":
            activation_constructor = lambda x: nn.LeakyReLU(0.05)
        elif callable(activation_fn):
            activation_constructor = lambda x: activation_fn
        elif activation_fn == "prelu": # must be initilalized in situ
            activation_constructor = nn.PReLU
        else:
            raise NotImplementedError("Activation function not implemented")
        return activation_constructor

    @staticmethod
    def maybe_add_activation(module, act_fn_constructor, final, C):
        if final:
            return module
        else:
            BN = nn.BatchNorm3d(C)
            return nn.Sequential(BN, module, act_fn_constructor(C))