from torch import nn
from pipeline.nn.res_simple import ResBlock, ResBlockStack3, ResBlock1x1
from pipeline.nn.res_complex import DRU, ResBlockSlim

class NNBuilder():
    """Class to build nn blocks"""

    def __init__(self):
        pass

    @staticmethod
    def conv(encode, conv_kwargs, dropout, batch_norm, ):
        if not encode:
            # First must update conv_kwargs for decoder.
            # i.e. switch in_channels <--> out_channels
            Cin = conv_kwargs["in_channels"]
            Cout = conv_kwargs["out_channels"]
            conv_kwargs["in_channels"] = Cout
            conv_kwargs["out_channels"] = Cin

        if not dropout and not batch_norm:
            return nn.Conv3d(**conv_kwargs) if encode else nn.ConvTranspose3d(**conv_kwargs)

        #else
        layer = OrderedDict()
        if dropout:
            #TODO - make dropout rate variable
            layer.update({"0": nn.Dropout3d(0.33)})
        if batch_norm:
            layer.update({"1": nn.BatchNorm3d(conv_kwargs["in_channels"])})
        if encode:
            layer.update({"2": nn.Conv3d(**conv_kwargs)})
        else:
            layer.update({"2": nn.ConvTranspose3d(**conv_kwargs)})
        conv = nn.Sequential(layer)
        return conv

    @staticmethod
    def conv1x1(D, I):
        channel_down = (channel // D) if (channel // D > 0) else 1
        return nn.Conv3d(I, channel_down, kernel_size=(1, 1, 1), stride=(1,1,1))


    @staticmethod
    def resB(activation_fn, C):
        """Returns Residual block of structure:
        conv -> activation -> conv -> sum both conv.

        These enforce that Cin == Cout == C"""

        return ResBlock(activation_fn, C)

    @staticmethod
    def resB_3(activation_fn, C):
        """Returns 3 stacked residual blocks each of structure:
            conv -> activation -> conv -> sum both conv.
        There is then a skip connection from first to output of stacked
        residual block as in 10.1109/CVPR.2018.00462

        Note: enforce that Cin == Cout == C"""

        return ResBlockStack3(activation_fn, C)

    @staticmethod
    def resB1x1(activation_fn, I, O):

        return ResBlock1x1(activation_fn, I, O)
    @staticmethod
    def resBslim(activation_fn, I, O):

        return ResBlockSlim(activation_fn, I, O)

    @staticmethod
    def DRU(activation_fn, C):
        """Returns A Dense Residual Unit

        Note: enforce that Cin == Cout == C"""

        return DRU(activation_fn, C)

