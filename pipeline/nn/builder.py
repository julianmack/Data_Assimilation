from torch import nn

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
