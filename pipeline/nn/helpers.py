from torch import nn
from pipeline.nn.pytorch_gdn.gdn import GDN

def get_activation(activation_constructor):
    act_init = activation_constructor(1, True)

    act = None

    if isinstance(act_init, type(nn.PReLU())):
        act = "prelu"
    elif isinstance(act_init, type(nn.ReLU())):
        act = "relu"
    elif isinstance(act_init, type(nn.LeakyReLU(0.1))):
        act = "lrelu"
    elif isinstance(act_init, type(GDN(1, "cpu", False))):
        act = "GDN"
    else:
        raise NotImplementedError("Activation function not recognised")
    return act

