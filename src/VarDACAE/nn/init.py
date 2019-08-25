from torch import nn



def conv(weight, activation_constructor):
    return # the changes below reduced the accuracy
    act = __get_activation(activation_constructor)

    if act == "prelu":
        a = 0.25
        nonlinearity = "leaky_relu"
    elif act == "lrelu":
        a = 0.05
        nonlinearity = "leaky_relu"
    elif act == "relu":
        a = 0
        nonlinearity = "relu"
    else:
        raise NotImplementedError("activation not implemented")

    nn.init.kaiming_uniform_(weight, a=a, nonlinearity=nonlinearity)

