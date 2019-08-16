from torch import nn

class Empty(nn.Module):

    """Empty module - used for debug
    (i.e. printing and logging w/o needing to alter functionality)"""

    def __init__(self):
        super(Empty, self).__init__()

    def forward(self, x):
        ##################
        #Debug inside here:

        print(x[0,0,0,0])
        print(x.shape)


        ###################
        return x