from torch import nn

class Empty(nn.Module):

    """Empty module - used for debug
    (i.e. printing and logging w/o needing to alter functionality)"""

    def __init__(self, b, c):
        super(Empty, self).__init__()
        self.b = b
        self.c = c
    def forward(self, x):
        ##################
        #Debug inside here:


        print(self.b, self.c, x.shape)


        ###################
        return x