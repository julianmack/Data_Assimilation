from pipeline.settings.block_base import Block
from pipeline.AEs.AE_general import MODES as M


class Baseline1Block(Block):
    """Replica of Baseline1 w. Block build"""
    def __init__(self):
        super(Baseline1Block, self).__init__()
        self.ACTIVATION = "lrelu"
        self.BLOCKS = [M.S, (7, "conv")]
        down = [0, 0, 1, 1, 1, 1, 1]
        down_z = [0, 0, 1, 1, 1, 0, 0]
        self.DOWNSAMPLE = (down, down, down_z)
        self.get_channels()
        self.CHANNELS[1] = 1
        self.CHANNELS[2] = 16
        #self.DOWNSAMPLE = down_z


class ResNeXt(Baseline1Block):
    """Settings class for ResNext variants
    Args:
        layers - number of layers of ResNeXt
        cardinality - width of each layer (in terms of number of res blocks)
    """

    def __init__(self, layers, cardinality):
        super(ResNeXt, self).__init__()
        kwargs = {"C": 32, "L": layers, "N": cardinality}
        self.BLOCKS = [M.S, (5, "conv"), (1, "resResNeXt", kwargs), (2, "conv")]
        down = [[0, 0, 1, 1, 1,], [], [1, 1]]
        down_z = [[0, 0, 1, 1, 1,], [], [0, 0]]
        self.DOWNSAMPLE__  = (down, down, down_z)

