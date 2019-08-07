
from pipeline.AEs.AE_general import MODES as M
from pipeline.settings.block_base import Block

class BaselineBlock(Block):
    def __init__(self):
        super(BaselineBlock, self).__init__()
        self.BLOCKS = [M.S, (9, "conv")]
        self.DOWNSAMPLE = [0, 1, 0, 1, 0, 1, 0, 0, 0]

class Baseline2Block(Block):
    """Replica of Baseline2 w. Block build"""
    def __init__(self):
        super(Baseline2_replicate, self).__init__()
        self.BLOCKS = [M.S, (6, "conv")]
        down = [0, 1, 1, 1, 1, 1, ]
        down_z = [0, 1, 1, 1, 0, 0, ]
        self.DOWNSAMPLE = (down, down, down_z)
        self.get_channels()
        self.CHANNELS[1] = 16
        self.CHANNELS[2] = 32

class BaselineRes(Block):
    def __init__(self):
        super(BaselineRes, self).__init__()
        self.BLOCKS = [M.S, (7, "conv"), (2, "resB", {"C": 32}), (2, "conv")]
        self.DOWNSAMPLE__ = [[0, 1, 0, 1, 0, 1, 0], [], [0, 0]]

class BaselineResDown(Block):
    def __init__(self):
        super(BaselineResDown, self).__init__()
        self.BLOCKS = [M.S, (7, "conv"), (2, "resB1x1", {"I": 32, "O": 4}), (2, "conv")]
        self.DOWNSAMPLE__ = [[0, 1, 0, 1, 0, 1, 0], [], [0, 0]]

class BaselineResSlim(Block):
    def __init__(self):
        super(BaselineResSlim, self).__init__()
        self.BLOCKS = [M.S, (7, "conv"), (2, "resBslim", {"I": 32, "O": 4}), (2, "conv")]
        self.DOWNSAMPLE__ = [[0, 1, 0, 1, 0, 1, 0], [], [0, 0]]

class Res34AE(Block):
    def __init__(self):
        super(Res34AE, self).__init__()
        self.ACTIVATION = "lrelu"
        self.BLOCKS = [M.S, (8, "conv"), (13, "resB", {"C": 32})]

        self.DOWNSAMPLE = [0, 1, 0, 1, 0, 1, 0, 0] #downsample for conv
        #i.e. strides =   [1, 2, 1, 2, 1, 2, 1, 1]

class Res34AE_Stacked(Block):
    def __init__(self):
        super(Res34AE_Stacked, self).__init__()
        self.ACTIVATION = "lrelu"
        self.BLOCKS = [M.S, (8, "conv"), (4, "resB_3", {"C": 32})]

        self.DOWNSAMPLE = [0, 1, 0, 1, 0, 1, 0, 0] #downsample for conv
        #i.e. strides =   [1, 2, 1, 2, 1, 2, 1, 1]

class Cho2019(Block):
    """Initilaizes model as in paper:
            "Low Bit-rate Image Compression based on
            Post-processing with Grouped Residual Dense Network"
    Uses DRUs (Dense Residual Blocks)

    """
    def __init__(self):
        super(Cho2019, self).__init__()
        self.ACTIVATION = "lrelu"
        self.BLOCKS = [M.S, (8, "conv"), (4, "DRU", {"C": 32})]

        self.DOWNSAMPLE = [0, 1, 0, 1, 0, 1, 0, 0] #downsample for conv
        #i.e. strides =   [1, 2, 1, 2, 1, 2, 1, 1]

