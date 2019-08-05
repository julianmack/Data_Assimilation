
from pipeline.AEs.AE_general import MODES as M
from pipeline.settings.block_base import Block
class BaselineBlock(Block):
    def __init__(self):
        super(BaselineBlock, self).__init__()
        self.BLOCKS = [M.S, (8, "conv")]
        self.DOWNSAMPLE = [0, 1, 0, 1, 0, 1, 0, 0]

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

