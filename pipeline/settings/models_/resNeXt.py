from pipeline.settings.block_base import Block
from pipeline.AEs.AE_general import MODES as M


class Baseline1Block(Block):
    """Replica of Baseline1 w. Block build"""
    def __init__(self):
        super(Baseline1Block, self).__init__()
        self.ACTIVATION = "prelu"
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


class ResStack3(ResNeXt):
    def __init__(self, layers, cardinality, block_type="NeXt",
                    module_type="ResNeXt3", Csmall=None, k=None,
                    subBlock="vanilla", attenuation=True, sigmoid=None):
        #NOTE: the block refered to as an RNAB is actually a RAB by the
        #definition in http://arxiv.org/abs/1903.10082 so therefore:
        if block_type == "RNAB":
            block_type = "RAB"
        assert block_type in ["NeXt", "vanilla", "RAB", "CBAM_NeXt", "CBAM_vanilla"]

        if module_type in ["ResNeXt3", "RDB3"]:
            subBlock = None #this is not used
            assert layers % 3 == 0, "layers must be a multiple of three for `ResNeXt3` and `RDB3`"
        elif module_type in ["Bespoke"]:
            pass
        else:
            raise NotImplementedError("Only `ResNeXt3`, `RDB3` and `Bespoke` implemented for config ResStack3")
        super(ResStack3, self).__init__(layers, cardinality)
        kwargs = {"C": 32, "L": layers, "N": cardinality, "B":
                    block_type, "CS": Csmall, "k": k, "SB": subBlock,
                    "A": attenuation, "S": sigmoid}

        self.BLOCKS = [M.S, (5, "conv"), (1, module_type, kwargs), (2, "conv")]




