from VarDACAE.settings.models.resNeXt import Baseline1Block
from VarDACAE.AEs.AE_general import MODES as M
from VarDACAE import GetData

class CLIC(Baseline1Block):
    """Settings class for ResNext variants
    Args:
    """

    def __init__(self, model_name, block_type, Cstd, loader = None, sigmoid=None,
                    activation="prelu", aug_scheme=None):
        super(CLIC, self).__init__(loader)
        assert model_name in ["Tucodec"]
        assert block_type in ["vanilla", "NeXt", "CBAM_vanilla", "CBAM_NeXt",]
        #assert sigmoid is not None, "Comment out this line if you would like to overrule"

        self.BLOCKS = [M.S, (1, model_name, {"B": block_type,
                                            "Cstd": Cstd,
                                            "S": sigmoid,
                                            "A": activation,
                                            "AS": aug_scheme})]
        self.ACTIVATION = activation
        self.AUG_SCHEME = aug_scheme

        self.REM_FINAL = False
        self.CHANNELS = "see model def"

class GRDNBaseline(CLIC):
    def __init__(self, block_type, Cstd,  loader = None, activation="prelu", aug_scheme=None):
        super(CLIC, self).__init__()
        assert block_type in ["vanilla", "NeXt", "CBAM_vanilla", "CBAM_NeXt",]
        GRDN_kwargs = {"B": block_type, "Cstd": Cstd, "A": activation, "AS": aug_scheme}

        self.BLOCKS = [M.S, (1, "GRDN", GRDN_kwargs), (7, "conv")]
        down = [[], [0, 0, 1, 1, 1, 1, 1]]
        down_z = [[], [0, 0, 1, 1, 1, 0, 0]]
        self.DOWNSAMPLE__  = (down, down, down_z)
        channels  = self.get_channels()
        channels[0] = 1
        channels[1] = 1
        self.update_channels(channels)
        self.ACTIVATION = activation
        self.AUG_SCHEME = aug_scheme
