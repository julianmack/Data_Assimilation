from pipeline.settings.models_.resNeXt import Baseline1Block
from pipeline.AEs.AE_general import MODES as M


class CLIC(Baseline1Block):
    """Settings class for ResNext variants
    Args:
        layers - number of layers of ResNeXt
        cardinality - width of each layer (in terms of number of res blocks)
    """

    def __init__(self, model_name, block_type):
        super(CLIC, self).__init__()
        assert model_name in ["Tucodec"]
        assert block_type in ["vanilla", "NeXt"]
        self.BLOCKS = [M.S, (1, model_name, {"B": block_type})]
        self.REM_FINAL = False