from pipeline.settings.config import CAEConfig
from pipeline.AEs.AE_general import GenCAE
from pipeline.AEs.AE_general import MODES as M

class Block(Config3D):
    """This model is the baseline CAE in my report and is *similar* to
    the model described in:

    LOSSY IMAGE COMPRESSION WITH COMPRESSIVE AUTOENCODERS (2017), Theis et al.
    http://arxiv.org/abs/1703.00395


    Note: this class is not to be confused with the settings.Config() class
    which is the base cofiguration class """
    def __init__(self):
        super(Block, self).__init__()
        self.REDUCED_SPACE = True
        self.COMPRESSION_METHOD = "AE"
        self.AE_MODEL_TYPE = GenCAE

        self.ACTIVATION = "lrelu"  #default negative_slope = 0.05
        self.BATCH_NORM = False
        self.AUGMENTATION = False
        self.DROPOUT = False


    def get_kwargs(self):
        kernel = (3, 3, 3)
        padding = None
        stride = (2, 2, 2)
        conv_kwargs = {"kernel_size": kernel,
                     "padding": padding,
                     "stride": stride}

        blocks = [M.S, (2, "conv", conv_kwargs)]

        latent_sz = None

        kwargs =   {"blocks": blocks,
                    "activation": self.ACTIVATION,
                    "latent_sz": latent_sz,
                    "batch_norm": self.BATCH_NORM,
                    "dropout": self.DROPOUT}
        return kwargs