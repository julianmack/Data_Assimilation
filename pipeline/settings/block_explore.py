from pipeline.settings.config import Config3D
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
        conv1 = conv_kwargs.copy()
        conv2 = conv_kwargs.copy()
        conv1["in_channels"] = 1
        conv1["out_channels"] = 56
        conv2["in_channels"] = 56
        conv2["out_channels"] = 24
        kwargs_ls = [conv1, conv2]

        blocks = [M.S, (2, "conv", kwargs_ls)]

        latent_sz = None

        kwargs =   {"blocks": blocks,
                    "activation": self.ACTIVATION,
                    "latent_sz": latent_sz,
                    "batch_norm": self.BATCH_NORM,
                    "dropout": self.DROPOUT}
        return kwargs