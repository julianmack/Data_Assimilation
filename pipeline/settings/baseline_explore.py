from pipeline.settings.config import CAEConfig

class Baseline1(CAEConfig):
    """This model is the baseline CAE in my report and is *similar* to
    the model described in:

    LOSSY IMAGE COMPRESSION WITH COMPRESSIVE AUTOENCODERS (2017), Theis et al.
    http://arxiv.org/abs/1703.00395


    Note: this class is not to be confused with the settings.Config() class
    which is the base cofiguration class """
    def __init__(self):
        super(Baseline1, self).__init__()
        self.REDUCED_SPACE = True
        self.ACTIVATION = "lrelu"  #default negative_slope = 0.05
        self.CHANGEOVER_DEFAULT = 2
        self.DEBUG = False

        self.BATCH_NORM = True
        self.AUGMENTATION = True
        self.DROPOUT = True

        self.get_channels()

    def gen_channels(self):
        num_layers_dec = self.get_num_layers_decode()
        idx_half = int((num_layers_dec + 1) / 2)

        channels = [64] * (num_layers_dec+ 1)
        channels[idx_half:] = [32] *  len(channels[idx_half:])

        #update bespoke vals

        channels[0] = 1
        channels[1] = 1
        channels[2] = 16
        return channels

class Baseline2(CAEConfig):
    """This model is the baseline CAE in my report and is *similar* to
    the model described in:

    LOSSY IMAGE COMPRESSION WITH COMPRESSIVE AUTOENCODERS (2017), Theis et al.
    http://arxiv.org/abs/1703.00395


    Note: this class is not to be confused with the settings.Config() class
    which is the base cofiguration class """
    def __init__(self):
        super(Baseline2, self).__init__()
        self.REDUCED_SPACE = True
        self.ACTIVATION = "lrelu"  #default negative_slope = 0.05
        self.CHANGEOVER_DEFAULT = 0
        self.DEBUG = False

        self.BATCH_NORM = True
        self.AUGMENTATION = True
        self.DROPOUT = True

        self.get_channels()

    def gen_channels(self):
        num_layers_dec = self.get_num_layers_decode()
        idx_half = int((num_layers_dec + 1) / 2)

        channels = [64] * (num_layers_dec+ 1)
        channels[idx_half:] = [32] *  len(channels[idx_half:])

        #update bespoke vals

        channels[0] = 1
        channels[1] = 16
        channels[2] = 32
        return channels