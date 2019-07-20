from pipeline.settings import config

class CAE7(config.CAEConfig):
    """CAE with default values that were most successful in testing.
    CAE5 gave the best results - as this had the widest number of channels
    of the CAE settings in CAE config I will now even wider systems"""
    def __init__(self):
        super(CAE7, self).__init__()
        self.ACTIVATION = "lrelu"
        self.CHANGEOVER_DEFAULT = 0
        self.BATCH_NORM = False

    def gen_channels(self):
        channels = [16] * (self.get_num_layers_decode() + 1)
        half_layers = int((self.get_num_layers_decode() + 1) / 2)
        channels[half_layers:] = [32] *  len(channels[half_layers:])
        channels[-1] = 8 #This gives latent dim of 32

        channels[0] = 1
        return channels