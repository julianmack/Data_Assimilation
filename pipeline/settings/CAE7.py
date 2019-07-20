from pipeline.settings import config

class CAE7(config.CAEConfig):
    """CAE with default values that were most successful in testing.
    CAE5 gave the best results - as this had the widest number of channels
    of the CAE settings in CAE config I will now even wider systems"""
    def __init__(self, CAE5_mult=2, latent_mult=1):
        super(CAE7, self).__init__()
        self.ACTIVATION = "lrelu"
        self.CHANGEOVER_DEFAULT = 0
        self.BATCH_NORM = False
        self.CAE5_mult = CAE5_mult
        self.latent_mult = latent_mult

    def gen_channels(self):
        channels = [8 * self.CAE5_mult] * (self.get_num_layers_decode() + 1)
        half_layers = int((self.get_num_layers_decode() + 1) / 2)
        channels[half_layers:] = [16 * self.CAE5_mult] *  len(channels[half_layers:])
        channels[-1] = 8 * self.latent_mult #This gives latent dim of 32 * latent_mult

        channels[0] = 1
        return channels