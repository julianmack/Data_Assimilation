from pipeline.settings.config import CAEConfig

class Baseline(CAEConfig):
    """This model is the baseline CAE in my report and is *similar* to
    the model described in:

    LOSSY IMAGE COMPRESSION WITH COMPRESSIVE AUTOENCODERS (2017), Theis et al.
    http://arxiv.org/abs/1703.00395


    Note: this class is not to be confused with the settings.Config() class
    which is the base cofiguration class """
    def __init__(self):
        super(Baseline, self).__init__()
        self.REDUCED_SPACE = True
        self.ACTIVATION = "lrelu"  #default negative_slope = 0.05
        self.CHANGEOVER_DEFAULT = 0
        #self.CHANGEOVERS = ?

        self.BATCH_NORM = False

        self.gen_channels()
#
# def gen_channels(self):
#     channels = [8 * self.CAE5_mult] * (self.get_num_layers_decode() + 1)
#     half_layers = int((self.get_num_layers_decode() + 1) / 2)
#     channels[half_layers:] = [16 * self.CAE5_mult] *  len(channels[half_layers:])
#     channels[-1] = 8 * self.latent_mult #This gives latent dim of 32 * latent_mult
#
#     channels[0] = 1
#     return channels