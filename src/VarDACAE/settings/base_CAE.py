"""The base classes here are not actually used to init the final models.
 The functionality herein was replaced by base_block as
 initializing models with CAEConfig() had started to get quite messy. """

from VarDACAE.AEs import VanillaAE, ToyAE, CAE_3D
from VarDACAE.settings.base import Config
from VarDACAE import ML_utils

class ConfigAE(Config):
    def __init__(self, loader=None):
        super(ConfigAE, self).__init__(loader)
        self.BATCH_NORM = False
        self.COMPRESSION_METHOD = "AE"
        self.NUMBER_MODES = 4
        #self.AE_MODEL_FP = self.HOME_DIR + "models/AE_dim{}_epoch120.pth".format(self.NUMBER_MODES) #AE_dim40_epoch120.pth"
        self.AE_MODEL_TYPE = VanillaAE #this must match
        self.HIDDEN = [1000, 200]
        self.ACTIVATION = "lrelu"
        self.DROPOUT = False

        #define getter for __kwargs since they may change after initialization
    def get_kwargs(self):
        return  {"input_size": self.get_n(), "latent_dim": self.NUMBER_MODES,
                "hidden":self.HIDDEN, "activation": self.ACTIVATION, "batch_norm": self.BATCH_NORM}



class CAEConfig(ConfigAE):
    def __init__(self, loader=None):
        super(CAEConfig, self).__init__(loader)
        self.AE_MODEL_TYPE = CAE_3D
        self.n3d = (91, 85, 32)
        self.FACTOR_INCREASE = 2.43 #interpolation ratio of oridinal # points to final
        self.__n = self.n3d #This overrides FACTOR_INCREASE
        self.CHANNELS = None
        self.THREE_DIM = True
        model_name  = self.__class__.__name__
        #self.AE_MODEL_FP = self.HOME_DIR + "models/{}_{}.pth".format(model_name, self.NUMBER_MODES)
        self.SAVE = True


        #define getter for __kwargs since they may change after initialization
    def get_num_layers_decode(self):
        return len(self.get_conv_schedule()[0])

    def get_number_modes(self):
        modes = self.calc_modes()
        return modes

    def get_kwargs(self):
        conv_data = self.get_conv_schedule()
        init_data = ML_utils.ConvScheduler.get_init_data_from_schedule(conv_data)
        channels = self.get_channels()
        latent_sz = self.__get_latent_sz(conv_data, channels)
        if hasattr(self, "DROPOUT"):
            dropout = self.DROPOUT
        else:
            dropout = False

        kwargs =   {"layer_data": init_data,
                    "channels": channels,
                    "activation": self.ACTIVATION,
                    "latent_sz": latent_sz,
                    "batch_norm": self.BATCH_NORM,
                    "dropout": dropout}
        return kwargs

    def get_n(self):
        return self.get_n_3D()
    def set_n(self, n):
        self.n3d = n
    def get_n_3D(self):
        return self.n3d

    def get_conv_schedule(self):
        if hasattr(self, "CHANGEOVERS"):
            changeovers = self.CHANGEOVERS
        else:
            changeovers = None
        if hasattr(self, "CHANGEOVER_DEFAULT"):
            changeover_out_def = self.CHANGEOVER_DEFAULT
        else:
            changeover_out_def = 10
        return ML_utils.ConvScheduler.conv_scheduler3D(self.get_n(), changeovers, 1, False, changeover_out_def=changeover_out_def )

    def gen_channels(self):
        channels = [8] * (self.get_num_layers_decode() + 1)
        channels[0] = 1
        return channels

    def calc_modes(self):
        #lantent dim is Channels_latent * (x_size_latent ) x (y_size_latent ) x (z_size_latent )
        [x_data, y_data, z_data] = self.get_conv_schedule()
        return self.get_channels()[-1] * x_data[-1]["out"] * y_data[-1]["out"] * z_data[-1]["out"]
    def __get_latent_sz(self, conv_data, channels):
        Cout = channels[-1]
        latent_sz = (Cout, )
        for dim_data in conv_data:
            final_layer = dim_data[-1]
            out = final_layer["out"]
            latent_sz = latent_sz + (out, )
        return latent_sz



class ToyAEConfig(ConfigAE):
    def __init__(self):
        super(ToyAEConfig, self).__init__()
        self.JAC_NOT_IMPLEM = False
        self.NUMBER_MODES = 3
        self.HIDDEN = 4
        #self.AE_MODEL_FP = self.HOME_DIR + "models/AE_toy_{}_{}_{}.pth".format(self.NUMBER_MODES, self.HIDDEN, self.FIELD_NAME)
        self.DEBUG = True
        self.AE_MODEL_TYPE = ToyAE
        self.ACTIVATION = "relu"

    def get_number_modes(self):
        return self.NUMBER_MODES