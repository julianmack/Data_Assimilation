"""Configuration file for VarDA. Custom data structure that holds configuration options.

User can create new classes that inherit from class Config and override class variables
in order to create new combinations of config options. Alternatively, individual config
options can be altered one at a time on an ad-hoc basis."""

from pipeline.AutoEncoders import VanillaAE, ToyAE, CAE_3D
from pipeline import utils

import socket
import os, sys

class Config():

    def __init__(self):
        self.HOME_DIR = utils.get_home_dir()
        self.RESULTS_FP = self.HOME_DIR + "results/"
        self.DATA_FP = self.HOME_DIR + "data/small3DLSBU/"
        self.INTERMEDIATE_FP = self.HOME_DIR + "data/small3D_intermediate/"
        self.FIELD_NAME = "Pressure"
        self.X_FP = self.INTERMEDIATE_FP + "X_1D_{}.npy".format(self.FIELD_NAME)
        self.FORCE_GEN_X = False
        self.__n = 100040
        self.THREE_DIM = False # i.e. is representation in 3D tensor or 1D array
        self.SAVE = True
        self.DEBUG = True

        self.SEED = 42
        self.NORMALIZE = True #Whether to normalize input data
        self.UNDO_NORMALIZE = self.NORMALIZE

        #config options to divide up data between "History", "observation" and "control_state"
        #User is responsible for checking that these regions do not overlap
        self.HIST_FRAC = 4.0 / 5.0 #fraction of data used as "history"
        self.TDA_IDX_FROM_END = 0 #timestep index of u_c (control state from which
                            #observations are selcted). Value given as integer offset
                            #from final timestep (since number of historical timesteps M is
                            #not known by config file)
        self.OBS_MODE = "rand" #Observation mode: "single_max" or "rand" - i.e. use a single
                         # observation or a random subset
        self.OBS_FRAC = 0.5 # (with OBS_MODE=rand). fraction of state used as "observations".
                        # This is ignored when OBS_MODE = single_max

        #VarDA hyperparams
        self.ALPHA = 1.0
        self.OBS_VARIANCE = 0.01 #TODO - CHECK this is specific to the sensors (in this case - the error in model predictions)

        self.COMPRESSION_METHOD = "SVD" # "SVD"/"AE"
        self.NUMBER_MODES = 2 #Number of modes to retain.
            # If NUMBER_MODES = None (and COMPRESSION_METHOD = "SVD"), we use
            # the Rossella et al. method for selection of truncation parameter

        self.TOL = 1e-3 #Tolerance in VarDA minimization routine
        self.JAC_NOT_IMPLEM = True #whether explicit jacobian has been implemented
        self.export_env_vars()

        self.AZURE_STORAGE_ACCOUNT = "vtudata"
        self.AZURE_STORAGE_KEY = "yguh9jwnySH8FsHxX25rPOf4qE3wwdB6G+pqf/spwx/ofYHPBYHzAl32lx1swOETlqC3qorH1JKwJoWOWt4L4Q=="
        self.AZURE_CONTAINER = "x-data"
        self.AZURE_DOWNLOAD = True

    def get_n(self):
        return self.__n
    def set_n(self, n):
        self.__n = n
    def get_number_modes(self):
        return self.NUMBER_MODES

    def export_env_vars(self):
        self.env_vars = {"SEED": self.SEED}

        env = os.environ
        for k, v in self.env_vars.items():
            env[str(k)] = str(v)



class ConfigExample(Config):
    """Override and add relevant configuration options."""
    def __init__(self):
        super(ConfigExample, self).__init__()
        self.ALPHA = 2.0 #override
        self.NEW_OPTION = "FLAG" #Add new

class ConfigAE(Config):
    def __init__(self):
        super(ConfigAE, self).__init__()
        self.BATCH_NORM = False
        self.COMPRESSION_METHOD = "AE"
        self.NUMBER_MODES = 4
        #self.AE_MODEL_FP = self.HOME_DIR + "models/AE_dim{}_epoch120.pth".format(self.NUMBER_MODES) #AE_dim40_epoch120.pth"
        self.AE_MODEL_TYPE = VanillaAE #this must match
        self.HIDDEN = [1000, 200]
        self.ACTIVATION = "lrelu"
        #define getter for __kwargs since they may change after initialization
    def get_kwargs(self):
        return  {"input_size": self.get_n(), "latent_dim": self.NUMBER_MODES,
                "hidden":self.HIDDEN, "activation": self.ACTIVATION, "batch_norm": self.BATCH_NORM}

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

class CAEConfig(ConfigAE):
    def __init__(self):
        super(CAEConfig, self).__init__()
        self.AE_MODEL_TYPE = CAE_3D
        self.n3d = (91, 85, 32)
        self.FACTOR_INCREASE = 2.43 #interpolation ratio of oridinal # points to final
        self.__n = self.n3d #This overrides FACTOR_INCREASE
        self.CHANNELS = None
        self.THREE_DIM = True
        self.X_FP = self.INTERMEDIATE_FP + "X_3D_{}.npy".format(self.FIELD_NAME)
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
        init_data = utils.ML_utils.get_init_data_from_schedule(conv_data)
        channels = self.get_channels()
        latent_sz = self.__get_latent_sz(conv_data, channels)
        kwargs =   {"layer_data": init_data, "channels": channels,
                    "activation": self.ACTIVATION, "latent_sz": latent_sz,
                    "batch_norm": self.BATCH_NORM}
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

        return utils.ML_utils.conv_scheduler3D(self.get_n(), changeovers, 1, False, changeover_out_def=changeover_out_def )

    def get_channels(self):
        if self.CHANNELS != None and hasattr(self, "CHANNELS"):
            return self.CHANNELS
        elif hasattr(self, "gen_channels"): #gen random channels
            self.CHANNELS = self.gen_channels()
            return self.CHANNELS
        else:
            raise NotImplementedError("No default channel init")
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

class SmallTestDomain(Config):
    def __init__(self):
        super(SmallTestDomain, self).__init__()
        self.SAVE = False
        self.DEBUG = True
        self.NORMALIZE = True
        self.X_FP = self.INTERMEDIATE_FP + "X_small3D_{}_TINY.npy".format(self.FIELD_NAME)
        self.__n = 4
        self.OBS_FRAC = 0.3
        self.NUMBER_MODES = 3
