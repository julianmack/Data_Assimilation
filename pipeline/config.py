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
        self.HOME_DIR = self.__get_home_dir()
        self.RESULTS_FP = self.HOME_DIR + "results/"
        self.DATA_FP = self.HOME_DIR + "data/small3DLSBU/"
        self.INTERMEDIATE_FP = self.HOME_DIR + "data/small3D_intermediate/"
        self.FIELD_NAME = "Pressure"
        self.X_FP = self.INTERMEDIATE_FP + "X_small3D_{}.npy".format(self.FIELD_NAME)
        self.FORCE_GEN_X = False
        self.n = 100040
        self.SAVE = True
        self.DEBUG = True

        self.SEED = 42
        self.NORMALIZE = True #Whether to normalize input data
        self.UNDO_NORMALIZE = self.NORMALIZE

        #config options to divide up data between "History", "observation" and "control_state"
        #User is responsible for checking that these regions do not overlap
        self.HIST_FRAC = 2.0 / 4.0 #fraction of data used as "history"
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

        self.export_env_vars()

    def export_env_vars(self):
        self.env_vars = {"SEED": self.SEED}

        env = os.environ
        for k, v in self.env_vars.items():
            env[str(k)] = str(v)

    def __get_home_dir(self):
        wd = os.getcwd()
        if sys.platform[0:3] == 'win': #i.e. windows
            #replace the backslashes with forward slashes
            wd = wd.replace("\\", '/')
            wd = wd.replace("C:", "")
        wd += "/"
        return wd




class ConfigExample(Config):
    """Override and add relevant configuration options."""
    def __init__(self):
        super(ConfigExample, self).__init__()
        self.ALPHA = 2.0 #override
        self.NEW_OPTION = "FLAG" #Add new

class ConfigAE(Config):
    def __init__(self):
        super(ConfigAE, self).__init__()
        self.COMPRESSION_METHOD = "AE"
        self.NUMBER_MODES = 4
        self.AE_MODEL_FP = self.HOME_DIR + "models/AE_dim{}_epoch120.pth".format(self.NUMBER_MODES) #AE_dim40_epoch120.pth"
        self.AE_MODEL_TYPE = VanillaAE #this must match
        self.HIDDEN = [1000, 200]
        self.ACTIVATION = "lrelu"
        #define getter for __kwargs since they may change after initialization
    def get_kwargs(self):
        return  {"input_size": self.n, "latent_dim": self.NUMBER_MODES, "hidden":self.HIDDEN, "activation": self.ACTIVATION}

class ToyAEConfig(ConfigAE):
    def __init__(self):
        super(ToyAEConfig, self).__init__()
        self.NUMBER_MODES = 2
        self.HIDDEN = 32
        self.AE_MODEL_FP = self.HOME_DIR + "models/AE_toy_{}_{}_{}.pth".format(self.NUMBER_MODES, self.HIDDEN, self.FIELD_NAME)
        self.DEBUG = True
        self.AE_MODEL_TYPE = ToyAE
        self.ACTIVATION = "relu"

class CAEConfig(ConfigAE):
    def __init__(self):
        super(CAEConfig, self).__init__()
        self.NUMBER_MODES = 4
        self.AE_MODEL_FP = self.HOME_DIR + "models/CAE_toy_{}_{}_{}.pth".format(self.NUMBER_MODES, self.HIDDEN, self.FIELD_NAME)
        self.AE_MODEL_TYPE = CAE_3D
        self.FACTOR_INCREASE = 2.43 #interpolation ratio of oridinal # points to final
        self.n = self.get_n_3D()
        self.CHANNELS = None
        #define getter for __kwargs since they may change after initialization
    def get_kwargs(self):
        conv_data = self.get_conv_schedule()
        init_data = utils.ML_utils.get_init_data_from_schedule(conv_data)
        channels = self.get_channels()
        kwargs =   {"layer_data": init_data, "channels": channels, "activation": self.ACTIVATION}
        return kwargs

    def get_n_3D(self):
        return (91, 85, 32) #TODO - use self.FACTOR_INCREASE

    def get_conv_schedule(self):
        #TODO add self.CROSSOVER != None
        #TODO add self.lowest_out != None
        #TODO add self.MAX_Layers
        #TODO - give ability to set bespoke schedule
        return utils.ML_utils.conv_scheduler3D(self.n, None, 1, True)

    def get_channels(self):
        if self.CHANNELS != None:
            return self.CHANNELS
        else: #gen random channels
            n_layers_decode = len(self.get_conv_schedule()[0])
            channels = [2] * (n_layers_decode + 1)
            return channels

class SmallTestDomain(Config):
    def __init__(self):
        super(SmallTestDomain, self).__init__()
        self.SAVE = False
        self.DEBUG = True
        self.NORMALIZE = True
        self.X_FP = self.INTERMEDIATE_FP + "X_small3D_{}_TINY.npy".format(self.FIELD_NAME)
        self.n = 4
        self.OBS_FRAC = 0.3
        self.NUMBER_MODES = 3
